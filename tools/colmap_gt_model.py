#!/usr/bin/env python3
"""
placecell — colmap_gt_model.py

Author:  Alejandro Fontan
Assisted by: Claude (Opus 5.5)
Created: 2026-09-30
License: Apache-2.0

A COLMAP text model built from a COLMAP database's verified matches and the ground truth
of a synthetic VSLAM-LAB sequence (perfect poses and depths, e.g. REPLICA), in place of
COLMAP's own reconstruction, so tools/colmap_information_kernel.py computes the
shared-information kernel on the true geometry.

  1. poses and depths: groundtruth.csv (TUM, world-from-camera) at each image's timestamp,
     converted to COLMAP's camera-from-world; every keypoint back-projected with its
     ground-truth z-depth (depth_0 PNG / depth_factor, nearest pixel);
  2. verification: the median transfer error of sampled verified matches (a keypoint's
     back-projection reprojected into the matching image) must be below --max-reproj,
     otherwise the pose convention or the depth type is wrong and nothing is written;
     --pixel-offset auto picks 0 or 0.5 here;
  3. matches: the verified inlier matches of two_view_geometries (COLMAP's WATERMARK pairs
     excluded) that the ground truth confirms: every side with a depth transfers into the
     other within --max-reproj. Filtering BEFORE the union-find matters: a few false
     matches otherwise chain unrelated tracks into giant ones;
  4. tracks and points: union-find over the confirmed matches; a track's point is the
     median of its depth-backed back-projections. An observation is dropped when it
     reprojects more than --max-reproj px from the point or its depth disagrees by more
     than --max-depth-rel (two passes); of two keypoints of one image in a track the one
     closer to the point is kept; a track needs >= 2 surviving observations in different
     images;
  5. output: cameras.txt / images.txt / points3D.txt in --out (default
     <database_dir>/gt_model). images.txt lists every keypoint, -1 for those not in a
     surviving track, like COLMAP's export.

Images are matched by file name against the sequence's rgb.csv (path_rgb_0), which gives
their timestamp and depth file. --pixel-offset is subtracted from the keypoints to reach the
calibration's pixel convention (COLMAP puts pixel centres at +0.5, but on REPLICA an offset
of 0 fits better, 0.30 vs 0.40 px), hence auto. Intrinsics
and depth_factor come from calibration.yaml; the camera written is the database's.
Pillow (16-bit PNG) comes with matplotlib in the default pixi environment.

Usage:
    colmap_gt_model.py <colmap_database.db> --sequence <VSLAM-LAB-Benchmark/<DATASET>/<sequence>>
        [--out dir] [--max-reproj 2.0] [--max-depth-rel 0.02] [--pixel-offset auto|0|0.5]

Exit code 1 when an input is missing or the verification fails.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

MAX_IMAGE_ID = 2147483647                    # COLMAP's pair_id = id1 * MAX + id2 (id1 < id2)
VALID_CONFIGS = {2, 3, 4, 5, 6}              # CALIBRATED, UNCALIBRATED, PLANAR, PANORAMIC, PLANAR_OR_PANORAMIC
CAMERA_MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------

def read_calibration(path: Path) -> dict:
    """fx, fy, cx, cy, width, height, depth_factor of the first camera of a VSLAM-LAB calibration.yaml."""
    text = open(path).read()

    def pair(key: str) -> tuple[float, float]:
        m = re.search(key + r"\s*:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]", text)
        if not m:
            raise ValueError(f"{path}: no {key}")
        return float(m.group(1)), float(m.group(2))

    fx, fy = pair("focal_length")
    cx, cy = pair("principal_point")
    w, h = pair("image_dimension")
    m = re.search(r"depth_factor\s*:\s*([-\d.eE+]+)", text)
    if not m:
        raise ValueError(f"{path}: no depth_factor (not an RGB-D sequence?)")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h), depth_factor=float(m.group(1)))


def read_sequence(sequence: Path) -> tuple[dict, dict]:
    """rgb.csv rows by image file name (timestamp, depth path) and ground-truth poses by timestamp."""
    frames = {}
    with open(sequence / "rgb.csv", newline="") as f:
        reader = csv.DictReader(f)
        ts_col = next(c for c in reader.fieldnames if c.startswith("ts_rgb_0"))
        depth_col = "path_depth_0" if "path_depth_0" in reader.fieldnames else None
        if depth_col is None:
            raise ValueError(f"{sequence / 'rgb.csv'}: no path_depth_0 column")
        for row in reader:
            frames[os.path.basename(row["path_rgb_0"])] = (int(row[ts_col]), row[depth_col])
    poses = {}
    with open(sequence / "groundtruth.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                poses[int(row[0])] = np.array(list(map(float, row[1:8])))   # tx ty tz qx qy qz qw
    return frames, poses


def read_database(path: Path):
    """images {image_id: name}, keypoints {image_id: (n,2) xy}, the camera (model, w, h, params),
    and the verified inlier matches as (image_id1, image_id2, (m,2) keypoint indices)."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    images = dict(db.execute("SELECT image_id, name FROM images"))
    keypoints = {}
    for image_id, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        keypoints[image_id] = (np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2].astype(np.float64)
                               if rows else np.zeros((0, 2)))
    cameras = list(db.execute("SELECT model, width, height, params FROM cameras"))
    if len(cameras) != 1:
        raise ValueError(f"{path}: {len(cameras)} cameras; one shared camera is supported")
    model, width, height, params = cameras[0]
    camera = (CAMERA_MODEL_NAMES.get(model, str(model)), width, height, np.frombuffer(params, np.float64).copy())
    matches = []
    for pair_id, rows, cols, data, config in db.execute(
            "SELECT pair_id, rows, cols, data, config FROM two_view_geometries WHERE rows > 0"):
        if config not in VALID_CONFIGS:
            continue
        id2 = pair_id % MAX_IMAGE_ID
        id1 = (pair_id - id2) // MAX_IMAGE_ID
        matches.append((int(id1), int(id2), np.frombuffer(data, np.uint32).reshape(rows, cols)))
    return images, keypoints, camera, matches


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------

def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """(qw, qx, qy, qz), qw >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def connected_components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Label of every node (the smallest node id of its component) for the edges (a, b):
    min-label propagation along the edges plus pointer jumping, vectorised."""
    labels = np.arange(n, dtype=np.int64)
    while True:
        m = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        while True:                          # pointer jumping: label <- label of label
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, labels):
            return labels
        labels = new


def group_median(keys: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group median of `values` (m, d) for integer group `keys` in [0, n_groups)."""
    out = np.full((n_groups, values.shape[1]), np.nan)
    for d in range(values.shape[1]):
        order = np.lexsort((values[:, d], keys))
        k, v = keys[order], values[order, d]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        lengths = np.diff(np.r_[starts, len(k)])
        out[k[starts], d] = 0.5 * (v[starts + (lengths - 1) // 2] + v[starts + lengths // 2])
    return out


# ---------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("database", type=Path, help="COLMAP database (keypoints + verified matches)")
    ap.add_argument("--sequence", type=Path, required=True,
                    help="VSLAM-LAB sequence folder: rgb.csv, groundtruth.csv, calibration.yaml, depth_0/")
    ap.add_argument("--out", type=Path, default=None, help="output folder [<database_dir>/gt_model]")
    ap.add_argument("--max-reproj", type=float, default=2.0, help="max reprojection error in px [2.0]")
    ap.add_argument("--max-depth-rel", type=float, default=0.02, help="max relative depth disagreement [0.02]")
    ap.add_argument("--pixel-offset", default="auto",
                    help="subtracted from COLMAP keypoints to reach the calibration's pixel convention; "
                         "auto tries 0 and 0.5 on sampled matches and keeps the better [auto]")
    args = ap.parse_args()
    t0 = time.time()

    for p in (args.database, args.sequence / "rgb.csv", args.sequence / "groundtruth.csv", args.sequence / "calibration.yaml"):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1
    calib = read_calibration(args.sequence / "calibration.yaml")
    frames, gt = read_sequence(args.sequence)
    images, keypoints, camera, matches = read_database(args.database)
    print(f"database: {len(images)} images, {sum(len(k) for k in keypoints.values())} keypoints, "
          f"{len(matches)} verified pairs, {sum(len(m[2]) for m in matches)} inlier matches")

    # ---- images: ground-truth pose (camera-from-world) and depth image ---------------------
    image_ids = sorted(images)
    index_of = {image_id: k for k, image_id in enumerate(image_ids)}
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    R_all = np.zeros((len(image_ids), 3, 3))
    t_all = np.zeros((len(image_ids), 3))
    depth_images = []
    for k, image_id in enumerate(image_ids):
        name = os.path.basename(images[image_id])
        if name not in frames:
            print(f"ERROR: database image {name} is not in {args.sequence / 'rgb.csv'}", file=sys.stderr)
            return 1
        ts, depth_path = frames[name]
        if ts not in gt:
            print(f"ERROR: no ground-truth pose at timestamp {ts} ({name})", file=sys.stderr)
            return 1
        tx, ty, tz, qx, qy, qz, qw = gt[ts]
        R_wc = quaternion_to_rotation(qx, qy, qz, qw)
        R_all[k] = R_wc.T
        t_all[k] = -R_wc.T @ np.array([tx, ty, tz])
        depth_images.append(np.asarray(Image.open(args.sequence / depth_path), dtype=np.float64) / calib["depth_factor"])
    print(f"poses and depths for {len(image_ids)} images ({time.time() - t0:.1f}s)")

    # ---- nodes: every keypoint, with its image, pixel, depth and back-projection -----------
    counts = np.array([len(keypoints[i]) for i in image_ids])
    offset = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_nodes = int(counts.sum())
    node_img = np.repeat(np.arange(len(image_ids)), counts)          # index into image_ids
    node_kp = np.concatenate([np.arange(c) for c in counts])
    node_raw = np.concatenate([keypoints[i] for i in image_ids])      # COLMAP convention
    edge_a = np.concatenate([offset[index_of[i1]] + m[:, 0].astype(np.int64) for i1, i2, m in matches])
    edge_b = np.concatenate([offset[index_of[i2]] + m[:, 1].astype(np.int64) for i1, i2, m in matches])

    def nodes_at(pixel_offset: float):
        xy = node_raw - pixel_offset
        z = np.zeros(n_nodes)
        for k, D in enumerate(depth_images):
            sel = slice(offset[k], offset[k] + counts[k])
            u, v = xy[sel, 0], xy[sel, 1]
            inside = (u > -0.5) & (u < D.shape[1] - 0.5) & (v > -0.5) & (v < D.shape[0] - 0.5)
            ui = np.clip(np.rint(u).astype(int), 0, D.shape[1] - 1)
            vi = np.clip(np.rint(v).astype(int), 0, D.shape[0] - 1)
            z[sel] = np.where(inside, D[vi, ui], 0.0)
        Xc = np.stack([(xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy, np.ones(n_nodes)], axis=1) * z[:, None]
        Xw = np.einsum("nji,nj->ni", R_all[node_img], Xc - t_all[node_img])   # R^T (Xc - t)
        return xy, z, Xw

    def project(Xw: np.ndarray, img: np.ndarray):
        Pc = np.einsum("nij,nj->ni", R_all[img], Xw) + t_all[img]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([fx * Pc[:, 0] / Pc[:, 2] + cx, fy * Pc[:, 1] / Pc[:, 2] + cy], axis=1)
        return uv, Pc[:, 2]

    def transfer_error(xy, z, Xw, src, dst):
        """Reprojection error of src's back-projection in dst's image (inf without depth or behind)."""
        uv, zc = project(Xw[src], node_img[dst])
        err = np.linalg.norm(uv - xy[dst], axis=1)
        err[(z[src] <= 0) | ~(zc > 0)] = np.inf
        return err

    # ---- conventions: pixel offset, and the pose / depth sanity check on sampled matches ----
    rng = np.random.default_rng(0)
    sample = rng.choice(len(edge_a), size=min(200_000, len(edge_a)), replace=False)
    offsets = [0.0, 0.5] if args.pixel_offset == "auto" else [float(args.pixel_offset)]
    medians = {}
    for off in offsets:
        xy, z, Xw = nodes_at(off)
        e = transfer_error(xy, z, Xw, edge_a[sample], edge_b[sample])
        medians[off] = float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else np.inf
    pixel_offset = min(medians, key=medians.get)
    print("pairwise transfer error of sampled matches, median: "
          + ", ".join(f"offset {o}: {m:.3f} px" for o, m in medians.items()) + f" -> offset {pixel_offset}")
    if not medians[pixel_offset] < args.max_reproj:
        print(f"ERROR: the median transfer error of the verified matches is {medians[pixel_offset]:.2f} px "
              f"(> {args.max_reproj}): the pose convention (world-from-camera expected) or the depth type "
              f"(z-depth expected) is wrong; nothing written", file=sys.stderr)
        return 1
    xy, z, Xw = nodes_at(pixel_offset)
    has_depth = z > 0

    # ---- edges: keep the matches the ground truth confirms ---------------------------------
    # A match passes when every side that has a depth transfers into the other within
    # --max-reproj; a match with no depth on either side cannot be checked and is dropped.
    edge_ok = np.zeros(len(edge_a), bool)
    for s in range(0, len(edge_a), 1_000_000):
        a, b = edge_a[s:s + 1_000_000], edge_b[s:s + 1_000_000]
        e_ab = transfer_error(xy, z, Xw, a, b)
        e_ba = transfer_error(xy, z, Xw, b, a)
        da, db_ = has_depth[a], has_depth[b]
        ok = (da | db_) & (~da | (e_ab <= args.max_reproj)) & (~db_ | (e_ba <= args.max_reproj))
        edge_ok[s:s + 1_000_000] = ok
    print(f"matches confirmed by the ground truth: {int(edge_ok.sum())} of {len(edge_a)} "
          f"({100.0 * edge_ok.mean():.1f}%) ({time.time() - t0:.1f}s)")

    # ---- tracks: union-find over the confirmed matches -------------------------------------
    ea, eb = edge_a[edge_ok], edge_b[edge_ok]
    labels = connected_components(n_nodes, ea, eb)
    in_edge = np.zeros(n_nodes, bool)
    in_edge[ea] = True
    in_edge[eb] = True
    obs = np.flatnonzero(in_edge)
    _, track = np.unique(labels[obs], return_inverse=True)
    n_tracks_raw = int(track.max()) + 1 if len(track) else 0
    o_img, o_xy, o_z, o_Xw, o_depth = node_img[obs], xy[obs], z[obs], Xw[obs], has_depth[obs]
    print(f"tracks: {n_tracks_raw} from {len(obs)} keypoints ({time.time() - t0:.1f}s)")

    # ---- points: median back-projection, then filter (two passes) --------------------------
    # Pass 1 takes a track's point from all its depth-backed observations, pass 2 from those
    # that passed pass 1. An observation passes when it reprojects within --max-reproj of the
    # point and, with a depth, agrees with it within --max-depth-rel; one without depth can
    # pass too (it then enters the track, not the point).
    use = o_depth.copy()
    for it in range(2):
        point = group_median(track[use], o_Xw[use], n_tracks_raw)
        has_point = ~np.isnan(point[track]).any(axis=1)
        uv, zc = project(np.nan_to_num(point[track]), o_img)
        err = np.linalg.norm(uv - o_xy, axis=1)
        err[~has_point | ~(zc > 0)] = np.inf
        rel = np.where(o_depth, np.abs(zc - o_z) / np.maximum(o_z, 1e-9), 0.0)
        bad_reproj = err > args.max_reproj
        bad_depth = (rel > args.max_depth_rel) & ~bad_reproj
        passed = ~bad_reproj & ~bad_depth
        use = o_depth & passed
    dropped_reproj = int(bad_reproj.sum())
    dropped_depth = int(bad_depth.sum())
    point = group_median(track[use], o_Xw[use], n_tracks_raw)
    uv, _ = project(np.nan_to_num(point[track]), o_img)
    err_all = np.linalg.norm(uv - o_xy, axis=1)
    keep = passed.copy()

    # one keypoint per image per track: the one closer to the point
    k_idx = np.flatnonzero(keep)
    order = np.lexsort((err_all[k_idx], o_img[k_idx], track[k_idx]))
    k_sorted = k_idx[order]
    first = np.r_[True, (track[k_sorted][1:] != track[k_sorted][:-1]) | (o_img[k_sorted][1:] != o_img[k_sorted][:-1])]
    dropped_dup = int((~first).sum())
    keep[:] = False
    keep[k_sorted[first]] = True
    # tracks with >= 2 images and a point
    obs_per_track = np.bincount(track[keep], minlength=n_tracks_raw)
    good_track = (obs_per_track >= 2) & ~np.isnan(point).any(axis=1)
    keep &= good_track[track]
    final_err = err_all[keep]
    o_kp = node_kp[obs]
    o_image_id = np.array(image_ids)[o_img]

    # ---- write the COLMAP text model -----------------------------------------------------
    out = args.out if args.out is not None else args.database.parent / "gt_model"
    out.mkdir(parents=True, exist_ok=True)
    point_ids = np.full(n_tracks_raw, -1, np.int64)
    point_ids[good_track] = np.arange(1, int(good_track.sum()) + 1)
    model, width, height, params = camera
    with open(out / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 {model} {width} {height} " + " ".join(f"{p:.10g}" for p in params) + "\n")
    point2d_pid = {i: np.full(len(keypoints[i]), -1, np.int64) for i in image_ids}
    for o in np.flatnonzero(keep):
        point2d_pid[o_image_id[o]][o_kp[o]] = point_ids[track[o]]
    with open(out / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in image_ids:
            q = rotation_to_quaternion(R_all[index_of[i]])
            t = t_all[index_of[i]]
            f.write(f"{i} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {images[i]}\n")
            f.write(" ".join(f"{x:.6f} {y:.6f} {p}" for (x, y), p in zip(keypoints[i], point2d_pid[i])) + "\n")
    with open(out / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        by_track = {}
        for o in np.flatnonzero(keep):
            by_track.setdefault(int(track[o]), []).append(o)
        for tr in np.flatnonzero(good_track):
            X = point[tr]
            members = by_track.get(int(tr), [])
            e = float(np.mean(err_all[members])) if members else 0.0
            f.write(f"{point_ids[tr]} {X[0]:.9g} {X[1]:.9g} {X[2]:.9g} 128 128 128 {e:.4f} "
                    + " ".join(f"{o_image_id[o]} {o_kp[o]}" for o in members) + "\n")

    # ---- report --------------------------------------------------------------------------
    lengths = obs_per_track[good_track]
    keep_node = np.zeros(n_nodes, bool)
    keep_node[obs[keep]] = True
    kept_matches = keep_node[edge_a] & keep_node[edge_b]
    print(f"observations: {len(obs)} in tracks, {int(o_depth.sum())} with depth; dropped {dropped_reproj} for reprojection, "
          f"{dropped_depth} for depth, {dropped_dup} duplicates in one image; kept {int(keep.sum())}")
    print(f"reprojection error of the kept observations: median "
          f"{np.median(final_err):.3f} px, p95 {np.quantile(final_err, 0.95):.3f} px")
    print(f"points: {int(good_track.sum())} of {n_tracks_raw} tracks; track length median {int(np.median(lengths))}, "
          f"p95 {int(np.quantile(lengths, 0.95))}, max {int(lengths.max())}; "
          f"{100.0 * kept_matches.mean():.1f}% of the verified inlier matches kept")
    print(f"wrote {out / 'cameras.txt'}, images.txt, points3D.txt ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
