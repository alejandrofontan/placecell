#!/usr/bin/env python3
"""
placecell — colmap_information_kernel.py

Author:  Alejandro Fontan
Assisted by: Claude (Fable 5)
Created: 2026-09-07
License: Apache-2.0

Pairwise shared-information kernel of the images of a COLMAP reconstruction, for a
kernel-only placecell store (PlaceCell::set_kernel via load_npy; examples/kernel_demo
--kind similarity --ids <out>/ids.csv).

Model. For a pair of images (i, j) take the variables x = (pose_i, pose_j, every 3D point
seen by either image). Each image's observations give a Gauss-Newton information matrix
H_i = J_i^T J_i / sigma_pix^2 from the pinhole reprojection Jacobians, and an isotropic
prior L0 (sigma_t, sigma_rot on the poses, sigma_point on the points) fixes the 7-DoF
monocular gauge. Under the Gaussian (Laplace) approximation the mutual information
between the two images' measurement sets is

    I_ij = 1/2 [ logdet(L0+H_i) + logdet(L0+H_j) - logdet(L0) - logdet(L0+H_i+H_j) ]

and the kernel entry is the normalised mutual information

    K_ij = I_ij / sqrt(G_i G_j),   G_i = I_ii = 1/2 [ logdet(L0+H_i) - logdet(L0) ],

the fraction of each image's own information gain that the pair shares. Data processing
gives I_ij <= min(G_i, G_j), so K is in [0,1] with unit diagonal (PSD is not guaranteed;
set_kernel checks it, kernel_demo --clip projects). Every logdet is a Schur complement of
the independent 3x3 point blocks onto the pose block; the single-image reductions are
precomputed once per image and every pair-point incidence is processed in numpy batches
(pairwise_mutual_information), so the cost is linear in sum over tracks of L(L-1)/2 —
a 1180-image model with 38 M incidences takes about a minute and ~1.5 GB.

Conventions. Only registered images get a row. Row ids are the data-row index of the
sequence's rgb.csv (--rgb-csv; VSLAM-LAB's frame_id numbering, same as VPR-LAB's D.npy),
matched by image file name against the path_rgb_0 column; without --rgb-csv the tool
looks for rgb_exp.csv next to the model folder and warns that its numbering is the
experiment list's, and falls back to COLMAP image ids otherwise. Non-pinhole camera
models are approximated by their focal length / principal point (warned).

Usage:
    colmap_information_kernel.py <model_dir> [--rgb-csv rgb.csv] [--out dir]
        [--sigma-pix 1.0] [--prior-scale 1.0] [--sigma-t S] [--sigma-rot S] [--sigma-point S]
        [--check N] [--chunk 200000] [--quiet]

<model_dir> holds cameras/images/points3D as .bin or .txt (the .bin is preferred). Output:
<out>/kernel.npy (float32 NMI), <out>/mutual_information.npy (I_ij in nats, diagonal G_i),
<out>/ids.csv (internal_id, external_id, image_id, name, information_gain, observations)
and a summary on stdout. --check N verifies N random pairs against a direct dense
evaluation of the formula. Default --out: <model_dir>/information_kernel.
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------
# COLMAP model reading (text and binary; no pycolmap)
# ---------------------------------------------------------------------------------------

CAMERA_MODELS = {
    # model_id: (name, num_params, focal/pp param layout)
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
MODEL_ID_BY_NAME = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}
EXACT_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}


def pinhole_params(model: str, params: np.ndarray) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy for any COLMAP model (distortion ignored for the others)."""
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def qvec_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_model(model_dir: Path):
    """cameras {id: (model, params)}, images {id: dict(R, t, name, camera, xy(n,2), pid(n,))},
    points {id: xyz}. Prefers the .bin files."""
    if (model_dir / "images.bin").exists():
        return _read_bin(model_dir)
    if (model_dir / "images.txt").exists():
        return _read_txt(model_dir)
    raise FileNotFoundError(f"{model_dir}: no images.bin / images.txt")


def _read_txt(model_dir: Path):
    cameras = {}
    for line in open(model_dir / "cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cameras[int(e[0])] = (e[1], np.array(list(map(float, e[4:]))))
    images = {}
    lines = [l for l in open(model_dir / "images.txt") if not l.startswith("#") and l.strip()]
    for a, b in zip(lines[0::2], lines[1::2]):
        e = a.split()
        q = np.array(list(map(float, e[1:5])))
        t = np.array(list(map(float, e[5:8])))
        obs = np.array(list(map(float, b.split()))).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        images[int(e[0])] = dict(R=qvec_to_rotation(q), t=t, camera=int(e[8]), name=e[9],
                                 xy=obs[:, :2], pid=obs[:, 2].astype(np.int64))
    points = {}
    for line in open(model_dir / "points3D.txt"):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        points[int(e[0])] = np.array(list(map(float, e[1:4])))
    return cameras, images, points


def _read_bin(model_dir: Path):
    def read(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    cameras = {}
    with open(model_dir / "cameras.bin", "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = read(f, "iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            params = np.array(read(f, "d" * num_params))
            cameras[cam_id] = (name, params)
    images = {}
    with open(model_dir / "images.bin", "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            image_id = read(f, "i")[0]
            q = np.array(read(f, "dddd"))
            t = np.array(read(f, "ddd"))
            cam_id = read(f, "i")[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or not c:
                    break
                name += c
            (m,) = read(f, "Q")
            raw = np.frombuffer(f.read(24 * m), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")]))
            images[image_id] = dict(R=qvec_to_rotation(q), t=t, camera=cam_id, name=name.decode(),
                                    xy=np.stack([raw["x"], raw["y"]], axis=1) if m else np.zeros((0, 2)),
                                    pid=raw["pid"].copy() if m else np.zeros(0, np.int64))
    points = {}
    with open(model_dir / "points3D.bin", "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            pid = read(f, "Q")[0]
            xyz = np.array(read(f, "ddd"))
            f.read(3)  # rgb
            f.read(8)  # error
            (track_len,) = read(f, "Q")
            f.read(8 * track_len)
            points[pid] = xyz
    return cameras, images, points


# ---------------------------------------------------------------------------------------
# Information geometry
# ---------------------------------------------------------------------------------------

def logdet(M: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(M)
    return float(ld) if sign > 0 else -np.inf


def batched_logdet(M: np.ndarray) -> np.ndarray:
    sign, ld = np.linalg.slogdet(M)
    ld = np.where(sign > 0, ld, -np.inf)
    return ld


class ImageInfo:
    """Per-image precomputation. For every observed point k (with a 3D point):
    Jx_k (2x6, perturbation [dtheta, dt] on the left of cam_from_world), Jp_k (2x3),
    the single-image point block P_k = lambda_p I + Jp^T Jp, its logdet, and the Schur
    term T_k = B_k P_k^-1 B_k^T with B_k = Jx^T Jp (6x6). A_full = prior_c + sum_k (Jx^T Jx - T_k)
    is the image's reduced pose information; ld_full = sum_k logdet(P_k) - 3 n_k log(lambda_p)
    + logdet(A_full) - logdet(prior_c) is logdet(L0+H_i) - logdet(L0)."""

    def __init__(self, image, fx, fy, cx, cy, points, lambda_p, prior_c, inv_sigma_pix, keep_jacobians=False):
        valid = image["pid"] >= 0
        pids = image["pid"][valid]
        keep = np.array([p in points for p in pids], dtype=bool) if len(pids) else np.zeros(0, bool)
        # Per OBSERVATION (COLMAP can map two keypoints of one image to the same 3D point,
        # so observations and unique points differ): Jacobians are formed per observation,
        # the information blocks are aggregated per unique point below. Only the per-point
        # blocks a pair needs are kept (B, P_k^-1, JpJp, logdet P_k: ~300 B per point);
        # keep_jacobians retains Jx/Jp/obs_pids for the dense --check.
        self.obs_pids = pids[keep]
        P = np.array([points[p] for p in self.obs_pids]) if len(self.obs_pids) else np.zeros((0, 3))
        R, t = image["R"], image["t"]
        pc = P @ R.T + t                              # (n,3) in camera frame
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        n = len(self.obs_pids)
        A = np.zeros((n, 2, 3))                       # d pixel / d pc
        A[:, 0, 0] = fx / Z
        A[:, 0, 2] = -fx * X / Z**2
        A[:, 1, 1] = fy / Z
        A[:, 1, 2] = -fy * Y / Z**2
        A *= inv_sigma_pix
        skew = np.zeros((n, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -Z, Y
        skew[:, 1, 0], skew[:, 1, 2] = Z, -X
        skew[:, 2, 0], skew[:, 2, 1] = -Y, X
        dpc_dxi = np.concatenate([-skew, np.broadcast_to(np.eye(3), (n, 3, 3))], axis=2)   # (n,3,6)
        Jx = A @ dpc_dxi                              # (n,2,6) per observation
        Jp = A @ R                                    # (n,2,3) per observation
        JxJx = np.einsum("nki,nkj->nij", Jx, Jx)      # (n,6,6)
        JpJp = np.einsum("nki,nkj->nij", Jp, Jp)      # (n,3,3)
        B = np.einsum("nki,nkj->nij", Jx, Jp)         # (n,6,3)
        # Aggregate per unique 3D point (m <= n); pids come out sorted (searchsorted lookups)
        self.pids, inverse = np.unique(self.obs_pids, return_inverse=True)
        m = len(self.pids)
        JxJx_sum = JxJx.sum(0)
        self.JpJp = np.zeros((m, 3, 3)); np.add.at(self.JpJp, inverse, JpJp)
        self.B = np.zeros((m, 6, 3)); np.add.at(self.B, inverse, B)
        Pk = self.JpJp + lambda_p * np.eye(3)
        self.Pk_inv = np.linalg.inv(Pk)
        self.ldP = batched_logdet(Pk)                                # (m,)
        T_sum = np.einsum("nij,njk,nlk->il", self.B, self.Pk_inv, self.B)   # sum_k B_k P_k^-1 B_k^T
        self.A_full = prior_c + JxJx_sum - T_sum
        self.ldA = logdet(self.A_full)
        self.ld_full = float(self.ldP.sum()) - 3 * m * np.log(lambda_p) + self.ldA - logdet(prior_c)
        self.lambda_p = lambda_p
        self.prior_c = prior_c
        if keep_jacobians:
            self.Jx, self.Jp = Jx, Jp
        else:
            self.obs_count = n
            self.obs_pids = None

    @property
    def gain(self) -> float:
        return 0.5 * self.ld_full

    def rows(self, shared: np.ndarray) -> np.ndarray:
        """Row indices (into pids/B/...) of the given point ids (must all be observed)."""
        return np.searchsorted(self.pids, shared)


def pair_logdet(a: ImageInfo, b: ImageInfo, shared: np.ndarray) -> float:
    """logdet(L0 + H_a + H_b) - logdet(L0) for the pair problem over the union of their points
    (reference single-pair implementation; the batched pairwise_mutual_information is the one
    used for the kernel). Exclusive points keep their single-image Schur terms; shared points
    are re-done jointly."""
    ka, kb = a.rows(shared), b.rows(shared)
    S = np.zeros((12, 12))
    # undo the single-image Schur terms of the shared points, then add the joint ones
    S[:6, :6] = a.A_full + np.einsum("nij,njk,nlk->il", a.B[ka], a.Pk_inv[ka], a.B[ka])
    S[6:, 6:] = b.A_full + np.einsum("nij,njk,nlk->il", b.B[kb], b.Pk_inv[kb], b.B[kb])
    P = a.JpJp[ka] + b.JpJp[kb] + a.lambda_p * np.eye(3)     # joint point blocks (m,3,3)
    P_inv = np.linalg.inv(P)
    B = np.concatenate([a.B[ka], b.B[kb]], axis=1)    # (m,12,3)
    S -= np.einsum("nij,njk,nlk->il", B, P_inv, B)
    ld_points = float(a.ldP.sum() - a.ldP[ka].sum() + b.ldP.sum() - b.ldP[kb].sum() + batched_logdet(P).sum())
    n_points = len(a.pids) + len(b.pids) - len(shared)
    prior_cc = np.zeros((12, 12))
    prior_cc[:6, :6] = a.prior_c
    prior_cc[6:, 6:] = b.prior_c
    return ld_points - 3 * n_points * np.log(a.lambda_p) + logdet(S) - logdet(prior_cc)


def pairwise_mutual_information(infos: list, lambda_p: float, chunk_incidences: int = 200_000,
                                progress=None) -> tuple[np.ndarray, int]:
    """I_ij for every pair of images sharing a point, batched. Returns (I, n_pairs) with
    the images' own gains G_i on the diagonal.

    With the single-image reductions in hand, a pair only needs its shared points:
        I_ab = 1/2 [ logdet A_a + logdet A_b - logdet S_ab
                     - sum_shared ( logdet P_k^ab - logdet P_k^a - logdet P_k^b - 3 log lambda_p ) ]
    where S_ab = blockdiag(A_a, A_b) + [shared single-image Schur terms put back]
                 - sum_shared B_k^ab (P_k^ab)^-1 (B_k^ab)^T  (12x12),
    P_k^ab = lambda_p I + JpJp_k^a + JpJp_k^b, B_k^ab = [B_k^a; B_k^b].
    (The two prior log-dets and the exclusive points' terms cancel between the singles and
    the pair.) Implementation: every (image, point) incidence gets a global row in stacked
    per-point arrays; incidences are sorted by point, the pairs of every track are
    enumerated with triu indices (vectorised per track length), sorted by pair key, and
    processed in chunks of <= chunk_incidences rows with batched einsums and
    np.add.reduceat over the pair segments. Memory: ~30 B per pair-point incidence for the
    index arrays plus ~200 B per incidence of the current chunk. Cost is linear in the
    number of incidences, sum over points of L(L-1)/2."""
    n = len(infos)
    counts = np.array([len(inf.pids) for inf in infos])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    N = int(offsets[-1])
    B_all = np.concatenate([inf.B for inf in infos]) if N else np.zeros((0, 6, 3))
    Pinv_all = np.concatenate([inf.Pk_inv for inf in infos]) if N else np.zeros((0, 3, 3))
    JpJp_all = np.concatenate([inf.JpJp for inf in infos]) if N else np.zeros((0, 3, 3))
    ldP_all = np.concatenate([inf.ldP for inf in infos]) if N else np.zeros(0)
    pid_all = np.concatenate([inf.pids for inf in infos]) if N else np.zeros(0, np.int64)
    img_of = np.repeat(np.arange(n, dtype=np.int32), counts)
    A_all = np.stack([inf.A_full for inf in infos]) if n else np.zeros((0, 6, 6))
    ldA = np.array([inf.ldA for inf in infos])
    G = np.array([inf.gain for inf in infos])
    I = np.zeros((n, n))
    np.fill_diagonal(I, G)
    if N == 0:
        return I, 0

    # ---- incidences: for every track, all image pairs -----------------------------------
    order = np.argsort(pid_all, kind="stable")
    pid_sorted = pid_all[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(pid_sorted)) + 1])
    lengths = np.diff(np.concatenate([starts, [N]]))
    keys, ka_list, kb_list = [], [], []
    for L in np.unique(lengths):
        if L < 2:
            continue
        seg = starts[lengths == L]                                   # (S,)
        rows = order[seg[:, None] + np.arange(L)[None, :]]           # (S, L) global rows
        iu, ju = np.triu_indices(L, 1)
        ka = rows[:, iu].ravel()
        kb = rows[:, ju].ravel()
        ia, ib = img_of[ka], img_of[kb]
        swap = ia > ib                                               # a < b (a point is at most once per image)
        ka, kb = np.where(swap, kb, ka), np.where(swap, ka, kb)
        ia, ib = np.minimum(ia, ib), np.maximum(ia, ib)
        keys.append(ia.astype(np.int64) * n + ib)
        ka_list.append(ka.astype(np.int32))
        kb_list.append(kb.astype(np.int32))
    if not keys:
        return I, 0
    key = np.concatenate(keys); ka_all = np.concatenate(ka_list); kb_all = np.concatenate(kb_list)
    del keys, ka_list, kb_list
    perm = np.argsort(key, kind="stable")
    key, ka_all, kb_all = key[perm], ka_all[perm], kb_all[perm]
    del perm
    pair_starts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1])
    pair_keys = key[pair_starts]
    n_pairs = len(pair_starts)
    pair_ends = np.concatenate([pair_starts[1:], [len(key)]])
    del key
    log_lambda = 3.0 * np.log(lambda_p)
    eye3 = np.eye(3)

    # ---- chunks of whole pairs ------------------------------------------------------------
    p0 = 0
    while p0 < n_pairs:
        # extend the chunk while it stays under the incidence budget (always >= 1 pair)
        p1 = int(np.searchsorted(pair_ends, pair_starts[p0] + chunk_incidences, side="right"))
        p1 = max(p1, p0 + 1)
        s, e = pair_starts[p0], pair_ends[p1 - 1]
        ka, kb = ka_all[s:e], kb_all[s:e]
        seg = pair_starts[p0:p1] - s                                 # segment starts within the chunk
        Ba, Bb = B_all[ka], B_all[kb]                                # (M,6,3)
        Ta = np.einsum("nij,njk,nlk->nil", Ba, Pinv_all[ka], Ba)      # single-image Schur terms to put back
        Tb = np.einsum("nij,njk,nlk->nil", Bb, Pinv_all[kb], Bb)
        P = JpJp_all[ka] + JpJp_all[kb] + lambda_p * eye3            # joint point blocks
        P_inv = np.linalg.inv(P)
        B12 = np.concatenate([Ba, Bb], axis=1)                       # (M,12,3)
        joint = np.einsum("nij,njk,nlk->nil", B12, P_inv, B12)       # (M,12,12)
        ld_term = batched_logdet(P) - ldP_all[ka] - ldP_all[kb] - log_lambda
        sum_Ta = np.add.reduceat(Ta, seg, axis=0)
        sum_Tb = np.add.reduceat(Tb, seg, axis=0)
        sum_joint = np.add.reduceat(joint, seg, axis=0)
        sum_ld = np.add.reduceat(ld_term, seg)
        ia = (pair_keys[p0:p1] // n).astype(np.int64)
        ib = (pair_keys[p0:p1] % n).astype(np.int64)
        S = -sum_joint
        S[:, :6, :6] += A_all[ia] + sum_Ta
        S[:, 6:, 6:] += A_all[ib] + sum_Tb
        I_ab = 0.5 * (ldA[ia] + ldA[ib] - batched_logdet(S) - sum_ld)
        I[ia, ib] = I_ab
        I[ib, ia] = I_ab
        if progress:
            progress(p1, n_pairs)
        p0 = p1
    return I, n_pairs


def dense_pair_logdet(a: ImageInfo, b: ImageInfo, shared: np.ndarray) -> float:
    """Direct evaluation of logdet(L0+H_a+H_b) - logdet(L0) on the full (12 + 3m) system (--check):
    one column block per unique point, one residual per observation."""
    union = sorted(set(a.pids.tolist()) | set(b.pids.tolist()))
    col = {p: 12 + 3 * k for k, p in enumerate(union)}
    n = 12 + 3 * len(union)
    L0 = np.zeros((n, n))
    L0[:6, :6] = a.prior_c
    L0[6:12, 6:12] = b.prior_c
    L0[12:, 12:] = a.lambda_p * np.eye(3 * len(union))
    H = np.zeros((n, n))
    for img, off in ((a, 0), (b, 6)):
        if img.obs_pids is None:
            raise RuntimeError("dense_pair_logdet needs ImageInfo(..., keep_jacobians=True)")
        for k, p in enumerate(img.obs_pids):
            J = np.zeros((2, n))
            J[:, off:off + 6] = img.Jx[k]
            J[:, col[int(p)]:col[int(p)] + 3] = img.Jp[k]
            H += J.T @ J
    return logdet(L0 + H) - logdet(L0)


# ---------------------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------------------

def rgb_csv_rows(path: Path) -> dict[str, int]:
    """basename of path_rgb_0 -> data-row index (0-based)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index("path_rgb_0") if "path_rgb_0" in header else 1
        return {os.path.basename(row[col]): i for i, row in enumerate(reader) if row}


# ---------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="COLMAP model folder (.bin or .txt files)")
    ap.add_argument("--rgb-csv", type=Path, default=None,
                    help="sequence rgb.csv whose data-row index becomes each image's id (matched by file name)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <model_dir>/information_kernel)")
    ap.add_argument("--sigma-pix", type=float, default=1.0, help="observation noise in pixels [1.0]")
    ap.add_argument("--prior-scale", type=float, default=1.0, help="multiply every prior sigma [1.0]")
    ap.add_argument("--sigma-t", type=float, default=None, help="pose translation prior sigma [median consecutive baseline]")
    ap.add_argument("--sigma-rot", type=float, default=1.0, help="pose rotation prior sigma in rad [1.0]")
    ap.add_argument("--sigma-point", type=float, default=None, help="point prior sigma [median distance to the point cloud centroid]")
    ap.add_argument("--check", type=int, default=0, help="verify N random pairs against the dense formula (slow)")
    ap.add_argument("--chunk", type=int, default=200_000,
                    help="pair-point incidences per batch (memory ~200 B each during the batch) [200000]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def say(*a):
        if not args.quiet:
            print(*a)

    t0 = time.time()
    cameras, images, points = read_model(args.model_dir)
    say(f"model {args.model_dir}: {len(images)} registered images, {len(points)} points, "
        f"{sum(int((im['pid'] >= 0).sum()) for im in images.values())} observations")
    if (args.model_dir / "best_model").exists() or any((args.model_dir / str(k) / "images.bin").exists() for k in range(3)):
        print(f"WARNING: {args.model_dir} looks like a COLMAP output folder with sub-models; pass the best sub-model "
              f"(see its best_model file) unless the model files here are the ones you want", file=sys.stderr)
    for cam_id, (model, _) in cameras.items():
        if model not in EXACT_MODELS:
            print(f"WARNING: camera {cam_id} is {model}; its distortion is ignored (pinhole Jacobians from f, c)", file=sys.stderr)

    # ---- ids ---------------------------------------------------------------------------
    image_ids = sorted(images)
    id_source = "colmap image_id"
    name_to_row: dict[str, int] | None = None
    rgb_csv = args.rgb_csv
    if rgb_csv is None:
        candidate = args.model_dir.resolve().parent / "rgb_exp.csv"
        if candidate.exists():
            rgb_csv = candidate
            print(f"WARNING: no --rgb-csv given; using {candidate} — its row numbering is the experiment's input "
                  f"list, not the sequence's rgb.csv (VSLAM-LAB frame_id / D.npy numbering)", file=sys.stderr)
    if rgb_csv is not None:
        name_to_row = rgb_csv_rows(rgb_csv)
        id_source = f"data-row index of {rgb_csv}"
        missing = [images[i]["name"] for i in image_ids if os.path.basename(images[i]["name"]) not in name_to_row]
        if missing:
            print(f"ERROR: {len(missing)} registered images are not in {rgb_csv} (e.g. {missing[0]})", file=sys.stderr)
            return 1
    external_ids = np.array([name_to_row[os.path.basename(images[i]["name"])] if name_to_row else i for i in image_ids],
                            dtype=np.int64)
    if len(set(external_ids.tolist())) != len(external_ids):
        print("ERROR: duplicate ids (two registered images map to the same csv row)", file=sys.stderr)
        return 1

    # ---- priors ------------------------------------------------------------------------
    centres = np.array([-images[i]["R"].T @ images[i]["t"] for i in image_ids])
    order = np.argsort(external_ids)                    # consecutive in sequence order
    baselines = np.linalg.norm(np.diff(centres[order], axis=0), axis=1)
    P = np.array(list(points.values()))
    sigma_t = args.sigma_t if args.sigma_t is not None else float(np.median(baselines)) if len(baselines) else 1.0
    sigma_point = args.sigma_point if args.sigma_point is not None else float(np.median(np.linalg.norm(P - P.mean(0), axis=1)))
    sigma_t *= args.prior_scale
    sigma_rot = args.sigma_rot * args.prior_scale
    sigma_point *= args.prior_scale
    if sigma_t <= 0 or sigma_point <= 0:
        print("ERROR: degenerate prior sigma", file=sys.stderr)
        return 1
    prior_c = np.diag([1 / sigma_rot**2] * 3 + [1 / sigma_t**2] * 3)
    lambda_p = 1 / sigma_point**2
    say(f"priors: sigma_t {sigma_t:.4f}, sigma_rot {sigma_rot:.3f} rad, sigma_point {sigma_point:.4f}; sigma_pix {args.sigma_pix}")

    # ---- per-image reductions ------------------------------------------------------------
    infos: list[ImageInfo] = []
    for i in image_ids:
        model, params = cameras[images[i]["camera"]]
        fx, fy, cx, cy = pinhole_params(model, params)
        infos.append(ImageInfo(images[i], fx, fy, cx, cy, points, lambda_p, prior_c, 1.0 / args.sigma_pix,
                               keep_jacobians=args.check > 0))
    n = len(infos)
    G = np.array([inf.gain for inf in infos])
    say(f"per-image information gain G_i: min {G.min():.1f} median {np.median(G):.1f} max {G.max():.1f} nats "
        f"({time.time() - t0:.1f}s)")

    # ---- pairs (batched over all pair-point incidences) ----------------------------------
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if not args.quiet and time.time() - last[0] > 5.0:
            last[0] = time.time()
            print(f"  pairs {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    I, n_pairs = pairwise_mutual_information(infos, lambda_p, chunk_incidences=args.chunk, progress=progress)
    K = I / np.sqrt(np.outer(G, G))
    np.fill_diagonal(K, 1.0)
    say(f"pairs sharing points: {n_pairs} of {n * (n - 1) // 2} ({time.time() - t0:.1f}s)")

    # ---- self-check against the dense formula --------------------------------------------
    # Random overlapping pairs: the batched value vs the single-pair reduction vs the dense
    # (12 + 3m)-dim system. Slow (dense logdets), hence opt-in.
    if args.check > 0 and n_pairs > 0:
        rng = np.random.default_rng(0)
        iu, ju = np.triu_indices(n, 1)
        overlapping = np.flatnonzero(I[iu, ju] > 0)
        picks = rng.choice(overlapping, size=min(args.check, len(overlapping)), replace=False)
        worst_dense = worst_batch = 0.0
        for idx in picks:
            a, b = int(iu[idx]), int(ju[idx])
            shared = np.intersect1d(infos[a].pids, infos[b].pids)
            fast = pair_logdet(infos[a], infos[b], shared)
            dense = dense_pair_logdet(infos[a], infos[b], shared)
            worst_dense = max(worst_dense, abs(fast - dense) / max(1.0, abs(dense)))
            I_single = 0.5 * (infos[a].ld_full + infos[b].ld_full - fast)
            worst_batch = max(worst_batch, abs(I_single - I[a, b]) / max(1.0, abs(I_single)))
        say(f"check: {len(picks)} pairs, worst relative error single-pair vs dense {worst_dense:.2e}, "
            f"batched vs single-pair {worst_batch:.2e}")
        if worst_dense > 1e-6 or worst_batch > 1e-6:
            print("ERROR: Schur reduction disagrees with the dense formula", file=sys.stderr)
            return 1

    # ---- report ------------------------------------------------------------------------
    off = ~np.eye(n, dtype=bool)
    overlapping = I[off] > 0
    eig = np.linalg.eigvalsh(K)
    say(f"I_ij over overlapping pairs: min {I[off][overlapping].min():.2f} median {np.median(I[off][overlapping]):.2f} "
        f"max {I[off][overlapping].max():.2f} nats" if overlapping.any() else "no overlapping pairs")
    if overlapping.any():
        q = np.quantile(K[off][overlapping], [0, 0.25, 0.5, 0.75, 0.95, 1])
        say(f"K_ij over overlapping pairs: quantiles {np.round(q, 3).tolist()}; max {K[off].max():.3f}; "
            f"negative entries {(K[off] < -1e-9).sum()}")
    consecutive = [K[order[k], order[k + 1]] for k in range(n - 1)]
    say(f"kernel eigenvalues: min {eig.min():.4f} max {eig.max():.2f}, negative {(eig < -1e-9).sum()}; "
        f"consecutive-frame median K {np.median(consecutive):.3f}" if consecutive else "")

    # ---- write -------------------------------------------------------------------------
    out = args.out if args.out is not None else args.model_dir / "information_kernel"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "kernel.npy", K.astype(np.float32))
    np.save(out / "mutual_information.npy", I.astype(np.float32))
    with open(out / "ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["internal_id", "external_id", "image_id", "name", "information_gain", "observations"])
        for a, i in enumerate(image_ids):
            observations = len(infos[a].obs_pids) if infos[a].obs_pids is not None else infos[a].obs_count
            w.writerow([a, int(external_ids[a]), i, images[i]["name"], f"{G[a]:.6f}", observations])
    say(f"ids: {id_source}")
    say(f"wrote {out / 'kernel.npy'} ({n}x{n}), {out / 'mutual_information.npy'}, {out / 'ids.csv'} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
