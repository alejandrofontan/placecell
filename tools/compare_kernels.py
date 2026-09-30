#!/usr/bin/env python3
"""
placecell — compare_kernels.py

Author:  Alejandro Fontan
Assisted by: Claude (Fable 5)
Created: 2026-09-07
Updated: 2026-09-30 (any number of kernels, every pair compared)
License: Apache-2.0

Side-by-side of two or more kernels over the same images, every pair compared. An input is
either a kernel folder (kernel.npy + ids.csv, from tools/colmap_information_kernel.py: row i
is the image whose rgb.csv row is ids.csv's external_id) or a .npy matrix over the
sequence's rgb.csv rows (VPR-LAB's D.npy), converted to a similarity by --kind, or per input
with path::kind (similarity, squared-euclidean, cosine-distance, euclidean; see kernel_io.h).
Every kernel is symmetrised and restricted to the images common to ALL inputs, in frame-id
order, optionally double-centred (--centred: placecell's Pearson correlation of
mean-centred descriptors, what the culler marginalises) and min-max normalised to [0,1] over
its off-diagonal entries (the diagonal stays 1) unless --no-normalise: the VPR cosine has a
~0.37 common-mode floor, the information kernel lives in 0-0.6, so the heatmaps share a range.

Stdout: the images kept per input, each kernel's off-diagonal range, then for every pair the
Pearson / Spearman correlation of the paired off-diagonal entries, the mean, mean absolute
and largest difference, and the most disagreeing frame pairs; N x N Pearson and Spearman
matrices. <out>/pairs.csv has one row per pair. Figure <out>/compare_kernels[_centred].png:
an N x N grid, each kernel's heatmap on the diagonal, the difference heatmaps (row - column,
one diverging scale) below it and the scatter of paired entries above it.

Usage:
    compare_kernels.py <input> <input> [<input> ...] [--kind squared-euclidean] [--labels A B ...]
                       [--centred] [--no-normalise] [--out dir] [--show]
    compare_kernels.py <D.npy> <colmap_kernel_dir>          # the original two-kernel call

--out defaults to the first kernel folder among the inputs, else the current directory.
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

KINDS = ("similarity", "squared-euclidean", "squared_euclidean", "faiss", "cosine-distance", "cosine_distance",
         "cosine", "euclidean")


def load_ids(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = list(csv.DictReader(open(path, newline="")))
    return np.array([int(r["external_id"]) for r in rows]), [r["name"] for r in rows]


def similarity_from_distance(D: np.ndarray, kind: str) -> np.ndarray:
    if kind == "similarity":
        return D
    if kind in ("squared-euclidean", "squared_euclidean", "faiss"):
        return 1.0 - 0.5 * D
    if kind in ("cosine-distance", "cosine_distance", "cosine"):
        return 1.0 - D
    if kind == "euclidean":
        return 1.0 - 0.5 * D**2
    raise ValueError(f"unknown kind {kind}")


def centre(K: np.ndarray) -> np.ndarray:
    """placecell's double-centring + unit-diagonal renormalisation (PlaceCell::centre_kernel)."""
    n = len(K)
    J = np.eye(n) - np.ones((n, n)) / n
    C = J @ K @ J
    d = np.sqrt(np.maximum(np.diag(C), 1e-9))
    return C / np.outer(d, d)


def normalise(K: np.ndarray) -> np.ndarray:
    off = ~np.eye(len(K), dtype=bool)
    lo, hi = K[off].min(), K[off].max()
    N = (K - lo) / (hi - lo) if hi > lo else np.zeros_like(K)
    np.fill_diagonal(N, 1.0)
    return N


def load_input(spec: str, default_kind: str) -> dict:
    """A kernel folder (kernel.npy + ids.csv) or a matrix over rgb.csv rows (path[::kind])."""
    path_str, _, kind = spec.partition("::")
    path = Path(path_str)
    if path.is_dir():
        if kind:
            raise ValueError(f"{spec}: ::kind applies to a .npy matrix, not to a kernel folder")
        if not (path / "kernel.npy").exists() or not (path / "ids.csv").exists():
            raise FileNotFoundError(f"{path}: a kernel folder needs kernel.npy and ids.csv")
        ids, _ = load_ids(path / "ids.csv")
        K = np.load(path / "kernel.npy").astype(np.float64)
        if len(ids) != len(K):
            raise ValueError(f"{path}: {len(ids)} ids for a {len(K)} x {len(K)} kernel")
        return dict(label=path.name, path=path, is_folder=True, ids=ids, K=K, kind="similarity")
    if path.suffix != ".npy" or not path.exists():
        raise FileNotFoundError(f"{spec}: neither a kernel folder nor a .npy matrix")
    kind = kind or default_kind
    if kind not in KINDS:
        raise ValueError(f"{spec}: unknown kind {kind}")
    D = np.load(path).astype(np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{path}: not a square matrix ({D.shape})")
    return dict(label=path.stem, path=path, is_folder=False, ids=np.arange(len(D)),
                K=similarity_from_distance(D, kind), kind=kind)


def restrict(inp: dict, common: np.ndarray) -> np.ndarray:
    """The input's kernel over `common` frame ids (in that order), symmetrised, unit diagonal."""
    row_of = {int(i): r for r, i in enumerate(inp["ids"])}
    rows = np.array([row_of[int(i)] for i in common])
    K = inp["K"][np.ix_(rows, rows)]
    K = 0.5 * (K + K.T)                  # the rotation-min VPR matrix is asymmetric (see kernel_io.h)
    np.fill_diagonal(K, 1.0)
    return K


def pair_stats(A: np.ndarray, B: np.ndarray, frame_ids: np.ndarray, worst: int = 5) -> dict:
    n = len(A)
    off = ~np.eye(n, dtype=bool)
    a, b = A[off], B[off]
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    diff = A - B
    iu = np.triu_indices(n, 1)
    top = np.argsort(-np.abs(diff[iu]))[:worst]
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(np.corrcoef(ra, rb)[0, 1]),
                mean=float(diff[off].mean()), mean_abs=float(np.abs(diff[off]).mean()),
                max_abs=float(np.abs(diff[off]).max()),
                worst=[(int(frame_ids[iu[0][w]]), int(frame_ids[iu[1][w]]),
                        float(A[iu[0][w], iu[1][w]]), float(B[iu[0][w], iu[1][w]])) for w in top])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="kernel folders (kernel.npy + ids.csv) or .npy matrices over rgb.csv rows "
                                              "(path::kind to override --kind)")
    ap.add_argument("--kind", default="squared-euclidean", help="how to read a .npy matrix [squared-euclidean]")
    ap.add_argument("--labels", nargs="+", default=None, help="one label per input [folder / file name]")
    ap.add_argument("--out", type=Path, default=None, help="output folder [the first kernel folder, else .]")
    ap.add_argument("--centred", action="store_true", help="double-centre every kernel first (what the culler uses)")
    ap.add_argument("--no-normalise", action="store_true", help="show the kernels as they are instead of min-max 0-1")
    ap.add_argument("--show", action="store_true", help="open a matplotlib window as well")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        print("ERROR: give at least two inputs", file=sys.stderr)
        return 1
    try:
        inputs = [load_input(spec, args.kind) for spec in args.inputs]
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.labels is not None:
        if len(args.labels) != len(inputs):
            print(f"ERROR: {len(args.labels)} labels for {len(inputs)} inputs", file=sys.stderr)
            return 1
        for inp, label in zip(inputs, args.labels):
            inp["label"] = label
    labels = [inp["label"] for inp in inputs]
    if len(inputs) > 5:
        print(f"WARNING: {len(inputs)} inputs make a crowded grid", file=sys.stderr)

    # ---- the images common to every input -----------------------------------------------
    common = inputs[0]["ids"]
    for inp in inputs[1:]:
        common = np.intersect1d(common, inp["ids"])
    common = np.sort(common)
    n = len(common)
    if n < 3:
        print(f"ERROR: only {n} images are common to every input", file=sys.stderr)
        return 1
    label = "centred" if args.centred else "raw"
    print(f"{n} images common to all {len(inputs)} inputs ({label}{'' if args.no_normalise else ', min-max normalised'})")
    kernels, shown = [], []
    for inp in inputs:
        K = restrict(inp, common)
        if args.centred:
            K = centre(K)
        kernels.append(K)
        shown.append(K if args.no_normalise else normalise(K))
    off = ~np.eye(n, dtype=bool)
    width = max(len(l) for l in labels)
    for inp, K in zip(inputs, kernels):
        what = "kernel folder" if inp["is_folder"] else f"matrix, {inp['kind']}"
        print(f"  {inp['label']:<{width}}  {len(inp['ids'])} images ({len(inp['ids']) - n} not common; {what}); "
              f"off-diagonal min {K[off].min():.3f} median {np.median(K[off]):.3f} max {K[off].max():.3f}")

    # ---- every pair ------------------------------------------------------------------------
    N = len(inputs)
    stats = {}
    pearson = np.eye(N)
    spearman = np.eye(N)
    for i, j in combinations(range(N), 2):
        s = pair_stats(shown[i], shown[j], common)
        stats[(i, j)] = s
        pearson[i, j] = pearson[j, i] = s["pearson"]
        spearman[i, j] = spearman[j, i] = s["spearman"]
        print(f"\n{labels[i]} vs {labels[j]}: Pearson {s['pearson']:.3f}, Spearman {s['spearman']:.3f}; "
              f"difference {labels[i]} - {labels[j]}: mean {s['mean']:+.3f}, mean |.| {s['mean_abs']:.3f}, "
              f"max |.| {s['max_abs']:.3f}")
        print(f"  most disagreeing pairs (frame ids, {labels[i]}, {labels[j]}, diff):")
        for fi, fj, a, b in s["worst"]:
            print(f"    {fi:5d} {fj:5d}   {a:.3f}  {b:.3f}  {a - b:+.3f}")
    if N > 2:
        for name, M in (("Pearson", pearson), ("Spearman", spearman)):
            print(f"\n{name}:")
            print("  " + " " * width + "".join(f"  {l[:8]:>8}" for l in labels))
            for i in range(N):
                print(f"  {labels[i]:<{width}}" + "".join(f"  {M[i, j]:8.3f}" for j in range(N)))

    # ---- files -----------------------------------------------------------------------------
    out = args.out if args.out is not None else next((inp["path"] for inp in inputs if inp["is_folder"]), Path("."))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"pairs{'_centred' if args.centred else ''}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "images", "pearson", "spearman", "mean_diff", "mean_abs_diff", "max_abs_diff"])
        for (i, j), s in stats.items():
            w.writerow([labels[i], labels[j], n, f"{s['pearson']:.6f}", f"{s['spearman']:.6f}", f"{s['mean']:.6f}",
                        f"{s['mean_abs']:.6f}", f"{s['max_abs']:.6f}"])

    # ---- figure: heatmaps on the diagonal, differences below, scatters above ---------------
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(N, N, figsize=(4.4 * N, 4.2 * N), squeeze=False)
    tick_pos = np.linspace(0, n - 1, min(n, 6)).round().astype(int)
    tick_lab = [str(common[t]) for t in tick_pos]
    vmin, vmax = (None, None) if args.no_normalise else (0.0, 1.0)
    lim = max(float(np.abs(shown[i] - shown[j])[off].max()) for i, j in combinations(range(N), 2)) or 1.0
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            if i == j:
                im = ax.imshow(shown[i], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"{labels[i]} ({label})", fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            elif i > j:
                im = ax.imshow(shown[i] - shown[j], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.set_title(f"{labels[i]} − {labels[j]}", fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            else:
                s = stats[(i, j)]
                a, b = shown[i][off], shown[j][off]
                ax.scatter(b, a, s=4, alpha=0.3, color="#1f5fbf", edgecolors="none")
                lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
                ax.plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--")
                ax.set_xlabel(labels[j], fontsize=9)
                ax.set_ylabel(labels[i], fontsize=9)
                ax.set_title(f"Pearson {s['pearson']:.2f}, Spearman {s['spearman']:.2f}", fontsize=10)
                ax.set_aspect("equal", adjustable="box")
                continue
            ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=45, fontsize=7)
            ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=7)
    fig.suptitle(f"{n} images common to {', '.join(labels)} (frame ids = rgb.csv rows)"
                 f"{'' if args.no_normalise else ' — each min-max normalised to [0,1]'}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure = out / f"compare_kernels{'_centred' if args.centred else ''}.png"
    fig.savefig(figure, dpi=110)
    print(f"\nfigure: {figure}\npairs : {out / ('pairs' + ('_centred' if args.centred else '') + '.csv')}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
