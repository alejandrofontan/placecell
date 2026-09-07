#!/usr/bin/env python3
"""
placecell — compare_kernels.py

Author:  Alejandro Fontan
Assisted by: Claude (Fable 5)
Created: 2026-09-07
License: Apache-2.0

Side-by-side of two kernels over the same images: the appearance kernel from a VPR
distance matrix (VPR-LAB D.npy -> S = 1 - D/2, see kernel_io.h) and the geometric
shared-information kernel of a COLMAP reconstruction (tools/colmap_information_kernel.py).
The COLMAP kernel's ids.csv gives the rgb.csv row of every registered image, and D.npy is
indexed by the same rows, so the VPR kernel is restricted to those images. Both kernels
are min-max normalised to [0,1] over their off-diagonal entries (the diagonal stays 1) —
the VPR cosine has a ~0.37 common-mode floor, the information kernel lives in 0-0.6 — so
the heatmaps show the same range. Optionally both are double-centred first (placecell's
Pearson correlation of mean-centred descriptors, what the culler marginalises).

Figure: VPR heatmap | COLMAP heatmap | difference (VPR - COLMAP, diverging) | scatter of
the paired off-diagonal entries. Stats on stdout: Pearson / Spearman correlation of the
off-diagonal entries, mean and max absolute difference, the most disagreeing pairs.

Usage:
    compare_kernels.py <D.npy> <colmap_kernel_dir> [--kind squared-euclidean] [--out fig.png]
                       [--centred] [--no-normalise] [--show]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vpr_matrix", type=Path, help="VPR-LAB D.npy (rows = rgb.csv rows)")
    ap.add_argument("colmap_kernel_dir", type=Path, help="output folder of colmap_information_kernel.py (kernel.npy + ids.csv)")
    ap.add_argument("--kind", default="squared-euclidean", help="distance convention of the VPR matrix [squared-euclidean]")
    ap.add_argument("--out", type=Path, default=None, help="figure path [<colmap_kernel_dir>/compare_kernels.png]")
    ap.add_argument("--centred", action="store_true", help="double-centre both kernels first (what the culler uses)")
    ap.add_argument("--no-normalise", action="store_true", help="show the kernels as they are instead of min-max 0-1")
    ap.add_argument("--show", action="store_true", help="open a matplotlib window as well")
    args = ap.parse_args()

    ids, names = load_ids(args.colmap_kernel_dir / "ids.csv")
    K_colmap = np.load(args.colmap_kernel_dir / "kernel.npy").astype(np.float64)
    D = np.load(args.vpr_matrix).astype(np.float64)
    if ids.max() >= len(D):
        print(f"ERROR: ids reach {ids.max()} but {args.vpr_matrix} has {len(D)} rows", file=sys.stderr)
        return 1
    S = similarity_from_distance(D, args.kind)
    S = 0.5 * (S + S.T)                    # the rotation-min matrix is asymmetric (see kernel_io.h)
    order = np.argsort(ids)                # sequence order along both axes
    ids_sorted = ids[order]
    K_vpr = S[np.ix_(ids_sorted, ids_sorted)].copy()
    np.fill_diagonal(K_vpr, 1.0)
    K_col = K_colmap[np.ix_(order, order)].copy()
    np.fill_diagonal(K_col, 1.0)
    n = len(ids_sorted)
    label = "raw"
    if args.centred:
        K_vpr, K_col = centre(K_vpr), centre(K_col)
        label = "centred"
    A = K_vpr if args.no_normalise else normalise(K_vpr)
    B = K_col if args.no_normalise else normalise(K_col)
    diff = A - B
    off = ~np.eye(n, dtype=bool)

    # ---- stats -------------------------------------------------------------------------
    a, b = A[off], B[off]
    pearson = np.corrcoef(a, b)[0, 1]
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    spearman = np.corrcoef(ra, rb)[0, 1]
    print(f"{n} images shared by both kernels ({label}{'' if args.no_normalise else ', min-max normalised'})")
    print(f"VPR    off-diagonal: min {K_vpr[off].min():.3f} median {np.median(K_vpr[off]):.3f} max {K_vpr[off].max():.3f}")
    print(f"COLMAP off-diagonal: min {K_col[off].min():.3f} median {np.median(K_col[off]):.3f} max {K_col[off].max():.3f}")
    print(f"correlation of paired entries: Pearson {pearson:.3f}, Spearman {spearman:.3f}")
    print(f"difference VPR - COLMAP: mean {diff[off].mean():+.3f}, mean |.| {np.abs(diff[off]).mean():.3f}, max |.| {np.abs(diff[off]).max():.3f}")
    iu = np.triu_indices(n, 1)
    worst = np.argsort(-np.abs(diff[iu]))[:5]
    print("most disagreeing pairs (frame ids, VPR, COLMAP, diff):")
    for w in worst:
        i, j = iu[0][w], iu[1][w]
        print(f"  {ids_sorted[i]:5d} {ids_sorted[j]:5d}   {A[i, j]:.3f}  {B[i, j]:.3f}  {diff[i, j]:+.3f}")

    # ---- figure ------------------------------------------------------------------------
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    tick_pos = np.linspace(0, n - 1, min(n, 8)).round().astype(int)
    tick_lab = [str(ids_sorted[t]) for t in tick_pos]
    vmin, vmax = (None, None) if args.no_normalise else (0.0, 1.0)
    for ax, M, title, cmap in ((axes[0], A, f"VPR (MegaLoc) kernel, {label}", "viridis"),
                               (axes[1], B, f"COLMAP shared-information kernel, {label}", "viridis")):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=45, fontsize=8)
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=8)
        ax.set_xlabel("frame id (rgb.csv row)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    lim = float(np.abs(diff[off]).max()) or 1.0
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axes[2].set_title("difference: VPR − COLMAP")
    axes[2].set_xticks(tick_pos); axes[2].set_xticklabels(tick_lab, rotation=45, fontsize=8)
    axes[2].set_yticks(tick_pos); axes[2].set_yticklabels(tick_lab, fontsize=8)
    axes[2].set_xlabel("frame id (rgb.csv row)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    axes[3].scatter(a, b, s=6, alpha=0.35, color="#1f5fbf", edgecolors="none")
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    axes[3].plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--")
    axes[3].set_xlabel("VPR entry"); axes[3].set_ylabel("COLMAP entry")
    axes[3].set_title(f"paired off-diagonal entries\nPearson {pearson:.2f}, Spearman {spearman:.2f}")
    axes[3].set_aspect("equal", adjustable="box")
    fig.suptitle(f"{n} images registered by COLMAP — {args.vpr_matrix.name} ({args.kind}) vs {args.colmap_kernel_dir / 'kernel.npy'}"
                 f"{'' if args.no_normalise else ' — both min-max normalised to [0,1]'}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = args.out if args.out is not None else args.colmap_kernel_dir / f"compare_kernels{'_centred' if args.centred else ''}.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
