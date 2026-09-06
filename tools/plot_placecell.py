"""
Module: placecell - plot_placecell.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-09-05
- License: Apache-2.0

Offline visualizer for a PlaceCell::dump() directory (kernel.npy, kernel_centred.npy,
views.csv, queries.csv, culls.csv, cull_calls.csv, alive_information.csv, decisions.csv,
thresholds.csv, profile.csv). Same three views as the C++ placecell::viz module plus the
profile distributions, with matplotlib:

    python tools/plot_placecell.py <dump_dir>            # interactive windows
    python tools/plot_placecell.py <dump_dir> --save     # PNGs next to the CSVs
    python tools/plot_placecell.py <dump_dir> --raw      # raw instead of centred kernel

Needs numpy + matplotlib only (no placecell import).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    sys.exit("plot_placecell.py needs matplotlib (pip install matplotlib)")


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Columns as float arrays (non-numeric columns as object arrays); {} when missing/empty."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        values = [r[key] for r in rows]
        try:
            out[key] = np.array([float(v) for v in values])
        except ValueError:
            out[key] = np.array(values, dtype=object)
    return out


def plot_kernel(ax, dump: Path, centred: bool) -> None:
    name = "kernel_centred.npy" if centred else "kernel.npy"
    kernel = np.load(dump / name)
    views = read_csv(dump / "views.csv")
    n = kernel.shape[0]
    if n == 0:
        ax.text(0.5, 0.5, "empty store", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    vmin, vmax = (-1.0, 1.0) if centred else (float(np.nanmin(kernel)), 1.0)
    image = ax.imshow(kernel, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    if views:
        culled = views["culled"].astype(bool)
        # dim history rows/cols with a translucent overlay
        mask = np.zeros((n, n, 4))
        mask[culled, :, 3] = 0.55
        mask[:, culled, 3] = 0.55
        ax.imshow(mask, interpolation="nearest")
        protected = np.flatnonzero(views["protected"].astype(bool))
        if protected.size:
            ax.plot(protected, np.full_like(protected, -0.5), "|", color="white", markersize=8)
        alive = int((~culled).sum())
        ax.set_title(f"kernel ({'centred' if centred else 'raw'})  n={n}  alive={alive}  history={n - alive}")
    else:
        ax.set_title(f"kernel ({'centred' if centred else 'raw'})  n={n}")
    ax.set_xlabel("internal id")
    ax.set_ylabel("internal id")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def plot_information(ax, dump: Path) -> None:
    queries = read_csv(dump / "queries.csv")
    if not queries:
        ax.text(0.5, 0.5, "no queries", ha="center", va="center", transform=ax.transAxes)
        return
    x = queries["index"]
    v = queries["unexplained"]
    t = queries["time_s"]
    ax.plot(x, v, color="#3cd2ff", linewidth=0.9, label="v (query)")

    thresholds = read_csv(dump / "thresholds.csv")
    if thresholds:
        # step lines mapped from time to query index
        for key, color, label in (("tau", "#ff5050", "tau"), ("min_information", "#50dc50", "min info")):
            values = thresholds[key]
            times = thresholds["time_s"]
            for k, value in enumerate(values):
                if np.isnan(value):
                    continue
                start = 0 if k == 0 else int(np.searchsorted(t, times[k]))
                end = int(np.searchsorted(t, times[k + 1])) if k + 1 < len(values) else len(x) - 1
                ax.hlines(value, x[start], x[max(start, end)], colors=color, linestyles="--", linewidth=1,
                          label=label if k == 0 else None)

    decisions = read_csv(dump / "decisions.csv")
    if decisions:
        inserted = decisions["inserted"].astype(bool)
        qi = decisions["query_index"][inserted].astype(int)
        qi = np.clip(qi, 0, len(v) - 1)
        vv = decisions["unexplained"][inserted]
        vv = np.where(np.isnan(vv), v[qi], vv)
        ax.plot(qi, vv, "^", color="#ffdc00", markersize=5, label=f"inserted ({inserted.sum()})")

    culls = read_csv(dump / "culls.csv")
    if culls:
        qi = np.searchsorted(t, culls["time_s"])
        qi = np.clip(qi, 0, len(v) - 1)
        ax.plot(qi, culls["unique_information"], ".", color="#ff50ff", markersize=4, label=f"culled ({len(qi)})")
        ax.vlines(qi, 0.0, 0.03, colors="#ff50ff", linewidth=0.8)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x[0], max(x[-1], x[0] + 1))
    ax.set_xlabel("query")
    ax.set_ylabel("unexplained information")
    ax.set_title(f"unexplained information  queries={len(x)}  last={v[-1]:.3f}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=5)


def plot_alive(ax, dump: Path) -> None:
    alive = read_csv(dump / "alive_information.csv")
    calls = read_csv(dump / "cull_calls.csv")
    if not alive:
        ax.text(0.5, 0.5, "no cull_keyframes call", ha="center", va="center", transform=ax.transAxes)
        return
    values = alive["unique_information"]
    tau = float(calls["tau"][-1]) if calls else float("nan")
    colors = np.where(np.isnan(values), "#6e6e6e", np.where(values > tau, "#3cd2ff", "#a0a0a0"))
    ax.bar(np.arange(len(values)), np.nan_to_num(values, nan=1.0), color=colors, width=0.9)
    if not np.isnan(tau):
        ax.axhline(tau, color="#ff5050", linestyle="--", linewidth=1, label=f"tau={tau:.2f}")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("alive view (insertion order)")
    ax.set_ylabel("unique information")
    title = f"unique information of alive views  alive={len(values)}"
    if calls:
        title += f"  after cull #{int(calls['index'][-1])}  culled={int(calls['culled'][-1])}/{int(calls['candidates'][-1])}"
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")


def plot_profile(ax, dump: Path) -> None:
    profile = read_csv(dump / "profile.csv")
    if not profile:
        ax.text(0.5, 0.5, "no profile samples", ha="center", va="center", transform=ax.transAxes)
        return
    functions = [f for f in dict.fromkeys(profile["function"].tolist())]
    data = [profile["ms"][profile["function"] == f] for f in functions]
    # (tick labels set separately: the boxplot `labels` kwarg was renamed across matplotlib versions)
    ax.boxplot(data, vert=False, showfliers=True, flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_yticks(range(1, len(functions) + 1), ["   " + f.split("/")[-1] if "/" in f else f for f in functions])
    ax.set_xscale("log")
    ax.set_xlabel("ms (log)")
    ax.set_title("profile: per-call time of the main entry points")
    ax.grid(alpha=0.25, axis="x")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a PlaceCell::dump() directory")
    parser.add_argument("dump", type=Path)
    parser.add_argument("--save", action="store_true", help="write PNGs into the dump directory instead of showing")
    parser.add_argument("--raw", action="store_true", help="raw kernel instead of the centred one")
    args = parser.parse_args()
    if not args.dump.is_dir():
        sys.exit(f"{args.dump} is not a directory")
    if args.save:
        matplotlib.use("Agg")

    plt.style.use("dark_background")
    fig_kernel, ax_kernel = plt.subplots(figsize=(8, 7.5))
    plot_kernel(ax_kernel, args.dump, centred=not args.raw)
    fig_history, (ax_info, ax_alive) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    plot_information(ax_info, args.dump)
    plot_alive(ax_alive, args.dump)
    fig_profile, ax_profile = plt.subplots(figsize=(9, 4))
    plot_profile(ax_profile, args.dump)
    for fig in (fig_kernel, fig_history, fig_profile):
        fig.tight_layout()

    if args.save:
        fig_kernel.savefig(args.dump / "kernel_mpl.png", dpi=110)
        fig_history.savefig(args.dump / "information_mpl.png", dpi=110)
        fig_profile.savefig(args.dump / "profile_mpl.png", dpi=110)
        print(f"wrote kernel_mpl.png, information_mpl.png, profile_mpl.png to {args.dump}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
