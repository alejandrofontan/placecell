#!/usr/bin/env python3
"""
placecell — colmap_kernel_demo.py

Author:  Alejandro Fontan
Assisted by: Claude (Fable 5)
Created: 2026-09-07
License: Apache-2.0

Offline keyframe selection from a COLMAP reconstruction — the counterpart of
`kernel_demo <D.npy>` for the geometric kernel. Chains the two existing pieces and
checks the result:

  1. tools/colmap_information_kernel.py <model_dir> --rgb-csv <csv>
         -> <out>/kernel.npy (normalised BA mutual information), ids.csv
  2. build/examples/kernel_demo <out>/kernel.npy --kind similarity --ids <out>/ids.csv
         --rgb-csv <csv> --tau T [...] --out <out>/selection
         -> the information culler over every registered image, dump, and
            <out>/selection/rgb.csv with the frames that survive
  3. checks: the survivors are registered images (names match ids.csv), the culled
     ones are exactly the complement, the first frame of the sequence is kept
     (kernel_demo protects it), and the count agrees with kernel_demo's report.

tau is on the information kernel's own scale: an image's unique information is
typically 0.5-0.9 here (most of its gain is points/pose the pair does not share), so the
useful range is about 0.7-0.95; the default is 0.9 (see CLAUDE.md for the sweep).

Usage:
    colmap_kernel_demo.py <model_dir> --rgb-csv <sequence>/rgb.csv [--tau 0.9]
        [--min-keyframes 5] [--raw] [--clip] [--out dir] [--kernel-demo path] [--verbosity L]
        [--skip-kernel]   (reuse an existing <out>/kernel.npy + ids.csv)

Exit code 1 when any step or check fails.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run(cmd: list[str], label: str) -> str:
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[{label}] failed with exit code {proc.returncode}", file=sys.stderr)
        sys.exit(1)
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="COLMAP model folder (the best sub-model, see its best_model file)")
    ap.add_argument("--rgb-csv", type=Path, required=True, help="sequence rgb.csv (or rgb_raw.csv): ids and surviving rows")
    ap.add_argument("--tau", type=float, default=0.9, help="max unexplained information per view [0.9]")
    ap.add_argument("--min-keyframes", type=int, default=5)
    ap.add_argument("--raw", action="store_true", help="marginalise on the raw kernel instead of the centred one")
    ap.add_argument("--clip", action="store_true", help="project the kernel to PSD before culling (not needed when PSD)")
    ap.add_argument("--out", type=Path, default=None, help="output folder [<model_dir>/information_kernel]")
    ap.add_argument("--kernel-demo", type=Path, default=REPO / "build" / "examples" / "kernel_demo")
    ap.add_argument("--verbosity", default="warn", help="placecell log level for kernel_demo [warn]")
    ap.add_argument("--skip-kernel", action="store_true", help="reuse <out>/kernel.npy and ids.csv")
    args = ap.parse_args()

    out = args.out if args.out is not None else args.model_dir / "information_kernel"
    if not args.kernel_demo.exists():
        print(f"ERROR: {args.kernel_demo} not found — build placecell first (pixi run build)", file=sys.stderr)
        return 1
    if not args.rgb_csv.exists():
        print(f"ERROR: {args.rgb_csv} not found", file=sys.stderr)
        return 1

    # ---- 1. kernel ---------------------------------------------------------------------
    if not args.skip_kernel or not (out / "kernel.npy").exists():
        run([sys.executable, str(HERE / "colmap_information_kernel.py"), str(args.model_dir),
             "--rgb-csv", str(args.rgb_csv), "--out", str(out)], "kernel")
    else:
        print(f"[kernel] reusing {out / 'kernel.npy'}")

    # ---- 2. cull -----------------------------------------------------------------------
    selection = out / f"selection_tau{args.tau:g}{'_raw' if args.raw else ''}"
    cmd = [str(args.kernel_demo), str(out / "kernel.npy"), "--kind", "similarity", "--ids", str(out / "ids.csv"),
           "--rgb-csv", str(args.rgb_csv), "--tau", str(args.tau), "--min-keyframes", str(args.min_keyframes),
           "--out", str(selection), "--verbosity", args.verbosity]
    if args.raw:
        cmd.append("--raw")
    if args.clip:
        cmd.append("--clip")
    stdout = run(cmd, "cull")
    m = re.search(r"-> (\d+) keyframes survive", stdout)
    reported = int(m.group(1)) if m else -1

    # ---- 3. checks ---------------------------------------------------------------------
    ok = True
    ids_rows = list(csv.DictReader(open(out / "ids.csv", newline="")))
    registered = {int(r["external_id"]): os.path.basename(r["name"]) for r in ids_rows}
    views = list(csv.DictReader(open(selection / "views.csv", newline="")))
    alive = sorted(int(v["external_id"]) for v in views if v["culled"] == "0")
    culled = sorted(int(v["external_id"]) for v in views if v["culled"] != "0")
    with open(args.rgb_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index("path_rgb_0") if "path_rgb_0" in header else 1
        source = [row for row in reader if row]
    with open(selection / "rgb.csv", newline="") as f:
        reader = csv.reader(f)
        out_header = next(reader)
        survivors = [row for row in reader if row]

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"CHECK FAILED: {msg}", file=sys.stderr)

    if out_header != header:
        fail("surviving rgb.csv header differs from the source")
    if len(survivors) != len(alive) or (reported >= 0 and reported != len(alive)):
        fail(f"{len(survivors)} rows written, {len(alive)} alive views, kernel_demo reported {reported}")
    if sorted(alive + culled) != sorted(registered):
        fail("alive + culled views are not exactly the registered images")
    if alive and alive[0] != min(registered):
        fail("the first registered frame was culled (kernel_demo protects it)")
    for row, i in zip(survivors, alive):
        if row != source[i]:
            fail(f"row for id {i} is not source row {i}")
            break
        if os.path.basename(row[col]) != registered[i]:
            fail(f"id {i}: csv image {os.path.basename(row[col])} != registered {registered[i]}")
            break

    print(f"\ncolmap kernel demo: {len(registered)} registered images -> {len(alive)} kept, {len(culled)} culled at tau {args.tau:g}"
          f" ({'raw' if args.raw else 'centred'} kernel)")
    print(f"  kept   : {alive}")
    print(f"  culled : {culled}")
    print(f"  surviving frames: {selection / 'rgb.csv'}")
    print(f"  dump + plots     : {selection}")
    print("colmap kernel demo OK" if ok else "colmap kernel demo FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
