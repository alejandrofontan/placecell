"""
Module: placecell - download_models.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-10-13
- License: Apache-2.0

Fetch the published MegaLoc ONNX files + sidecars (HF vslamlab/megaloc-models) into
megaloc_models/ WITHOUT ever overwriting a file that is already there: a local
tools/export_megaloc.py run produces the same names, and `hf download --local-dir`
replaces them with the published versions (it did once, 2026-10-13). Every file is
checked individually; existing ones are reported and kept.

Usage: python tools/download_models.py [--repo vslamlab/megaloc-models] [--dir megaloc_models]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "megaloc_models" / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--repo", default="vslamlab/megaloc-models")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "megaloc_models")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    args.dir.mkdir(parents=True, exist_ok=True)
    names = [f for f in list_repo_files(args.repo) if f.endswith((".onnx", ".yaml", ".txt"))]
    for name in sorted(names):
        target = args.dir / name
        if target.exists():
            print(f"kept     {target} ({target.stat().st_size / 2**20:.1f} MB, already present)")
            continue
        print(f"download {args.repo}/{name} -> {target}")
        hf_hub_download(repo_id=args.repo, filename=name, local_dir=str(args.dir))


if __name__ == "__main__":
    main()
