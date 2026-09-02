# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`placecell` is a C++17 library (with Python bindings) for keyframe lifecycle management in VSLAM / 3D reconstruction: keyframe creation, connectivity, redundancy detection, and marginalization. Current state:

- `placecell::PlaceCell` (core, Eigen-only): id-mapped, append-only store of global descriptors (`add(external_id, descriptor)` -> internal id; `descriptor(id)` returns a stable pointer) plus the similarity kernel over them (`kernel()`, Gram matrix grown incrementally on every add; rows are never removed — culled views stay as history). `external_ids()` maps kernel rows back to host ids; `clear()` resets everything. Thread-safe.
- `placecell::MegaLocEmbedder` (megaloc module, `PLACECELL_WITH_MEGALOC=ON`): cv::Mat -> 8448-d MegaLoc descriptor via TensorRT (pimpl hides TensorRT/CUDA from the public header); engine built on first use and cached next to the ONNX.
- `placecell::MegaLocPlaceCell` (megaloc module): PlaceCell + embedder; `add_image(id, cv::Mat)` = embed + store, idempotent.
- Consumed by AllFeature-VSLAM as `Thirdparty/placecell` (System owns a `MegaLocPlaceCell`; keyframe descriptors and the VPR kernel live here, not in the SLAM code).

The `megaloc` pixi environment provides CUDA/TensorRT/OpenCV; `pixi run -e megaloc build-megaloc` builds it, `download-models` fetches the ONNX from HF `vslamlab/megaloc-models`, and `examples/megaloc_embedder_smoke` is the store/kernel verification harness. There are no unit tests yet beyond that smoke example. The Python bindings still expose only the empty `PlaceCell()` constructor — NOT yet in sync with the C++ API.

## Build and run

pixi is the primary workflow (linux-64 only, per `pixi.toml`). It enables the Python bindings; plain CMake leaves them off by default.

```bash
pixi run build          # cmake configure (Ninja, Release, PLACECELL_BUILD_PYTHON=ON) + build everything
pixi run example        # run examples/main.cpp (build/examples/placecell_example)
pixi run python-smoke   # import the compiled _placecell module straight from build/python
pixi run clean          # rm -rf build
```

Plain CMake (needs a C++17 compiler and Eigen3; add `-DPLACECELL_BUILD_PYTHON=ON` for bindings, which also needs Python + nanobind):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Python wheel via scikit-build-core (turns examples OFF, bindings ON):

```bash
pip install .
python -c "import placecell; print(placecell.PlaceCell())"
```

CMake options: `PLACECELL_BUILD_EXAMPLES` (default ON), `PLACECELL_BUILD_PYTHON` (default OFF). `compile_commands.json` is exported into `build/`.

## Architecture

- `include/placecell/` — public headers; the only include path consumers get. Downstream CMake projects link `placecell::placecell` (alias of the `placecell` static library).
- `src/` — library implementation. Eigen is a PUBLIC dependency of the library target.
- `examples/main.cpp` — scratch executable for development, not a demo for users.
- `python/bindings.cpp` — nanobind module named `_placecell`. `python/placecell/__init__.py` is the pure-Python package that re-exports from `._placecell`. The compiled module only lands inside the package on `pip install` (the `SKBUILD` install rule); in a plain build it sits in `build/python/`, which is why `pixi run python-smoke` imports `_placecell` directly rather than `placecell`.
- `python/CMakeLists.txt` locates nanobind via `python -m nanobind --cmake_dir`, so the Python interpreter that CMake finds must have nanobind installed (pixi provides this).

When adding public API: declare in `include/placecell/placecell.h`, implement in `src/`, and expose it in `python/bindings.cpp` so the Python package stays in sync.

## Conventions

- Allman braces (opening brace on its own line for namespaces, classes, functions), 4-space indent, trailing-underscore private members (`kernel_`).
- Every source file starts with a doc-comment header (`placecell — ...`, Author, Created, License).
- Library and example compile with `-Wall -Wextra -Wpedantic`; keep new code warning-free.
- `pixi.lock` is marked binary for merges in `.gitattributes`; don't hand-edit it.
