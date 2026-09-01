# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`placecell` is a C++17 library (with Python bindings) for keyframe lifecycle management in VSLAM / 3D reconstruction: keyframe creation, connectivity, redundancy detection, and marginalization. It is currently a skeleton — `placecell::PlaceCell` holds a similarity kernel (`Eigen::MatrixXf` Gram matrix of view descriptors) and nothing else yet. There are no tests yet.

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
