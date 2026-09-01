# placecell

Keyframe lifecycle management for VSLAM and 3D reconstruction — creation, connectivity, redundancy, marginalization.

## Layout

```
include/placecell/   public headers
src/                 library sources
examples/            scratch executables for development
python/              nanobind Python bindings
```

## Build (pixi)

Install [pixi](https://pixi.sh), then from the repo root:

```bash
pixi run build          # configure + compile library, example, and Python module
pixi run example        # run examples/main.cpp
pixi run python-smoke   # import the Python module from the build tree
pixi run clean          # remove the build directory
```

## Build (plain CMake)

Requires a C++17 compiler and Eigen3.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CMake options: `PLACECELL_BUILD_EXAMPLES` (default `ON`), `PLACECELL_BUILD_PYTHON` (default `OFF`, needs Python + nanobind).

## Use from another CMake project

```cmake
add_subdirectory(placecell)
target_link_libraries(your_target PRIVATE placecell::placecell)
```

## Python package

```bash
pip install .           # builds the wheel via scikit-build-core
python -c "import placecell; print(placecell.PlaceCell())"
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
