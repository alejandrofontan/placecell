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
pixi run build          # configure + compile library, examples, and Python module
pixi run example        # run examples/main.cpp
pixi run demo           # GPU-free synthetic run: queries, insertions, culls, profile table, dump
pixi run plot           # matplotlib plots of that dump (tools/plot_placecell.py)
pixi run python-smoke   # import the Python module from the build tree
pixi run clean          # remove the build directory
```

## Diagnostics

Every `PlaceCell` carries a profiler (timing of the main entry points) and a recorder
(query / cull / decision history); logging is process-wide with a `[placecell]` tag.

```bash
PLACECELL_VERBOSITY=debug ./your_host ...     # off | error | warn (default) | info | debug | trace
```

```cpp
placecell::PlaceCell::Options options;
options.verbosity = placecell::LogLevel::info;
options.report_on_destruction = true;         // profile table when the store dies
placecell::PlaceCell cell(options);
cell.recorder().set_thresholds(tau, min_information);
cell.recorder().record_decision(id, inserted, v);   // what only the host knows
cell.print_profile();
cell.dump("placecell_out");                   // kernel .npy + CSVs -> tools/plot_placecell.py
```

With `-DPLACECELL_WITH_VIZ=ON` (OpenCV) the `placecell::viz` target renders the kernel heatmap,
the unexplained-information history and the alive-information strip to `cv::Mat`, or shows them
live through `placecell::viz::Visualizer`.

## Build (plain CMake)

Requires a C++17 compiler and Eigen3.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CMake options: `PLACECELL_BUILD_EXAMPLES` (default `ON`), `PLACECELL_BUILD_PYTHON` (default `OFF`, needs Python + nanobind), `PLACECELL_WITH_MEGALOC` (default `OFF`, CUDA + TensorRT + OpenCV + yaml-cpp), `PLACECELL_WITH_VIZ` (default `OFF`, OpenCV).

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
