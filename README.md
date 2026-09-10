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
pixi run kernel-demo <matrix.npy> [--kind K] [--tau T]   # kernel-only store from a precomputed pairwise matrix; offline cull -> <out>/rgb.csv of surviving frames
pixi run colmap-kernel <colmap_model_dir> --rgb-csv <sequence>/rgb.csv   # shared-information kernel of a COLMAP reconstruction -> kernel.npy + ids.csv
pixi run compare-kernels <D.npy> <colmap_kernel_dir> [--centred]         # both kernels on the registered images: heatmaps, difference, scatter
pixi run colmap-demo <colmap_model_dir> --rgb-csv <sequence>/rgb.csv [--tau 0.9]   # kernel -> offline cull -> rgb.csv of surviving frames, with checks
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

## Kernel-only stores

A store can also be initialised from a precomputed pairwise matrix instead of descriptors
(`include/placecell/kernel_io.h`, `PlaceCell::set_kernel`):

```cpp
Eigen::MatrixXf D = placecell::load_npy("vpr-lab/D.npy");                      // e.g. faiss squared L2
Eigen::MatrixXf S = placecell::similarity_from_distance(D, placecell::DistanceKind::squared_euclidean);
placecell::PlaceCell cell;
auto report = cell.set_kernel(S);      // ids 0..n-1; symmetrises, unit diagonal, checks the spectrum
cell.cull_keyframes(params, [](placecell::PlaceCell::ExternalId) { return true; });   // offline selection
```

Such a store has no descriptors: `add()` is refused and the descriptor query returns NaN, while
culling, snapshots and dumps work as usual. `params.max_unexplained` (tau) bounds what a cull may
leave unexplained; alternatively `params.target_alive = N` runs the same greedy order until N views
remain (count-driven, tau ignored) — the "keep the N least redundant frames" selection.

`tools/colmap_information_kernel.py <model_dir> --rgb-csv <sequence>/rgb.csv` builds such a kernel
from a COLMAP reconstruction: the normalised mutual information between every two images'
measurements in the joint bundle-adjustment problem (poses + points, Gauss-Newton Hessian from
the reprojection Jacobians). It writes `kernel.npy` + `ids.csv` for
`kernel_demo <out>/kernel.npy --kind similarity --ids <out>/ids.csv`.

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

The bindings cover the core (no MegaLoc / OpenCV), including the kernel-only path. Offline
selection of the N least redundant frames of a sequence from a precomputed pairwise matrix:

```python
import numpy as np, placecell as pc

D = np.load("vpr-lab/D.npy").astype(np.float32)               # faiss squared L2 on unit descriptors
S = pc.similarity_from_distance(D, "squared-euclidean")       # S = 1 - D/2
cell = pc.PlaceCell()
options = pc.PlaceCell.KernelOptions(); options.clip_to_psd = True
cell.set_kernel(S, None, options)                             # ids 0..n-1, symmetrised, PSD-clipped

params = pc.PlaceCell.CullParameters()
params.target_alive = 100; params.min_keyframes = 1           # count-driven: stop at 100 alive views
report = cell.cull_keyframes(params, lambda frame: True)
kept = [i for i in range(len(cell)) if not cell.is_culled(i)]
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
