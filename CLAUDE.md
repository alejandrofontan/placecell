# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`placecell` is a C++17 library (with nanobind Python bindings) for keyframe lifecycle management in VSLAM / 3D reconstruction: keyframe creation, connectivity, redundancy detection, and marginalization. It is a standalone first-party repo (`github.com/alejandrofontan/placecell`, Apache-2.0) that AllFeature-VSLAM consumes as the git submodule `Thirdparty/placecell`. CMake targets:

- `placecell::placecell` — the Eigen-only core (`src/placecell.cpp`, `include/placecell/placecell.h`) plus the three diagnostic managers (`log.h`, `profiler.h`, `recorder.h`) and `kernel_io.h` (`.npy` read/write, distance → similarity conversion). Descriptors come in already computed; no image or model code here.
- `placecell::megaloc` — optional (`PLACECELL_WITH_MEGALOC=ON`), needs CUDA 12 + TensorRT + OpenCV + yaml-cpp: the MegaLoc image frontend (`src/megaloc/`). Links the core transitively.
- `placecell::viz` — optional (`PLACECELL_WITH_VIZ=ON`), needs OpenCV only: renders the kernel heatmap, the unexplained-information history and the alive-information strip (`include/placecell/viz.h`).

Design goal for every change: placecell must stay independent of any particular SLAM/SfM system, and the kernel's metric (MegaLoc cosine today) must become swappable — GitHub issues #1–#5 track the gaps (pluggable similarity strategy, host-supplied kernel rows, the PSD/unit-diagonal kernel contract, centring as a per-metric default, the singular-`K_AA` degeneracy).

## Documentation map

| Where | What |
|---|---|
| this file | how to build, verify and change things; the architecture in one screen |
| header comments in `include/placecell/*.h` | the **contracts** (idempotence, pointer stability, thread-safety, NaN semantics) — update them when behaviour changes |
| the comment block at the top of `cull_keyframes` in `src/placecell.cpp` | the culler's derivation, centring rationale and threshold-change handling — read before touching the loop |
| `docs/notes/2026-09-25_kernels.md` | dated reference numbers, sweeps and design decisions for the three store modes, the COLMAP kernel and the tools (append a dated paragraph, never rewrite) |
| `README.md` | user-facing usage, C++ and Python snippets |
| `docs/megaloc.md` | placeholder TODO |
| GitHub issues (`gh issue list`) | anything actionable; file instead of writing TODOs into files |

Do not append investigations or measurements to this file — they go into a dated `docs/notes/` page (the kernels note above is the first); keep here only what changes how to work.

## Build and run

pixi is the primary workflow (linux-64 with CUDA only, per `pixi.toml`). Two environments, two build directories — don't mix them:

```bash
# default env: core + examples + Python bindings, in build/
pixi run build          # cmake configure (Ninja, Release, PLACECELL_BUILD_PYTHON=ON) + build
pixi run example        # build/examples/placecell_example (examples/main.cpp, a skeleton check)
pixi run demo           # build/examples/synthetic_demo -> placecell_demo_out/ (GPU-free smoke test, exit 1 on mismatch)
pixi run kernel-demo <matrix.npy> [--kind K] [--tau T] [--clip] [--raw] [--out dir] [--rgb-csv f] [--ids f]
pixi run colmap-kernel <model_dir> --rgb-csv <sequence>/rgb.csv [--out dir] [--check N]
pixi run colmap-demo <model_dir> --rgb-csv <sequence>/rgb.csv [--tau 0.9] [--raw] [--skip-kernel]
pixi run compare-kernels <D.npy> <colmap_kernel_dir> [--centred]
pixi run plot           # tools/plot_placecell.py placecell_demo_out (matplotlib windows; --save for PNGs)
pixi run python-smoke   # import the compiled _placecell module straight from build/python
pixi run clean          # rm -rf build

# megaloc env: everything above + PLACECELL_WITH_MEGALOC=ON + PLACECELL_WITH_VIZ=ON, in build-megaloc/
pixi run -e megaloc build-megaloc   # runs download-models first (HF vslamlab/megaloc-models -> megaloc_models/, skipped once an .onnx is there)
pixi run -e megaloc megaloc-smoke   # build-megaloc/examples/megaloc_test with no args -> prints usage only; real checks below
pixi run -e megaloc demo-viz        # synthetic_demo with the OpenCV windows (kernel / information / alive)

# export-megaloc env (adds PyTorch/ONNX): regenerate and publish the model
pixi run -e export-megaloc export-models   # tools/export_megaloc.py -> megaloc_models/megaloc_322x322.onnx + sidecar .yaml (+ self-check with --test_images)
pixi run -e export-megaloc upload-models   # hf upload of *.onnx/*.yaml only (needs `hf auth login`)
```

Plain CMake works for the core (C++17 compiler + Eigen3 only): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`. Options: `PLACECELL_BUILD_EXAMPLES` (ON), `PLACECELL_BUILD_PYTHON` (OFF; needs Python + nanobind, located via `python -m nanobind --cmake_dir`), `PLACECELL_WITH_MEGALOC` (OFF; TensorRT has no CMake config, so `CMakeLists.txt` finds it under `$CONDA_PREFIX` and fails hard outside the pixi env), `PLACECELL_WITH_VIZ` (OFF). `pip install .` builds the wheel via scikit-build-core (examples OFF, bindings ON). A standalone core build outside pixi works with system g++ 11 + `/usr/include/eigen3` (viz against a conda OpenCV needs the env's libstdc++ on the link line, see `docs/notes/2026-09-25_kernels.md`). Heavy jobs often run on this machine: check `free -h` and build with `nice` and `--parallel 2` when in doubt.

## Verification (no unit tests)

There are no unit tests and no linter. The harnesses are examples with exit codes; run the one that covers what you touched:

| After touching | Run | What it checks |
|---|---|---|
| the core or any manager | `build/examples/synthetic_demo [<out_dir>] [--views N] [--places P] [--tau T] [--min-info M] [--verbosity L] [--windows]` (`pixi run demo`) | no GPU, no models: synthetic places in R^64, every view queried with `unexplained_information`, the demo plays host (inserts when novel or periodically, records its decisions), culls every 4 insertions, prints the profile table, dumps the store (+ the three PNGs with viz). Checks store / recorder / profiler counts agree, the centred kernel has unit diagonal, and that a **kernel-only twin** built with `set_kernel(kernel(), external_ids())` + the same `set_culled` flags reproduces the raw/centred kernels and the culler's `alive_unique_information`, plus the kernel-only rules (`descriptor()` null, `add()` refused, descriptor query NaN). Also reproduces issue #5 live: the first cull reports a unique information of ~1e-5 and the once-per-process warning fires. |
| `kernel_io` or `set_kernel` | `build/examples/kernel_demo <matrix.npy> [--kind similarity\|squared-euclidean\|cosine-distance\|euclidean] [--tau T] [--min-keyframes N] [--raw] [--clip] [--no-psd-check] [--out dir] [--rgb-csv file] [--ids file] [--verbosity L]` | `load_npy` → `similarity_from_distance` → `set_kernel` (row i = id i, or the 2nd column of the `--ids` csv written by `tools/colmap_information_kernel.py`) → prints the `KernelReport` → one offline `cull_keyframes` over every view → profile table + dump → `<out>/rgb.csv` of the surviving frames (from `--rgb-csv`, or by default `<sequence>/rgb.csv` next to a VSLAM-LAB `<sequence>/vpr-lab/<matrix>.npy`, then `rgb_raw.csv` if the row count no longer equals n; exit 1 only when an explicit `--rgb-csv` is unusable). Checks the stored kernel equals the symmetrised unit-diagonal input, the id mapping, the kernel-only rules and the cull bookkeeping. Reference input: `VSLAM-LAB-Benchmark/ETH/table_3/vpr-lab/D.npy` **with `--clip`** (see gotchas). |
| the COLMAP tools | `pixi run colmap-demo <model_dir> --rgb-csv <sequence>/rgb.csv` | kernel → `kernel_demo --kind similarity --ids` → survivors' `rgb.csv`, with consistency checks (survivors are registered images, alive + culled = registered set, first frame kept); exit 1 on failure. Pass the best sub-model (`<colmap_dir>/best_model` names it). |
| the public MegaLoc API | `build-megaloc/examples/megaloc_embedder_smoke <model.onnx> <img>...` | embeds, prints pairwise cosines, round-trips `add_image`/`descriptor`/idempotence, checks the grown `kernel()` against independently computed cosines, and (≥3 images, first two of the same place) a `cull_keyframes` smoke expecting exactly one of the near-duplicate pair culled. |
| the export or the TensorRT/preprocessing code | `build-megaloc/examples/megaloc_test <model.onnx> <img>... [--precision fp16\|fp32] [--reference <stem>_reference.txt] [--iters N]` | the TensorRT backend directly (includes `src/megaloc/tensorrt_megaloc.hpp`): engine build/load time, steady-state ms/image, cosine against the PyTorch reference written by `export_megaloc.py --test_images`. |
| the bindings | `pixi run python-smoke`, then the README's Python snippet | the item-backed and kernel-only paths have been exercised from Python with the same parity/refusal/frozen-set checks as C++ (`docs/notes/2026-09-25_kernels.md`). |

The count-driven cull path (`target_alive`) is only covered by VSLAM-LAB's `rgb_placecell` smoke test (`configs/test_exp_placecell.yaml` in the parent) — `synthetic_demo`/`kernel_demo` exercise the tau path only.

## Architecture

### Core: `PlaceCell` (`include/placecell/placecell.h`, `src/placecell.cpp`)

An id-mapped, **append-only** store of views plus the similarity kernel over them. External ids are the host's (keyframe ids); each maps to a contiguous internal id = kernel row. Rows are never removed: a culled view keeps its row and becomes **history** (`set_culled`/`is_culled`); `set_protected` marks views the culler may use as explainers but never propose; `clear()` is the host-reset hook.

**Three ways to fill a store, never mixed** (the first call decides, `clear()` resets; a call of the wrong kind logs an ERROR and returns `invalid_id`, `set_kernel` throws `std::logic_error` on a non-empty store):

- **descriptor-backed** — `add(external_id, descriptor)`: idempotent; grows the n×n Gram kernel by one row of double-accumulated dots (unit descriptors ⇒ cosine) and maintains per-row sums + total sum for out-of-sample centring. A descriptor-size mismatch writes NaN into the kernel and the descriptor query then returns NaN.
- **kernel-only** — `set_kernel(similarity, ids, KernelOptions)` on an EMPTY store: a host n×n similarity (row i → `ids[i]`, or `0..n-1`), all views alive, no descriptors, cannot grow and cannot answer insertion queries (loader only, by decision — issue #2). `KernelOptions` fixes what a host matrix typically gets wrong (`symmetrise`, `unit_diagonal`, `psd_check` — O(n³) eigen-solve, WARN when negative — and `clip_to_psd`, off by default) and `KernelReport` says what it found. `kernel_only()` tells.
- **item-backed** — `set_items(id, items)`: a view is the sorted unique set of `ItemId`s (`uint64_t`, the host's map-point ids) and the kernel is the cosine of the indicator vectors `K_ij = |P_i ∩ P_j| / sqrt(|P_i| |P_j|)` — a Gram matrix, PSD with unit diagonal, exactly 0 for disjoint views, so the host passes `centred = false`. **Sets are mutable**: a later call for an alive id replaces its set and the diff flows through an inverted index into an integer shared-count matrix; the float kernel is rebuilt lazily by every kernel reader, so a burst of `set_items` costs one rebuild. **A culled view's set is frozen** (WARN once) because the culler prices history rows against the kernel they were culled under. **An empty set is a NaN row** that `usable_rows` drops alone (a row is unusable only when ALL its off-diagonal entries are NaN; the other rows just lose that column). `items(id)` is a stable pointer, `item_mode()` tells. `unexplained_information(items, window, centred=false)` is the matching insertion query (empty query → NaN).

`kernel_io.h` (`src/kernel_io.cpp`): `load_npy`/`save_npy` (2-D `<f4`/`<f8`, C or Fortran order, npy 1.0–3.0) and `DistanceKind {similarity, squared_euclidean, cosine_distance, euclidean}` / `parse_distance_kind` / `similarity_from_distance` for unit-norm descriptors.

**`cull_keyframes(params, callback, local_window)`** — the "gram-greedy" joint-information culler. Under a Gaussian model the information of view x not explained by alive set A is `v_x = K_xx − k_xA K_AA⁻¹ k_Ax ∈ [0,1]`; for an alive view that is `1/(K_AA⁻¹)_ii`. Greedily culls the alive view with the smallest `v_i` such that every view ever inserted (history and itself) stays ≤ `tau = max_unexplained`, using rank-one Schur downdates of `M = K_AA⁻¹` and `W = K_HA M`. The **host executes each cull** through the callback (return false to refuse; the view then stays alive for the rest of the call). `local_window` restricts marginalisation to the host's covisibility window. **Count-driven mode:** `CullParameters::target_alive > 0` runs the same greedy order with tau = +inf and stops when that many views are alive in scope, never below `min_keyframes` (set `min_keyframes = 1` for an offline "keep N" selection); protections still apply and `report.alive_after` says how many survived. The maths/design history also lives in `/home/alejandro/cull_keyframes_math.pdf` (rev. 5, outside the repo).

**`unexplained_information(descriptor, window, centred)`** — the culler's dual for a view NOT in the store (keyframe-insertion decisions): same `v_x` over the alive views (or the alive views among `window`), read-only. Out-of-sample centring uses the maintained row/total sums, so the query never sweeps the n×n kernel; cost = one dot per stored view + a |explainers|³ solve. On a raw kernel `v_x` equals the unique information the view would have right after `add()` (Schur identity); on a centred kernel only approximately. Returns `Information{unexplained, explainers, best_explainer, best_similarity}`.

**Centring** (`centred`, default true in both descriptor APIs, false for items): the kernel is double-centred over every usable stored view (alive AND history) and renormalised to unit diagonal — Pearson correlation of mean-centred descriptors — which removes MegaLoc's common-mode floor (~0.37). Only applied with ≥3 stored views; the centred kernel has rank n−1, hence the `1e-6` diagonal jitter. **Known degeneracy (issue #5):** double-centring over a set and marginalising over that SAME set makes `K_AA` singular and every `1/M_ii` jitter-scale — the culler's situation whenever alive set == centring set (map scope, no history yet, e.g. the very first cull after start/clear, or every offline run). The insertion query is unaffected. Not fixed.

**Locking rule** (keep it when adding methods): a single `mutex_` guards the store; methods take a snapshot under the lock and do the linear algebra outside it. `cull_keyframes` invokes the host callback with the lock **released** (the host will typically take its own map mutex there). The managers have their own mutexes and are never called under `mutex_`.

### The three managers (log / profile / record + viz)

Always compiled in, runtime switches, no compile-time kill flags; the Logger is process-wide, the Profiler and the Recorder are **per `PlaceCell`** (`profiler()` / `recorder()`); `PlaceCell::Options` (verbosity, `profile`, `record`, `report_on_destruction`, `name`) selects what is on.

- **Logger** (`log.h`, `Logger::instance()`): levels `off < error < warn < info < debug < trace`, default `warn`. `PLACECELL_VERBOSITY` (name or 0–5) overrides at startup and beats `Options::verbosity`; `set_level()` beats both. One unbuffered write to **stderr** per line (`[placecell][LEVEL] component: message`), colours only on a TTY; `set_sink()` forwards into a host logger; `print()` bypasses the gate for reports. Macros `PLACECELL_{ERROR,WARN,INFO,DEBUG,TRACE}(component, stream-expr)` evaluate the expression only when enabled; `PLACECELL_WARN_ONCE` for degraded paths that would repeat every frame. Level policy: WARN = degraded path (NaN row, singular `K_AA`); INFO = one-off facts (engine built/loaded, `clear()`, dumps); DEBUG = one line per add / cull / `cull_keyframes` call; TRACE = per query. TensorRT's own messages arrive as component `TensorRT`. There is deliberately no FATAL: a library never aborts the host.
- **Profiler** (`profiler.h`): times only the main entry points (`add`, `set_kernel`, `set_items`, `unexplained_information[_items]`, `cull_keyframes` + sub-rows `snapshot+centring`, `inverse`, `greedy`, `host_callback`; in the megaloc module `embed`, `add_image`, `unexplained_information_image`). Host callback time is subtracted from the parent (`Scope::exclude`). Every sample carries the problem size (`size_a`/`size_b`). `report()` = count/median/p95/max table; `dump_csv` writes every sample. Adding a timed function: `Profiler::Scope timer(profiler_, "name"); timer.set_sizes(...)`, and `declare()` it in the `PlaceCell` constructor so the report order stays fixed. Helpers stay untimed.
- **Recorder** (`recorder.h`): dependency-free history for the plots — every query, every `cull_keyframes` call and each cull inside it, plus what only the host knows: `record_decision(id, inserted, unexplained, reason)` and `set_thresholds(tau, min_information)` (kept as a change history). `dump_csv(dir)` writes `queries.csv`, `culls.csv`, `cull_calls.csv`, `alive_information.csv`, `decisions.csv`, `thresholds.csv`; `PlaceCell::dump(dir)` adds `kernel.npy`, `kernel_centred.npy`, `views.csv`, `profile.csv`.
- **Event fan-out** (`src/placecell_events.cpp`): the maths calls `on_add` / `on_set_kernel` / `on_set_items` / `on_query` / `on_cull_call` once per event; those feed the Recorder and the Logger. Timing stays inline in `placecell.cpp` because it has to bracket the work. Keep new instrumentation behind these hooks rather than sprinkling recorder/log calls through the culler.
- **viz** (`viz.h`, OpenCV): `render_kernel`, `render_information_history`, `render_alive_information`; `Visualizer(cell, options)` bundles them (`update()` re-renders on change at most `max_hz`, `windows` shows them via `cv::imshow` from one thread, `save(dir)` writes PNGs). `PlaceCell::snapshot(centred)` exists so the renderer gets kernel + ids + culled/protected flags under one lock. `tools/plot_placecell.py <dump_dir> [--save] [--raw]` is the offline twin.

### MegaLoc module (`include/placecell/megaloc_*.h`, `src/megaloc/`)

- `MegaLocEmbedder` — `cv::Mat` (BGR) → 8448-d L2-normalised descriptor. Pimpl over `megaloc::TensorRTMegaLoc` (`src/megaloc/tensorrt_megaloc.{hpp,cpp}`, internal), so the public header needs neither TensorRT nor CUDA. First construction for an ONNX + precision **builds** the engine (~1–2 min, GPU-specific) and caches it as `<onnx>.<precision>.engine` next to the ONNX; a failed load rebuilds and overwrites. The constructor also runs a warmup inference. `embed()` serialises on the single execution context.
- The sidecar `<onnx>.yaml` written by the export script carries input size, normalisation, tensor names, descriptor dim — the C++ side hardcodes none of that. The ONNX is traced at one fixed resolution (322×322) and is only valid there.
- `MegaLocPlaceCell : PlaceCell` — `add_image(id, image)` = embed + `add()` (never re-embeds a known id); `unexplained_information(image, window, centred, descriptor_out)` embeds without storing and can hand the embedding back so a subsequent `add()` needs no second inference; `embedder()` exposes the embedder for transient queries. Constructible from a shared `MegaLocEmbedder` so the host can share one engine.
- **Model pipeline** (`tools/export_megaloc.py`, run via the `export-models` task — its docstring "Usage" shows an old path): clones `gmberton/MegaLoc` (pinned commit) into `tools/megaloc/`, downloads the HF weights into `megaloc_models/.cache` (never `~/.cache`), rewrites three graph fragments into TensorRT-friendly form (the attention q/k/v slicing and the fp32 aggregator — the reasons are in `docs/notes/2026-09-25_kernels.md`), exports the ONNX + sidecar, and with `--test_images` self-checks torch vs wrapper vs ONNX Runtime and writes `<stem>_reference.txt` for `megaloc_test`. `megaloc_models/`, `*.onnx`, `*.engine`, `tools/megaloc/` are gitignored; the published artifacts live in HF `vslamlab/megaloc-models`.

### Offline tools (`tools/`, numpy + matplotlib only)

- `colmap_information_kernel.py <model_dir> --rgb-csv <csv>` — the pairwise shared-information kernel of a COLMAP reconstruction (normalised mutual information of the joint bundle-adjustment problem; own `.bin`/`.txt` reader, no pycolmap in any env) → `kernel.npy`, `mutual_information.npy`, `ids.csv`, consumed by `kernel_demo --kind similarity --ids`. Ids are data-row indices of the rgb csv matched by image name; unregistered images get no row. **Its tau lives on a different scale** than the MegaLoc kernel (useful range ≈ 0.7–0.95, see `docs/notes/2026-09-25_kernels.md`).
- `colmap_kernel_demo.py` — the chained kernel → cull → survivors' `rgb.csv` run with checks (`colmap-demo`).
- `compare_kernels.py <D.npy> <colmap_kernel_dir>` — VPR vs COLMAP kernel heatmaps, difference, scatter, Pearson/Spearman.
- `plot_placecell.py <dump_dir>` — plots of a `PlaceCell::dump` directory.

### How AllFeature-VSLAM consumes this

The parent `CMakeLists.txt` sets `PLACECELL_WITH_MEGALOC ON`, `PLACECELL_BUILD_EXAMPLES OFF`, `PLACECELL_BUILD_PYTHON OFF`, then `add_subdirectory(Thirdparty/placecell)` and links `placecell::megaloc`; it compiles this tree into its own `build/placecell` (no local build dir here). Its `System` owns the `MegaLocPlaceCell`; keyframe descriptors and the VPR kernel live here, not in the SLAM code, and keyframe ids are the external ids. `cull_keyframes` (LocalMapping's information-based culling) and `unexplained_information` (Tracking's keyframe-insertion decision) are called from there; the host passes `Options` from its `PlaceCell.*` settings keys, calls `record_decision`/`set_thresholds` from `Tracking::need_new_keyframe`, prints the profile table at shutdown, dumps through `PlaceCell::dump` + `Visualizer::save`, and shows the viz images in a second Pangolin window (its OpenCV is headless, so `Visualizer::Options::windows` stays off there). Its own `AF_INFO` lines for culls/insertions duplicate placecell's DEBUG lines, which is why placecell defaults to `warn`. **API changes here ripple into `PlaceRecognitionMegaLoc`/`LocalMapping`/`Tracking` in the parent**, and the parent's `CLAUDE.md` documents the settings keys that map onto `CullParameters`. VSLAM-LAB's `rgb_placecell: <n>` experiment parameter (`Capabilities/placecell.py`) is the other consumer, through the Python bindings. Bumping the parent's submodule pointer is part of landing a change here.

### Python bindings

`python/bindings.cpp` (nanobind module `_placecell`) exposes the core: `PlaceCell` (+ `Options`, `Information`, `CullParameters` incl. `target_alive`, `CullReport`, `KernelOptions`, `KernelReport`, `invalid_id`), `add`/`set_kernel(similarity, ids=None, options=None)`/`kernel_only`/`set_items`/`items` (a copy, `None` when absent)/`item_mode`/`unexplained_information_items(items, window=None, centred=False)` (a distinct name because nanobind cannot disambiguate an Eigen vector from a `std::vector<uint64_t>` overload)/`kernel`/`centred_kernel`/`external_ids`/`unexplained_information`/`cull_keyframes` (a Python callable as the cull callback)/`set_culled`/`is_culled`/`set_protected`/`__len__`, `Profiler`/`Recorder` access, `dump`, `set_verbosity`/`verbosity`, and the module-level `similarity_from_distance(matrix, kind)`. Eigen ↔ NumPy via `nanobind/eigen/dense.h`. `python/placecell/__init__.py` re-exports from `._placecell`; the compiled module only lands inside the package on `pip install`, otherwise it sits in `build/python/` (hence `python-smoke` imports `_placecell` directly). VSLAM-LAB's `placecell` env pip-installs it for Python 3.11 with `pip install --no-build-isolation --no-deps` + `CMAKE_BUILD_PARALLEL_LEVEL=2`. When adding public API: declare in the header, implement in `src/`, expose it in `bindings.cpp` and re-export it in `__init__.py`.

## Gotchas

- **VPR-LAB's `<sequence>/vpr-lab/D.npy` is squared L2 with a rotation-min**, hence asymmetric and indefinite (3 negative eigenvalues on ETH `table_3`): use `--kind squared-euclidean` (the default, `S = 1 − D/2`) **and `--clip`** / `KernelOptions::clip_to_psd`. On an unclipped indefinite kernel the culler is broken, not degraded (`1/M_ii` scores > 1, the greedy loop jams whatever tau is). `D_0.npy` is the symmetric unrotated matrix and already PSD.
- **Centred offline runs hit the issue-#5 degeneracy on the first cull** (every view alive, no history); `--raw` avoids it, count-driven mode ignores tau anyway.
- **An `add()`/`set_items()`/`set_kernel()` of the wrong mode is refused, not converted** — check `kernel_only()`/`item_mode()` when a host reset is involved; `clear()` is the only way back.
- **stdout is fully buffered under a redirect** — that is why the Logger writes to stderr unbuffered; don't switch it.
- **Name shadowing in Python:** a host script called `placecell.py` shadows the package when run as a script (`sys.path[0]` is its own directory) — it has to drop that entry before `import placecell`.
- **TensorRT 10.3** miscompiles the upstream attention reshape/permute pattern and uniform fp16 loses accuracy; the export encodes both fixes — re-verify with `megaloc_test --reference` after touching it.

## Conventions

- Allman braces (opening brace on its own line for namespaces, classes, functions), 4-space indent, trailing-underscore private members (`kernel_`). The megaloc-module files and parts of `cull_keyframes` use K&R `if(...){` — match the surrounding file rather than reformatting.
- Every source file starts with a doc-comment header (`placecell — ...` or `Module: placecell - <file>`, Author, Created, License). Public headers carry their **contracts** in that header comment — update them when behaviour changes.
- Core library and its examples compile with `-Wall -Wextra -Wpedantic`; the megaloc target with `-Wall -Wextra`. Keep new code warning-free.
- Linear algebra is accumulated in `double` even though storage is `float` (kernel, descriptors); keep that for anything feeding `K_AA` solves.
- New instrumentation goes through the `on_*` event hooks; new timed entry points are `declare()`d in the constructor.
- `pixi.lock` is marked binary for merges in `.gitattributes`; don't hand-edit it.
- Never `git push` or mutate the remote; commit only when asked. New features: discuss the design first (questions one at a time, each with a recommended default), then implement exactly the chosen options.
