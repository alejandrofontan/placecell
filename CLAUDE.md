# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`placecell` is a C++17 library (with nanobind Python bindings) for keyframe lifecycle management in VSLAM / 3D reconstruction: keyframe creation, connectivity, redundancy detection, and marginalization. It is a standalone first-party repo (`github.com/alejandrofontan/placecell`, Apache-2.0) that AllFeature-VSLAM consumes as the git submodule `Thirdparty/placecell`. There are two CMake targets:

- `placecell::placecell` — the Eigen-only core (`src/placecell.cpp`, header `include/placecell/placecell.h`) plus the three diagnostic managers (`log.h`, `profiler.h`, `recorder.h`, see below). Descriptors come in already computed; no image or model code here.
- `placecell::megaloc` — optional (`PLACECELL_WITH_MEGALOC=ON`), needs CUDA 12 + TensorRT + OpenCV + yaml-cpp: the MegaLoc image frontend (`src/megaloc/`). Links the core transitively.
- `placecell::viz` — optional (`PLACECELL_WITH_VIZ=ON`), needs OpenCV only: renders the kernel heatmap, the unexplained-information history and the alive-information strip from a `PlaceCell` + its `Recorder` (`include/placecell/viz.h`, `src/viz/viz.cpp`).

Design goal to keep in mind for every change: placecell must stay independent of any particular SLAM/SfM system, and the kernel's metric (MegaLoc cosine today) must become swappable — issues #1–#5 in the GitHub repo track the gaps (pluggable similarity strategy, host-supplied kernel rows, the PSD/unit-diagonal kernel contract, centring as a per-metric default, and the singular-`K_AA` degeneracy).

## Build and run

pixi is the primary workflow (linux-64 with CUDA only, per `pixi.toml`). Two environments, two build directories — don't mix them:

```bash
# default env: core + examples + Python bindings, in build/
pixi run build          # cmake configure (Ninja, Release, PLACECELL_BUILD_PYTHON=ON) + build
pixi run example        # build/examples/placecell_example (examples/main.cpp, a skeleton check)
pixi run demo           # build/examples/synthetic_demo -> placecell_demo_out/ (GPU-free smoke test, exit 1 on mismatch)
pixi run plot           # tools/plot_placecell.py placecell_demo_out (matplotlib windows; --save for PNGs)
pixi run python-smoke   # import the compiled _placecell module straight from build/python
pixi run clean          # rm -rf build

# megaloc env: everything above + PLACECELL_WITH_MEGALOC=ON + PLACECELL_WITH_VIZ=ON, in build-megaloc/
pixi run -e megaloc build-megaloc   # also runs download-models first (HF vslamlab/megaloc-models -> megaloc_models/, skipped once an .onnx is there)
pixi run -e megaloc megaloc-smoke   # runs build-megaloc/examples/megaloc_test with no args -> just prints usage; see below for a real check
pixi run -e megaloc demo-viz        # synthetic_demo with the OpenCV windows (kernel / information / alive)

# export-megaloc env (adds PyTorch/ONNX): regenerate and publish the model
pixi run -e export-megaloc export-models   # tools/export_megaloc.py -> megaloc_models/megaloc_322x322.onnx + sidecar .yaml (+ self-check with --test_images)
pixi run -e export-megaloc upload-models   # hf upload of *.onnx/*.yaml only (needs `hf auth login`)
```

Plain CMake works for the core (C++17 compiler + Eigen3 only): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`. Options: `PLACECELL_BUILD_EXAMPLES` (ON), `PLACECELL_BUILD_PYTHON` (OFF; needs Python + nanobind, located via `python -m nanobind --cmake_dir`), `PLACECELL_WITH_MEGALOC` (OFF; TensorRT has no CMake config, so `CMakeLists.txt` finds it under `$CONDA_PREFIX` and fails hard outside the pixi env). `pip install .` builds the wheel via scikit-build-core (examples OFF, bindings ON).

**There are no unit tests.** The verification harnesses are one GPU-free example and two megaloc examples that need a real ONNX and images:

- `build/examples/synthetic_demo [<out_dir>] [--views N] [--places P] [--tau T] [--min-info M] [--verbosity L] [--windows]` — no GPU, no models: synthetic places in R^64, every view queried with `unexplained_information`, the demo plays host (inserts when novel or periodically while not redundant, records its decisions), culls every 4 insertions, then prints the profile table, dumps the store and (with viz) writes the three PNGs. Checks that store / recorder / profiler counts agree with what it did and that the centred kernel has unit diagonal; exit 1 otherwise. **Run this after touching the core or any manager.** It also reproduces issue #5 live: the first cull reports a unique information of ~1e-5 (the singular-`K_AA` case) and the once-per-process warning fires.
- `build-megaloc/examples/megaloc_embedder_smoke <model.onnx> <img>...` — exercises the PUBLIC API end to end: embeds, prints pairwise cosines, round-trips `MegaLocPlaceCell::add_image`/`descriptor`/idempotence, checks the grown `kernel()` against the independently computed cosines, and (with ≥3 images, first two of the same place) runs a `cull_keyframes` smoke expecting exactly one of the near-duplicate pair culled. Exit code is the pass/fail signal.
- `build-megaloc/examples/megaloc_test <model.onnx> <img>... [--precision fp16|fp32] [--reference <stem>_reference.txt] [--iters N]` — tests the TensorRT backend directly (includes `src/megaloc/tensorrt_megaloc.hpp`, not the public header): engine build/load time, steady-state ms/image, and cosine against the PyTorch reference descriptors written by `export_megaloc.py --test_images`. Run this after touching the export or the TensorRT/preprocessing code.

Expected numbers on real sequence frames: self ≈1.0, consecutive frames ≈0.9+, unrelated places ≈0.35–0.4 (MegaLoc's common-mode floor — the reason the kernel is centred, see below); TensorRT-vs-PyTorch cosine ≈0.99+ (worst seen 0.994, fp16 backbone / fp32 aggregator, ~4 ms/image).

## Architecture

### Core: `PlaceCell` (`include/placecell/placecell.h`, `src/placecell.cpp`)

An id-mapped, **append-only** store of global descriptors plus the similarity kernel over them:

- `add(external_id, descriptor)` → contiguous internal id (= kernel row). Idempotent. Grows the n×n Gram kernel by one row of double-accumulated dots (unit descriptors ⇒ cosine) and maintains per-row sums + total sum for out-of-sample centring. A descriptor-size mismatch writes NaN into the kernel and poisons later queries (`unexplained` = NaN).
- Rows are never removed. A culled view keeps its row and becomes **history** (`set_culled`/`is_culled`); `set_protected` marks views the culler may use as explainers but never propose. `clear()` is the host-reset hook.
- `cull_keyframes(params, callback, local_window)` — the "gram-greedy" joint-information culler. Under a Gaussian model the information of view x not explained by alive set A is `v_x = K_xx − k_xA K_AA⁻¹ k_Ax ∈ [0,1]`; for an alive view that is `1/(K_AA⁻¹)_ii`. Greedily culls the alive view with the smallest `v_i` such that every view ever inserted (history and itself) stays ≤ `tau = max_unexplained`, using rank-one Schur downdates of `M = K_AA⁻¹` and `W = K_HA M`. The **host executes each cull** through the callback (return false to refuse; the view then stays alive for the rest of the call). `local_window` restricts marginalisation to the host's covisibility window; history is reduced to rows whose best alive explainer lies in it. Full derivation, kernel centring rationale, and the online-threshold-change (tau lowered/raised) handling are in the comment block at the top of the function body — read it before changing the loop. The maths/design history also lives in `/home/alejandro/cull_keyframes_math.pdf` (rev. 5, outside the repo).
- `unexplained_information(descriptor, window, centred)` — the culler's dual for a view NOT in the store (keyframe-insertion decisions): same `v_x` over the alive views (or the alive views among `window`), on the same kernel/centring. Read-only. Out-of-sample centring uses the row/total sums `add()` maintains, so the query never sweeps the n×n kernel; cost = one dot per stored view + a |explainers|³ solve. On a raw kernel `v_x` equals the unique information the view would have right after `add()` (Schur identity); on a centred kernel only approximately (the mean shifts). Returns `Information{unexplained, explainers, best_explainer, best_similarity}`.

**Centring** (`centred`, default true in both APIs): the kernel is double-centred over every stored view (alive AND history) and renormalised to unit diagonal — Pearson correlation of mean-centred descriptors — which removes MegaLoc's common-mode floor. Only applied with ≥3 stored views; the centred kernel has rank n−1, hence the `1e-6` diagonal jitter. **Known degeneracy:** double-centring over a set and marginalising over that SAME set makes the centred vectors sum to zero, so `K_AA` is singular and every `1/M_ii` is jitter-scale — this is the culler's situation whenever alive set == centring set (map scope, no history yet, e.g. the very first cull after start/clear). The insertion query is unaffected because it excludes itself from the centring set. Not fixed.

**Locking rule** (keep it when adding methods): a single `mutex_` guards the store; methods take a snapshot under the lock and do the linear algebra outside it. `cull_keyframes` invokes the host callback with the lock **released** (the host will typically take its own map mutex there — holding ours would invite deadlock). The managers have their own mutexes and are never called under `mutex_`.

### The three managers (log / profile / record + viz)

Added 2026-09-05. Conventions: always compiled in, runtime switches, no compile-time kill flags; the Logger is process-wide, the Profiler and the Recorder are **per `PlaceCell`** (`profiler()` / `recorder()`); `PlaceCell::Options` (verbosity, `profile`, `record`, `report_on_destruction`, `name`) selects what is on.

- **Logger** (`log.h`, `Logger::instance()`): levels `off < error < warn < info < debug < trace`, default `warn`. `PLACECELL_VERBOSITY` (name or 0–5) overrides at startup and beats `Options::verbosity`; `set_level()` beats both. Line format `[placecell][LEVEL] component: message`, one unbuffered write to **stderr** per line (stdout is fully buffered when redirected — the host's own `AF_INFO` lag bug), colours only when stderr is a TTY, optional elapsed-seconds prefix. `set_sink()` replaces the default (forward into a host logger); `print()` bypasses the gate for reports. Macros `PLACECELL_{ERROR,WARN,INFO,DEBUG,TRACE}(component, stream-expr)` evaluate the expression only when enabled; `PLACECELL_WARN_ONCE` for degraded paths that would repeat every frame. Level policy: WARN = degraded path (NaN kernel row, singular `K_AA`); INFO = one-off facts (engine built/loaded, `clear()`, over-budget history changed, dumps); DEBUG = one line per add / per cull / per `cull_keyframes` call; TRACE = per query. TensorRT's own messages arrive as component `TensorRT` (errors/warnings at their level, kINFO at debug, kVERBOSE at trace). There is deliberately no FATAL: a library never aborts the host.
- **Profiler** (`profiler.h`): times only the main entry points — `add`, `unexplained_information`, `cull_keyframes` (+ sub-rows `snapshot+centring`, `inverse`, `greedy`, `host_callback`), and in the megaloc module `embed` (embedder's own Profiler), `add_image`, `unexplained_information_image`. Host callback time inside `cull_keyframes` is subtracted from the parent (`Scope::exclude`) and reported as its own sub-row. Every sample carries the problem size (`size_a`/`size_b`: stored/alive/history/explainers or image width/height — see each call site). `report()` = count/median/p95/max table; `PlaceCell::print_profile()` sends it through `Logger::print`; `dump_csv` writes every sample. Adding a timed function: `Profiler::Scope timer(profiler_, "name"); timer.set_sizes(...)`, and `declare()` it in the `PlaceCell` constructor so the report order stays fixed. Helpers stay untimed.
- **Recorder** (`recorder.h`): dependency-free history for the plots — every query (`unexplained`, explainers, best explainer, stored, window), every `cull_keyframes` call (with the alive views' unique information kept for the latest call only) and each cull inside it, plus what only the host knows: `record_decision(id, inserted, unexplained, reason)` and `set_thresholds(tau, min_information)` (kept as a change history so plots can draw steps). `dump_csv(dir)` writes `queries.csv`, `culls.csv`, `cull_calls.csv`, `alive_information.csv`, `decisions.csv`, `thresholds.csv`; `PlaceCell::dump(dir)` adds `kernel.npy`, `kernel_centred.npy`, `views.csv`, `profile.csv` (`save_npy` is a 40-line NumPy writer, no dependency).
- **Event fan-out** (`src/placecell_events.cpp`): the maths calls `on_add` / `on_query` / `on_cull_call` once per event; those feed the Recorder and the Logger. Timing stays inline in `placecell.cpp` because it has to bracket the work. Keep new instrumentation behind these three hooks rather than sprinkling recorder/log calls through the culler.
- **viz** (`viz.h`, OpenCV): `render_kernel(snapshot|cell, KernelStyle)` (raw or centred, history rows/cols dimmed, protected ticks, colourbar), `render_information_history(recorder, HistoryStyle)` (v per query, tau/min-info step lines, host insertions as triangles, culls as ticks + dots), `render_alive_information(recorder, AliveStyle)` (bars of `1/M_ii` per alive view vs tau, from `CullReport::alive_unique_information`). `Visualizer(cell, options)` bundles them: `update()` re-renders only on change and at most `max_hz` (default 2 Hz), `windows` shows them with `cv::imshow` (call from one thread — the host's viewer thread), `save(dir)` writes PNGs. `PlaceCell::snapshot(centred)` exists so the renderer gets kernel + ids + culled/protected flags under one lock. `tools/plot_placecell.py <dump_dir> [--save] [--raw]` is the offline twin (numpy + matplotlib; also plots the profile distributions).
- **Host wiring status**: AllFeature-VSLAM passes `Options` from its `PlaceCell.*` settings keys, calls `record_decision`/`set_thresholds` from `Tracking::need_new_keyframe`, prints the profile table at shutdown, dumps through `PlaceCell::dump` + `Visualizer::save`, and shows the three viz images in a second Pangolin window (its OpenCV is headless, so `Visualizer::Options::windows` stays off there). Its own `AF_INFO` lines for culls/insertions duplicate placecell's DEBUG lines, which is why placecell defaults to `warn`.

Build note for a standalone build outside pixi (as done for the first verification): system g++ 11 + `/usr/include/eigen3` work for the core; the viz module linked against a conda OpenCV needs the env's libstdc++ (`-DCMAKE_EXE_LINKER_FLAGS="-L<env>/lib -Wl,-rpath,<env>/lib"`), otherwise `__cxa_call_terminate@CXXABI_1.3.15` is undefined at link time.

### MegaLoc module (`include/placecell/megaloc_*.h`, `src/megaloc/`)

- `MegaLocEmbedder` — `cv::Mat` (BGR) → 8448-d L2-normalised descriptor. Pimpl over `megaloc::TensorRTMegaLoc` (`src/megaloc/tensorrt_megaloc.{hpp,cpp}`, internal, `namespace megaloc`), so the public header needs neither TensorRT nor CUDA. First construction for an ONNX + precision **builds** the engine (~1–2 min, GPU-specific) and caches it as `<onnx>.<precision>.engine` next to the ONNX; later runs load it in ms. A failed load rebuilds and overwrites. The constructor also runs a warmup inference. `embed()` serialises on the single execution context (thread-safe, blocking).
- The sidecar `<onnx>.yaml` written by the export script carries input size, normalisation, tensor names, descriptor dim — the C++ side reads it and hardcodes none of that. The ONNX is traced at one fixed resolution (322×322) and is only valid there.
- `MegaLocPlaceCell : PlaceCell` — `add_image(id, image)` = embed + `add()` (idempotent, never re-embeds a known id); `unexplained_information(image, window, centred, descriptor_out)` embeds without storing and can hand the embedding back so a subsequent `add()` needs no second inference; `embedder()` exposes the embedder for transient queries (relocalization). Constructible from a shared `MegaLocEmbedder` so the host can share one engine.

### Model pipeline (`tools/export_megaloc.py`)

Clones `gmberton/MegaLoc` (pinned commit) into `tools/megaloc/` (gitignored), downloads the HF weights into `megaloc_models/.cache` (never `~/.cache`), rewrites three graph fragments into TensorRT-friendly form, exports the ONNX + sidecar, and with `--test_images` self-checks torch vs wrapper vs ONNX Runtime and writes `<stem>_reference.txt` for `megaloc_test`. **TensorRT 10.3 gotcha this encodes:** the upstream attention `qkv.reshape(...).permute(...)[0..2]` pattern is miscompiled by TensorRT's Myelin fuser (descriptor cosine 0.05 vs ORT, while every isolated sub-graph looks fine); the export slices q/k/v from the projection output instead. Uniform fp16 was a separate loss (cos 0.93), fixed by pinning the aggregator to fp32. The script's docstring "Usage" still shows an old path — invoke it via the `export-models` pixi task.

`megaloc_models/`, `*.onnx`, `*.engine`, `tools/megaloc/` are all gitignored; the published artifacts live in HF `vslamlab/megaloc-models`.

### How AllFeature-VSLAM consumes this

The parent `CMakeLists.txt` sets `PLACECELL_WITH_MEGALOC ON`, `PLACECELL_BUILD_EXAMPLES OFF`, `PLACECELL_BUILD_PYTHON OFF`, then `add_subdirectory(Thirdparty/placecell)` and links `placecell::megaloc`. Its `System` owns the `MegaLocPlaceCell`; keyframe descriptors and the VPR kernel live here, not in the SLAM code. Keyframe ids are the external ids. Both `cull_keyframes` (LocalMapping's information-based culling) and `unexplained_information` (Tracking's keyframe-insertion decision) are called from there — API changes here ripple into `PlaceRecognitionMegaLoc`/`LocalMapping`/`Tracking` in the parent, and the parent's `CLAUDE.md` documents the settings keys that map onto `CullParameters`. Note the parent has no local build dir here: it compiles this tree into its own `build/placecell`.

### Python bindings

`python/bindings.cpp` (nanobind module `_placecell`) exposes the core: `PlaceCell` (+ `Options`, `Information`, `CullParameters`, `CullReport`), `add`/`kernel`/`centred_kernel`/`external_ids`/`unexplained_information`/`cull_keyframes` (Python callable as the cull callback), `Profiler`/`Recorder` access, `dump`, and `set_verbosity`/`verbosity`. Eigen ↔ NumPy via `nanobind/eigen/dense.h`. **Not compiled in the 2026-09-05 pass** (no env with nanobind was available) — build with `pixi run build` and fix any binding compile error before relying on it. `python/placecell/__init__.py` re-exports from `._placecell`; the compiled module only lands inside the package on `pip install`, otherwise it sits in `build/python/` (hence `python-smoke` imports `_placecell` directly). When adding public API: declare in the header, implement in `src/`, and expose it in `bindings.cpp`.

## Conventions

- Allman braces (opening brace on its own line for namespaces, classes, functions), 4-space indent, trailing-underscore private members (`kernel_`). The megaloc-module files and parts of `cull_keyframes` use K&R `if(...){` — match the surrounding file rather than reformatting.
- Every source file starts with a doc-comment header (`placecell — ...` or `Module: placecell - <file>`, Author, Created, License). Public headers carry their **contracts** (idempotence, pointer stability, thread-safety, NaN semantics) in that header comment — update them when behaviour changes.
- Core library and its example compile with `-Wall -Wextra -Wpedantic`; the megaloc target with `-Wall -Wextra`. Keep new code warning-free.
- Linear algebra is accumulated in `double` even though storage is `float` (kernel, descriptors); keep that for anything feeding `K_AA` solves.
- `pixi.lock` is marked binary for merges in `.gitattributes`; don't hand-edit it.
- `docs/megaloc.md` is a placeholder TODO; this file and the header comments are the real documentation for now.
