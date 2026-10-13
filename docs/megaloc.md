# MegaLoc embedder: from the git repository to a C++ library on ONNX Runtime

How placecell turns [MegaLoc](https://github.com/gmberton/MegaLoc) (Berton & Masone, 2025: a DINOv2 ViT-B/14
backbone with an optimal-transport aggregator, 8448-d L2-normalised global descriptors, MIT) into
`placecell::MegaLocEmbedder`, a C++ class that embeds a `cv::Mat` on Linux (CUDA) and on Apple Silicon (Core ML,
CPU) through one code path. Written as a walkthrough so the pipeline can be re-run and understood end to end; the
measurements it rests on are the dated notes
[`2026-10-13_megaloc_onnx_export.md`](notes/2026-10-13_megaloc_onnx_export.md) and
[`2026-10-13_megaloc_onnxruntime_backend.md`](notes/2026-10-13_megaloc_onnxruntime_backend.md).

## 0. The pieces

| Piece | What it is | Where |
|---|---|---|
| **MegaLoc** | the trained model: Python class `megaloc_model.py` in the git repo + `model.safetensors` on Hugging Face (`gberton/MegaLoc`); the published usage is `torch.hub.load("gmberton/MegaLoc", "get_trained_model")` and the README's preprocessing (`ToTensor`, ImageNet `Normalize`, `Resize(322, 322)`) | `tools/megaloc/` (pinned clone, gitignored) |
| **ONNX** | a file format for a computation graph: nodes are standard operators of an *opset*, edges are tensors, weights are embedded constants. Ours is static: `[1,3,322,322] float32 → [1,8448] float32` | `megaloc_models/megaloc_322x322.onnx` (fp32, 912 MB) and `megaloc_322x322_fp16.onnx` (fp16 weights, 456 MB) + sidecar `.yaml` each |
| **ONNX Runtime (ORT)** | the library that runs the graph. At session creation it partitions the graph among *execution providers*: the CUDA provider on NVIDIA, the Core ML provider on Apple, the CPU provider everywhere (and as the fallback for nodes the others decline). One C++ API; the provider is a runtime choice | conda-forge `onnxruntime-cpp` 1.31 (`cuda129` build on linux-64, `cpu` build with the Core ML provider on osx-arm64) |
| **placecell** | `MegaLocEmbedder` (public, pimpl) → `megaloc::OnnxRuntimeMegaLoc` (the session) + `megaloc_common` (sidecar, preprocessing, normalisation); `MegaLocPlaceCell` stores the descriptors and answers the VPR and information queries | `include/placecell/megaloc_*.h`, `src/megaloc/` |

The ONNX file is the contract: everything upstream of it is Python and runs once per export (sections 1–3),
everything downstream is C++ and must reproduce, for the same input, the descriptor the Python side produced
(sections 4–6). `megaloc_322x322_reference.txt`, written by the export's self-check, is what "reproduce" is
measured against.

## 1. Getting the model out of the repository (`tools/export_megaloc.py`)

`torch.hub.load` done by hand, so it is reproducible and leaves nothing in `~/.cache`:

1. clone `gmberton/MegaLoc` into `tools/megaloc/` and check out one pinned commit (`MEGALOC_COMMIT`);
2. `import megaloc_model` from that clone — the one file the repository contributes;
3. download `model.safetensors` from Hugging Face into `megaloc_models/.cache` (`HF_HOME` is set before
   `huggingface_hub` is imported), `load_state_dict`, `eval()`.

The commit goes into the sidecar (`source:`), so a model file always says where it came from.

## 2. Making it exportable (`ExportWrapper`)

`torch.onnx.export` captures the program (`torch.export`, the exporter PyTorch defaults to since 2.9) and
onnxscript translates it. Every learned module is called unchanged — `patch_embed`, each block's
`norm1/attn.qkv/attn.proj/ls1/mlp/ls2/norm2`, the final `norm`, `cluster_features`, `score`, `token_features`,
`dust_bin`, `linear` — the wrapper rewrites only the glue between them:

| Upstream | Exported as is | Wrapper |
|---|---|---|
| resize if not a multiple of 14, then `interpolate_pos_encoding` (bicubic resize of the 37×37 positional grid) at every forward | a runtime `Resize` node that every backend implements with its own cubic kernel | resolution fixed (322 = 23 × 14); the interpolated embedding computed **once** and stored as a constant |
| Sinkhorn with slice writes into `torch.empty()` and `.squeeze()` | `ScatterND` + `ScatterElements` | the same three iterations with `cat` / `logsumexp` |
| attention q/k/v via `reshape(B,N,3,heads,hd).permute(...)[0..2]` | 38 `Gather` nodes | three slices of the `[B,N,3C]` projection — the pattern runtimes fuse into one attention kernel |

Measured: the untouched upstream module **does** export under `torch.export` and agrees to 4e-7, so the rewrites
exist for the graph the runtimes get (Core ML handles `Resize`/scatter/`Gather` badly; TensorRT 10.3 miscompiled the
upstream attention pattern), not for the exporter. Export settings: opset 18, `external_data=False` (one
self-contained file), `optimize=True` (478 nodes against 807 from the deprecated TorchScript tracer, same maths).
Normalisation stays **outside** the graph: the C++ side applies it, and the sidecar carries `mean`/`std`.

## 3. Export, fp16 twin, sidecar, reference

```bash
pixi run -e export-megaloc export-models -- --test_images <4 frames>     # Linux; torch 2.10 CPU build, ~10 min
pixi run -e export-megaloc upload-models                                # hf upload (needs write access to vslamlab)
```

- `megaloc_322x322.onnx` + `.yaml`: the fp32 export and its sidecar (input size, normalisation, tensor names,
  descriptor dim, exporter, precision — nothing of that is hardcoded in C++).
- `megaloc_322x322_fp16.onnx` + `.yaml`: `onnxconverter_common.float16.convert_float_to_float16` with
  `keep_io_types=True` and a **topological** block list: the backbone's final `LayerNormalization` and everything
  downstream of it (the aggregator: Sinkhorn `exp`/`logsumexp`, the convs, the 16640→8448 linear) stays fp32 — 473
  nodes converted, 65 kept. The converter leaves the graph output float16 despite `keep_io_types`;
  `ensure_float32_io` re-points that cast. fp16 weights cost ≤ 6e-5 in cosine.
- `--test_images`: torch original with the README transform **(the reference)**, the wrapper with the OpenCV
  preprocessing, ONNX Runtime on the fp32 and fp16 files. Every ONNX path agrees with the wrapper to 1e-6 and all sit
  at **0.9953–0.9958 against the reference**: that floor is the preprocessing (OpenCV area resize vs torchvision's
  antialiased bilinear, PIL vs OpenCV decoding), identical for every backend and irrelevant inside the SLAM, which
  only compares its own descriptors with each other. The reference descriptors go to `megaloc_322x322_reference.txt`.
- `tools/download_models.py` (the `download-models` task) fetches the published files per file and never overwrites
  a local export.

## 4. The C++ backend (`src/megaloc/onnxruntime_megaloc.{hpp,cpp}`)

```cpp
placecell::MegaLocEmbedder::Settings s;      // provider "auto", 4 threads, Core ML units "ALL"
s.provider = "cuda";                          // or "coreml", "cpu"
placecell::MegaLocEmbedder embedder("megaloc_models/megaloc_322x322_fp16.onnx", s);
Eigen::VectorXf d = embedder.embed(cv::imread(path));          // BGR in, 8448-d unit vector out
float c = placecell::MegaLocEmbedder::cosine(d, d2);
```

Inside, in order:

1. **Sidecar** (`megaloc_common::loadSidecar`): `<model>.onnx.yaml` → mean/std, tensor names, input size,
   descriptor dim, precision.
2. **`Ort::Env`** with a logging callback that forwards ORT's messages into the placecell Logger (component `ORT`).
3. **`Ort::SessionOptions`**: `ORT_ENABLE_ALL` graph optimisation, `SetIntraOpNumThreads(threads)` — a fixed small
   number, never the core count, the SLAM's own threads need the rest —, inter-op 1, sequential execution.
4. **Providers** (`appendProviders`): `auto` = CUDA if ORT has it, else Core ML if it has it, else CPU; a requested
   provider that is missing falls back to `auto` with a warning. CUDA through the V2 provider options
   (`device_id`, `cudnn_conv_use_max_workspace`); Core ML through `AppendExecutionProvider("CoreML", {...})` with
   `ModelFormat=MLProgram`, `MLComputeUnits`, `RequireStaticInputShapes=1` and `ModelCacheDirectory` next to the model
   (the compiled model is cached per model file; a repeat start costs ~35 s instead of 1–3 min). The CPU provider is
   implicit and last.
5. **`Ort::Session`**, then the model's input/output names, shapes and element types are read back and
   cross-checked against the sidecar (float32 in, float32 out, `[1,3,H,W]` → `[1,D]`).
6. **`infer()`**: `megaloc_common::preprocess` (3-channel BGR → area resize → RGB → /255 → (x−mean)/std → CHW
   planar, the exact twin of `preprocess_cv2()` in the export script) into a host buffer that an `Ort::Value` wraps
   without copying; `session.Run`; the `[1,D]` output copied out and re-normalised on the host (`normalizeInPlace`:
   fp16 paths leave |d| slightly off 1).

`MegaLocEmbedder` adds the one mutex (one session, one inference at a time), the warmup inference in the constructor,
the `embed` profiler row, and `provider()` / `precision()` / `model_path()`. The model file is loaded as given: the
fp32 export or its `_fp16` twin, the sidecar says which — there is no precision switch.

## 5. Building it

- CMake: `PLACECELL_WITH_MEGALOC=ON` → `find_package(onnxruntime CONFIG)` (conda-forge's package ships it; the
  source-layout include dir `include/onnxruntime/core/session` is added so `<onnxruntime_cxx_api.h>` works), plus
  OpenCV and yaml-cpp. No CUDA toolkit: the CUDA provider lives inside `libonnxruntime` and dlopens `libcudart` /
  cuDNN at run time — which is why the `megaloc` environment lists `cuda-cudart` and `cudnn` by hand on linux-64.
- pixi: `pixi run -e megaloc build-megaloc` (env `megaloc`, build dir `build-megaloc/`); `megaloc-test` and
  `megaloc-smoke` are the harnesses; `demo-viz` the viz windows. The host, AllFeature-VSLAM, compiles this tree as a
  subdirectory with the same option and reads the provider from its settings (`megaloc_provider`,
  `megaloc_threads`, `megaloc_coreml_units`).

## 6. Measured

`examples/test_megaloc.cpp`, four `table_3` frames, steady state over 20–50 inferences, cosine against the torch
reference (floor 0.9955 = preprocessing) and between providers:

| Machine | Provider, file | ms/image | cos vs reference (worst) | notes |
|---|---|---|---|---|
| Linux, RTX 4000 Ada | cuda, fp32 | 11.4 | 0.99525 | agrees with CPU fp32 to 0.999998 |
| | cuda, fp16 | **7.6** | 0.99553 | 0.9998 vs cuda fp32 |
| | cpu, fp32, 2 / 4 / 8 threads | 391 / 210 / 120 | 0.99527 | near-linear scaling |
| | cpu, fp16 | 230 | 0.99522 | x86 emulates half precision: fp16 is for the accelerators |
| MacBook Pro M1 Pro | cpu, fp32, 2 / 4 / 6 threads | 611 / 328 / 229 | 0.99527 | identical to Linux to the fifth decimal |
| | coreml, fp16, `CPUAndGPU` | **46.0** | 0.99545 | the accurate GPU path |
| | coreml, fp32, `ALL` | 47.9 | 0.99527 | fp32 keeps Core ML on the GPU |
| | coreml, fp16, `ALL` / `CPUAndNeuralEngine` | 41.2 / 42.2 | 0.98229 | **the Neural Engine loses accuracy** |

Session creation: 1.2–1.6 s on CPU and CUDA; Core ML 90–175 s on the first compile of a file, ~36 s afterwards.
17 of 545 nodes stay on the CPU provider in 13 Core ML partitions (shape plumbing around the attention and the
Sinkhorn); cutting those partitions is the known lever for the Mac. Inside AllFeature-VSLAM on `table_3`, the
per-keyframe `add_image` costs 12 ms on CUDA and 60 ms on Core ML, on the mapping thread.

## 7. What is not here any more

The first backend was TensorRT (`tensorrt_megaloc.{hpp,cpp}`, 2026-09): an ahead-of-time engine per GPU (1–2 min
build, cached as `<onnx>.<precision>.engine`), NVIDIA-only, with a Myelin miscompile of the upstream attention that
the export wrapper still works around. It was deleted on 2026-10-13 once the ONNX Runtime backend reproduced its
descriptors (its "mixed fp16" engine had in fact been fp32: the tracer export carried no `/backbone/` node names for
its layer-pinning rule to match). The AllFeature-VSLAM plan `docs/plans/megaloc_onnxruntime.md` has the step-by-step
history.
