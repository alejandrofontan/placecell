# MegaLoc on ONNX Runtime: the C++ backend, parity and timing on Linux

- **Date:** 2026-10-13
- **Context:** AllFeature-VSLAM `docs/plans/megaloc_onnxruntime.md`, steps 4–5 (phase 2 of the macOS port). The
  MegaLoc embedder gets a second backend, ONNX Runtime, selected at build time with
  `PLACECELL_MEGALOC_BACKEND=onnxruntime|tensorrt` (default `onnxruntime`); the preprocessing, sidecar parsing,
  re-normalisation and cosine moved into `src/megaloc/megaloc_common.{hpp,cpp}` so both backends run the same
  code around the graph. Environment `megaloc-onnx`: `onnxruntime-cpp` 1.31.0 (`cuda129` build + `cuda-cudart` +
  `cudnn` 9.10 on linux-64, `cpu` build with the CoreML provider on osx-arm64), OpenCV 4.12, yaml-cpp, no CUDA
  toolkit, no TensorRT, no Python. Models: the exports of `docs/notes/2026-10-13_megaloc_onnx_export.md`.

## Harness

`examples/test_megaloc.cpp` now goes through the public `MegaLocEmbedder` API (any backend) and takes
`--provider auto|cuda|coreml|cpu`, `--threads N`, `--coreml-units U`, `--reference <file>`, `--dump <file>`,
`--iters N` (`--precision` only matters to the TensorRT backend's engine). The model file is loaded as given: the
fp32 export or its `_fp16` twin — precision is a property of the file, read from its sidecar, not a switch (the
`_fp16` substitution that existed for one afternoon was removed after review: one key, one file). A dump has the
reference file's format, so a run on one provider / machine is the reference of another:

```bash
pixi run -e megaloc-onnx megaloc-test megaloc_models/megaloc_322x322.onnx <4 table_3 frames> \
    --provider cuda --reference megaloc_models/megaloc_322x322_reference.txt --dump cuda_fp32.txt
pixi run -e megaloc-onnx megaloc-test megaloc_models/megaloc_322x322_fp16.onnx <4 table_3 frames> --provider cuda ...
```

## Results (Linux: RTX 4000 Ada, CUDA 12.9; CPU runs with 4 intra-op threads)

| Provider, file | Session + warmup | Steady state ms/image | cos vs torch reference (per frame; worst) | cos vs ORT-CPU fp32 (worst; max abs diff) |
|---|---|---|---|---|
| cuda, fp32 | 1.6 s | **11.4** | 0.99525 / 0.99580 / 0.99563 / 0.99579; 0.99525 | 0.999998; 8.4e-5 |
| cuda, fp16 twin | 1.2 s | **7.6** | 0.99553 / 0.99565 / 0.99554 / 0.99561; 0.99553 | vs cuda fp32: 0.99982; 9.5e-4 |
| cpu, fp32 | 1.5 s | 208.5 | 0.99527 / 0.99581 / 0.99564 / 0.99579; 0.99527 | (reference) |
| cpu, fp16 twin | 1.4 s | 230.4 | 0.99522 / 0.99580 / 0.99564 / 0.99576; 0.99522 | 0.99999; 2.1e-4 |

Pairwise cosine of the four frames is the same matrix as the Python self-check (0.576 for the first/last frame,
0.21–0.33 elsewhere).

## Reading

- **Parity criteria met** (restated in the export note): every provider ≥ 0.995 against the README reference
  (the preprocessing floor), fp32 providers agree to 0.999998, fp16 ones to ≥ 0.9998.
- **CPU fp32 reproduces the Python ONNX Runtime numbers to the fifth decimal**: the C++ preprocessing equals
  `preprocess_cv2()` exactly.
- **CUDA fp16 is 1.5× faster than CUDA fp32** (7.6 vs 11.4 ms) at a 1.8e-4 cosine cost. For scale, the TensorRT
  engine measured ~5 ms on the same GPU — but that engine was effectively fp32 (see the export note); TensorRT's
  ahead-of-time kernel selection is its remaining advantage, worth ~2× at a per-keyframe rate that does not need it.
- **fp16 on the CPU is slower than fp32** (230 vs 209 ms): x86 has no native half arithmetic, ORT inserts casts.
  The fp16 twin is the file for the accelerators (CUDA, Core ML / Neural Engine), the fp32 file is the CPU's.
- **Session creation is 1.2–1.6 s** on every provider, versus TensorRT's 1–2 min engine build on first use, and
  nothing machine-specific is written next to the model (CoreML will add a compiled cache on the Mac).

## Two things found on the way

- The `cuda129` build of `onnxruntime-cpp` dlopens `libcudart.so.12` at runtime without declaring the dependency:
  without `cuda-cudart` in the env the CUDA provider fails with `ProviderLibrary::Get() [ONNXRuntimeError] : 1`.
- `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)` keeps the inputs float32 but, when a
  graph output is produced by a node on the block list, still appends a `Cast` to float16 and declares the output
  float16. `export_megaloc.py::ensure_float32_io` re-points that cast; the C++ side keeps its float32 I/O check.

## Thread sweep (Linux CPU, fp32 file)

| threads | 2 | 4 | 8 |
|---|---|---|---|
| ms/image | 391 | 210 | 120 |

Near-linear: the ViT's matrix multiplies parallelise cleanly on the CPU provider.

## Results (macOS: MacBook Pro M1 Pro 6P+2E, 16 GB; Apple clang via conda, `onnxruntime-cpp` 1.31 `cpu` build with the CoreML provider)

Same four frames, same reference file (published with the models and downloaded by `tools/download_models.py`).

| Provider, file, compute units | Session + warmup | Steady state ms/image | cos vs torch reference (per frame; worst) |
|---|---|---|---|
| cpu, fp32, 2 threads | 1.4 s | 611 | (not dumped) |
| cpu, fp32, 4 threads | 1.5 s | 328 | 0.99527 / 0.99581 / 0.99564 / 0.99579; **0.99527** — identical to Linux to the fifth decimal |
| cpu, fp32, 6 threads | 0.9 s | 229 | |
| coreml, fp16, `ALL` | 171 s | **41.2** | 0.98289 / 0.98916 / 0.98229 / 0.98308; 0.98229 |
| coreml, fp16, `CPUAndGPU` | 114 s | **46.0** | 0.99550 / 0.99573 / 0.99545 / 0.99562; **0.99545** |
| coreml, fp16, `CPUAndNeuralEngine` | 89 s | 42.2 | 0.98288 / 0.98916 / 0.98229 / 0.98307; 0.98229 |
| coreml, fp32, `ALL` | 174 s | **47.9** | 0.99527 / 0.99581 / 0.99564 / 0.99579; **0.99527** |

Every CoreML session logs `number of partitions supported by CoreML: 13, nodes 545, supported 528` (fp32 file:
526 / 511): 17 nodes stay on the CPU provider, split into 13 CoreML partitions — the shape/reshape plumbing around
the attention and the Sinkhorn, at a cost of 13 boundary copies per inference that the steady-state numbers already
include.

## Reading the Mac

- **The Neural Engine is where the accuracy goes, not the fp16 file.** The fp16 file on `CPUAndGPU` scores 0.9955,
  the fp32-level figure; the same file on `ALL` or `CPUAndNeuralEngine` drops to 0.982–0.989, and `ALL` on the fp32
  file is exact again (0.9953) because Core ML runs an fp32 model on the GPU. So the ANE computes with less internal
  precision than fp16 weights imply. The pairwise structure survives (0.571 vs 0.576 for the near pair, 0.24 vs 0.25
  for the far ones), so VPR would most likely still work on it, but 0.982 against a 0.9955 floor is a real loss
  for 5 ms of gain.
- **The GPU path is the one to use on the Mac:** fp16 file + `CPUAndGPU` (46 ms, 0.9955) or fp32 file + `ALL`
  (48 ms, 0.9953, i.e. the defaults of the settings keys). Both are 5–7× faster than the 6-thread CPU (229 ms) and
  ~4× slower than CUDA on the Linux workstation (8–12 ms). At 1–5 keyframes/s that is a few percent of one core's
  time on the mapping thread.
- **CPU: 6 threads is the number** on the M1 Pro (229 ms, vs 328 at 4): the two efficiency cores add nothing after
  that. CPU fp32 reproduces Linux to the fifth decimal — same code, same arithmetic.
- **Session creation is the open problem: 90–175 s per configuration.** Each of the four CoreML runs was a different
  (file, compute units) pair, so each may have been a first compile; whether `ModelCacheDirectory` makes the second
  session of the same configuration fast is the next thing to check (a repeat run of fp32 + `ALL` and a look into
  `megaloc_models/coreml_cache/`). If it does not, the host would pay 2–3 minutes at every start on the Mac.

**Compile cache, checked:** `megaloc_models/coreml_cache/` holds one entry per model file (two hash-named
directories, 16 items each — the 13 partitions plus bookkeeping), so the compute-unit choice is not part of the
key. A repeat of fp32 + `ALL` started in **36 s** instead of 174 s: the Core ML compile is cached, what remains is
ONNX Runtime's conversion of the graph into the MLProgram partitions (weights included) plus Core ML's load and
specialisation for the compute units. 36 s at every host start on the Mac is acceptable for now; fewer partitions
(the phase-6 item below) would also shrink it.

## Open (phase 2)
- `ProfileComputePlan=1` once, to see which 17 nodes stay on the CPU and whether the attention matmuls run on the
  GPU (expected) or fall back.
