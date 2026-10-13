# MegaLoc ONNX export: dynamo exporter, fp16 twin, and what the export wrapper is for

- **Date:** 2026-10-13
- **Context:** AllFeature-VSLAM `docs/plans/megaloc_onnxruntime.md` (phase 2 of the macOS port), step 3. The export
  moves to PyTorch's `torch.export`-based ONNX exporter and gains an fp16-weights twin; this note records the
  side-by-side measurements the decision rests on. Export environment: torch 2.10.0 (CPU build), onnxscript 0.7.2,
  onnx 1.20.0, onnxruntime 1.28.0 (CPU), onnxconverter-common 1.16.0. Command:

```bash
pixi run -e export-megaloc export-models -- --exporter both --raw --test_images <4 ETH table_3 frames>
# frames: rgb_0/2989.023949.png 3000.046089.png 3011.105092.png 3029.536764.png (first, 300th, 600th, 1100th)
```

The two comparison switches (`--exporter torchscript|both`, `--raw`) were removed from the script right after this
run: the TorchScript path is deprecated upstream and the question they answered is settled here. To repeat the
experiment, check out the commit before the cleanup or re-add ~40 lines around `export()`/`self_check()`.

## The four exports

| File | Exporter | Nodes | Initialisers | Op types | Size |
|---|---|---|---|---|---|
| `megaloc_322x322.onnx` (the published fp32) | dynamo, opset 18, `optimize=True` | 478 | 217 | 23 | 870.2 MB |
| `megaloc_322x322_torchscript.onnx` | TorchScript tracer (deprecated), opset 18 | 807 | 191 | 29 | 869.8 MB |
| `megaloc_322x322_raw.onnx` | dynamo, **upstream module as is** (no `ExportWrapper`) | 496 | 225 | 27 | 873.2 MB |
| `megaloc_322x322_fp16.onnx` (published) | dynamo fp32 → `onnxconverter_common.float16`, `keep_io_types` | 478 | 217 | | 435.4 MB |

Operator histograms:

- dynamo: Add×92, MatMul×72, Mul×60, Reshape×53, Transpose×50, Slice×38, LayerNormalization×25, Div×16, Softmax×12,
  Erf×12, Sub×7, ReduceLogSumExp×6, Unsqueeze×6, Conv×5, ReduceL2×4, Clip×4, Expand×4, Concat×3, Relu×3, Gemm×3,
  Gather×1, Exp×1, Einsum×1.
- torchscript: the same plus Constant×296, Shape×7, ConstantOfShape×4, Equal×4, Where×4, Flatten×1, and a few more
  Expand/Concat/Slice — shape arithmetic and constants the dynamo exporter folds into initialisers.
- raw upstream: the dynamo set plus **Resize×1** (the positional-embedding interpolation evaluated at runtime),
  **ScatterND×1 + ScatterElements×1** (the Sinkhorn solver's in-place slice writes, functionalised), **Gather×38 +
  Squeeze×6** (the 5-D `reshape/permute/index` attention split) instead of Slice×38 / Transpose×50.

## Numerical self-check (4 frames; reference = original torch model with the README transform)

| | 2989.02 | 3000.05 | 3011.11 | 3029.54 |
|---|---|---|---|---|
| wrapper vs original, max abs diff (same input) | 2.3e-7 | 2.3e-7 | 4.1e-7 | 4.1e-7 |
| cos(wrapper + cv2 preprocessing, reference) | 0.99527 | 0.99581 | 0.99564 | 0.99579 |
| cos(ONNX dynamo fp32, reference) | 0.99527 | 0.99581 | 0.99564 | 0.99579 |
| cos(ONNX dynamo fp32, wrapper cv2) | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| cos(ONNX fp16 twin, reference) | 0.99521 | 0.99580 | 0.99564 | 0.99577 |
| cos(ONNX torchscript, reference); max abs diff to dynamo | 0.99527; 2.5e-7 | 0.99581; 3.7e-7 | 0.99564; 3.1e-7 | 0.99579; 3.4e-7 |
| cos(ONNX raw upstream, reference); max abs diff to dynamo | 0.99527; 3.7e-7 | 0.99581; 2.5e-7 | 0.99564; 3.6e-7 | 0.99579; 2.6e-7 |

Pairwise cosine of the dynamo descriptors: 0.576 between the first and last frame (same part of the table), 0.21–0.33
elsewhere.

## What this settles

1. **Exporter: dynamo.** Same function to 4e-7, 40 % fewer nodes, no shape arithmetic in the graph, and it is what
   PyTorch maintains. `--exporter torchscript|both` stays for comparisons only.
2. **The `ExportWrapper` is no longer needed for correctness** — `torch.export` captures the upstream `forward` as
   is (the resize branch is specialised away, the in-place writes are functionalised) — **but it is still wanted for
   the graph**: the raw export carries a runtime `Resize`, two scatter ops and 38 `Gather`s, exactly the operators
   with the patchiest support on CoreML and the ones TensorRT 10.3 miscompiled. The three rewrites (precomputed
   positional embedding, Sinkhorn with `cat`, sliced q/k/v) remain the right default.
3. **fp16 weights in the backbone are free:** ≤ 6e-5 in cosine on these frames, with the aggregator (65 nodes: the
   final LayerNormalization and everything downstream) kept fp32 by the topological rule. TensorRT's "uniform fp16
   loses 2–7 %" was a TensorRT-kernel matter, not a property of half-precision weights. The fp16 file is the one to
   hand to CoreML and halves the download.
4. **The dominant error is preprocessing, identical for every backend:** 0.9953–0.9958 against the README reference
   comes from OpenCV's area resize versus torchvision's antialiased bilinear resize (plus PIL vs OpenCV decoding),
   and every ONNX path agrees with the cv2-preprocessed wrapper to 1e-6. Inside the SLAM only our own descriptors
   are compared with each other, so this does not affect retrieval consistency; it only matters for reproducing
   the paper's numbers. **Parity criteria for the C++ backends, restated:** cos(backend, reference) ≥ 0.995 (the
   preprocessing floor) and cos(backend, ORT-CPU fp32) ≥ 0.9999 for fp32 providers, ≥ 0.999 for fp16/CoreML.
5. Harmless noise in the log: `onnxconverter_common/float16.py` warns for ~70 constants below 1e-7 that it clamps
   (the fp16 column shows no effect); the four `_schemas.py` warnings at start are torchvision registering its own
   `roi_align`/`roi_pool` ops with the exporter.

## Open

- Preprocessing fidelity (item 4) is a lever for paper-parity only; a C++ antialiased bilinear resize matching
  torchvision would close most of the 0.45 % if ever needed.
- The TensorRT backend's "mixed precision" never matched any layer (the tracer export has no `/backbone/` node
  names), so the `fp16.engine` in use is effectively fp32 — recorded in the AllFeature plan; moot once TensorRT
  leaves.
