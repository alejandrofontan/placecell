/**
 * Module: placecell - onnxruntime_megaloc.hpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-10-13
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * OnnxRuntimeMegaLoc: image -> MegaLoc global descriptor (8448-d, L2-normalised), the
 * ONNX Runtime backend of placecell's MegaLoc embedder. One C++ path on every
 * platform; the hardware is an execution-provider choice made at session creation:
 *
 *   provider  "auto"   CUDA where ONNX Runtime was built with it, else CoreML (macOS),
 *                      else CPU
 *             "cuda"   NVIDIA GPU (the CUDA execution provider; cuDNN at runtime)
 *             "coreml" Apple GPU / Neural Engine through Core ML (MLProgram format,
 *                      compiled once per model file and cached next to it)
 *             "cpu"    the CPU provider only (`threads` intra-op threads)
 *
 * A requested provider that is not available falls back to "auto" with a warning; the
 * CPU provider is always appended last, so any node a provider does not support still
 * runs. Unlike TensorRT there is no engine build: session creation takes seconds (CPU,
 * CUDA) to a minute (CoreML's first compile), and nothing is GPU-specific on disk.
 *
 * The model file is loaded as given: precision is a property of the file (the export
 * writes 'megaloc_322x322.onnx' in fp32 and the twin 'megaloc_322x322_fp16.onnx' with
 * fp16 weights, I/O float32 in both), read back from the sidecar '<model>.onnx.yaml',
 * which also supplies input size, normalisation, tensor names and descriptor dim
 * (megaloc_common.hpp). The fp16 file is the one for CUDA / Core ML, the fp32 file the
 * faster one on the CPU provider.
 */
#ifndef PLACECELL_ONNXRUNTIME_MEGALOC_HPP
#define PLACECELL_ONNXRUNTIME_MEGALOC_HPP

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "megaloc_common.hpp"

namespace megaloc
{

class OnnxRuntimeMegaLoc
{
  public:
    /**
     * Create the session for an exported MegaLoc ONNX.
     * @param onnxPath   the model file to load (fp32 export or its fp16 twin, see above)
     * @param provider   "auto" | "cuda" | "coreml" | "cpu"
     * @param threads    intra-op threads of the CPU provider (also the fallback nodes); >= 1
     * @param coremlComputeUnits  "ALL" | "CPUAndGPU" | "CPUAndNeuralEngine" | "CPUOnly" (CoreML only)
     * @param sidecarYamlPath sidecar path; empty -> '<onnxPath>.yaml'
     * @throws std::runtime_error on any unrecoverable setup failure
     */
    explicit OnnxRuntimeMegaLoc(const std::string& onnxPath, const std::string& provider = "auto",
                                int threads = 4, const std::string& coremlComputeUnits = "ALL",
                                const std::string& sidecarYamlPath = "");
    ~OnnxRuntimeMegaLoc();

    OnnxRuntimeMegaLoc(const OnnxRuntimeMegaLoc&) = delete;
    OnnxRuntimeMegaLoc& operator=(const OnnxRuntimeMegaLoc&) = delete;

    /** Global descriptor of one image (BGR, cv::imread order; grayscale replicated,
     *  BGRA drops alpha), L2-normalised on the host. Synchronous. NOT thread-safe:
     *  callers serialise externally (MegaLocEmbedder does). */
    std::vector<float> infer(const cv::Mat& imageBGR);

    /** Cosine similarity of two descriptors (dot product for unit vectors). */
    static float cosine(const std::vector<float>& a, const std::vector<float>& b);

    /** One line per model input/output: name, shape, element type. */
    std::string describeIOTensors() const;

    /** The provider the session got: "cuda", "coreml" or "cpu". */
    const std::string& provider() const { return provider_; }
    /** The loaded file's weight precision, from its sidecar ("fp32" | "fp16"). */
    const std::string& precision() const { return sidecar_.precision; }
    /** The model file actually loaded (the fp16 twin when precision asked for it). */
    const std::string& enginePath() const { return modelPath_; }
    /** No engine cache in this backend: always true (the model file is the artefact). */
    bool loadedFromCache() const { return true; }
    int descriptorDim() const { return sidecar_.descriptorDim; }
    int inputWidth() const { return sidecar_.inputWidth; }
    int inputHeight() const { return sidecar_.inputHeight; }

  private:
    /** Append the execution providers for `provider` to `options`; returns the name
     *  of the one that will run the graph. */
    std::string appendProviders(Ort::SessionOptions& options, const std::string& provider,
                                const std::string& coremlComputeUnits) const;
    /** Read the session's I/O names and shapes and cross-check them against the sidecar. */
    void checkIOTensors();

    std::string modelPath_;
    std::string provider_;
    Sidecar sidecar_;

    // Declaration order is destruction order in reverse: session before options before env
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_;

    std::vector<int64_t> inputShape_;    // [1,3,H,W]
    std::vector<float> hostInput_;       // CHW planar, wrapped by the input Ort::Value
    const char* inputNames_[1]{nullptr};
    const char* outputNames_[1]{nullptr};
};

} // namespace megaloc

#endif // PLACECELL_ONNXRUNTIME_MEGALOC_HPP
