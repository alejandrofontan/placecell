/**
 * Module: placecell - megaloc_embedder.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 2.0
 * - Created: 2026-09-02
 * - Updated: 2026-10-13
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * MegaLocEmbedder: image -> MegaLoc global descriptor (8448-d, L2-normalised), the
 * image frontend that produces the descriptors placecell's core consumes. Thin
 * adapter over the ONNX Runtime backend (src/megaloc/onnxruntime_megaloc.hpp), hidden
 * behind the Impl pointer so this header needs neither ONNX Runtime nor any GPU SDK.
 *
 * Contracts:
 * - The model file is loaded as given; its weight precision is a property of the file
 *   (the export writes an fp32 file and an '_fp16' twin, both with float32 I/O; the
 *   sidecar '<model>.onnx.yaml' says which it is). Settings select the execution
 *   provider (auto | cuda | coreml | cpu), the CPU thread budget and the Core ML compute
 *   units. A provider that is not available falls back with a warning, never a failure.
 * - Construction is the slow path: the session is created (seconds on CPU / CUDA; up to
 *   a minute for Core ML's first compile of a model file, ~35 s on later starts with the
 *   compiled model cached next to it) and a dummy inference absorbs the one-time warmup,
 *   so the first real embed() runs at steady-state speed.
 * - embed() expects BGR pixel order (cv::imread's native order); grayscale is
 *   replicated to 3 channels and BGRA drops alpha internally, but RGB input is the
 *   caller's responsibility to convert.
 * - Thread-safe: embed() serialises internally on the single session, so concurrent
 *   callers (e.g. a mapping thread and a relocalization query) need no external lock.
 *   Calls block while another inference is in flight.
 * - Diagnostics: session setup and ONNX Runtime's own messages go through the placecell
 *   Logger (components "MegaLocEmbedder", "OnnxRuntimeMegaLoc", "ORT"); every embed() is
 *   timed in the embedder's own Profiler under "embed" (size_a = width, size_b = height).
 */
#pragma once

#include <string>
#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "placecell/profiler.h"

namespace placecell {
class MegaLocEmbedder {
    public:
        struct Settings
        {
            std::string provider{"auto"};            // "auto" | "cuda" | "coreml" | "cpu"
            int threads{4};                          // intra-op threads of the CPU provider (>= 1)
            std::string coreml_compute_units{"ALL"}; // CoreML: "ALL" | "CPUAndGPU" | "CPUAndNeuralEngine" | "CPUOnly"
        };

        // Default Settings (a nested struct's member initialisers cannot serve as a default argument of
        // the enclosing class, hence two constructors)
        explicit MegaLocEmbedder(const std::string& onnx_path);
        MegaLocEmbedder(const std::string& onnx_path, const Settings& settings);
        ~MegaLocEmbedder();
        MegaLocEmbedder(const MegaLocEmbedder&) = delete;
        MegaLocEmbedder& operator=(const MegaLocEmbedder&) = delete;

        Eigen::VectorXf embed(const cv::Mat& image_bgr);
        int descriptor_dim() const;

        // The provider that runs the graph: "cuda" | "coreml" | "cpu"
        const std::string& provider() const;
        // Weight precision of the loaded file, from its sidecar: "fp32" | "fp16"
        const std::string& precision() const;
        // The model file that was loaded
        const std::string& model_path() const;

        Profiler& profiler();
        const Profiler& profiler() const;

        // Cosine similarity of two descriptors (double accumulation; 0 when empty or
        // of mismatched size). Descriptors are L2-normalised, so this is ~their dot
        // product, but the explicit normalisation keeps it exact under fp16 rounding.
        // Eigen::Ref accepts VectorXf, blocks, and Eigen::Map over external buffers
        // (e.g. a std::vector<float>) without copying.
        static float cosine(const Eigen::Ref<const Eigen::VectorXf>& a,
                            const Eigen::Ref<const Eigen::VectorXf>& b);

    private:
        struct Impl;                    // forward declaration only
        std::unique_ptr<Impl> impl_;

};

}
