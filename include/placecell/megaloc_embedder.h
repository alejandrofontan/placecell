/**
 * Module: placecell - megaloc_embedder.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.1
 * - Created: 2026-09-02
 * - Updated: 2026-09-05
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * MegaLocEmbedder: image -> MegaLoc global descriptor (8448-d, L2-normalised), the
 * image frontend that produces the descriptors placecell's core consumes. Thin
 * adapter over one inference backend, chosen at build time
 * (PLACECELL_MEGALOC_BACKEND = onnxruntime | tensorrt) and hidden behind the Impl
 * pointer, so this header needs neither ONNX Runtime, TensorRT nor CUDA.
 *
 * Contracts:
 * - Settings select the model file by precision ("fp16" uses '<stem>_fp16.onnx' next
 *   to the given ONNX when it exists), the execution provider (ONNX Runtime:
 *   auto | cuda | coreml | cpu; TensorRT ignores it) and the CPU thread budget. A
 *   provider that is not available falls back with a warning, never a failure.
 * - Construction is the slow path: ONNX Runtime creates the session (seconds; a
 *   minute for CoreML's first compile, cached next to the model), TensorRT builds or
 *   loads its engine ('<onnx>.<precision>.engine', 1-2 min on first use). The
 *   constructor also absorbs the one-time warmup (a dummy inference), so the first
 *   real embed() runs at steady-state speed.
 * - embed() expects BGR pixel order (cv::imread's native order); grayscale is
 *   replicated to 3 channels and BGRA drops alpha internally, but RGB input is the
 *   caller's responsibility to convert.
 * - Thread-safe: embed() serialises internally on the single session/context, so
 *   concurrent callers (e.g. a mapping thread and a relocalization query) need no
 *   external lock. Calls block while another inference is in flight.
 * - Diagnostics: session/engine setup and the backend's own messages go through the
 *   placecell Logger (components "MegaLocEmbedder", "ORT" / "TensorRT"); every
 *   embed() is timed in the embedder's own Profiler under "embed" (size_a = width,
 *   size_b = height).
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
            std::string precision{"fp16"};          // "fp32" | "fp16" (model file choice; TensorRT: engine precision)
            std::string provider{"auto"};           // ONNX Runtime: "auto" | "cuda" | "coreml" | "cpu"
            int threads{4};                         // ONNX Runtime: intra-op threads of the CPU provider (>= 1)
            std::string coreml_compute_units{"ALL"}; // CoreML: "ALL" | "CPUAndGPU" | "CPUAndNeuralEngine" | "CPUOnly"
        };

        // Default Settings (a nested struct's member initialisers cannot serve as a default argument of
        // the enclosing class, hence two constructors)
        explicit MegaLocEmbedder(const std::string& onnx_path);
        MegaLocEmbedder(const std::string& onnx_path, const Settings& settings);
        // Shorthand for Settings{precision} (the pre-settings signature)
        MegaLocEmbedder(const std::string& onnx_path, const std::string& precision);
        ~MegaLocEmbedder();
        MegaLocEmbedder(const MegaLocEmbedder&) = delete;
        MegaLocEmbedder& operator=(const MegaLocEmbedder&) = delete;

        Eigen::VectorXf embed(const cv::Mat& image_bgr);
        int descriptor_dim() const;

        // "onnxruntime" | "tensorrt" (compile-time choice)
        static const char* backend();
        // The provider that runs the graph: ONNX Runtime "cuda" | "coreml" | "cpu"; TensorRT "tensorrt"
        const std::string& provider() const;
        // True if the engine came from the cache file, false if it was built now
        // (ONNX Runtime: always true, the model file is the artefact)
        bool loaded_from_cache() const;
        // The artefact actually loaded: the model file (ONNX Runtime) or the engine (TensorRT)
        const std::string& engine_path() const;

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
