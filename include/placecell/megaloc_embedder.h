/**
 * Module: placecell - megaloc_embedder.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * MegaLocEmbedder: image -> MegaLoc global descriptor (8448-d, L2-normalised), the
 * image frontend that produces the descriptors placecell's core consumes. Thin
 * adapter over the TensorRT backend (src/megaloc/tensorrt_megaloc.hpp), which stays
 * hidden behind the Impl pointer so this header needs neither TensorRT nor CUDA.
 *
 * Contracts:
 * - The first construction for a given ONNX + precision builds the TensorRT engine
 *   (~1-2 min, benchmarks kernels on THIS GPU); the blob is cached next to the ONNX
 *   ('<onnx>.<precision>.engine') and later constructions load it in milliseconds.
 * - embed() expects BGR pixel order (cv::imread's native order); grayscale is
 *   replicated to 3 channels and BGRA drops alpha internally, but RGB input is the
 *   caller's responsibility to convert.
 * - NOT thread-safe: one execution context, callers serialise externally.
 */
#pragma once

#include <string>
#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace placecell {
class MegaLocEmbedder {
    public:
        explicit MegaLocEmbedder(const std::string& onnx_path, const std::string& precision = "fp16");
        ~MegaLocEmbedder();
        MegaLocEmbedder(const MegaLocEmbedder&) = delete;
        MegaLocEmbedder& operator=(const MegaLocEmbedder&) = delete;

        Eigen::VectorXf embed(const cv::Mat& image_bgr);
        int descriptor_dim() const;

    private:
        struct Impl;                    // forward declaration only
        std::unique_ptr<Impl> impl_;

};

}