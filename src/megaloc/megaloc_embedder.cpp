/**
 * Module: placecell - megaloc_embedder.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0
 */
#include "placecell/megaloc_embedder.h"

#include <vector>

#include "tensorrt_megaloc.hpp"

namespace placecell {
    struct MegaLocEmbedder::Impl
    {
        Impl(const std::string& onnx_path, const std::string& precision)
            : engine(onnx_path, precision) {}

        megaloc::TensorRTMegaLoc engine;
    };

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path, const std::string& precision)
    : impl_(std::make_unique<Impl>(onnx_path, precision)){}

    // Defined here, below Impl's definition, so unique_ptr<Impl> destroys a complete type
    MegaLocEmbedder::~MegaLocEmbedder() = default;

    Eigen::VectorXf MegaLocEmbedder::embed(const cv::Mat& image_bgr)
    {
        const std::vector<float> descriptor = impl_->engine.infer(image_bgr);
        return Eigen::Map<const Eigen::VectorXf>(descriptor.data(),
                                                Eigen::Index(descriptor.size()));
    }

    int MegaLocEmbedder::descriptor_dim() const
    {
        return impl_->engine.descriptorDim();
    }
}