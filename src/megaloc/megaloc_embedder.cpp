/**
 * Module: placecell - megaloc_embedder.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0
 */
#include "placecell/megaloc_embedder.h"

#include <mutex>
#include <vector>

#include "tensorrt_megaloc.hpp"

namespace placecell {
    struct MegaLocEmbedder::Impl
    {
        Impl(const std::string& onnx_path, const std::string& precision)
            : engine(onnx_path, precision) {}

        megaloc::TensorRTMegaLoc engine;
        std::mutex mutex;   // TensorRTMegaLoc is not thread-safe: one execution context
    };

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path, const std::string& precision)
    : impl_(std::make_unique<Impl>(onnx_path, precision))
    {
        // Absorb the one-time lazy CUDA/TensorRT warmup here (constructor is already
        // the slow path) rather than on the first real embed()
        (void)impl_->engine.infer(cv::Mat::zeros(480, 640, CV_8UC3));
    }

    // Defined here, below Impl's definition, so unique_ptr<Impl> destroys a complete type
    MegaLocEmbedder::~MegaLocEmbedder() = default;

    Eigen::VectorXf MegaLocEmbedder::embed(const cv::Mat& image_bgr)
    {
        std::vector<float> descriptor;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            descriptor = impl_->engine.infer(image_bgr);
        }
        return Eigen::Map<const Eigen::VectorXf>(descriptor.data(),
                                                Eigen::Index(descriptor.size()));
    }

    int MegaLocEmbedder::descriptor_dim() const
    {
        return impl_->engine.descriptorDim();
    }

    bool MegaLocEmbedder::loaded_from_cache() const
    {
        return impl_->engine.loadedFromCache();
    }

    const std::string& MegaLocEmbedder::engine_path() const
    {
        return impl_->engine.enginePath();
    }

    float MegaLocEmbedder::cosine(const Eigen::Ref<const Eigen::VectorXf>& a,
                                  const Eigen::Ref<const Eigen::VectorXf>& b)
    {
        // Same semantics as TensorRTMegaLoc::cosine (parity for consumers migrating
        // from the wrapper): double accumulation, 0 on empty or mismatched sizes
        if(a.size() == 0 || a.size() != b.size())
            return 0.0f;
        const double dot = a.cast<double>().dot(b.cast<double>());
        const double den = a.cast<double>().norm() * b.cast<double>().norm();
        return den > 0.0 ? static_cast<float>(dot / den) : 0.0f;
    }
}