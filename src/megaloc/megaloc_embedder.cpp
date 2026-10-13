/**
 * Module: placecell - megaloc_embedder.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 2.0
 * - Created: 2026-09-02
 * - Updated: 2026-10-13
 * - License: Apache-2.0
 */
#include "placecell/megaloc_embedder.h"

#include <mutex>
#include <vector>

#include "placecell/log.h"
#include "onnxruntime_megaloc.hpp"

namespace placecell {
    struct MegaLocEmbedder::Impl
    {
        explicit Impl(const std::string& onnx_path, const Settings& s)
            : engine(onnx_path, s.provider, s.threads, s.coreml_compute_units) {}

        megaloc::OnnxRuntimeMegaLoc engine;
        std::mutex mutex;   // the backend is not thread-safe: one session, one inference at a time
        Profiler profiler{"MegaLocEmbedder"};
    };

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path, const Settings& settings)
    : impl_(std::make_unique<Impl>(onnx_path, settings))
    {
        impl_->profiler.declare({"embed"});
        // Absorb the one-time lazy warmup (kernel selection, allocator growth, CoreML
        // compile) here (the constructor is already the slow path) rather than on the
        // first real embed()
        const Profiler::Stopwatch warmup;
        (void)impl_->engine.infer(cv::Mat::zeros(480, 640, CV_8UC3));
        PLACECELL_INFO("MegaLocEmbedder", "ready (provider " << impl_->engine.provider() << ", "
                       << impl_->engine.precision() << " weights, " << impl_->engine.descriptorDim() << "-d, input "
                       << impl_->engine.inputWidth() << "x" << impl_->engine.inputHeight() << "): "
                       << impl_->engine.modelPath() << "; warmup " << warmup.ms() << " ms");
    }

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path)
    : MegaLocEmbedder(onnx_path, Settings())
    {
    }

    // Defined here, below Impl's definition, so unique_ptr<Impl> destroys a complete type
    MegaLocEmbedder::~MegaLocEmbedder() = default;

    Eigen::VectorXf MegaLocEmbedder::embed(const cv::Mat& image_bgr)
    {
        Profiler::Scope timer(impl_->profiler, "embed");
        timer.set_sizes(image_bgr.cols, image_bgr.rows);
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

    const std::string& MegaLocEmbedder::provider() const
    {
        return impl_->engine.provider();
    }

    const std::string& MegaLocEmbedder::precision() const
    {
        return impl_->engine.precision();
    }

    const std::string& MegaLocEmbedder::model_path() const
    {
        return impl_->engine.modelPath();
    }

    Profiler& MegaLocEmbedder::profiler()
    {
        return impl_->profiler;
    }

    const Profiler& MegaLocEmbedder::profiler() const
    {
        return impl_->profiler;
    }

    float MegaLocEmbedder::cosine(const Eigen::Ref<const Eigen::VectorXf>& a,
                                  const Eigen::Ref<const Eigen::VectorXf>& b)
    {
        // Same semantics as megaloc::cosine (parity for consumers migrating from the
        // backend): double accumulation, 0 on empty or mismatched sizes
        if(a.size() == 0 || a.size() != b.size())
            return 0.0f;
        const double dot = a.cast<double>().dot(b.cast<double>());
        const double den = a.cast<double>().norm() * b.cast<double>().norm();
        return den > 0.0 ? static_cast<float>(dot / den) : 0.0f;
    }
}
