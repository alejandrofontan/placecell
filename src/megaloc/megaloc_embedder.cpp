/**
 * Module: placecell - megaloc_embedder.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.2
 * - Created: 2026-09-02
 * - Updated: 2026-10-13
 * - License: Apache-2.0
 */
#include "placecell/megaloc_embedder.h"

#include <mutex>
#include <vector>

#include "placecell/log.h"

#if PLACECELL_MEGALOC_TENSORRT
#include "tensorrt_megaloc.hpp"
#else
#include "onnxruntime_megaloc.hpp"
#endif

namespace placecell {
    struct MegaLocEmbedder::Impl
    {
#if PLACECELL_MEGALOC_TENSORRT
        explicit Impl(const std::string& onnx_path, const Settings& s)
            : engine(onnx_path, s.precision), provider_name("tensorrt") {}
        megaloc::TensorRTMegaLoc engine;
#else
        explicit Impl(const std::string& onnx_path, const Settings& s)
            : engine(onnx_path, s.provider, s.threads, s.coreml_compute_units),
              provider_name(engine.provider()) {}
        megaloc::OnnxRuntimeMegaLoc engine;
#endif
        std::string provider_name;
        std::mutex mutex;   // the backends are not thread-safe: one session / execution context
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
        PLACECELL_INFO("MegaLocEmbedder", backend() << " ready (" << (impl_->engine.loadedFromCache() ? "loaded" : "built")
                       << ", provider " << impl_->provider_name << ", " << impl_->engine.precision() << " weights, "
                       << impl_->engine.descriptorDim() << "-d, input " << impl_->engine.inputWidth() << "x"
                       << impl_->engine.inputHeight() << "): " << impl_->engine.enginePath() << "; warmup "
                       << warmup.ms() << " ms");
    }

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path)
    : MegaLocEmbedder(onnx_path, Settings())
    {
    }

    MegaLocEmbedder::MegaLocEmbedder(const std::string& onnx_path, const std::string& precision)
    : MegaLocEmbedder(onnx_path, Settings{precision, "auto", 4, "ALL"})
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

    const char* MegaLocEmbedder::backend()
    {
#if PLACECELL_MEGALOC_TENSORRT
        return "tensorrt";
#else
        return "onnxruntime";
#endif
    }

    const std::string& MegaLocEmbedder::provider() const
    {
        return impl_->provider_name;
    }

    const std::string& MegaLocEmbedder::precision() const
    {
        return impl_->engine.precision();
    }

    bool MegaLocEmbedder::loaded_from_cache() const
    {
        return impl_->engine.loadedFromCache();
    }

    const std::string& MegaLocEmbedder::engine_path() const
    {
        return impl_->engine.enginePath();
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
        // Same semantics as the backends' cosine (parity for consumers migrating
        // from the wrapper): double accumulation, 0 on empty or mismatched sizes
        if(a.size() == 0 || a.size() != b.size())
            return 0.0f;
        const double dot = a.cast<double>().dot(b.cast<double>());
        const double den = a.cast<double>().norm() * b.cast<double>().norm();
        return den > 0.0 ? static_cast<float>(dot / den) : 0.0f;
    }
}
