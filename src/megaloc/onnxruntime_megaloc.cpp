/**
 * Module: placecell - onnxruntime_megaloc.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-10-13
 * - License: Apache-2.0
 */
#include "onnxruntime_megaloc.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "placecell/log.h"

namespace megaloc
{

namespace
{
/** ONNX Runtime's log channel, mapped onto placecell's levels (component "ORT"). */
void ortLog(void* /*param*/, OrtLoggingLevel severity, const char* category, const char* /*logid*/,
            const char* /*code_location*/, const char* message)
{
    switch (severity)
    {
        case ORT_LOGGING_LEVEL_FATAL:
        case ORT_LOGGING_LEVEL_ERROR:
            PLACECELL_ERROR("ORT", category << ": " << message);
            break;
        case ORT_LOGGING_LEVEL_WARNING:
            PLACECELL_WARN("ORT", category << ": " << message);
            break;
        case ORT_LOGGING_LEVEL_INFO:
            PLACECELL_DEBUG("ORT", category << ": " << message);
            break;
        case ORT_LOGGING_LEVEL_VERBOSE:
        default:
            PLACECELL_TRACE("ORT", category << ": " << message);
            break;
    }
}

bool hasProvider(const std::vector<std::string>& available, const char* name)
{
    return std::find(available.begin(), available.end(), name) != available.end();
}

std::string shapeToString(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
        oss << (i ? "," : "") << shape[i];
    oss << "]";
    return oss.str();
}
} // namespace

OnnxRuntimeMegaLoc::OnnxRuntimeMegaLoc(const std::string& onnxPath, const std::string& precision,
                                       const std::string& provider, int threads,
                                       const std::string& coremlComputeUnits,
                                       const std::string& sidecarYamlPath)
    : modelPath_(resolveModelPath(onnxPath, precision)),
      sidecar_(loadSidecar(sidecarYamlPath.empty() ? modelPath_ + ".yaml" : sidecarYamlPath)),
      env_(ORT_LOGGING_LEVEL_WARNING, "placecell", &ortLog, nullptr),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    if (precision != "fp16" && precision != "fp32")
        throw std::runtime_error("OnnxRuntimeMegaLoc: precision must be fp16 or fp32, got " + precision);
    if (threads < 1)
        throw std::runtime_error("OnnxRuntimeMegaLoc: threads must be >= 1");

    // Threads: a fixed small number, never the core count - the embedder shares the
    // machine with the tracking and mapping threads. Sequential execution: one graph
    // at a time, no inter-op parallelism to reason about.
    options_.SetIntraOpNumThreads(threads);
    options_.SetInterOpNumThreads(1);
    options_.SetExecutionMode(ORT_SEQUENTIAL);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    provider_ = appendProviders(options_, provider, coremlComputeUnits);

    PLACECELL_INFO("OnnxRuntimeMegaLoc", "creating session for " << modelPath_ << " (provider " << provider_
                   << ", " << threads << " threads)...");
    try
    {
        session_ = std::make_unique<Ort::Session>(env_, modelPath_.c_str(), options_);
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("OnnxRuntimeMegaLoc: session creation failed for ") + modelPath_ +
                                 ": " + e.what());
    }
    checkIOTensors();
    hostInput_.assign(static_cast<size_t>(3) * sidecar_.inputHeight * sidecar_.inputWidth, 0.0f);
    inputShape_ = {1, 3, sidecar_.inputHeight, sidecar_.inputWidth};
}

OnnxRuntimeMegaLoc::~OnnxRuntimeMegaLoc() = default;

std::string OnnxRuntimeMegaLoc::appendProviders(Ort::SessionOptions& options, const std::string& provider,
                                                const std::string& coremlComputeUnits) const
{
    const std::vector<std::string> available = Ort::GetAvailableProviders();
    const bool hasCuda = hasProvider(available, "CUDAExecutionProvider");
    const bool hasCoreML = hasProvider(available, "CoreMLExecutionProvider");

    std::string chosen = provider;
    if (chosen != "auto" && chosen != "cuda" && chosen != "coreml" && chosen != "cpu")
        throw std::runtime_error("OnnxRuntimeMegaLoc: provider must be auto|cuda|coreml|cpu, got " + provider);
    if ((chosen == "cuda" && !hasCuda) || (chosen == "coreml" && !hasCoreML))
    {
        std::ostringstream list;
        for (const auto& p : available)
            list << " " << p;
        PLACECELL_WARN("OnnxRuntimeMegaLoc", "provider '" << chosen << "' is not available in this ONNX Runtime ("
                       << list.str() << " ) - falling back to auto");
        chosen = "auto";
    }
    if (chosen == "auto")
        chosen = hasCuda ? "cuda" : hasCoreML ? "coreml" : "cpu";

    if (chosen == "cuda")
    {
        // V2 options through the C API (the C++ wrapper only forwards them)
        OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
        try
        {
            const char* keys[] = {"device_id", "cudnn_conv_use_max_workspace"};
            const char* values[] = {"0", "1"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cudaOptions, keys, values, 2));
            options.AppendExecutionProvider_CUDA_V2(*cudaOptions);
        }
        catch (...)
        {
            Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
            throw;
        }
        Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
    }
    else if (chosen == "coreml")
    {
        // MLProgram: the Core ML format with the widest operator coverage (macOS 12+).
        // Static shapes let Core ML specialise; the compiled model is cached next to
        // the ONNX so later sessions skip the compile.
        const std::filesystem::path cacheDir =
            std::filesystem::path(modelPath_).parent_path() / "coreml_cache";
        std::unordered_map<std::string, std::string> coremlOptions{
            {"ModelFormat", "MLProgram"},
            {"MLComputeUnits", coremlComputeUnits},
            {"RequireStaticInputShapes", "1"},
            {"EnableOnSubgraphs", "0"},
            {"ModelCacheDirectory", cacheDir.string()},
        };
        options.AppendExecutionProvider("CoreML", coremlOptions);
    }
    // The CPU provider is implicit and last: it takes every node the others decline.
    return chosen;
}

void OnnxRuntimeMegaLoc::checkIOTensors()
{
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
        throw std::runtime_error("OnnxRuntimeMegaLoc: expected one input and one output - wrong ONNX?\n" +
                                 describeIOTensors());
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr inName = session_->GetInputNameAllocated(0, allocator);
    const Ort::AllocatedStringPtr outName = session_->GetOutputNameAllocated(0, allocator);
    if (sidecar_.inputName != inName.get() || sidecar_.outputName != outName.get())
        throw std::runtime_error("OnnxRuntimeMegaLoc: sidecar tensor names (" + sidecar_.inputName + ", " +
                                 sidecar_.outputName + ") disagree with the model:\n" + describeIOTensors());
    // The names must outlive the session calls: the sidecar strings do.
    inputNames_[0] = sidecar_.inputName.c_str();
    outputNames_[0] = sidecar_.outputName.c_str();

    // GetTensorTypeAndShapeInfo() is a view into the TypeInfo: keep the owners alive
    const Ort::TypeInfo inType = session_->GetInputTypeInfo(0);
    const Ort::TypeInfo outType = session_->GetOutputTypeInfo(0);
    const auto inInfo = inType.GetTensorTypeAndShapeInfo();
    const auto outInfo = outType.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> inShape = inInfo.GetShape();
    const std::vector<int64_t> outShape = outInfo.GetShape();
    if (inShape.size() != 4 || outShape.size() != 2)
        throw std::runtime_error("OnnxRuntimeMegaLoc: unexpected tensor ranks - wrong ONNX?\n" + describeIOTensors());
    if (inShape[2] != sidecar_.inputHeight || inShape[3] != sidecar_.inputWidth)
        throw std::runtime_error("OnnxRuntimeMegaLoc: sidecar input size disagrees with the model:\n" +
                                 describeIOTensors());
    if (outShape[1] != sidecar_.descriptorDim)
        throw std::runtime_error("OnnxRuntimeMegaLoc: sidecar descriptor_dim disagrees with the model:\n" +
                                 describeIOTensors());
    if (inInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        outInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("OnnxRuntimeMegaLoc: I/O tensors are not float32 - wrong ONNX export?");
}

std::vector<float> OnnxRuntimeMegaLoc::infer(const cv::Mat& imageBGR)
{
    preprocess(imageBGR, sidecar_, hostInput_);
    // The input Ort::Value wraps our own buffer (no allocation per call); the output
    // is allocated by the session and copied out.
    Ort::Value input = Ort::Value::CreateTensor<float>(memoryInfo_, hostInput_.data(), hostInput_.size(),
                                                       inputShape_.data(), inputShape_.size());
    std::vector<Ort::Value> outputs;
    try
    {
        outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames_, &input, 1, outputNames_, 1);
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("OnnxRuntimeMegaLoc: Run failed: ") + e.what());
    }
    const float* data = outputs.front().GetTensorData<float>();
    std::vector<float> descriptor(data, data + sidecar_.descriptorDim);
    normalizeInPlace(descriptor);
    return descriptor;
}

float OnnxRuntimeMegaLoc::cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    return megaloc::cosine(a, b);
}

std::string OnnxRuntimeMegaLoc::describeIOTensors() const
{
    std::ostringstream oss;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_->GetInputCount(); ++i)
    {
        const Ort::TypeInfo type = session_->GetInputTypeInfo(i);
        const auto info = type.GetTensorTypeAndShapeInfo();
        oss << "  input  " << session_->GetInputNameAllocated(i, allocator).get() << " "
            << shapeToString(info.GetShape()) << " dtype=" << static_cast<int>(info.GetElementType()) << "\n";
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i)
    {
        const Ort::TypeInfo type = session_->GetOutputTypeInfo(i);
        const auto info = type.GetTensorTypeAndShapeInfo();
        oss << "  output " << session_->GetOutputNameAllocated(i, allocator).get() << " "
            << shapeToString(info.GetShape()) << " dtype=" << static_cast<int>(info.GetElementType()) << "\n";
    }
    return oss.str();
}

} // namespace megaloc
