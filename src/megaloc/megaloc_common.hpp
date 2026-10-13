/**
 * Module: placecell - megaloc_common.hpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-10-13
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * What every MegaLoc backend shares and none may reimplement: the sidecar the export
 * script writes next to the ONNX, the image preprocessing that must match
 * tools/export_megaloc.py::preprocess_cv2 exactly, the host-side re-normalisation of
 * the descriptor, the cosine helper, and the choice of model file by precision. A
 * backend (ONNX Runtime, TensorRT) only runs the graph.
 */
#ifndef PLACECELL_MEGALOC_COMMON_HPP
#define PLACECELL_MEGALOC_COMMON_HPP

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace megaloc
{

/** The sidecar '<model>.onnx.yaml' (tools/export_megaloc.py::write_sidecar_yaml):
 *  everything inference must know but never hardcode. */
struct Sidecar
{
    std::array<float, 3> mean{};   // ImageNet normalisation, RGB order
    std::array<float, 3> std{};
    std::string inputName{"input"};
    std::string outputName{"descriptor"};
    int inputWidth{0};
    int inputHeight{0};
    int descriptorDim{0};
    std::string precision{"fp32"};   // weights/internal tensors of that file (I/O are float32 either way)
};

/** Parse the sidecar; throws std::runtime_error with the path on any problem. */
Sidecar loadSidecar(const std::string& yamlPath);

/** The model file for a requested precision: 'fp16' picks '<stem>_fp16.onnx' next to
 *  `onnxPath` when it exists (logs which file is used), otherwise the given path. */
std::string resolveModelPath(const std::string& onnxPath, const std::string& precision);

/** Image -> normalised CHW float planar input ([1,3,H,W], R plane first). Must match
 *  preprocess_cv2() in tools/export_megaloc.py: 1/4-channel inputs are converted to
 *  3-channel BGR, area-interpolated resize to the sidecar size, /255, (x-mean)/std.
 *  `out` is resized to 3*H*W. */
void preprocess(const cv::Mat& imageBGR, const Sidecar& sidecar, std::vector<float>& out);

/** L2-normalise in place (double accumulation). The graph already normalises, but
 *  fp16 paths leave |d| slightly off 1; a zero vector stays zero. */
void normalizeInPlace(std::vector<float>& descriptor);

/** Cosine similarity (double accumulation); 0 when empty or of mismatched sizes. */
float cosine(const std::vector<float>& a, const std::vector<float>& b);

} // namespace megaloc

#endif // PLACECELL_MEGALOC_COMMON_HPP
