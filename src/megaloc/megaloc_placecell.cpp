/**
 * Module: placecell - megaloc_placecell.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.1
 * - Created: 2026-09-02
 * - Updated: 2026-09-05
 * - License: Apache-2.0
 */
#include "placecell/megaloc_placecell.h"

#include <utility>

namespace placecell {

MegaLocPlaceCell::MegaLocPlaceCell(const std::string& onnx_path, const std::string& precision)
    : MegaLocPlaceCell(onnx_path, precision, Options{})
{
}

MegaLocPlaceCell::MegaLocPlaceCell(const std::string& onnx_path, const std::string& precision, const Options& options)
    : PlaceCell(options), embedder_(std::make_shared<MegaLocEmbedder>(onnx_path, precision))
{
    profiler().declare({"add_image", "unexplained_information_image"});
}

MegaLocPlaceCell::MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder)
    : MegaLocPlaceCell(std::move(embedder), Options{})
{
}

MegaLocPlaceCell::MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder, const Options& options)
    : PlaceCell(options), embedder_(std::move(embedder))
{
    profiler().declare({"add_image", "unexplained_information_image"});
}

PlaceCell::InternalId MegaLocPlaceCell::add_image(const ExternalId id, const cv::Mat& image_bgr)
{
    const InternalId existing = internal_id(id);
    if(existing != invalid_id)
        return existing;   // known id: never re-embed
    Profiler::Scope timer(profiler(), "add_image");
    timer.set_sizes(image_bgr.cols, image_bgr.rows);
    return add(id, embedder_->embed(image_bgr));
}

PlaceCell::Information MegaLocPlaceCell::unexplained_information(const cv::Mat& image_bgr,
                                                                 const std::vector<ExternalId>* window,
                                                                 const bool centred,
                                                                 Eigen::VectorXf* descriptor_out) const
{
    Profiler::Scope timer(mutable_profiler(), "unexplained_information_image");
    timer.set_sizes(image_bgr.cols, image_bgr.rows);
    Eigen::VectorXf descriptor = embedder_->embed(image_bgr);
    const Information information = PlaceCell::unexplained_information(descriptor, window, centred);
    if(descriptor_out)
        *descriptor_out = std::move(descriptor);
    return information;
}

} // namespace placecell
