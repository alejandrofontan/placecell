/**
 * Module: placecell - megaloc_placecell.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0
 */
#include "placecell/megaloc_placecell.h"

#include <utility>

namespace placecell {

MegaLocPlaceCell::MegaLocPlaceCell(const std::string& onnx_path, const std::string& precision)
    : embedder_(std::make_shared<MegaLocEmbedder>(onnx_path, precision))
{
}

MegaLocPlaceCell::MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder)
    : embedder_(std::move(embedder))
{
}

PlaceCell::InternalId MegaLocPlaceCell::add_image(const ExternalId id, const cv::Mat& image_bgr)
{
    const InternalId existing = internal_id(id);
    if(existing != invalid_id)
        return existing;   // known id: never re-embed
    return add(id, embedder_->embed(image_bgr));
}

} // namespace placecell
