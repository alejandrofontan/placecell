/**
 * Module: placecell - megaloc_placecell.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * MegaLocPlaceCell: the PlaceCell core with a MegaLoc image frontend — one call takes
 * the host's id and an OpenCV image, embeds it (MegaLocEmbedder) and stores the
 * descriptor in the core under that id. Everything PlaceCell offers (descriptor(),
 * has(), internal_id(), ...) is inherited.
 *
 * add_image() is idempotent like PlaceCell::add — a known id returns its internal id
 * without re-embedding. (Two threads racing to add the SAME unknown id may both run
 * the embedding; only one result is stored, the other is discarded — harmless.)
 * Thread-safe, same as the core and the embedder.
 */
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "placecell/megaloc_embedder.h"
#include "placecell/placecell.h"

namespace placecell
{

class MegaLocPlaceCell : public PlaceCell
{
public:
    // Builds/loads the TensorRT engine and warms it up (see MegaLocEmbedder).
    explicit MegaLocPlaceCell(const std::string& onnx_path, const std::string& precision = "fp16");
    // Share an already-constructed embedder instead.
    explicit MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder);

    // Embed the image (BGR, see MegaLocEmbedder::embed) and store its descriptor
    // under the host's id; returns the internal id. Idempotent: a known id returns
    // immediately without re-embedding.
    InternalId add_image(ExternalId id, const cv::Mat& image_bgr);

    // The underlying embedder, e.g. for transient queries that must not enter the
    // store (a relocalization query frame).
    MegaLocEmbedder& embedder() { return *embedder_; }
    const MegaLocEmbedder& embedder() const { return *embedder_; }

private:
    std::shared_ptr<MegaLocEmbedder> embedder_;
};

} // namespace placecell
