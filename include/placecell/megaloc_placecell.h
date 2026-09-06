/**
 * Module: placecell - megaloc_placecell.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.1
 * - Created: 2026-09-02
 * - Updated: 2026-09-05
 * - License: Apache-2.0 (MegaLoc itself is MIT)
 *
 * MegaLocPlaceCell: the PlaceCell core with a MegaLoc image frontend — one call takes
 * the host's id and an OpenCV image, embeds it (MegaLocEmbedder) and stores the
 * descriptor in the core under that id. Everything PlaceCell offers (descriptor(),
 * has(), internal_id(), profiler(), recorder(), ...) is inherited.
 *
 * unexplained_information(image) embeds without storing: the information a keyframe
 * made from that image would add to the alive views (keyframe-insertion decisions).
 *
 * add_image() is idempotent like PlaceCell::add — a known id returns its internal id
 * without re-embedding. (Two threads racing to add the SAME unknown id may both run
 * the embedding; only one result is stored, the other is discarded — harmless.)
 * Thread-safe, same as the core and the embedder.
 *
 * Profiling: the image-taking entry points are timed as wholes in the store's Profiler
 * ("add_image", "unexplained_information_image"; size_a = width, size_b = height) on
 * top of the core's rows; the embedding itself is in the embedder's own Profiler.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

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
    MegaLocPlaceCell(const std::string& onnx_path, const std::string& precision, const Options& options);
    // Share an already-constructed embedder instead.
    explicit MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder);
    MegaLocPlaceCell(std::shared_ptr<MegaLocEmbedder> embedder, const Options& options);

    // Embed the image (BGR, see MegaLocEmbedder::embed) and store its descriptor
    // under the host's id; returns the internal id. Idempotent: a known id returns
    // immediately without re-embedding.
    InternalId add_image(ExternalId id, const cv::Mat& image_bgr);

    // Embed the image and return its unexplained information given the alive views
    // (or the alive views among `window`) — see PlaceCell::unexplained_information.
    // Read-only: the descriptor is NOT stored and the kernel is untouched (the host
    // decides afterwards whether the view becomes a keyframe). `descriptor_out`, when
    // given, receives the embedding so a later add() needs no second embedding.
    using PlaceCell::unexplained_information;   // keep the descriptor overload visible
    Information unexplained_information(const cv::Mat& image_bgr,
                                        const std::vector<ExternalId>* window = nullptr,
                                        bool centred = true,
                                        Eigen::VectorXf* descriptor_out = nullptr) const;

    // The underlying embedder, e.g. for transient queries that must not enter the
    // store (a relocalization query frame).
    MegaLocEmbedder& embedder() { return *embedder_; }
    const MegaLocEmbedder& embedder() const { return *embedder_; }

private:
    std::shared_ptr<MegaLocEmbedder> embedder_;
};

} // namespace placecell
