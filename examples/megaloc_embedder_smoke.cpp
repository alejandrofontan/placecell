/**
 * Module: placecell - megaloc_embedder_smoke.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0
 *
 * Smoke test of the PUBLIC placecell API (no TensorRT/CUDA headers): embeds every
 * image given and prints the descriptor dimension and the pairwise cosine-similarity
 * matrix, then round-trips the same images through MegaLocPlaceCell (add_image /
 * descriptor / idempotence). Rough expectations on real sequence frames: self ~1.0,
 * consecutive frames ~0.9+, unrelated places ~0.35-0.4 (MegaLoc's common-mode floor).
 *
 * Usage:
 *   megaloc_embedder_smoke <model.onnx> <image> [<image> ...]
 */
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <placecell/megaloc_embedder.h>
#include <placecell/megaloc_placecell.h>

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "usage: megaloc_embedder_smoke <model.onnx> <image> [<image> ...]" << std::endl;
        return 1;
    }

    auto embedder = std::make_shared<placecell::MegaLocEmbedder>(argv[1]);
    std::cout << "engine " << (embedder->loaded_from_cache() ? "cached" : "built")
              << ": " << embedder->engine_path()
              << " (descriptor dim " << embedder->descriptor_dim() << ")" << std::endl;

    std::vector<Eigen::VectorXf> descriptors;
    std::vector<cv::Mat> images;
    for(int i = 2; i < argc; i++)
    {
        const cv::Mat image = cv::imread(argv[i]);
        if(image.empty())
        {
            std::cerr << "could not read image: " << argv[i] << std::endl;
            return 1;
        }
        images.push_back(image);
        descriptors.push_back(embedder->embed(image));
        std::cout << "embedded " << argv[i] << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    }

    std::cout << "pairwise cosine similarity:" << std::endl;
    for(size_t i = 0; i < descriptors.size(); i++)
    {
        for(size_t j = 0; j < descriptors.size(); j++)
            std::printf("  %.4f", placecell::MegaLocEmbedder::cosine(descriptors[i], descriptors[j]));
        std::printf("\n");
    }

    // MegaLocPlaceCell round-trip: store under ids 100, 101, ..., read back, check
    // idempotence (a second add_image of a known id must not re-embed or re-map).
    placecell::MegaLocPlaceCell store(embedder);
    for(size_t i = 0; i < images.size(); i++)
    {
        const placecell::PlaceCell::ExternalId id = 100 + i;
        const placecell::PlaceCell::InternalId internal = store.add_image(id, images[i]);
        const placecell::PlaceCell::InternalId again = store.add_image(id, images[i]);
        const Eigen::VectorXf* stored = store.descriptor(id);
        if(!stored || internal != i || again != internal || !store.has(id))
        {
            std::cerr << "store round-trip FAILED for id " << id << std::endl;
            return 1;
        }
        std::printf("store id %zu -> internal %zu, stored-vs-fresh cosine %.4f\n",
                    size_t(id), size_t(internal),
                    placecell::MegaLocEmbedder::cosine(*stored, descriptors[i]));
    }
    std::cout << "store round-trip OK (" << store.size() << " views, unknown id -> "
              << (store.descriptor(9999) == nullptr ? "nullptr" : "ERROR") << ")" << std::endl;

    // The kernel placecell grew on add_image must reproduce the pairwise cosines
    // (unit descriptors: dot == cosine) computed independently above.
    const Eigen::MatrixXf kernel = store.kernel();
    float max_diff = 0.0f;
    for(size_t i = 0; i < descriptors.size(); i++)
        for(size_t j = 0; j < descriptors.size(); j++)
            max_diff = std::max(max_diff, std::abs(kernel(int(i), int(j))
                                - placecell::MegaLocEmbedder::cosine(descriptors[i], descriptors[j])));
    std::cout << "kernel " << kernel.rows() << "x" << kernel.cols()
              << ", max |kernel - pairwise cosine| = " << max_diff
              << (max_diff < 1e-5f ? " (OK)" : " (ERROR)") << std::endl;
    if(max_diff >= 1e-5f)
        return 1;

    // Culling smoke (needs >= 3 views, the first two of the same place): gram-greedy
    // with a mid tau should cull exactly one of the near-duplicate pair -- the other
    // explains it -- and stop, since every remaining view is then poorly explained.
    if(images.size() >= 3)
    {
        placecell::PlaceCell::CullParameters cull_params;
        cull_params.max_unexplained = 0.5f;
        cull_params.centred = false;      // 3 views is too few for stable centring
        cull_params.min_keyframes = 1;
        cull_params.protect_last = 1;
        cull_params.protect_first = false;
        const auto cull_report = store.cull_keyframes(
            cull_params, [](placecell::PlaceCell::ExternalId) { return true; });
        std::cout << "cull smoke: culled " << cull_report.culled.size()
                  << " of " << cull_report.candidates << " candidates:";
        for(const auto& culled : cull_report.culled)
            std::printf(" [id %zu, unique info %.3f]", size_t(culled.id), double(culled.unique_information));
        const bool ok = cull_report.culled.size() == 1
                        && store.is_culled(cull_report.culled[0].id)
                        && (cull_report.culled[0].id == 100 || cull_report.culled[0].id == 101);
        std::cout << (ok ? " (OK)" : " (ERROR)") << std::endl;
        if(!ok)
            return 1;
    }
    return 0;
}
