/**
 * Module: placecell - megaloc_embedder_smoke.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-02
 * - License: Apache-2.0
 *
 * Smoke test of the PUBLIC MegaLocEmbedder API (includes only
 * <placecell/megaloc_embedder.h> -- no TensorRT/CUDA headers): embeds every image
 * given and prints the descriptor dimension and the pairwise cosine-similarity
 * matrix. Rough expectations on real sequence frames: self ~1.0, consecutive frames
 * ~0.9+, unrelated places ~0.35-0.4 (MegaLoc's common-mode floor).
 *
 * Usage:
 *   megaloc_embedder_smoke <model.onnx> <image> [<image> ...]
 */
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <placecell/megaloc_embedder.h>

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "usage: megaloc_embedder_smoke <model.onnx> <image> [<image> ...]" << std::endl;
        return 1;
    }

    placecell::MegaLocEmbedder embedder(argv[1]);
    std::cout << "engine " << (embedder.loaded_from_cache() ? "cached" : "built")
              << ": " << embedder.engine_path()
              << " (descriptor dim " << embedder.descriptor_dim() << ")" << std::endl;

    std::vector<Eigen::VectorXf> descriptors;
    for(int i = 2; i < argc; i++)
    {
        const cv::Mat image = cv::imread(argv[i]);
        if(image.empty())
        {
            std::cerr << "could not read image: " << argv[i] << std::endl;
            return 1;
        }
        descriptors.push_back(embedder.embed(image));
        std::cout << "embedded " << argv[i] << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    }

    std::cout << "pairwise cosine similarity:" << std::endl;
    for(size_t i = 0; i < descriptors.size(); i++)
    {
        for(size_t j = 0; j < descriptors.size(); j++)
            std::printf("  %.4f", placecell::MegaLocEmbedder::cosine(descriptors[i], descriptors[j]));
        std::printf("\n");
    }
    return 0;
}
