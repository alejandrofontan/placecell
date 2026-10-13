/**
 * Module: placecell - test_megaloc.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 2.0
 * - Created: 2026-09-02
 * - Updated: 2026-10-13
 * - License: Apache-2.0
 *
 * Standalone harness for the MegaLoc embedder through its public API (whichever
 * backend the build selected): creates the embedder, embeds every image given, times
 * the steady state, prints the pairwise cosine-similarity matrix and - with
 * --reference - the cosine between each descriptor and the one of the same image in
 * a reference file: the PyTorch reference written by tools/export_megaloc.py
 * --test_images (README preprocessing; expect ~0.995, the preprocessing floor), or a
 * --dump of another run of this program (another provider / precision / machine;
 * expect >= 0.9999 between fp32 providers, >= 0.999 against fp16 ones).
 *
 * Usage:
 *   megaloc_test <model.onnx> <image> [<image> ...] [--precision fp32|fp16]
 *                [--provider auto|cuda|coreml|cpu] [--threads N] [--coreml-units U]
 *                [--reference <file>] [--dump <file>] [--iters N]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>

#include "placecell/megaloc_embedder.h"

namespace
{
double msSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/** '<path> v0 v1 ...' per line, '#' comments (export_megaloc.py and --dump write this). */
std::map<std::string, Eigen::VectorXf> readReference(const std::string& path)
{
    std::map<std::string, Eigen::VectorXf> ref;
    std::ifstream in(path);
    if (!in)
    {
        std::printf("cannot read reference file %s\n", path.c_str());
        return ref;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::string imagePath;
        iss >> imagePath;
        std::vector<float> d;
        float v;
        while (iss >> v)
            d.push_back(v);
        ref[imagePath] = Eigen::Map<const Eigen::VectorXf>(d.data(), Eigen::Index(d.size()));
    }
    return ref;
}

std::string baseName(const std::string& path)
{
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <model.onnx> <image> [<image> ...] [--precision fp32|fp16] "
                    "[--provider auto|cuda|coreml|cpu] [--threads N] [--coreml-units U] "
                    "[--reference <file>] [--dump <file>] [--iters N]\n",
                    argv[0]);
        return 1;
    }
    const std::string onnxPath = argv[1];
    placecell::MegaLocEmbedder::Settings settings;
    settings.precision = "fp32";
    std::string referencePath, dumpPath;
    int iters = 50;
    std::vector<std::string> images;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc)
            settings.precision = argv[++i];
        else if (std::strcmp(argv[i], "--provider") == 0 && i + 1 < argc)
            settings.provider = argv[++i];
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            settings.threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--coreml-units") == 0 && i + 1 < argc)
            settings.coreml_compute_units = argv[++i];
        else if (std::strcmp(argv[i], "--reference") == 0 && i + 1 < argc)
            referencePath = argv[++i];
        else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc)
            dumpPath = argv[++i];
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else
            images.emplace_back(argv[i]);
    }

    auto t0 = std::chrono::steady_clock::now();
    placecell::MegaLocEmbedder embedder(onnxPath, settings);
    const std::string label = std::string(placecell::MegaLocEmbedder::backend()) + "/" + embedder.provider() + " " +
                              settings.precision;
    std::printf("%s ready in %.0f ms (%s, %d-d): %s\n", label.c_str(), msSince(t0),
                embedder.loaded_from_cache() ? "loaded" : "built", embedder.descriptor_dim(),
                embedder.engine_path().c_str());

    std::vector<Eigen::VectorXf> descs;
    std::vector<std::string> names;
    for (const auto& path : images)
    {
        const cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            std::printf("cannot read image %s\n", path.c_str());
            return 1;
        }
        t0 = std::chrono::steady_clock::now();
        descs.push_back(embedder.embed(img));
        std::printf("%s: %dx%dx%d -> %ld-d in %.1f ms\n", baseName(path).c_str(), img.cols, img.rows,
                    img.channels(), static_cast<long>(descs.back().size()), msSince(t0));
        names.push_back(path);
    }

    if (iters > 0)
    {
        const cv::Mat img = cv::imread(images.front(), cv::IMREAD_UNCHANGED);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            (void)embedder.embed(img);
        std::printf("steady state: %.2f ms/image over %d iters (%s)\n", msSince(t0) / iters, iters, label.c_str());
    }

    if (descs.size() > 1)
    {
        std::printf("pairwise cosine similarity (%s):\n", label.c_str());
        for (size_t i = 0; i < descs.size(); ++i)
        {
            std::printf("  ");
            for (size_t j = 0; j < descs.size(); ++j)
                std::printf(" %6.3f", placecell::MegaLocEmbedder::cosine(descs[i], descs[j]));
            std::printf("   %s\n", baseName(names[i]).c_str());
        }
    }

    if (!dumpPath.empty())
    {
        std::ofstream out(dumpPath);
        out << "# " << label << " descriptors, dim " << embedder.descriptor_dim() << "\n";
        out << "# <image path> <v0> <v1> ... one image per line\n";
        for (size_t i = 0; i < descs.size(); ++i)
        {
            out << names[i];
            for (Eigen::Index k = 0; k < descs[i].size(); ++k)
                out << " " << std::scientific << descs[i][k];
            out << "\n";
        }
        std::printf("wrote %s\n", dumpPath.c_str());
    }

    if (!referencePath.empty())
    {
        const auto ref = readReference(referencePath);
        std::printf("cosine(%s, %s) per image:\n", label.c_str(), baseName(referencePath).c_str());
        float worst = 1.0f;
        for (size_t i = 0; i < descs.size(); ++i)
        {
            // match by basename so the reference can have been written from another cwd / machine
            const Eigen::VectorXf* r = nullptr;
            for (const auto& [p, d] : ref)
                if (baseName(p) == baseName(names[i]))
                    r = &d;
            if (r == nullptr)
            {
                std::printf("  %s: not in reference file\n", baseName(names[i]).c_str());
                continue;
            }
            const float c = placecell::MegaLocEmbedder::cosine(descs[i], *r);
            worst = std::min(worst, c);
            std::printf("  %s: %.5f\n", baseName(names[i]).c_str(), c);
        }
        std::printf("worst: %.5f\n", worst);
    }
    return 0;
}
