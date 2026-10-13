/**
 * Module: placecell - megaloc_common.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-10-13
 * - License: Apache-2.0
 */
#include "megaloc_common.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace megaloc
{

Sidecar loadSidecar(const std::string& yamlPath)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(yamlPath);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("MegaLoc: cannot read sidecar " + yamlPath + ": " + e.what());
    }
    Sidecar s;
    try
    {
        for (int c = 0; c < 3; ++c)
        {
            s.mean[c] = root["mean"][c].as<float>();
            s.std[c] = root["std"][c].as<float>();
        }
        s.inputName = root["input_tensor"].as<std::string>();
        s.outputName = root["output_tensor"].as<std::string>();
        s.inputHeight = root["input_height"].as<int>();
        s.inputWidth = root["input_width"].as<int>();
        s.descriptorDim = root["descriptor_dim"].as<int>();
        if (root["precision"])
            s.precision = root["precision"].as<std::string>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("MegaLoc: sidecar " + yamlPath + " is missing a key: " + e.what());
    }
    if (s.inputWidth <= 0 || s.inputHeight <= 0 || s.descriptorDim <= 0)
        throw std::runtime_error("MegaLoc: sidecar " + yamlPath + " has a non-positive size");
    return s;
}

void preprocess(const cv::Mat& imageBGR, const Sidecar& sidecar, std::vector<float>& out)
{
    // 3-channel BGR -> area resize -> RGB -> /255 -> (x-mean)/std -> CHW planar
    cv::Mat bgr = imageBGR;
    if (imageBGR.channels() == 1)
        cv::cvtColor(imageBGR, bgr, cv::COLOR_GRAY2BGR);
    else if (imageBGR.channels() == 4)
        cv::cvtColor(imageBGR, bgr, cv::COLOR_BGRA2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(sidecar.inputWidth, sidecar.inputHeight), 0, 0, cv::INTER_AREA);

    const int w = sidecar.inputWidth, h = sidecar.inputHeight;
    const int hw = w * h;
    out.resize(static_cast<size_t>(3) * hw);
    float* rPlane = out.data();   // CHW: R plane first (BGR->RGB swap)
    float* gPlane = out.data() + hw;
    float* bPlane = out.data() + 2 * hw;
    const auto& mean = sidecar.mean;
    const auto& sd = sidecar.std;
    for (int y = 0; y < h; ++y)
    {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x)
        {
            const int i = y * w + x;
            bPlane[i] = (row[x][0] / 255.0f - mean[2]) / sd[2];
            gPlane[i] = (row[x][1] / 255.0f - mean[1]) / sd[1];
            rPlane[i] = (row[x][2] / 255.0f - mean[0]) / sd[0];
        }
    }
}

void normalizeInPlace(std::vector<float>& descriptor)
{
    double sq = 0.0;
    for (const float v : descriptor)
        sq += static_cast<double>(v) * v;
    const float inv = sq > 0.0 ? static_cast<float>(1.0 / std::sqrt(sq)) : 0.0f;
    for (float& v : descriptor)
        v *= inv;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size())
        return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0.0 ? static_cast<float>(dot / den) : 0.0f;
}

} // namespace megaloc
