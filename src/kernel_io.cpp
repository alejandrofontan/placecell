/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-07
 * License: Apache-2.0
 */
#include "placecell/kernel_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace placecell
{

void save_npy(const std::string& path, const Eigen::MatrixXf& matrix)
{
    // NPY format 1.0: magic, version, little-endian uint16 header length, then a Python
    // dict literal padded with spaces so that the data starts on a 64-byte boundary.
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << matrix.rows() << ", " << matrix.cols() << "), }";
    std::string header = dict.str();
    const std::size_t preamble = 6 + 2 + 2;   // magic + version + header length
    std::size_t padding = 64 - ((preamble + header.size() + 1) % 64);
    if(padding == 64)
        padding = 0;
    header.append(padding, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    if(!out)
        return;
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.write(reinterpret_cast<const char*>(magic), 8);
    const std::uint16_t length = static_cast<std::uint16_t>(header.size());
    const unsigned char length_le[2] = {static_cast<unsigned char>(length & 0xFF),
                                        static_cast<unsigned char>((length >> 8) & 0xFF)};
    out.write(reinterpret_cast<const char*>(length_le), 2);
    out.write(header.data(), std::streamsize(header.size()));
    // C order: row by row (Eigen's default storage is column-major)
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major = matrix;
    out.write(reinterpret_cast<const char*>(row_major.data()),
              std::streamsize(sizeof(float) * std::size_t(row_major.size())));
}

namespace
{

[[noreturn]] void fail(const std::string& path, const std::string& what)
{
    throw std::runtime_error("placecell::load_npy: " + path + ": " + what);
}

// Value of a `'key': value` entry of the header dict, up to the next top-level ',' or '}'
// (the shape tuple's own commas are inside parentheses and skipped)
std::string dict_value(const std::string& header, const std::string& key, const std::string& path)
{
    const std::string quoted = "'" + key + "'";
    std::size_t pos = header.find(quoted);
    if(pos == std::string::npos)
        fail(path, "header has no '" + key + "' entry");
    pos = header.find(':', pos + quoted.size());
    if(pos == std::string::npos)
        fail(path, "malformed header near '" + key + "'");
    pos++;
    int depth = 0;
    std::size_t end = pos;
    for(; end < header.size(); end++)
    {
        const char c = header[end];
        if(c == '(' || c == '[') depth++;
        else if(c == ')' || c == ']') depth--;
        else if((c == ',' || c == '}') && depth == 0) break;
    }
    std::string value = header.substr(pos, end - pos);
    // trim spaces and quotes
    const auto first = value.find_first_not_of(" \t'");
    const auto last = value.find_last_not_of(" \t'");
    return first == std::string::npos ? std::string() : value.substr(first, last - first + 1);
}

} // namespace

Eigen::MatrixXf load_npy(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
        fail(path, "cannot open file");

    unsigned char preamble[8];
    in.read(reinterpret_cast<char*>(preamble), 8);
    if(!in || preamble[0] != 0x93 || std::memcmp(preamble + 1, "NUMPY", 5) != 0)
        fail(path, "not a .npy file (bad magic)");
    const int major = preamble[6];
    std::uint32_t header_length = 0;
    if(major == 1)
    {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        header_length = std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8);
    }
    else if(major == 2 || major == 3)
    {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        header_length = std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) | (std::uint32_t(b[2]) << 16)
                        | (std::uint32_t(b[3]) << 24);
    }
    else
        fail(path, "unsupported .npy format version " + std::to_string(major));
    if(!in)
        fail(path, "truncated header");

    std::string header(header_length, '\0');
    in.read(header.data(), std::streamsize(header_length));
    if(!in)
        fail(path, "truncated header");

    const std::string descr = dict_value(header, "descr", path);
    const std::string fortran = dict_value(header, "fortran_order", path);
    const std::string shape = dict_value(header, "shape", path);

    std::size_t item_size = 0;
    if(descr == "<f4" || descr == "=f4" || descr == "f4")
        item_size = 4;
    else if(descr == "<f8" || descr == "=f8" || descr == "f8")
        item_size = 8;
    else
        fail(path, "unsupported dtype '" + descr + "' (need little-endian float32 or float64)");
    const bool fortran_order = fortran.find("True") != std::string::npos;

    // shape: "(1180, 1180)" -> two dimensions required
    std::vector<std::size_t> dims;
    {
        std::string digits;
        for(const char c : shape)
        {
            if(c >= '0' && c <= '9')
                digits.push_back(c);
            else if(!digits.empty())
            {
                dims.push_back(std::size_t(std::stoull(digits)));
                digits.clear();
            }
        }
        if(!digits.empty())
            dims.push_back(std::size_t(std::stoull(digits)));
    }
    if(dims.size() != 2)
        fail(path, "expected a 2-D array, got shape " + shape);
    const Eigen::Index rows = Eigen::Index(dims[0]);
    const Eigen::Index cols = Eigen::Index(dims[1]);

    const std::size_t count = dims[0] * dims[1];
    std::vector<char> raw(count * item_size);
    in.read(raw.data(), std::streamsize(raw.size()));
    if(!in)
        fail(path, "truncated data (expected " + std::to_string(raw.size()) + " bytes)");

    Eigen::MatrixXf matrix(rows, cols);
    // Both orders are handled by mapping the buffer with the matching storage order
    if(item_size == 4)
    {
        const float* data = reinterpret_cast<const float*>(raw.data());
        if(fortran_order)
            matrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(data, rows, cols);
        else
            matrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, rows, cols);
    }
    else
    {
        const double* data = reinterpret_cast<const double*>(raw.data());
        if(fortran_order)
            matrix = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(data, rows, cols).cast<float>();
        else
            matrix = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, rows, cols).cast<float>();
    }
    return matrix;
}

std::optional<DistanceKind> parse_distance_kind(const std::string_view name)
{
    if(name == "similarity") return DistanceKind::similarity;
    if(name == "squared-euclidean" || name == "squared_euclidean" || name == "faiss") return DistanceKind::squared_euclidean;
    if(name == "cosine-distance" || name == "cosine_distance" || name == "cosine") return DistanceKind::cosine_distance;
    if(name == "euclidean") return DistanceKind::euclidean;
    return std::nullopt;
}

const char* distance_kind_name(const DistanceKind kind)
{
    switch(kind)
    {
        case DistanceKind::similarity: return "similarity";
        case DistanceKind::squared_euclidean: return "squared-euclidean";
        case DistanceKind::cosine_distance: return "cosine-distance";
        case DistanceKind::euclidean: return "euclidean";
    }
    return "?";
}

Eigen::MatrixXf similarity_from_distance(const Eigen::MatrixXf& distance, const DistanceKind kind)
{
    switch(kind)
    {
        case DistanceKind::similarity: return distance;
        case DistanceKind::squared_euclidean: return (1.0f - 0.5f * distance.array()).matrix();
        case DistanceKind::cosine_distance: return (1.0f - distance.array()).matrix();
        case DistanceKind::euclidean: return (1.0f - 0.5f * distance.array().square()).matrix();
    }
    return distance;
}

} // namespace placecell
