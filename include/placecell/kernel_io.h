/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-07
 * License: Apache-2.0
 *
 * kernel_io: getting a similarity kernel in and out of files, and turning a host's
 * pairwise DISTANCE matrix into the similarity PlaceCell::set_kernel expects.
 *
 *   load_npy / save_npy         NumPy .npy <-> Eigen::MatrixXf (float32 on disk when
 *                               writing; float32 or float64, C or Fortran order,
 *                               format versions 1.0-3.0 when reading)
 *   similarity_from_distance    D -> S for the common distance conventions, so the
 *                               semantics of a matrix produced elsewhere (VPR-LAB's
 *                               faiss squared-L2 matrices, a cosine-distance matrix, ...)
 *                               are stated explicitly by the caller, never guessed here
 *
 * The conversions assume unit-norm descriptors, i.e. the cosine kernel every other
 * part of placecell is built for (self-similarity 1):
 *   similarity          S = D                (already a similarity; identity)
 *   squared_euclidean   S = 1 - D / 2        ||a-b||^2 = 2 - 2 a.b  (faiss pairwise_distances)
 *   cosine_distance     S = 1 - D            D = 1 - a.b
 *   euclidean           S = 1 - D^2 / 2      ||a-b||  = sqrt(2 - 2 a.b)
 * Nothing is symmetrised, clipped or re-diagonalised here: set_kernel does that, and
 * reports what it had to fix (PlaceCell::KernelReport).
 *
 * load_npy throws std::runtime_error (message names the file and the problem) for a
 * missing/unreadable file, a non-.npy file, an unsupported dtype (anything but little-
 * endian float32/float64), or a non-2-D array.
 */
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <Eigen/Core>

namespace placecell
{

// Minimal NumPy .npy writer (float32, C order) — kernel dumps
void save_npy(const std::string& path, const Eigen::MatrixXf& matrix);

// Minimal NumPy .npy reader: 2-D '<f4' or '<f8' arrays, C or Fortran order, format
// versions 1.0 / 2.0 / 3.0. Returns the matrix as float32 (float64 input is narrowed).
Eigen::MatrixXf load_npy(const std::string& path);

enum class DistanceKind
{
    similarity,           // the matrix is already a similarity (identity)
    squared_euclidean,    // ||a-b||^2 on unit vectors -> S = 1 - D/2   (faiss, VPR-LAB D.npy)
    cosine_distance,      // 1 - a.b                  -> S = 1 - D
    euclidean             // ||a-b|| on unit vectors   -> S = 1 - D^2/2
};

// Names as accepted on a command line / in a settings file: "similarity",
// "squared-euclidean" (also "squared_euclidean", "faiss"), "cosine-distance"
// (also "cosine_distance", "cosine"), "euclidean". std::nullopt for anything else.
std::optional<DistanceKind> parse_distance_kind(std::string_view name);
const char* distance_kind_name(DistanceKind kind);

// Elementwise conversion (see the table in the header comment). Shape is preserved.
Eigen::MatrixXf similarity_from_distance(const Eigen::MatrixXf& distance, DistanceKind kind);

} // namespace placecell
