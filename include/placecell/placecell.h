/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Created: 2026-09-02
 * License: Apache-2.0
 */
#pragma once

#include <Eigen/Core>

namespace placecell
{

class PlaceCell
{
public:
    PlaceCell();

private:
    // Similarity kernel over the inserted views (Gram matrix of their descriptors)
    Eigen::MatrixXf kernel_;
};

} // namespace placecell
