/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-02
 * License: Apache-2.0
 */
#include "placecell/placecell.h"

#include <limits>
#include <utility>

namespace placecell
{

PlaceCell::PlaceCell() = default;
PlaceCell::~PlaceCell() = default;

PlaceCell::InternalId PlaceCell::add(const ExternalId id, Eigen::VectorXf descriptor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [it, inserted] = ids_.try_emplace(id, descriptors_.size());
    if(!inserted)
        return it->second;

    // Grow the kernel by this view's row/column of dot products (double accumulation;
    // unit descriptors make the dot the cosine). NaN marks a descriptor-size mismatch.
    const int n = int(descriptors_.size());
    kernel_.conservativeResize(n + 1, n + 1);
    for(int i = 0; i < n; i++)
    {
        float s = std::numeric_limits<float>::quiet_NaN();
        if(descriptors_[i].size() == descriptor.size())
            s = float(descriptor.cast<double>().dot(descriptors_[i].cast<double>()));
        kernel_(i, n) = s;
        kernel_(n, i) = s;
    }
    kernel_(n, n) = 1.0f;

    descriptors_.push_back(std::move(descriptor));
    external_ids_.push_back(id);
    return it->second;
}

bool PlaceCell::has(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.count(id) > 0;
}

const Eigen::VectorXf* PlaceCell::descriptor(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    return it == ids_.end() ? nullptr : &descriptors_[it->second];
}

PlaceCell::InternalId PlaceCell::internal_id(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    return it == ids_.end() ? invalid_id : it->second;
}

std::size_t PlaceCell::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_.size();
}

Eigen::MatrixXf PlaceCell::kernel() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return kernel_;
}

std::vector<PlaceCell::ExternalId> PlaceCell::external_ids() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<ExternalId>(external_ids_.begin(), external_ids_.end());
}

void PlaceCell::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    descriptors_.clear();
    external_ids_.clear();
    ids_.clear();
    kernel_.resize(0, 0);
}

} // namespace placecell
