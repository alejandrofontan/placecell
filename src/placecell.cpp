/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-02
 * License: Apache-2.0
 */
#include "placecell/placecell.h"

#include <utility>

namespace placecell
{

PlaceCell::PlaceCell() = default;
PlaceCell::~PlaceCell() = default;

PlaceCell::InternalId PlaceCell::add(const ExternalId id, Eigen::VectorXf descriptor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [it, inserted] = ids_.try_emplace(id, descriptors_.size());
    if(inserted)
        descriptors_.push_back(std::move(descriptor));
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

} // namespace placecell
