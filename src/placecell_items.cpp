/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Opus 5.5)
 * Created: 2026-09-30
 * License: Apache-2.0
 *
 * The item-backed mode: set_items (the sorted-unique set, the inverted index and the
 * shared-count matrix), the items / item_mode accessors, and materialise_kernel_locked,
 * which rebuilds the cosine kernel and its centring sums from the counts. Moved out of
 * placecell.cpp; the item query is in placecell_query.cpp.
 */
#include "placecell/placecell.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace placecell
{

PlaceCell::InternalId PlaceCell::set_items(const ExternalId id, std::vector<ItemId> items)
{
    return set_items(id, std::move(items), ItemOptions());
}

PlaceCell::InternalId PlaceCell::set_items(const ExternalId id, std::vector<ItemId> items, const ItemOptions& options)
{
    // Profile sizes: size_a = stored views before the call, size_b = items of the set.
    (void)options;   // cosine is the only normalisation; the parameter fixes the API
    Profiler::Scope timer(profiler_, "set_items");
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());

    InternalId internal = invalid_id;
    std::size_t added = 0, removed = 0, item_count = 0;
    bool new_view = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int n = int(external_ids_.size());
        timer.set_sizes(n, std::int64_t(items.size()));
        if(n > 0 && !item_mode_)
        {
            timer.cancel();
            PLACECELL_ERROR("set_items", "view " << id << " refused: the store is "
                            << (kernel_only_ ? "kernel-only (set_kernel)" : "descriptor-backed (add)")
                            << ", item sets cannot be mixed into it");
            return invalid_id;
        }

        const auto [it, inserted] = ids_.try_emplace(id, InternalId(n));
        internal = it->second;
        if(inserted)
        {
            // New row: zero shared counts with everybody until its items are indexed
            new_view = true;
            item_mode_ = true;
            external_ids_.push_back(id);
            culled_.push_back(0);
            protected_.push_back(0);
            items_.emplace_back();
            item_norms_.push_back(0);
            item_counts_.conservativeResize(n + 1, n + 1);
            item_counts_.row(n).setZero();
            item_counts_.col(n).setZero();
        }
        else if(culled_[internal])
        {
            // Frozen: the history rows the culler prices must not move under it
            timer.cancel();
            PLACECELL_WARN_ONCE("set_items", "view " << id << " is culled: its item set is frozen and the new set is ignored"
                                " (the host should not refresh culled views)");
            return internal;
        }

        // A refresh with the same set changes nothing; a NEW view must still go through the
        // rest, even with an empty set (its row is NaN and the kernel must be rebuilt)
        std::vector<ItemId>& current = items_[internal];
        if(!inserted && current == items)
        {
            timer.cancel();
            return internal;
        }

        // Diff old/new (both sorted, unique) and apply each change through the inverted
        // index: an item shared with view j moves counts_(i, j) and counts_(j, i) by one.
        const int i = int(internal);
        auto index_remove = [&](const ItemId item) {
            const auto posting = item_index_.find(item);
            for(const InternalId j : posting->second)
                if(int(j) != i)
                {
                    item_counts_(i, int(j))--;
                    item_counts_(int(j), i)--;
                }
            std::vector<InternalId>& views = posting->second;
            views.erase(std::find(views.begin(), views.end(), internal));
            if(views.empty())
                item_index_.erase(posting);
            item_counts_(i, i)--;
        };
        auto index_add = [&](const ItemId item) {
            std::vector<InternalId>& views = item_index_[item];
            for(const InternalId j : views)
            {
                item_counts_(i, int(j))++;
                item_counts_(int(j), i)++;
            }
            views.push_back(internal);
            item_counts_(i, i)++;
        };
        std::vector<ItemId> gone, came;
        std::set_difference(current.begin(), current.end(), items.begin(), items.end(), std::back_inserter(gone));
        std::set_difference(items.begin(), items.end(), current.begin(), current.end(), std::back_inserter(came));
        for(const ItemId item : gone)
            index_remove(item);
        for(const ItemId item : came)
            index_add(item);
        removed = gone.size();
        added = came.size();

        current = std::move(items);
        item_norms_[internal] = std::uint32_t(current.size());
        item_count = current.size();
        kernel_dirty_ = true;
    }
    on_set_items(id, internal, item_count, added, removed, new_view);
    return internal;
}

const std::vector<PlaceCell::ItemId>* PlaceCell::items(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!item_mode_)
        return nullptr;
    const auto it = ids_.find(id);
    return it == ids_.end() ? nullptr : &items_[it->second];
}

bool PlaceCell::item_mode() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return item_mode_;
}

void PlaceCell::materialise_kernel_locked() const
{
    // Item mode only: K_ij = counts_ij / sqrt(n_i n_j), NaN row/column for an empty set
    // (0/0), diagonal exactly 1 otherwise; then the centring sums add() would have kept.
    if(!item_mode_ || !kernel_dirty_)
        return;
    const int n = int(external_ids_.size());
    kernel_.resize(n, n);
    std::vector<double> inv_norm(static_cast<std::size_t>(n));
    for(int i = 0; i < n; i++)
        inv_norm[std::size_t(i)] = item_norms_[std::size_t(i)] > 0 ? 1.0 / std::sqrt(double(item_norms_[std::size_t(i)]))
                                                                  : std::numeric_limits<double>::quiet_NaN();
    for(int i = 0; i < n; i++)
    {
        kernel_(i, i) = item_norms_[std::size_t(i)] > 0 ? 1.0f : std::numeric_limits<float>::quiet_NaN();
        for(int j = 0; j < i; j++)
        {
            const float s = float(double(item_counts_(i, j)) * inv_norm[std::size_t(i)] * inv_norm[std::size_t(j)]);
            kernel_(i, j) = s;
            kernel_(j, i) = s;
        }
    }
    // Centring statistics over the usable (non-empty) views only: an empty view's row
    // sum is NaN and query_system_locked leaves it out of the means.
    kernel_row_sums_.assign(std::size_t(n), std::numeric_limits<double>::quiet_NaN());
    kernel_total_sum_ = 0.0;
    for(int i = 0; i < n; i++)
    {
        if(item_norms_[std::size_t(i)] == 0)
            continue;
        double row = 0.0;
        for(int j = 0; j < n; j++)
            if(item_norms_[std::size_t(j)] > 0)
                row += double(kernel_(i, j));
        kernel_row_sums_[std::size_t(i)] = row;
        kernel_total_sum_ += row;
    }
    kernel_dirty_ = false;
}

} // namespace placecell
