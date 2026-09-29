/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Opus 5.5)
 * Created: 2026-09-30
 * License: Apache-2.0
 *
 * The insertion query: unexplained_information for a descriptor or an item set (the
 * timed entry points and their compute_ halves), and the shared tail
 * explainers_locked / query_system_locked / solve_information. Moved out of
 * placecell.cpp; the store itself stays there, the item mode is in placecell_items.cpp.
 */
#include "placecell/placecell.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

namespace placecell
{

PlaceCell::Information PlaceCell::unexplained_information(const Eigen::VectorXf& descriptor,
                                                          const std::vector<ExternalId>* window,
                                                          const bool centred) const
{
    // Instrumented entry point; the maths is compute_unexplained_information below.
    // Profile sizes: size_a = stored views, size_b = explainers.
    Profiler::Scope timer(profiler_, "unexplained_information");
    int stored = 0;
    const Information information = compute_unexplained_information(descriptor, window, centred, stored);
    timer.set_sizes(stored, information.explainers);
    on_query(information, stored, window ? int(window->size()) : -1, centred, timer.elapsed_ms());
    return information;
}

PlaceCell::Information PlaceCell::compute_unexplained_information(const Eigen::VectorXf& descriptor,
                                                                  const std::vector<ExternalId>* window,
                                                                  const bool centred, int& stored_out) const
{
    // v_x = K_xx - k_xA K_AA^-1 k_Ax for a view x that is not stored, over the alive
    // explainers A (all alive views, or the alive views named by `window`). Same kernel
    // as cull_keyframes: raw dot products, or the double-centred correlation over U =
    // every stored view (alive AND history),
    //     C_ij = S_ij - r_i - r_j + t,   c_ij = C_ij / sqrt(C_ii C_jj),
    // with r the row means and t the total mean of the stored kernel — both kept
    // incrementally by add(), so this never sweeps the n x n kernel. The query is
    // centred out of sample against the same statistics (r_x = mean of its dots with
    // U; C_xx = k_xx - 2 r_x + t), i.e. as if it were an extra row that does not
    // contribute to the means.
    //
    // Locking: the snapshot (explainer rows, the query's dots) is taken under the lock —
    // the dots are the expensive part (one per stored view) but they read append-only
    // storage the culler also reads, and add() is rare compared with queries. The solve
    // runs outside the lock.
    Information information{};
    Eigen::MatrixXd K_AA;                    // kernel over the explainers
    Eigen::VectorXd k_xA;                    // query vs explainers
    double k_xx = 0.0;
    std::vector<ExternalId> explainer_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int n = int(external_ids_.size());
        stored_out = n;
        if(n == 0)
            return information;              // nothing alive: unexplained = 1, explainers = 0
        if(kernel_only_ || item_mode_ || size_mismatch_ || descriptor.size() != descriptor_size_)
        {
            information.unexplained = std::numeric_limits<float>::quiet_NaN();
            return information;
        }

        const std::vector<int> explainers = explainers_locked(window);
        if(explainers.empty())
            return information;

        // Query dots with every stored view (needed for the explainers and, when
        // centring, for the query's row mean over the whole store)
        const Eigen::VectorXd x = descriptor.cast<double>();
        const bool use_centred = centred && n >= 3;     // the culler centres only over >= 3 views
        Eigen::VectorXd k(n);
        if(use_centred)
        {
            for(int i = 0; i < n; i++)
                k(i) = x.dot(descriptors_[i].cast<double>());
        }
        else
        {
            k.setZero();
            for(const int i : explainers)
                k(i) = x.dot(descriptors_[i].cast<double>());
        }
        query_system_locked(explainers, k, x.squaredNorm(), use_centred, K_AA, k_xA, k_xx, explainer_ids);
    }

    // Marginalise outside the lock (same jitter as the culler)
    return solve_information(K_AA, k_xA, k_xx, explainer_ids);
}

std::vector<int> PlaceCell::explainers_locked(const std::vector<ExternalId>* window) const
{
    // Explainers: alive views, restricted to the window when given; sorted, unique
    std::vector<int> explainers;
    const int n = int(external_ids_.size());
    if(window)
    {
        explainers.reserve(window->size());
        for(const ExternalId id : *window)
        {
            const auto it = ids_.find(id);
            if(it != ids_.end() && culled_[it->second] == 0)
                explainers.push_back(int(it->second));
        }
        std::sort(explainers.begin(), explainers.end());
        explainers.erase(std::unique(explainers.begin(), explainers.end()), explainers.end());
    }
    else
    {
        for(int i = 0; i < n; i++)
            if(culled_[i] == 0)
                explainers.push_back(i);
    }
    return explainers;
}

void PlaceCell::query_system_locked(const std::vector<int>& explainers, const Eigen::VectorXd& k, const double k_xx,
                                    const bool use_centred, Eigen::MatrixXd& K_AA, Eigen::VectorXd& k_xA,
                                    double& k_xx_out, std::vector<ExternalId>& explainer_ids) const
{
    // Same kernel as cull_keyframes: raw, or the double-centred correlation over U = every
    // stored view (alive AND history),
    //     C_ij = S_ij - r_i - r_j + t,   c_ij = C_ij / sqrt(C_ii C_jj),
    // with r the row means and t the total mean of the stored kernel. The query is
    // centred out of sample against the same statistics (r_x = mean of its dots with U;
    // C_xx = k_xx - 2 r_x + t), i.e. as if it were an extra row that does not contribute
    // to the means. Reads kernel_ / the sums: item mode callers materialise them first.
    // U excludes the views whose row sum is NaN (empty item sets), so m <= n.
    const int n = int(external_ids_.size());
    const int na = int(explainers.size());
    K_AA.resize(na, na);
    k_xA.resize(na);
    explainer_ids.resize(na);
    for(int a = 0; a < na; a++)
        explainer_ids[a] = external_ids_[explainers[a]];

    if(use_centred)
    {
        int m = 0;
        double k_sum = 0.0;
        for(int i = 0; i < n; i++)
            if(!std::isnan(kernel_row_sums_[std::size_t(i)]))
            {
                m++;
                k_sum += k(i);
            }
        const double t = kernel_total_sum_ / (double(m) * double(m));
        const double r_x = k_sum / double(m);
        const double C_xx = std::max(k_xx - 2.0 * r_x + t, 1e-9);
        auto r = [&](const int i) { return kernel_row_sums_[i] / double(m); };
        auto C_diag = [&](const int i) { return std::max(double(kernel_(i, i)) - 2.0 * r(i) + t, 1e-9); };
        std::vector<double> d(na);
        for(int a = 0; a < na; a++)
            d[a] = std::sqrt(C_diag(explainers[a]));
        const double d_x = std::sqrt(C_xx);
        for(int a = 0; a < na; a++)
        {
            const int i = explainers[a];
            k_xA(a) = (k(i) - r_x - r(i) + t) / (d_x * d[a]);
            for(int b = 0; b < na; b++)
            {
                const int j = explainers[b];
                K_AA(a, b) = (double(kernel_(i, j)) - r(i) - r(j) + t) / (d[a] * d[b]);
            }
        }
        k_xx_out = 1.0;                      // the query's centred, normalised self-similarity
    }
    else
    {
        for(int a = 0; a < na; a++)
        {
            const int i = explainers[a];
            k_xA(a) = k(i);
            for(int b = 0; b < na; b++)
                K_AA(a, b) = double(kernel_(i, explainers[b]));
        }
        k_xx_out = k_xx;
    }
}

PlaceCell::Information PlaceCell::solve_information(Eigen::MatrixXd& K_AA, const Eigen::VectorXd& k_xA, const double k_xx,
                                                    const std::vector<ExternalId>& explainer_ids)
{
    // v_x = k_xx - k_xA K_AA^-1 k_Ax with the culler's diagonal jitter
    Information information{};
    const int na = int(explainer_ids.size());
    constexpr double jitter = 1e-6;
    for(int a = 0; a < na; a++)
        K_AA(a, a) += jitter;
    const Eigen::VectorXd w = K_AA.ldlt().solve(k_xA);
    const double v = k_xx - k_xA.dot(w);

    information.unexplained = float(std::min(std::max(v, 0.0), 1.0));
    information.explainers = na;
    int best = 0;
    for(int a = 1; a < na; a++)
        if(k_xA(a) > k_xA(best))
            best = a;
    information.best_explainer = explainer_ids[best];
    information.best_similarity = float(k_xA(best));
    return information;
}

// ---- Item query ----------------------------------------------------------------------

PlaceCell::Information PlaceCell::unexplained_information(const std::vector<ItemId>& items,
                                                          const std::vector<ExternalId>* window,
                                                          const bool centred) const
{
    // Instrumented entry point for the item query; the maths is the overload below.
    // Profile sizes: size_a = stored views, size_b = explainers.
    Profiler::Scope timer(profiler_, "unexplained_information_items");
    int stored = 0;
    const Information information = compute_unexplained_information(items, window, centred, stored);
    timer.set_sizes(stored, information.explainers);
    on_query(information, stored, window ? int(window->size()) : -1, centred, timer.elapsed_ms());
    return information;
}

PlaceCell::Information PlaceCell::compute_unexplained_information(const std::vector<ItemId>& items,
                                                                  const std::vector<ExternalId>* window,
                                                                  const bool centred, int& stored_out) const
{
    // The query's "descriptor" is its indicator vector over the items: its dot with a
    // stored view is their shared-item count, read off the inverted index, and its cosine
    // is that count over sqrt(|Q| |P_i|). k_xx = 1. Centring reuses the kernel's row/total
    // sums exactly as the descriptor query does.
    Information information{};
    Eigen::MatrixXd K_AA;
    Eigen::VectorXd k_xA;
    double k_xx = 0.0;
    std::vector<ExternalId> explainer_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int n = int(external_ids_.size());
        stored_out = n;
        if(n == 0)
            return information;              // nothing alive: unexplained = 1, explainers = 0
        if(!item_mode_)
        {
            information.unexplained = std::numeric_limits<float>::quiet_NaN();
            return information;
        }
        std::vector<ItemId> query(items);
        std::sort(query.begin(), query.end());
        query.erase(std::unique(query.begin(), query.end()), query.end());
        if(query.empty())
        {
            PLACECELL_WARN_ONCE("unexplained_information", "an item query with no items has no cosine (0/0): NaN returned");
            information.unexplained = std::numeric_limits<float>::quiet_NaN();
            return information;
        }

        std::vector<int> explainers = explainers_locked(window);
        explainers.erase(std::remove_if(explainers.begin(), explainers.end(),
                                        [&](const int i) { return item_norms_[std::size_t(i)] == 0; }),
                         explainers.end());   // empty views have no cosine: they explain nothing
        if(explainers.empty())
            return information;

        // Shared counts with every stored view through the inverted index
        Eigen::VectorXd k = Eigen::VectorXd::Zero(n);
        for(const ItemId item : query)
        {
            const auto posting = item_index_.find(item);
            if(posting == item_index_.end())
                continue;
            for(const InternalId j : posting->second)
                k(int(j)) += 1.0;
        }
        const double inv_norm_x = 1.0 / std::sqrt(double(query.size()));
        for(int i = 0; i < n; i++)
            k(i) = item_norms_[std::size_t(i)] > 0 ? k(i) * inv_norm_x / std::sqrt(double(item_norms_[std::size_t(i)]))
                                                   : std::numeric_limits<double>::quiet_NaN();

        // The explainer block (and, when centring, the row/total sums) come from the
        // materialised kernel; between refreshes this is a no-op.
        materialise_kernel_locked();
        const bool use_centred = centred && n >= 3;
        query_system_locked(explainers, k, 1.0, use_centred, K_AA, k_xA, k_xx, explainer_ids);
    }
    return solve_information(K_AA, k_xA, k_xx, explainer_ids);
}

} // namespace placecell
