/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-02
 * License: Apache-2.0
 *
 * The store: construction and diagnostics (print_profile, dump), the descriptor mode
 * (add) and the kernel-only mode (set_kernel), clear, the culled / protected flags, and
 * the kernel views (kernel, snapshot, centred_kernel, usable_rows, centre_kernel). The
 * item mode is in placecell_items.cpp, the insertion query in placecell_query.cpp, the
 * culler in placecell_cull.cpp / placecell_gram_greedy.cpp, the event hooks in
 * placecell_events.cpp.
 */
#include "placecell/placecell.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

namespace placecell
{

PlaceCell::PlaceCell() : PlaceCell(Options{}) {}

PlaceCell::PlaceCell(const Options& options)
    : options_(options), profiler_(options.name)
{
    if(options_.verbosity)
        Logger::instance().set_level_unless_environment(*options_.verbosity);
    profiler_.set_enabled(options_.profile);
    recorder_.set_enabled(options_.record);
    // Fixed report order: main entry points, then cull_keyframes' stages as sub-rows
    profiler_.declare({"add", "set_kernel", "set_items", "unexplained_information", "unexplained_information_items",
                       "cull_keyframes",
                       "cull_keyframes/snapshot+centring", "cull_keyframes/inverse", "cull_keyframes/greedy",
                       "cull_keyframes/host_callback"});
}

PlaceCell::~PlaceCell()
{
    if(options_.report_on_destruction)
        print_profile();
}

void PlaceCell::print_profile() const
{
    const std::string report = profiler_.report();
    if(!report.empty())
        Logger::instance().print(report);
}

void PlaceCell::dump(const std::string& directory) const
{
    std::filesystem::create_directories(directory);
    const std::filesystem::path dir(directory);
    const Snapshot raw = snapshot(false);
    save_npy((dir / "kernel.npy").string(), raw.kernel);
    save_npy((dir / "kernel_centred.npy").string(), centred_kernel());
    std::vector<std::uint32_t> item_sizes;   // item mode: |P_i| per view, 0 in the other modes
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(item_mode_)
            item_sizes.assign(item_norms_.begin(), item_norms_.end());
    }
    {
        std::ofstream out(dir / "views.csv");
        out << "internal_id,external_id,culled,protected,items\n";
        for(std::size_t i = 0; i < raw.ids.size(); i++)
            out << i << "," << raw.ids[i] << "," << int(raw.culled[i]) << "," << int(raw.protected_views[i]) << ","
                << (i < item_sizes.size() ? item_sizes[i] : 0u) << "\n";
    }
    recorder_.dump_csv(directory);
    profiler_.dump_csv((dir / "profile.csv").string());
    PLACECELL_INFO("dump", raw.ids.size() << " views, " << recorder_.query_count() << " queries, "
                           << recorder_.cull_count() << " culls written to " << directory);
}

PlaceCell::InternalId PlaceCell::add(const ExternalId id, Eigen::VectorXf descriptor)
{
    Profiler::Scope timer(profiler_, "add");   // size_a = stored views before the add
    InternalId internal = invalid_id;
    bool size_mismatch = false;
    bool refused = false;   // kernel-only / item-backed store: no descriptors to take the new row's dots against
    bool item_backed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        refused = kernel_only_ || item_mode_;
        item_backed = item_mode_;
    }
    if(refused)
    {
        timer.cancel();
        PLACECELL_ERROR("add", "view " << id << " refused: the store is "
                        << (item_backed ? "item-backed (set_items)" : "kernel-only (set_kernel)")
                        << ", descriptors cannot be added to it");
        return invalid_id;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [it, inserted] = ids_.try_emplace(id, descriptors_.size());
        if(!inserted)
        {
            timer.cancel();
            return it->second;
        }

        // Grow the kernel by this view's row/column of dot products (double accumulation;
        // unit descriptors make the dot the cosine). NaN marks a descriptor-size mismatch.
        const int n = int(descriptors_.size());
        timer.set_sizes(n);
        if(n == 0)
            descriptor_size_ = descriptor.size();
        else if(descriptor.size() != descriptor_size_)
            size_mismatch_ = size_mismatch = true;

        kernel_.conservativeResize(n + 1, n + 1);
        double row_sum = 1.0;   // the diagonal entry
        for(int i = 0; i < n; i++)
        {
            float s = std::numeric_limits<float>::quiet_NaN();
            if(descriptors_[i].size() == descriptor.size())
                s = float(descriptor.cast<double>().dot(descriptors_[i].cast<double>()));
            kernel_(i, n) = s;
            kernel_(n, i) = s;
            kernel_row_sums_[i] += double(s);
            row_sum += double(s);
        }
        kernel_(n, n) = 1.0f;
        kernel_row_sums_.push_back(row_sum);
        kernel_total_sum_ += 2.0 * row_sum - 1.0;   // the new row and column share the diagonal

        descriptors_.push_back(std::move(descriptor));
        external_ids_.push_back(id);
        culled_.push_back(0);
        protected_.push_back(0);
        internal = it->second;
    }
    on_add(id, internal, size_mismatch);
    return internal;
}

bool PlaceCell::has(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.count(id) > 0;
}

const Eigen::VectorXf* PlaceCell::descriptor(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(kernel_only_ || item_mode_)
        return nullptr;
    const auto it = ids_.find(id);
    return it == ids_.end() ? nullptr : &descriptors_[it->second];
}

bool PlaceCell::kernel_only() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return kernel_only_;
}

PlaceCell::KernelReport PlaceCell::set_kernel(const Eigen::MatrixXf& similarity, const std::vector<ExternalId>& ids)
{
    return set_kernel(similarity, ids, KernelOptions());
}

PlaceCell::KernelReport PlaceCell::set_kernel(const Eigen::MatrixXf& similarity, const std::vector<ExternalId>& ids,
                                              const KernelOptions& options)
{
    // Profile sizes: size_a = views. The decomposition (psd_check / clip_to_psd) is
    // the cost here, O(n^3); everything else is O(n^2).
    Profiler::Scope timer(profiler_, "set_kernel");
    const Eigen::Index n = similarity.rows();
    if(n == 0 || similarity.cols() != n)
        throw std::invalid_argument("placecell::PlaceCell::set_kernel: similarity must be a non-empty square matrix (got "
                                    + std::to_string(similarity.rows()) + " x " + std::to_string(similarity.cols()) + ")");
    if(!ids.empty() && Eigen::Index(ids.size()) != n)
        throw std::invalid_argument("placecell::PlaceCell::set_kernel: " + std::to_string(ids.size()) + " ids for a "
                                    + std::to_string(n) + " x " + std::to_string(n) + " kernel");
    if(!ids.empty())
    {
        std::unordered_set<ExternalId> unique(ids.begin(), ids.end());
        if(Eigen::Index(unique.size()) != n)
            throw std::invalid_argument("placecell::PlaceCell::set_kernel: duplicate ids");
    }
    if(similarity.hasNaN())
        throw std::invalid_argument("placecell::PlaceCell::set_kernel: the similarity has NaN entries");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!external_ids_.empty())
            throw std::logic_error("placecell::PlaceCell::set_kernel: the store is not empty (" + std::to_string(external_ids_.size())
                                   + " views) - call clear() first");
    }
    timer.set_sizes(n);

    // Fix-ups in double: symmetrise, unit diagonal, (optional) PSD projection
    KernelReport report{};
    report.views = int(n);
    Eigen::MatrixXd S = similarity.cast<double>();
    report.max_asymmetry = float((S - S.transpose()).cwiseAbs().maxCoeff());
    report.max_diagonal_deviation = float((S.diagonal().array() - 1.0).abs().maxCoeff());
    if(options.symmetrise)
        S = (0.5 * (S + S.transpose())).eval();
    if(options.unit_diagonal)
        S.diagonal().setOnes();
    if(options.psd_check || options.clip_to_psd)
    {
        // SelfAdjointEigenSolver reads the lower triangle: exact for a symmetrised
        // kernel, the lower triangle's spectrum otherwise
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
            S, options.clip_to_psd ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
        const Eigen::VectorXd& values = eigen.eigenvalues();
        report.min_eigenvalue = float(values.minCoeff());
        report.max_eigenvalue = float(values.maxCoeff());
        report.negative_eigenvalues = int((values.array() < 0.0).count());
        if(options.clip_to_psd && report.negative_eigenvalues > 0)
        {
            const Eigen::VectorXd clipped = values.cwiseMax(0.0);
            S = eigen.eigenvectors() * clipped.asDiagonal() * eigen.eigenvectors().transpose();
            if(options.unit_diagonal)
            {
                // Correlation-style renormalisation keeps PSD (D^-1/2 S D^-1/2)
                const Eigen::VectorXd d = S.diagonal().cwiseMax(1e-12).cwiseSqrt().cwiseInverse();
                S = d.asDiagonal() * S * d.asDiagonal();
                S.diagonal().setOnes();
            }
            report.clipped = true;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!external_ids_.empty())   // a concurrent add() slipped in while we decomposed
            throw std::logic_error("placecell::PlaceCell::set_kernel: the store was filled concurrently - call clear() first");
        kernel_ = S.cast<float>();
        // Centring statistics as add() maintains them: per-row sums and the total sum
        const Eigen::VectorXd row_sums = S.rowwise().sum();
        kernel_row_sums_.assign(row_sums.data(), row_sums.data() + n);
        kernel_total_sum_ = S.sum();
        for(Eigen::Index i = 0; i < n; i++)
        {
            const ExternalId id = ids.empty() ? ExternalId(i) : ids[std::size_t(i)];
            external_ids_.push_back(id);
            ids_.emplace(id, InternalId(i));
            culled_.push_back(0);
            protected_.push_back(0);
        }
        descriptor_size_ = 0;
        size_mismatch_ = false;
        kernel_only_ = true;
    }
    report.ms = timer.elapsed_ms();
    on_set_kernel(report, options);
    return report;
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
    return external_ids_.size();   // == descriptors_.size() unless the store is kernel-only
}

Eigen::MatrixXf PlaceCell::kernel() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    materialise_kernel_locked();
    return kernel_;
}

std::vector<PlaceCell::ExternalId> PlaceCell::external_ids() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<ExternalId>(external_ids_.begin(), external_ids_.end());
}

void PlaceCell::centre_kernel(Eigen::MatrixXf& kernel, const std::vector<char>& usable)
{
    // Double-centre over the usable set, K_c = J S J with J = I - 11^T/m, then renormalise
    // to unit diagonal (in double): the correlation of the mean-centred descriptors.
    std::vector<int> u;
    for(int i = 0; i < int(usable.size()); i++)
        if(usable[i])
            u.push_back(i);
    const int m = int(u.size());
    if(m < 3)
        return;
    Eigen::MatrixXd S(m, m);
    for(int a = 0; a < m; a++)
        for(int b = 0; b < m; b++)
            S(a, b) = double(kernel(u[a], u[b]));
    const Eigen::VectorXd row_mean = S.rowwise().mean();
    const double total_mean = row_mean.mean();
    Eigen::MatrixXd C = S;
    C.colwise() -= row_mean;
    C.rowwise() -= row_mean.transpose();
    C.array() += total_mean;
    const Eigen::VectorXd d = C.diagonal().cwiseMax(1e-9).cwiseSqrt();
    for(int a = 0; a < m; a++)
        for(int b = 0; b < m; b++)
            kernel(u[a], u[b]) = float(C(a, b) / (d(a) * d(b)));
}

std::vector<char> PlaceCell::usable_rows(const Eigen::MatrixXf& kernel)
{
    // Culprit rows first (every off-diagonal entry NaN), then anything still touching a NaN
    const int n = int(kernel.rows());
    std::vector<char> usable(std::size_t(n), 1);
    for(int i = 0; i < n; i++)
    {
        int nans = 0;
        for(int j = 0; j < n; j++)
            nans += (j != i && std::isnan(kernel(i, j))) ? 1 : 0;
        if(n > 1 && nans == n - 1)
            usable[std::size_t(i)] = 0;
        else if(n == 1 && std::isnan(kernel(i, i)))
            usable[std::size_t(i)] = 0;
    }
    for(int i = 0; i < n; i++)
    {
        if(!usable[std::size_t(i)])
            continue;
        for(int j = 0; j < n; j++)
            if(usable[std::size_t(j)] && std::isnan(kernel(i, j))) { usable[std::size_t(i)] = 0; break; }
    }
    return usable;
}

PlaceCell::Snapshot PlaceCell::snapshot(const bool centred) const
{
    Snapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        materialise_kernel_locked();
        snapshot.kernel = kernel_;
        snapshot.ids.assign(external_ids_.begin(), external_ids_.end());
        snapshot.culled.assign(culled_.begin(), culled_.end());
        snapshot.protected_views.assign(protected_.begin(), protected_.end());
    }
    if(centred)
    {
        centre_kernel(snapshot.kernel, usable_rows(snapshot.kernel));
        snapshot.centred = true;
    }
    return snapshot;
}

Eigen::MatrixXf PlaceCell::centred_kernel() const
{
    return snapshot(true).kernel;
}

void PlaceCell::clear()
{
    std::size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dropped = external_ids_.size();
        descriptors_.clear();
        external_ids_.clear();
        ids_.clear();
        kernel_.resize(0, 0);
        kernel_row_sums_.clear();
        kernel_total_sum_ = 0.0;
        descriptor_size_ = 0;
        size_mismatch_ = false;
        kernel_only_ = false;
        item_mode_ = false;
        items_.clear();
        item_norms_.clear();
        item_counts_.resize(0, 0);
        item_index_.clear();
        kernel_dirty_ = false;
        culled_.clear();
        protected_.clear();
    }
    last_history_over_budget_ = -1;
    PLACECELL_INFO("clear", "store reset (" << dropped << " views dropped; profile and history kept)");
}

void PlaceCell::set_protected(const ExternalId id, const bool value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    if(it != ids_.end())
        protected_[it->second] = value ? 1 : 0;
}

bool PlaceCell::is_protected(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    return it != ids_.end() && protected_[it->second] != 0;
}

void PlaceCell::set_culled(const ExternalId id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    if(it != ids_.end())
        culled_[it->second] = 1;
}

bool PlaceCell::is_culled(const ExternalId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = ids_.find(id);
    return it != ids_.end() && culled_[it->second] != 0;
}

} // namespace placecell
