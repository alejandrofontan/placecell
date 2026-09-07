/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-02
 * License: Apache-2.0
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
    profiler_.declare({"add", "set_kernel", "unexplained_information", "cull_keyframes",
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
    {
        std::ofstream out(dir / "views.csv");
        out << "internal_id,external_id,culled,protected\n";
        for(std::size_t i = 0; i < raw.ids.size(); i++)
            out << i << "," << raw.ids[i] << "," << int(raw.culled[i]) << "," << int(raw.protected_views[i]) << "\n";
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
    bool refused = false;   // kernel-only store: no descriptors to take the new row's dots against
    {
        std::lock_guard<std::mutex> lock(mutex_);
        refused = kernel_only_;
    }
    if(refused)
    {
        timer.cancel();
        PLACECELL_ERROR("add", "view " << id << " refused: the store is kernel-only (set_kernel), "
                        "descriptors cannot be added to it");
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
    if(kernel_only_)
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

PlaceCell::Snapshot PlaceCell::snapshot(const bool centred) const
{
    Snapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot.kernel = kernel_;
        snapshot.ids.assign(external_ids_.begin(), external_ids_.end());
        snapshot.culled.assign(culled_.begin(), culled_.end());
        snapshot.protected_views.assign(protected_.begin(), protected_.end());
    }
    if(centred)
    {
        const int n = int(snapshot.ids.size());
        std::vector<char> usable(n, 1);
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                if(std::isnan(snapshot.kernel(i, j))) { usable[i] = 0; break; }
        centre_kernel(snapshot.kernel, usable);
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
        dropped = descriptors_.size();
        descriptors_.clear();
        external_ids_.clear();
        ids_.clear();
        kernel_.resize(0, 0);
        kernel_row_sums_.clear();
        kernel_total_sum_ = 0.0;
        descriptor_size_ = 0;
        size_mismatch_ = false;
        kernel_only_ = false;
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
    std::vector<int> explainers;             // internal ids of the explainers
    Eigen::VectorXd k;                       // query dots with every stored view (u = internal id)
    Eigen::MatrixXd K_AA;                    // kernel over the explainers
    Eigen::VectorXd k_xA;                    // query vs explainers
    double k_xx = 0.0;
    std::vector<ExternalId> explainer_ids;
    bool use_centred = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int n = int(external_ids_.size());
        stored_out = n;
        if(n == 0)
            return information;              // nothing alive: unexplained = 1, explainers = 0
        if(kernel_only_ || size_mismatch_ || descriptor.size() != descriptor_size_)
        {
            information.unexplained = std::numeric_limits<float>::quiet_NaN();
            return information;
        }

        // Explainers: alive views, restricted to the window when given
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
        if(explainers.empty())
            return information;

        // Query dots with every stored view (needed for the explainers and, when
        // centring, for the query's row mean over the whole store)
        const Eigen::VectorXd x = descriptor.cast<double>();
        k_xx = x.squaredNorm();
        use_centred = centred && n >= 3;     // the culler centres only over >= 3 views
        k.resize(n);
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

        const int na = int(explainers.size());
        K_AA.resize(na, na);
        k_xA.resize(na);
        explainer_ids.resize(na);
        for(int a = 0; a < na; a++)
            explainer_ids[a] = external_ids_[explainers[a]];

        if(use_centred)
        {
            const double t = kernel_total_sum_ / (double(n) * double(n));
            const double r_x = k.sum() / double(n);
            const double C_xx = std::max(k_xx - 2.0 * r_x + t, 1e-9);
            auto r = [&](const int i) { return kernel_row_sums_[i] / double(n); };
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
            k_xx = 1.0;                      // the query's centred, normalised self-similarity
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
        }
    }

    // Marginalise outside the lock (same jitter as the culler)
    const int na = int(explainers.size());
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

PlaceCell::CullReport PlaceCell::cull_keyframes(const CullParameters& parameters,
                                                const CullCallback& try_cull,
                                                const std::vector<ExternalId>* local_window)
{
    // Greedy joint-information culling ("gram-greedy"). Let K be the (cosine, PSD)
    // similarity kernel over views, A the alive views and H the views culled earlier
    // (rows kept in the kernel). Under a Gaussian model the information of view x NOT
    // explained by the alive set is its conditional variance
    //     v_x = K_xx - k_xA K_AA^-1 k_Ax   (in [0,1]; exp(-2 I(x; A))).
    // For an alive view i that is v_i = 1 / (K_AA^-1)_ii, its unique information given
    // the other alive views; removing i raises every v_h by W_hi^2 / M_ii with
    // W = K_HA K_AA^-1 and M = K_AA^-1. Greedy rule: cull the alive view with the
    // smallest v_i among those for which, after the cull, every view ever inserted
    // (the culled ones AND i itself) stays at most tau = max_unexplained unexplained;
    // stop when none qualifies. M and W are updated by rank-one Schur downdates, so a
    // cull costs O(|H||A| + |A|^2).
    //
    // KERNEL (parameters.centred): image-embedding descriptors often share a large
    // common-mode component (unrelated places still score well above 0), which
    // compresses every v_i and makes tau over-sensitive. Double-centring the Gram
    // matrix over the usable views,
    //     K_c = J S J,  J = I - 11^T/n,   C_ij = K_c_ij / sqrt(K_c_ii K_c_jj)
    // is exactly the correlation of the mean-centred descriptors: unrelated pairs move
    // to ~0, near-duplicates stay high, the kernel stays PSD (rank n-1, hence the
    // diagonal jitter). The centring set grows with the map, so the kernel drifts
    // slightly as views arrive.
    //
    // ONLINE THRESHOLD CHANGES: culling is irreversible, so the history invariant
    // v_h <= tau only holds for the tau in force when h was culled. If tau is LOWERED
    // afterwards, rows with v_h > tau would make every candidate infeasible under a
    // plain "v_h + price <= tau" test and jam the culler. The constraint is therefore
    // relative: a cull may not raise any history row by more than max(tau - v_h,
    // slack), i.e. rows within budget behave as before and rows already over budget
    // only protect their actual explainers (slack absorbs the dense-W numerical dust
    // of unrelated candidates). RAISING tau would otherwise cull everything newly
    // feasible in one burst; max_per_call spreads that over successive calls.
    //
    // SCOPE (local_window): without a window the marginalisation runs over every alive
    // view; with one, over the window only (the host's covisibility neighbourhood),
    // with candidates drawn from the window and the history reduced to the culled
    // views whose best alive explainer (over ALL alive views) lies in it — so far-away
    // history cannot veto a local cull, and far-away views cannot explain a local one.
    if(parameters.method != "gram-greedy")
    {
        PLACECELL_ERROR("cull_keyframes", "unknown method '" << parameters.method << "' (options: gram-greedy)");
        throw std::invalid_argument("placecell::PlaceCell::cull_keyframes: unknown method '"
                                    + parameters.method + "' (options: gram-greedy)");
    }

    // Profile: the whole call (host callback time excluded) plus its stages as sub-rows.
    // Sizes: size_a = alive views in scope, size_b = history rows in scope.
    Profiler::Scope timer(profiler_, "cull_keyframes");
    Profiler::Stopwatch stage;
    double callback_ms = 0.0;
    CullReport report{};
    const bool local = local_window != nullptr;
    // Every early return still reports the call (with the stage timing so far)
    struct ReportOnExit
    {
        const PlaceCell& cell; const CullParameters& parameters; const CullReport& report;
        bool local; Profiler::Scope& timer;
        ~ReportOnExit() { cell.on_cull_call(parameters, report, local, timer.elapsed_ms()); }
    } report_on_exit{*this, parameters, report, local, timer};

    // Snapshot under the lock; the greedy loop and the callback run WITHOUT it
    Eigen::MatrixXf similarity;
    std::vector<ExternalId> row_ids;
    std::vector<char> row_culled, row_protected;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        similarity = kernel_;
        row_ids.assign(external_ids_.begin(), external_ids_.end());
        row_culled.assign(culled_.begin(), culled_.end());
        row_protected.assign(protected_.begin(), protected_.end());
    }
    const int n = int(row_ids.size());
    report.views_total = n;
    if(n < 3)
        return report;

    // Usable views: those with a complete kernel row (a descriptor-size mismatch leaves NaN)
    std::vector<int> alive, history;
    std::vector<char> usable(n, 1);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            if(std::isnan(similarity(i, j))) { usable[i] = 0; break; }
    if(std::count(usable.begin(), usable.end(), char(1)) < n)
        PLACECELL_WARN_ONCE("cull_keyframes", (n - std::count(usable.begin(), usable.end(), char(1)))
                            << " of " << n << " views have a NaN kernel row (descriptor-size mismatch) and are ignored");

    if(parameters.centred)
        centre_kernel(similarity, usable);
    profiler_.record("cull_keyframes/snapshot+centring", stage.ms(), n);
    stage.restart();

    for(int i = 0; i < n; i++){
        if(!usable[i]) continue;
        if(row_culled[i]) history.push_back(i);
        else alive.push_back(i);
    }
    if(local_window){
        std::unordered_set<ExternalId> window(local_window->begin(), local_window->end());
        // history rows stay only if their best alive explainer (over the whole map) is local
        std::vector<int> local_history;
        for(int h : history){
            int best = -1; float best_similarity = -std::numeric_limits<float>::infinity();
            for(int a : alive)
                if(similarity(h, a) > best_similarity){ best_similarity = similarity(h, a); best = a; }
            if(best >= 0 && window.count(row_ids[best]))
                local_history.push_back(h);
        }
        history.swap(local_history);
        std::vector<int> local_alive;
        for(int a : alive)
            if(window.count(row_ids[a]))
                local_alive.push_back(a);
        alive.swap(local_alive);
    }
    const int na = int(alive.size());
    report.alive_after = na;
    timer.set_sizes(na, std::int64_t(history.size()));
    if(na <= parameters.min_keyframes)
        return report;

    auto is_protected_row = [&](const int i) -> bool {
        if(parameters.protect_first && i == 0)
            return true;
        if(parameters.protect_last > 0 && i >= n - parameters.protect_last)
            return true;
        return row_protected[i] != 0;
    };
    std::vector<char> candidate(na, 0);
    int num_candidates = 0;
    for(int a = 0; a < na; a++){
        candidate[a] = !is_protected_row(alive[a]);
        num_candidates += candidate[a];
    }
    report.candidates = num_candidates;
    if(num_candidates == 0)
        return report;

    const double tau = double(parameters.max_unexplained);
    constexpr double jitter = 1e-6;
    constexpr double over_budget_slack = 0.01;   // max deterioration allowed for history rows already above tau
    const int max_per_call = parameters.max_per_call;

    // K_AA (double precision, jittered diagonal) and its inverse M; W = K_HA M and v_h for the history
    Eigen::MatrixXd K_AA(na, na);
    for(int a = 0; a < na; a++)
        for(int b = 0; b < na; b++)
            K_AA(a, b) = double(similarity(alive[a], alive[b])) + (a == b ? jitter : 0.0);
    Eigen::MatrixXd M = K_AA.ldlt().solve(Eigen::MatrixXd::Identity(na, na));
    profiler_.record("cull_keyframes/inverse", stage.ms(), na);
    stage.restart();
    if(history.empty() && parameters.centred && !local)
        PLACECELL_WARN_ONCE("cull_keyframes", "centring set == alive set (no history yet): K_AA is rank-deficient "
                            "and the unique-information scores of this call are jitter-scale (see issue #5)");

    // History rows are stored in a growable list; a culled view joins it during the loop
    std::vector<Eigen::VectorXd> W_rows;      // W_h over the alive columns (stale columns masked by `removed`)
    std::vector<double> v_h;                  // unexplained information of each history view
    auto add_history_row = [&](int idx, const Eigen::MatrixXd& M_now, double v) {
        Eigen::VectorXd k(na);
        for(int a = 0; a < na; a++) k(a) = double(similarity(idx, alive[a]));
        W_rows.push_back(M_now * k);           // M symmetric: W_h = k^T M
        v_h.push_back(v);
    };
    for(int h : history){
        Eigen::VectorXd k(na);
        for(int a = 0; a < na; a++) k(a) = double(similarity(h, alive[a]));
        const Eigen::VectorXd w = M * k;
        W_rows.push_back(w);
        v_h.push_back(std::max(0.0, double(similarity(h, h)) - w.dot(k)));
    }

    std::vector<char> removed(na, 0);
    int num_alive = na;
    int num_culled = 0;
    while(num_alive > parameters.min_keyframes && (max_per_call <= 0 || num_culled < max_per_call)){
        // Score every candidate: unique information v_i and the worst unexplained view after culling it.
        // Feasible iff v_i <= tau and no history row is raised by more than max(tau - v_h, slack).
        int best = -1;
        double best_v = std::numeric_limits<double>::infinity();
        double best_worst = 0.0;
        for(int a = 0; a < na; a++){
            if(removed[a] || !candidate[a]) continue;
            const double M_aa = M(a, a);
            if(M_aa <= 0.0) continue;
            const double v_i = 1.0 / M_aa;
            if(v_i > tau || v_i >= best_v) continue;
            bool feasible = true;
            double worst = v_i;
            for(size_t h = 0; h < W_rows.size(); h++){
                const double w = W_rows[h](a);
                const double price = w * w / M_aa;
                if(price > std::max(tau - v_h[h], over_budget_slack)){ feasible = false; break; }
                worst = std::max(worst, v_h[h] + price);
            }
            if(feasible){
                best = a; best_v = v_i; best_worst = worst;
            }
        }
        if(best < 0)
            break;

        // Hand the cull to the host (lock NOT held). A refusal (e.g. a deferred
        // erase) leaves the view alive and out of the running for this call.
        const Profiler::Stopwatch callback_watch;
        const bool culled_by_host = try_cull(row_ids[alive[best]]);
        callback_ms += callback_watch.ms();
        if(!culled_by_host){
            PLACECELL_DEBUG("cull_keyframes", "host refused to cull view " << row_ids[alive[best]]
                            << " (unique information " << best_v << "); skipped for this call");
            candidate[best] = 0;
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            culled_[alive[best]] = 1;
        }
        num_culled++;
        num_alive--;
        report.culled.push_back(CullReport::CulledView{row_ids[alive[best]], float(best_v),
                                                       float(best_worst), num_alive});

        // Rank-one downdate: M' = M - m m^T / M_ii ; W' = W - W_:,i m^T / M_ii ; v_h += W_hi^2 / M_ii
        const double M_ii = M(best, best);
        const Eigen::VectorXd m = M.col(best);
        for(size_t h = 0; h < W_rows.size(); h++){
            const double w = W_rows[h](best);
            v_h[h] += w * w / M_ii;
            W_rows[h] -= (w / M_ii) * m;
        }
        M -= (m * m.transpose()) / M_ii;
        M.row(best).setZero();
        M.col(best).setZero();
        removed[best] = 1;
        // The culled view joins the history with v_i = 1/M_ii (Schur identity)
        add_history_row(alive[best], M, best_v);
    }

    double worst = 0.0;
    int over_budget = 0;
    for(double v : v_h){ worst = std::max(worst, v); over_budget += (v > tau); }
    report.alive_after = num_alive;
    report.worst_history = float(worst);
    report.history_over_budget = over_budget;
    report.reached_max_per_call = max_per_call > 0 && num_culled >= max_per_call;

    // Unique information of what stays alive (the scores the next call would start from)
    report.alive_ids.reserve(std::size_t(num_alive));
    report.alive_unique_information.reserve(std::size_t(num_alive));
    for(int a = 0; a < na; a++){
        if(removed[a]) continue;
        report.alive_ids.push_back(row_ids[alive[a]]);
        report.alive_unique_information.push_back(
            M(a, a) > 0.0 ? float(1.0 / M(a, a)) : std::numeric_limits<float>::quiet_NaN());
    }

    profiler_.record("cull_keyframes/greedy", stage.ms() - callback_ms, na, std::int64_t(W_rows.size()));
    if(num_culled > 0 || callback_ms > 0.0)
        profiler_.record("cull_keyframes/host_callback", callback_ms, num_culled);
    timer.exclude(callback_ms);
    return report;
}

} // namespace placecell
