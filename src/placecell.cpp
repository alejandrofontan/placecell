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
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

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
    culled_.push_back(0);
    protected_.push_back(0);
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
    culled_.clear();
    protected_.clear();
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
        throw std::invalid_argument("placecell::PlaceCell::cull_keyframes: unknown method '"
                                    + parameters.method + "' (options: gram-greedy)");

    CullReport report{};

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

    if(parameters.centred){
        // Double-centre over the usable set, then renormalise to unit diagonal (in double)
        std::vector<int> u;
        for(int i = 0; i < n; i++) if(usable[i]) u.push_back(i);
        const int m = int(u.size());
        if(m >= 3){
            Eigen::MatrixXd S(m, m);
            for(int a = 0; a < m; a++)
                for(int b = 0; b < m; b++)
                    S(a, b) = double(similarity(u[a], u[b]));
            const Eigen::VectorXd row_mean = S.rowwise().mean();
            const double total_mean = row_mean.mean();
            Eigen::MatrixXd C = S;
            C.colwise() -= row_mean;
            C.rowwise() -= row_mean.transpose();
            C.array() += total_mean;
            Eigen::VectorXd d = C.diagonal().cwiseMax(1e-9).cwiseSqrt();
            for(int a = 0; a < m; a++)
                for(int b = 0; b < m; b++)
                    similarity(u[a], u[b]) = float(C(a, b) / (d(a) * d(b)));
        }
    }

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
        if(!try_cull(row_ids[alive[best]])){
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
    return report;
}

} // namespace placecell
