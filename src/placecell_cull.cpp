/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-26
 * License: Apache-2.0
 *
 * The culler's shell. PlaceCell::cull_keyframes is method-agnostic (validation,
 * profiling, the kernel snapshot, the scope and the candidates, the report); it hands
 * a CullScope and a CullExecutor to one culling method selected by
 * CullParameters::method. The methods live one per file (placecell_gram_greedy.cpp is
 * the only one); the snapshot, usable-row, centring and event helpers stay in
 * placecell.cpp and placecell_events.cpp.
 */
#include "placecell/placecell.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <Eigen/Dense>

namespace placecell
{

// ---- Method selection ---------------------------------------------------------------

PlaceCell::CullMethod PlaceCell::parse_cull_method(const std::string& name)
{
    if(name == "gram-greedy")
        return CullMethod::gram_greedy;
    PLACECELL_ERROR("cull_keyframes", "unknown method '" << name << "' (options: gram-greedy)");
    throw std::invalid_argument("placecell::PlaceCell::cull_keyframes: unknown method '"
                                + name + "' (options: gram-greedy)");
}

// ---- Shell ---------------------------------------------------------------------------

PlaceCell::CullReport PlaceCell::cull_keyframes(const CullParameters& parameters,
                                                const CullCallback& try_cull,
                                                const std::vector<ExternalId>* local_window)
{
    // What every method gets: the kernel snapshot (raw or centred), the usable rows in
    // scope split into alive and history, the candidates (alive and not protected), the
    // budget (tau, stop count, cap per call) and an executor for the culls. The method
    // decides the order and fills the report; the shell owns the invariants: the callback
    // runs without the lock, its time is excluded, every exit path reports the call.
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
    // SCOPE (local_window): without a window the marginalisation runs over every alive
    // view; with one, over the window only (the host's covisibility neighbourhood),
    // with candidates drawn from the window and the history reduced to the culled
    // views whose best alive explainer (over ALL alive views) lies in it — so far-away
    // history cannot veto a local cull, and far-away views cannot explain a local one.
    //
    // COUNT-DRIVEN (parameters.target_alive > 0): tau = +inf, so neither v_i nor the
    // history rows bound a cull; the loop stops when target_alive views are alive in
    // scope (never below min_keyframes). This is the offline "keep the N least redundant
    // views" selection; the report's alive_after says how many actually survived.
    const CullMethod method = parse_cull_method(parameters.method);

    // Profile: the whole call (host callback time excluded) plus its stages as sub-rows.
    // Sizes: size_a = alive views in scope, size_b = history rows in scope.
    Profiler::Scope timer(profiler_, "cull_keyframes");
    Profiler::Stopwatch stage;
    CullReport report{};
    const bool local = local_window != nullptr;
    // Every early return still reports the call (with the stage timing so far)
    struct ReportOnExit
    {
        const PlaceCell& cell; const CullParameters& parameters; const CullReport& report;
        bool local; Profiler::Scope& timer;
        ~ReportOnExit() { cell.on_cull_call(parameters, report, local, timer.elapsed_ms()); }
    } report_on_exit{*this, parameters, report, local, timer};

    // Snapshot under the lock; the method and the callback run WITHOUT it
    CullScope scope;
    std::vector<char> row_culled, row_protected;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        materialise_kernel_locked();
        scope.similarity = kernel_;
        scope.row_ids.assign(external_ids_.begin(), external_ids_.end());
        row_culled.assign(culled_.begin(), culled_.end());
        row_protected.assign(protected_.begin(), protected_.end());
    }
    const int n = int(scope.row_ids.size());
    report.views_total = n;
    if(n < 3)
        return report;

    // Usable views: those with a complete kernel row (a descriptor-size mismatch or an empty
    // item set leaves NaN)
    const std::vector<char> usable = usable_rows(scope.similarity);
    if(std::count(usable.begin(), usable.end(), char(1)) < n)
        PLACECELL_WARN_ONCE("cull_keyframes", (n - std::count(usable.begin(), usable.end(), char(1)))
                            << " of " << n << " views have a NaN kernel row (descriptor-size mismatch or empty item set) and are ignored");

    if(parameters.centred)
        centre_kernel(scope.similarity, usable);
    profiler_.record("cull_keyframes/snapshot+centring", stage.ms(), n);

    std::vector<int>& alive = scope.alive;
    std::vector<int>& history = scope.history;
    for(int i = 0; i < n; i++){
        if(!usable[i]) continue;
        if(row_culled[i]) history.push_back(i);
        else alive.push_back(i);
    }
    if(local_window){
        const Eigen::MatrixXf& similarity = scope.similarity;
        std::unordered_set<ExternalId> window(local_window->begin(), local_window->end());
        // history rows stay only if their best alive explainer (over the whole map) is local
        std::vector<int> local_history;
        for(int h : history){
            int best = -1; float best_similarity = -std::numeric_limits<float>::infinity();
            for(int a : alive)
                if(similarity(h, a) > best_similarity){ best_similarity = similarity(h, a); best = a; }
            if(best >= 0 && window.count(scope.row_ids[best]))
                local_history.push_back(h);
        }
        history.swap(local_history);
        std::vector<int> local_alive;
        for(int a : alive)
            if(window.count(scope.row_ids[a]))
                local_alive.push_back(a);
        alive.swap(local_alive);
    }
    const int na = int(alive.size());
    report.alive_after = na;
    timer.set_sizes(na, std::int64_t(history.size()));
    const bool count_driven = parameters.target_alive > 0;
    scope.stop_at = count_driven ? std::max(parameters.min_keyframes, parameters.target_alive)
                                 : parameters.min_keyframes;
    if(na <= scope.stop_at)
        return report;

    auto is_protected_row = [&](const int i) -> bool {
        if(parameters.protect_first && i == 0)
            return true;
        if(parameters.protect_last > 0 && i >= n - parameters.protect_last)
            return true;
        return row_protected[i] != 0;
    };
    scope.candidate.assign(std::size_t(na), 0);
    int num_candidates = 0;
    for(int a = 0; a < na; a++){
        scope.candidate[a] = !is_protected_row(alive[a]);
        num_candidates += scope.candidate[a];
    }
    report.candidates = num_candidates;
    if(num_candidates == 0)
        return report;

    // Count-driven: an infinite tau makes every candidate feasible (v_i <= tau and the
    // history price test both hold trivially); only the alive count stops the loop
    scope.tau = count_driven ? std::numeric_limits<double>::infinity() : double(parameters.max_unexplained);
    scope.max_per_call = parameters.max_per_call;
    scope.centred = parameters.centred;
    scope.local = local;

    CullExecutor execute(*this, try_cull, scope.row_ids);
    switch(method)
    {
        case CullMethod::gram_greedy:
            cull_gram_greedy(scope, execute, report);
            break;
    }

    if(execute.count() > 0 || execute.ms() > 0.0)
        profiler_.record("cull_keyframes/host_callback", execute.ms(), execute.count());
    timer.exclude(execute.ms());
    return report;
}

} // namespace placecell
