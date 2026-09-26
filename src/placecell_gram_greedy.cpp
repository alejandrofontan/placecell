/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-26
 * License: Apache-2.0
 *
 * The gram-greedy culling method: gram_greedy_seed / gram_greedy_propose /
 * gram_greedy_downdate and their driver cull_gram_greedy, called by the shell in
 * placecell_cull.cpp through CullParameters::method = "gram-greedy".
 */
#include "placecell/placecell.h"

#include <algorithm>
#include <limits>

#include <Eigen/Dense>

namespace placecell
{

PlaceCell::GramGreedyState PlaceCell::gram_greedy_seed(const CullScope& scope)
{
    const Eigen::MatrixXf& similarity = scope.similarity;
    const std::vector<int>& alive = scope.alive;
    const int na = int(alive.size());
    GramGreedyState state;

    Eigen::MatrixXd K_AA(na, na);
    for(int a = 0; a < na; a++)
        for(int b = 0; b < na; b++)
            K_AA(a, b) = double(similarity(alive[a], alive[b])) + (a == b ? gram_greedy_jitter : 0.0);
    state.M = K_AA.ldlt().solve(Eigen::MatrixXd::Identity(na, na));

    state.W_rows.reserve(scope.history.size());
    state.v_h.reserve(scope.history.size());
    for(int h : scope.history){
        Eigen::VectorXd k(na);
        for(int a = 0; a < na; a++) k(a) = double(similarity(h, alive[a]));
        const Eigen::VectorXd w = state.M * k;
        state.W_rows.push_back(w);
        state.v_h.push_back(std::max(0.0, double(similarity(h, h)) - w.dot(k)));
    }
    state.removed.assign(std::size_t(na), 0);
    return state;
}

PlaceCell::GramGreedyProposal PlaceCell::gram_greedy_propose(const CullScope& scope, const GramGreedyState& state,
                                                             const std::vector<char>& candidate)
{
    // Feasible iff v_i = 1/M_ii <= tau and no history row is raised by more than
    // max(tau - v_h, slack). Among the feasible candidates the winner minimises
    // scope.objective: v_i itself ("unique"), the worst unexplained view after the cull
    // ("minimax"), or v_i plus the rise of every other view ("total-loss": the history
    // prices W_hi^2 / M_ii and, for each alive j, 1/(M_jj - M_ji^2/M_ii) - 1/M_jj, the
    // diagonal of the downdated M). Ties go to the smaller v_i, then to the lowest
    // kernel row (insertion order).
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
    // COUNT-DRIVEN (scope.tau = +inf): neither test can fail; a plain argmin of the objective.
    const double tau = scope.tau;
    const int na = int(scope.alive.size());
    const CullObjective objective = scope.objective;
    constexpr double inf = std::numeric_limits<double>::infinity();
    GramGreedyProposal best{-1, inf, 0.0, inf};
    for(int a = 0; a < na; a++){
        if(state.removed[a] || !candidate[a]) continue;
        const double M_aa = state.M(a, a);
        if(M_aa <= 0.0) continue;
        const double v_i = 1.0 / M_aa;
        if(v_i > tau) continue;
        if(objective == CullObjective::unique && v_i >= best.score) continue;   // cannot win: skip the history scan
        bool feasible = true;
        double worst = v_i;
        double loss = v_i;
        for(std::size_t h = 0; h < state.W_rows.size(); h++){
            const double w = state.W_rows[h](a);
            const double price = w * w / M_aa;
            if(price > std::max(tau - state.v_h[h], gram_greedy_over_budget_slack)){ feasible = false; break; }
            worst = std::max(worst, state.v_h[h] + price);
            loss += price;
        }
        if(!feasible) continue;
        double score = v_i;
        if(objective == CullObjective::minimax)
            score = worst;
        else if(objective == CullObjective::total_loss){
            for(int j = 0; j < na; j++){
                if(j == a || state.removed[j] || state.M(j, j) <= 0.0) continue;
                const double M_jj = state.M(j, j);
                const double M_ja = state.M(j, a);
                const double M_jj_after = M_jj - M_ja * M_ja / M_aa;
                if(M_jj_after <= 0.0){ loss = inf; break; }
                loss += 1.0 / M_jj_after - 1.0 / M_jj;
            }
            score = loss;
        }
        if(score < best.score || (score == best.score && v_i < best.unique_information))
            best = GramGreedyProposal{a, v_i, worst, score};
    }
    return best;
}

void PlaceCell::gram_greedy_downdate(const CullScope& scope, GramGreedyState& state,
                                     const GramGreedyProposal& proposal)
{
    // Rank-one downdate for the accepted cull of alive[i], i = proposal.index:
    //     v_h += W_hi^2 / M_ii ;  W' = W - W_:,i m^T / M_ii ;  M' = M - m m^T / M_ii
    // with m = M_:,i taken BEFORE the update; then row/column i of M are zeroed (the
    // view explains nothing any more) and the view joins the history with v_i = 1/M_ii
    // (Schur identity), its W row computed against the updated M.
    const int i = proposal.index;
    const double M_ii = state.M(i, i);
    const Eigen::VectorXd m = state.M.col(i);
    for(std::size_t h = 0; h < state.W_rows.size(); h++){
        const double w = state.W_rows[h](i);
        state.v_h[h] += w * w / M_ii;
        state.W_rows[h] -= (w / M_ii) * m;
    }
    state.M -= (m * m.transpose()) / M_ii;
    state.M.row(i).setZero();
    state.M.col(i).setZero();
    state.removed[i] = 1;

    const std::vector<int>& alive = scope.alive;
    const int na = int(alive.size());
    Eigen::VectorXd k(na);
    for(int a = 0; a < na; a++) k(a) = double(scope.similarity(alive[i], alive[a]));
    state.W_rows.push_back(state.M * k);       // M symmetric: W_h = k^T M
    state.v_h.push_back(proposal.unique_information);
}

void PlaceCell::cull_gram_greedy(const CullScope& scope, CullExecutor& execute, CullReport& report)
{
    const std::vector<ExternalId>& row_ids = scope.row_ids;
    const std::vector<int>& alive = scope.alive;
    std::vector<char> candidate = scope.candidate;   // a refused view leaves it for this call
    const int na = int(alive.size());
    const int max_per_call = scope.max_per_call;
    Profiler::Stopwatch stage;

    GramGreedyState state = gram_greedy_seed(scope);
    profiler_.record("cull_keyframes/inverse", stage.ms(), na);
    stage.restart();
    if(scope.history.empty() && scope.centred && !scope.local)
        PLACECELL_WARN_ONCE("cull_keyframes", "centring set == alive set (no history yet): K_AA is rank-deficient "
                            "and the unique-information scores of this call are jitter-scale (see issue #5)");

    int num_alive = na;
    int num_culled = 0;
    while(num_alive > scope.stop_at && (max_per_call <= 0 || num_culled < max_per_call)){
        const GramGreedyProposal proposal = gram_greedy_propose(scope, state, candidate);
        if(proposal.index < 0)
            break;

        // Hand the cull to the host (lock NOT held). A refusal (e.g. a deferred
        // erase) leaves the view alive and out of the running for this call.
        if(!execute(alive[proposal.index])){
            PLACECELL_DEBUG("cull_keyframes", "host refused to cull view " << row_ids[alive[proposal.index]]
                            << " (unique information " << proposal.unique_information << "); skipped for this call");
            candidate[proposal.index] = 0;
            continue;
        }
        num_culled++;
        num_alive--;
        report.culled.push_back(CullReport::CulledView{row_ids[alive[proposal.index]],
                                                       float(proposal.unique_information),
                                                       float(proposal.worst_after), num_alive});
        gram_greedy_downdate(scope, state, proposal);
    }

    double worst = 0.0;
    int over_budget = 0;
    for(double v : state.v_h){ worst = std::max(worst, v); over_budget += (v > scope.tau); }
    report.alive_after = num_alive;
    report.worst_history = float(worst);
    report.history_over_budget = over_budget;
    report.reached_max_per_call = max_per_call > 0 && num_culled >= max_per_call;

    // Unique information of what stays alive (the scores the next call would start from)
    report.alive_ids.reserve(std::size_t(num_alive));
    report.alive_unique_information.reserve(std::size_t(num_alive));
    for(int a = 0; a < na; a++){
        if(state.removed[a]) continue;
        report.alive_ids.push_back(row_ids[alive[a]]);
        report.alive_unique_information.push_back(
            state.M(a, a) > 0.0 ? float(1.0 / state.M(a, a)) : std::numeric_limits<float>::quiet_NaN());
    }

    profiler_.record("cull_keyframes/greedy", stage.ms() - execute.ms(), na, std::int64_t(state.W_rows.size()));
}


} // namespace placecell
