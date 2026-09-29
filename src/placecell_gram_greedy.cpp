/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-26
 * License: Apache-2.0
 *
 * The gram-greedy culling method (placecell_gram_greedy.h): seed / propose / downdate /
 * report and their driver cull, called by the shell in placecell_cull.cpp through
 * CullParameters::method = "gram-greedy".
 */
#include "placecell_gram_greedy.h"

#include <algorithm>
#include <limits>

#include <Eigen/Dense>

namespace placecell::gram_greedy
{

State seed(const CullScope& scope)
{
    const Eigen::MatrixXf& similarity = scope.similarity;
    const std::vector<int>& alive = scope.alive;
    const int na = int(alive.size());
    State state;

    Eigen::MatrixXd K_AA(na, na);
    for(int a = 0; a < na; a++)
        for(int b = 0; b < na; b++)
            K_AA(a, b) = double(similarity(alive[a], alive[b])) + (a == b ? jitter : 0.0);
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

Proposal propose(const CullScope& scope, const State& state, const std::vector<char>& candidate)
{
    const double tau = scope.tau;
    const int na = int(scope.alive.size());
    const CullObjective objective = scope.objective;
    constexpr double inf = std::numeric_limits<double>::infinity();
    Proposal best{-1, inf, 0.0, inf};
    for(int a = 0; a < na; a++){
        if(state.removed[a] || !candidate[a]) continue;
        const double M_aa = state.M(a, a);
        if(M_aa <= 0.0) continue;
        const double v_i = 1.0 / M_aa;
        if(v_i > tau) continue;
        if(objective == CullObjective::unique && v_i >= best.score) continue;
        bool feasible = true;
        double worst = v_i;
        double loss = v_i;
        for(std::size_t h = 0; h < state.W_rows.size(); h++){
            const double w = state.W_rows[h](a);
            const double price = w * w / M_aa;
            const double allowed = state.v_h[h] > tau ? over_budget_slack : tau - state.v_h[h];
            if(price > allowed){ feasible = false; break; }
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
            best = Proposal{a, v_i, worst, score};
    }
    return best;
}

void downdate(const CullScope& scope, State& state, const Proposal& proposal)
{
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
    state.W_rows.push_back(state.M * k);
    state.v_h.push_back(proposal.unique_information);
}

void report(const CullScope& scope, const State& state, PlaceCell::CullReport& report)
{
    const std::vector<PlaceCell::ExternalId>& row_ids = scope.row_ids;
    const std::vector<int>& alive = scope.alive;
    const int na = int(alive.size());
    const int max_per_call = scope.max_per_call;
    const int num_culled = int(report.culled.size());
    const int num_alive = na - num_culled;

    double worst = 0.0;
    int over_budget = 0;
    for(double v : state.v_h){ worst = std::max(worst, v); over_budget += (v > scope.tau); }
    report.alive_after = num_alive;
    report.worst_history = float(worst);
    report.history_over_budget = over_budget;
    report.reached_max_per_call = max_per_call > 0 && num_culled >= max_per_call;

    report.alive_ids.reserve(std::size_t(num_alive));
    report.alive_unique_information.reserve(std::size_t(num_alive));
    for(int a = 0; a < na; a++){
        if(state.removed[a]) continue;
        report.alive_ids.push_back(row_ids[alive[a]]);
        report.alive_unique_information.push_back(
            state.M(a, a) > 0.0 ? float(1.0 / state.M(a, a)) : std::numeric_limits<float>::quiet_NaN());
    }
}

void cull(const CullScope& scope, CullExecutor& execute, PlaceCell::CullReport& report, Profiler& profiler)
{
    const std::vector<PlaceCell::ExternalId>& row_ids = scope.row_ids;
    const std::vector<int>& alive = scope.alive;
    std::vector<char> candidate = scope.candidate;
    const int na = int(alive.size());
    const int max_per_call = scope.max_per_call;
    Profiler::Stopwatch stage;

    State state = seed(scope);
    profiler.record("cull_keyframes/inverse", stage.ms(), na);
    stage.restart();
    if(scope.history.empty() && scope.centred && !scope.local)
        PLACECELL_WARN_ONCE("cull_keyframes", "centring set == alive set (no history yet): K_AA is rank-deficient "
                            "and the unique-information scores of this call are jitter-scale (see issue #5)");

    int num_alive = na;
    int num_culled = 0;
    while(num_alive > scope.stop_at && (max_per_call <= 0 || num_culled < max_per_call)){
        const Proposal proposal = propose(scope, state, candidate);
        if(proposal.index < 0)
            break;

        if(!execute(alive[proposal.index])){
            PLACECELL_DEBUG("cull_keyframes", "host refused to cull view " << row_ids[alive[proposal.index]]
                            << " (unique information " << proposal.unique_information << "); skipped for this call");
            candidate[proposal.index] = 0;
            continue;
        }
        num_culled++;
        num_alive--;
        report.culled.push_back(PlaceCell::CullReport::CulledView{row_ids[alive[proposal.index]],
                                                       float(proposal.unique_information),
                                                       float(proposal.worst_after), num_alive});
        downdate(scope, state, proposal);
    }

    gram_greedy::report(scope, state, report);
    profiler.record("cull_keyframes/greedy", stage.ms() - execute.ms(), na, std::int64_t(state.W_rows.size()));
}


} // namespace placecell::gram_greedy
