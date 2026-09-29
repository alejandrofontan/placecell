/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Opus 5.5)
 * Created: 2026-09-29
 * License: Apache-2.0
 *
 * Internal header (not installed): the gram-greedy culling method, selected by
 * CullParameters::method = "gram-greedy". cull() is the driver the shell calls; seed /
 * propose / downdate / report are its steps. None of them touches the store or its
 * lock; the executor is the only way a cull reaches the store. Description and
 * derivation: docs/reference/placecell_gram_greedy.md, paper/sec/03_methodology.tex.
 */
#pragma once

#include <vector>

#include <Eigen/Core>

#include "placecell/placecell.h"
#include "placecell/profiler.h"
#include "placecell_cull_method.h"

namespace placecell::gram_greedy
{

// Linear-algebra state of one call, indexed like scope.alive
struct State
{
    Eigen::MatrixXd M;                     // K_AA^-1 (jittered); zero row/column once removed
    std::vector<Eigen::VectorXd> W_rows;   // K_HA M, one row per history view (seeded, then one per cull)
    std::vector<double> v_h;               // unexplained information of each history row
    std::vector<char> removed;             // culled in this call
};

struct Proposal
{
    int index;                    // position in scope.alive, -1 = no feasible candidate
    double unique_information;    // v_i = 1/M_ii
    double worst_after;           // max unexplained view right after the cull
    double score;                 // the value of scope.objective that won (v_i, worst_after or the total loss)
};

inline constexpr double jitter = 1e-6;              // K_AA diagonal (same as the query's)
inline constexpr double over_budget_slack = 0.01;   // max deterioration of a history row already above tau

// The driver: seed, then propose / execute / downdate until no candidate is feasible,
// scope.stop_at views are alive or scope.max_per_call culls were accepted, then report.
// Fills report.culled and, through report(), the rest of the method's fields; records
// cull_keyframes/inverse and cull_keyframes/greedy in `profiler`.
void cull(const CullScope& scope, CullExecutor& execute, PlaceCell::CullReport& report, Profiler& profiler);

// The O(|A|^3) step: M = K_AA^-1, then W and v_h for the history in scope
State seed(const CullScope& scope);
// The greedy rule: the feasible candidate with the smallest scope.objective (index -1 when none)
Proposal propose(const CullScope& scope, const State& state, const std::vector<char>& candidate);
// After an accepted cull: rank-one downdate of M and W, v_h prices, the view joins the history
void downdate(const CullScope& scope, State& state, const Proposal& proposal);
// After the loop: alive_after, worst_history, history_over_budget, reached_max_per_call,
// alive_ids and alive_unique_information from the final state (counts from report.culled)
void report(const CullScope& scope, const State& state, PlaceCell::CullReport& report);

} // namespace placecell::gram_greedy
