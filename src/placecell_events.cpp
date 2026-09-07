/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 *
 * Event fan-out: the maths in placecell.cpp calls one of these per event; they feed the
 * Recorder (plot history) and the Logger (debug/trace narration). Timing itself stays
 * in placecell.cpp (Profiler::Scope), since it has to bracket the work.
 */
#include "placecell/placecell.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace placecell
{

namespace
{
std::string fixed3(const double x)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(3) << x;
    return s.str();
}
} // namespace

void PlaceCell::on_add(const ExternalId id, const InternalId internal, const bool size_mismatch) const
{
    if(size_mismatch)
        PLACECELL_WARN_ONCE("add", "view " << id << " has a descriptor size different from the store's: its "
                            "kernel row is NaN and unexplained_information will return NaN from now on");
    PLACECELL_DEBUG("add", "view " << id << " stored as row " << internal);
}

void PlaceCell::on_set_kernel(const KernelReport& report, const KernelOptions& options) const
{
    std::ostringstream line;
    line << report.views << " views from a host kernel (" << fixed3(report.ms) << "ms): max asymmetry "
         << fixed3(report.max_asymmetry) << (options.symmetrise ? " (symmetrised)" : "")
         << ", max |diag-1| " << fixed3(report.max_diagonal_deviation) << (options.unit_diagonal ? " (set to 1)" : "");
    if(!std::isnan(report.min_eigenvalue))
        line << ", eigenvalues [" << fixed3(report.min_eigenvalue) << ", " << fixed3(report.max_eigenvalue) << "]"
             << (report.negative_eigenvalues > 0 ? " with " + std::to_string(report.negative_eigenvalues) + " negative" : "")
             << (report.clipped ? " (clipped to PSD)" : "");
    else
        line << ", spectrum not checked";
    PLACECELL_INFO("set_kernel", line.str());

    if(report.max_asymmetry > options.asymmetry_warn)
        PLACECELL_WARN("set_kernel", "the supplied kernel is asymmetric (max |S_ij - S_ji| = " << fixed3(report.max_asymmetry)
                       << " > " << fixed3(options.asymmetry_warn) << ")"
                       << (options.symmetrise ? "; the average of S and S^T was stored" : "; stored as is"));
    if(report.negative_eigenvalues > 0 && !report.clipped)
        PLACECELL_WARN("set_kernel", report.negative_eigenvalues << " negative eigenvalues (smallest "
                       << fixed3(report.min_eigenvalue) << "): the kernel is not PSD, unique-information scores of "
                       "cull_keyframes may degrade (issue #3); KernelOptions::clip_to_psd projects it");
}

void PlaceCell::on_query(const Information& information, const int stored, const int window_size,
                         const bool centred, const double ms) const
{
    Recorder::Query query;
    query.unexplained = information.unexplained;
    query.explainers = information.explainers;
    query.best_explainer = information.best_explainer;
    query.best_similarity = information.best_similarity;
    query.centred = centred;
    query.stored = stored;
    query.window = window_size;
    recorder_.record_query(query);

    if(std::isnan(information.unexplained))
        PLACECELL_WARN_ONCE("unexplained_information", "the query cannot be compared with the store (descriptor-size "
                            "mismatch, or a kernel-only store without descriptors): NaN returned");
    PLACECELL_TRACE("unexplained_information", "v=" << fixed3(information.unexplained)
                    << " explainers=" << information.explainers << "/" << stored
                    << (window_size >= 0 ? " window=" + std::to_string(window_size) : std::string())
                    << " best=" << information.best_explainer << "(" << fixed3(information.best_similarity) << ")"
                    << (centred ? " centred" : " raw") << " " << fixed3(ms) << "ms");
}

void PlaceCell::on_cull_call(const CullParameters& parameters, const CullReport& report, const bool local,
                             const double ms) const
{
    Recorder::CullCall call;
    call.tau = parameters.max_unexplained;
    call.centred = parameters.centred;
    call.local = local;
    call.views_total = report.views_total;
    call.candidates = report.candidates;
    call.culled = int(report.culled.size());
    call.alive_after = report.alive_after;
    call.history_over_budget = report.history_over_budget;
    call.worst_history = report.worst_history;
    call.ms = ms;
    call.alive_ids = report.alive_ids;
    call.alive_unique_information = report.alive_unique_information;
    std::vector<Recorder::Cull> culls;
    culls.reserve(report.culled.size());
    for(const CullReport::CulledView& view : report.culled)
    {
        Recorder::Cull cull;
        cull.id = view.id;
        cull.unique_information = view.unique_information;
        cull.worst_unexplained_after = view.worst_unexplained_after;
        cull.alive_after = view.alive_after;
        cull.tau = parameters.max_unexplained;
        culls.push_back(cull);
    }
    recorder_.record_cull_call(std::move(call), culls);

    for(const CullReport::CulledView& view : report.culled)
        PLACECELL_DEBUG("cull_keyframes", "culled view " << view.id << ": unique information "
                        << fixed3(view.unique_information) << ", worst unexplained after "
                        << fixed3(view.worst_unexplained_after) << ", alive " << view.alive_after);
    PLACECELL_DEBUG("cull_keyframes", "[" << (local ? "local" : "map") << (parameters.centred ? ", centred" : ", raw")
                    << "] culled " << report.culled.size() << " of " << report.candidates << " candidates ("
                    << report.alive_after << " alive in scope, " << report.views_total << " views ever); worst history "
                    << fixed3(report.worst_history) << " (tau " << fixed3(parameters.max_unexplained) << ")"
                    << (report.reached_max_per_call ? " [per-call limit reached]" : "") << " " << fixed3(ms) << "ms");

    // Threshold lowered below earlier culls: say so once per change, not per call
    if(report.history_over_budget != last_history_over_budget_)
    {
        if(report.history_over_budget > 0)
            PLACECELL_INFO("cull_keyframes", report.history_over_budget << " culled views are above tau (worst "
                           << fixed3(report.worst_history) << " > tau " << fixed3(parameters.max_unexplained)
                           << "): only views that do not explain them can be culled");
        else if(last_history_over_budget_ > 0)
            PLACECELL_INFO("cull_keyframes", "history back within tau " << fixed3(parameters.max_unexplained));
        last_history_over_budget_ = report.history_over_budget;
    }
}

} // namespace placecell
