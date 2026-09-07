/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 */
#include "placecell/recorder.h"

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace placecell
{

Recorder::Recorder() : start_(std::chrono::steady_clock::now()) {}

void Recorder::set_enabled(const bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool Recorder::enabled() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

double Recorder::now_s() const
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

void Recorder::record_decision(const ExternalId id, const bool inserted, const float unexplained, std::string reason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!enabled_)
        return;
    Decision decision;
    decision.query_index = queries_.size();
    decision.time_s = now_s();
    decision.id = id;
    decision.inserted = inserted;
    decision.unexplained = unexplained;
    decision.reason = std::move(reason);
    decisions_.push_back(std::move(decision));
}

void Recorder::set_thresholds(const float tau, const float min_information)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!thresholds_.empty() && thresholds_.back().tau == tau && thresholds_.back().min_information == min_information)
        return;
    thresholds_.push_back(Thresholds{now_s(), tau, min_information});
}

Recorder::Thresholds Recorder::thresholds() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return thresholds_.empty() ? Thresholds{} : thresholds_.back();
}

void Recorder::record_query(const Query& query)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!enabled_)
        return;
    Query stored = query;
    stored.index = queries_.size();
    stored.time_s = now_s();
    queries_.push_back(stored);
}

void Recorder::record_cull_call(CullCall call, const std::vector<Cull>& culls)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!enabled_)
        return;
    call.index = cull_calls_.size();
    call.time_s = now_s();
    for(Cull cull : culls)
    {
        cull.call_index = call.index;
        cull.time_s = call.time_s;
        culls_.push_back(cull);
    }
    // Only the latest call keeps its per-view vectors
    if(!cull_calls_.empty())
    {
        cull_calls_.back().alive_ids.clear();
        cull_calls_.back().alive_ids.shrink_to_fit();
        cull_calls_.back().alive_unique_information.clear();
        cull_calls_.back().alive_unique_information.shrink_to_fit();
    }
    cull_calls_.push_back(std::move(call));
}

std::size_t Recorder::query_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queries_.size();
}

std::size_t Recorder::cull_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return culls_.size();
}

std::size_t Recorder::cull_call_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return cull_calls_.size();
}

std::vector<Recorder::Query> Recorder::queries() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queries_;
}

std::vector<Recorder::Cull> Recorder::culls() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return culls_;
}

std::vector<Recorder::CullCall> Recorder::cull_calls() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return cull_calls_;
}

std::optional<Recorder::CullCall> Recorder::latest_cull_call() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(cull_calls_.empty())
        return std::nullopt;
    return cull_calls_.back();
}

std::vector<Recorder::Decision> Recorder::decisions() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return decisions_;
}

std::vector<Recorder::Thresholds> Recorder::threshold_history() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return thresholds_;
}

double Recorder::elapsed_s() const
{
    return now_s();
}

void Recorder::dump_csv(const std::string& directory) const
{
    std::filesystem::create_directories(directory);
    const std::filesystem::path dir(directory);
    std::lock_guard<std::mutex> lock(mutex_);

    {
        std::ofstream out(dir / "queries.csv");
        out << "index,time_s,unexplained,explainers,best_explainer,best_similarity,centred,stored,window\n";
        for(const Query& q : queries_)
            out << q.index << "," << std::setprecision(9) << q.time_s << "," << q.unexplained << "," << q.explainers
                << "," << q.best_explainer << "," << q.best_similarity << "," << int(q.centred) << "," << q.stored
                << "," << q.window << "\n";
    }
    {
        std::ofstream out(dir / "culls.csv");
        out << "call_index,time_s,id,unique_information,worst_unexplained_after,alive_after,tau\n";
        for(const Cull& c : culls_)
            out << c.call_index << "," << std::setprecision(9) << c.time_s << "," << c.id << "," << c.unique_information
                << "," << c.worst_unexplained_after << "," << c.alive_after << "," << c.tau << "\n";
    }
    {
        std::ofstream out(dir / "cull_calls.csv");
        out << "index,time_s,tau,centred,local,views_total,candidates,culled,alive_after,history_over_budget,"
               "worst_history,ms\n";
        for(const CullCall& c : cull_calls_)
            out << c.index << "," << std::setprecision(9) << c.time_s << "," << c.tau << "," << int(c.centred) << ","
                << int(c.local) << "," << c.views_total << "," << c.candidates << "," << c.culled << ","
                << c.alive_after << "," << c.history_over_budget << "," << c.worst_history << "," << c.ms << "\n";
    }
    {
        std::ofstream out(dir / "alive_information.csv");
        out << "id,unique_information\n";
        if(!cull_calls_.empty())
        {
            const CullCall& latest = cull_calls_.back();
            for(std::size_t i = 0; i < latest.alive_ids.size() && i < latest.alive_unique_information.size(); i++)
                out << latest.alive_ids[i] << "," << latest.alive_unique_information[i] << "\n";
        }
    }
    {
        std::ofstream out(dir / "decisions.csv");
        out << "query_index,time_s,id,inserted,unexplained,reason\n";
        for(const Decision& d : decisions_)
            out << d.query_index << "," << std::setprecision(9) << d.time_s << "," << d.id << "," << int(d.inserted)
                << "," << d.unexplained << "," << d.reason << "\n";
    }
    {
        std::ofstream out(dir / "thresholds.csv");
        out << "time_s,tau,min_information\n";
        for(const Thresholds& t : thresholds_)
            out << std::setprecision(9) << t.time_s << "," << t.tau << "," << t.min_information << "\n";
    }
}

void Recorder::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    queries_.clear();
    culls_.clear();
    cull_calls_.clear();
    decisions_.clear();
    // thresholds are configuration, not data: keep the latest so plots stay annotated
    if(thresholds_.size() > 1)
        thresholds_.erase(thresholds_.begin(), thresholds_.end() - 1);
}

} // namespace placecell
