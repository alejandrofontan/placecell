/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 */
#include "placecell/profiler.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace placecell
{

namespace
{
double percentile(std::vector<double> values, const double fraction)
{
    if(values.empty())
        return 0.0;
    const std::size_t k = static_cast<std::size_t>(fraction * double(values.size() - 1) + 0.5);
    std::nth_element(values.begin(), values.begin() + std::ptrdiff_t(k), values.end());
    return values[k];
}
} // namespace

Profiler::Profiler(std::string name) : name_(std::move(name)) {}

void Profiler::declare(const std::initializer_list<const char*> functions)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(const char* function : functions)
    {
        if(entries_.try_emplace(function).second)
            order_.emplace_back(function);
    }
}

void Profiler::record(const std::string_view function, const double ms,
                      const std::int64_t size_a, const std::int64_t size_b)
{
    if(!enabled())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key(function);
    const auto [it, inserted] = entries_.try_emplace(key);
    if(inserted)
        order_.push_back(key);
    it->second.samples.push_back(Sample{ms, size_a, size_b});
}

Profiler::Stats Profiler::stats(const std::string_view function) const
{
    Stats stats{};
    std::vector<double> ms;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(std::string(function));
        if(it == entries_.end() || it->second.samples.empty())
            return stats;
        ms.reserve(it->second.samples.size());
        for(const Sample& sample : it->second.samples)
            ms.push_back(sample.ms);
        stats.last_ms = it->second.samples.back().ms;
    }
    stats.count = ms.size();
    stats.max_ms = *std::max_element(ms.begin(), ms.end());
    for(const double value : ms)
        stats.total_ms += value;
    stats.p95_ms = percentile(ms, 0.95);
    stats.median_ms = percentile(std::move(ms), 0.5);
    return stats;
}

double Profiler::last_ms(const std::string_view function) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(std::string(function));
    if(it == entries_.end() || it->second.samples.empty())
        return 0.0;
    return it->second.samples.back().ms;
}

std::vector<std::string> Profiler::functions() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return order_;
}

std::vector<Profiler::Sample> Profiler::samples(const std::string_view function) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(std::string(function));
    return it == entries_.end() ? std::vector<Sample>{} : it->second.samples;
}

std::string Profiler::report() const
{
    const std::vector<std::string> functions = this->functions();
    bool any = false;
    std::ostringstream out;
    out << "[placecell][PROFILE " << name_ << "]\n";
    out << "    |  " << std::left << std::setw(34) << "function"
        << std::right << std::setw(8) << "count" << std::setw(10) << "median" << std::setw(10) << "p95"
        << std::setw(10) << "max" << "  (ms)\n";
    for(const std::string& function : functions)
    {
        const Stats s = stats(function);
        if(s.count == 0)
            continue;
        any = true;
        // "parent/stage" sub-rows print indented with the stage name only
        const std::size_t slash = function.find('/');
        const std::string label = slash == std::string::npos ? function : "   " + function.substr(slash + 1);
        out << "    |  " << std::left << std::setw(34) << label
            << std::right << std::setw(8) << s.count
            << std::fixed << std::setprecision(3)
            << std::setw(10) << s.median_ms << std::setw(10) << s.p95_ms << std::setw(10) << s.max_ms << "\n";
    }
    out << "    |___\n";
    return any ? out.str() : std::string();
}

void Profiler::dump_csv(const std::string& path) const
{
    std::ofstream out(path);
    if(!out)
        return;
    out << "function,ms,size_a,size_b\n";
    std::lock_guard<std::mutex> lock(mutex_);
    for(const std::string& function : order_)
    {
        const auto it = entries_.find(function);
        if(it == entries_.end())
            continue;
        for(const Sample& sample : it->second.samples)
            out << function << "," << std::setprecision(6) << sample.ms << "," << sample.size_a << ","
                << sample.size_b << "\n";
    }
}

void Profiler::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto& [function, entry] : entries_)
        entry.samples.clear();
}

Profiler::Scope::Scope(Profiler& profiler, const std::string_view function)
    : profiler_(profiler), function_(function), active_(profiler.enabled())
{
}

Profiler::Scope::~Scope()
{
    if(active_)
        profiler_.record(function_, elapsed_ms(), size_a_, size_b_);
}

void Profiler::Scope::set_sizes(const std::int64_t size_a, const std::int64_t size_b)
{
    size_a_ = size_a;
    size_b_ = size_b;
}

} // namespace placecell
