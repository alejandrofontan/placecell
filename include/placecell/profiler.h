/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 *
 * Profiler: placecell's profile manager. One instance per PlaceCell (and per
 * MegaLocEmbedder), timing the MAIN entry points only — add, unexplained_information,
 * cull_keyframes (with its stages as sub-rows), embed — never helpers. Every sample
 * carries the problem size it was measured at (stored views, alive views, ...) so a
 * slow call can be explained by the size of its input, not just noticed.
 *
 * Sub-rows are named "parent/stage" (e.g. "cull_keyframes/greedy") and are printed
 * indented under their parent in report(). Time spent inside the host's callbacks is
 * excluded from the parent through Scope::exclude() and reported as its own sub-row,
 * so the numbers are placecell's own.
 *
 * Always compiled in; set_enabled(false) reduces each measurement to one atomic load.
 * report() returns the table (count / median / p95 / max in ms); PlaceCell prints it
 * on request and, with Options::report_on_destruction, when the store dies.
 *
 * Thread-safe.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace placecell
{

class Profiler
{
public:
    struct Sample
    {
        double ms{0.0};
        std::int64_t size_a{-1};   // meaning depends on the function (documented per call site)
        std::int64_t size_b{-1};
    };

    struct Stats
    {
        std::size_t count{0};
        double last_ms{0.0};
        double median_ms{0.0};
        double p95_ms{0.0};
        double max_ms{0.0};
        double total_ms{0.0};
    };

    explicit Profiler(std::string name = "placecell");

    const std::string& name() const { return name_; }
    void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    // Fix the report order (functions are otherwise listed in first-seen order)
    void declare(std::initializer_list<const char*> functions);

    void record(std::string_view function, double ms, std::int64_t size_a = -1, std::int64_t size_b = -1);

    Stats stats(std::string_view function) const;
    double last_ms(std::string_view function) const;
    std::vector<std::string> functions() const;
    std::vector<Sample> samples(std::string_view function) const;

    // Aligned table; empty string when nothing was recorded
    std::string report() const;
    // One row per sample: function,ms,size_a,size_b
    void dump_csv(const std::string& path) const;
    void clear();

    // Monotonic stopwatch in milliseconds
    class Stopwatch
    {
    public:
        Stopwatch() : start_(std::chrono::steady_clock::now()) {}
        void restart() { start_ = std::chrono::steady_clock::now(); }
        double ms() const
        {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        }

    private:
        std::chrono::steady_clock::time_point start_;
    };

    // RAII timer: records on destruction unless cancelled (or the profiler is disabled)
    class Scope
    {
    public:
        Scope(Profiler& profiler, std::string_view function);
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

        void set_sizes(std::int64_t size_a, std::int64_t size_b = -1);
        // Subtract time that is not placecell's (host callbacks)
        void exclude(double ms) { excluded_ms_ += ms; }
        double elapsed_ms() const { return watch_.ms() - excluded_ms_; }
        void cancel() { active_ = false; }
        bool active() const { return active_; }

    private:
        Profiler& profiler_;
        std::string function_;
        Stopwatch watch_;
        double excluded_ms_{0.0};
        std::int64_t size_a_{-1};
        std::int64_t size_b_{-1};
        bool active_;
    };

private:
    struct Entry
    {
        std::vector<Sample> samples;
    };

    mutable std::mutex mutex_;
    std::string name_;
    std::atomic<bool> enabled_{true};
    std::vector<std::string> order_;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace placecell
