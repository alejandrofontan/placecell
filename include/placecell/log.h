/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 *
 * Logger: placecell's print manager. One process-wide instance with a runtime
 * verbosity, a replaceable sink and a fixed line format that makes every line
 * unmistakably placecell's next to a host system's own output:
 *
 *     [placecell][WARN ] cull_keyframes: 3 culled keyframes are above tau ...
 *
 * Levels (loguru-style cutoff: a line is emitted when its level <= the verbosity):
 *     off < error < warn < info < debug < trace
 *   error  something the caller asked for could not be done (never aborts the host)
 *   warn   a degraded path was taken (descriptor-size mismatch, singular kernel, ...)
 *   info   one-off facts: engine built/loaded, store cleared, over-budget history changed
 *   debug  one line per event: each cull, each cull_keyframes call, each add
 *   trace  per-query chatter (every unexplained_information value)
 *
 * Verbosity: compiled default `warn`; the environment variable PLACECELL_VERBOSITY
 * (a level name or 0-5) overrides it at startup and wins over PlaceCell::Options so a
 * run can be made verbose without touching the host's configuration; set_level()
 * always wins (explicit host call).
 *
 * Sink: the default writes one line per call to stderr, unbuffered (stdout is fully
 * buffered when a runner redirects it to a file, which reorders lines against a
 * host's stderr output). Colours are on only when stderr is a terminal unless
 * set_colors() says otherwise. set_sink() replaces the default (e.g. to forward into
 * the host's logging macros or a GUI); print() bypasses the level gate for reports
 * the caller explicitly asked for (profiler tables).
 *
 * Thread-safe: formatting happens in the caller, the sink call is serialised.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace placecell
{

enum class LogLevel : int
{
    off = 0,
    error = 1,
    warn = 2,
    info = 3,
    debug = 4,
    trace = 5
};

class Logger
{
public:
    using Sink = std::function<void(LogLevel level, std::string_view component, std::string_view message)>;

    static Logger& instance();

    // Verbosity cutoff (atomic; safe to change while other threads log)
    void set_level(LogLevel level);
    LogLevel level() const { return level_.load(std::memory_order_relaxed); }
    bool enabled(LogLevel level) const { return static_cast<int>(level) <= static_cast<int>(this->level()); }
    // True when PLACECELL_VERBOSITY was set at startup (Options::verbosity then defers to it)
    bool level_from_environment() const { return level_from_environment_; }
    // PlaceCell::Options uses this: applies `level` unless the environment set one
    void set_level_unless_environment(LogLevel level);

    // Replace the default stderr sink (an empty function restores the default)
    void set_sink(Sink sink);
    // Force ANSI colours on/off (default: on iff stderr is a terminal)
    void set_colors(bool enabled);
    // Prefix every default-sink line with the seconds elapsed since the logger was created
    void set_show_elapsed(bool enabled);

    // Emit a line (the level gate is the caller's job — the macros below check it first)
    void log(LogLevel level, std::string_view component, std::string_view message);
    // Emit a preformatted multi-line block regardless of the verbosity (reports)
    void print(std::string_view block);

    static const char* name(LogLevel level);
    static std::optional<LogLevel> parse(std::string_view text);

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger();
    void default_sink(LogLevel level, std::string_view component, std::string_view message);

    std::atomic<LogLevel> level_{LogLevel::warn};
    bool level_from_environment_{false};
    std::mutex mutex_;
    Sink sink_;
    bool colors_{false};
    bool show_elapsed_{false};
    std::chrono::steady_clock::time_point start_;
};

} // namespace placecell

// Stream-style macros: PLACECELL_INFO("cull_keyframes", "culled " << n << " views");
// The expression is only evaluated when the level is enabled.
#define PLACECELL_LOG(level, component, expr)                                                   \
    do                                                                                          \
    {                                                                                           \
        if(::placecell::Logger::instance().enabled(level))                                      \
        {                                                                                       \
            std::ostringstream _placecell_oss;                                                  \
            _placecell_oss << expr;                                                             \
            ::placecell::Logger::instance().log(level, component, _placecell_oss.str());        \
        }                                                                                       \
    } while(0)

#define PLACECELL_ERROR(component, expr) PLACECELL_LOG(::placecell::LogLevel::error, component, expr)
#define PLACECELL_WARN(component, expr)  PLACECELL_LOG(::placecell::LogLevel::warn, component, expr)
#define PLACECELL_INFO(component, expr)  PLACECELL_LOG(::placecell::LogLevel::info, component, expr)
#define PLACECELL_DEBUG(component, expr) PLACECELL_LOG(::placecell::LogLevel::debug, component, expr)
#define PLACECELL_TRACE(component, expr) PLACECELL_LOG(::placecell::LogLevel::trace, component, expr)

// Warn the first time this call site is reached, then stay silent (degraded paths that
// would otherwise repeat every frame)
#define PLACECELL_WARN_ONCE(component, expr)                                                    \
    do                                                                                          \
    {                                                                                           \
        static std::atomic<bool> _placecell_warned{false};                                      \
        if(!_placecell_warned.exchange(true))                                                   \
            PLACECELL_WARN(component, expr);                                                    \
    } while(0)
