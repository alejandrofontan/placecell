/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 */
#include "placecell/log.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace placecell
{

namespace
{
constexpr const char* kReset = "\033[0m";
constexpr const char* kDim = "\033[2m";

const char* level_color(const LogLevel level)
{
    switch(level)
    {
        case LogLevel::error: return "\033[31m";     // red
        case LogLevel::warn:  return "\033[33m";     // yellow
        case LogLevel::info:  return "\033[32m";     // green
        case LogLevel::debug: return "\033[34m";     // blue
        case LogLevel::trace: return "\033[2m";      // dim
        case LogLevel::off:   break;
    }
    return "";
}

bool stderr_is_terminal()
{
#if defined(__unix__) || defined(__APPLE__)
    return isatty(fileno(stderr)) != 0;
#else
    return false;
#endif
}
} // namespace

Logger& Logger::instance()
{
    static Logger logger;
    return logger;
}

Logger::Logger()
    : colors_(stderr_is_terminal()), start_(std::chrono::steady_clock::now())
{
    if(const char* env = std::getenv("PLACECELL_VERBOSITY"))
    {
        if(const auto level = parse(env))
        {
            level_.store(*level, std::memory_order_relaxed);
            level_from_environment_ = true;
        }
        else
        {
            std::fprintf(stderr, "[placecell][WARN ] Logger: PLACECELL_VERBOSITY='%s' is not a level "
                                 "(off|error|warn|info|debug|trace or 0-5); keeping 'warn'\n", env);
        }
    }
}

void Logger::set_level(const LogLevel level)
{
    level_.store(level, std::memory_order_relaxed);
}

void Logger::set_level_unless_environment(const LogLevel level)
{
    if(!level_from_environment_)
        set_level(level);
}

void Logger::set_sink(Sink sink)
{
    std::lock_guard<std::mutex> lock(mutex_);
    sink_ = std::move(sink);
}

void Logger::set_colors(const bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    colors_ = enabled;
}

void Logger::set_show_elapsed(const bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    show_elapsed_ = enabled;
}

void Logger::log(const LogLevel level, const std::string_view component, const std::string_view message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(sink_)
        sink_(level, component, message);
    else
        default_sink(level, component, message);
}

void Logger::print(const std::string_view block)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(sink_)
    {
        sink_(LogLevel::info, "", block);
        return;
    }
    std::fwrite(block.data(), 1, block.size(), stderr);
    if(block.empty() || block.back() != '\n')
        std::fputc('\n', stderr);
    std::fflush(stderr);
}

void Logger::default_sink(const LogLevel level, const std::string_view component, const std::string_view message)
{
    // Build the whole line first and emit it with a single write so concurrent lines
    // from other libraries cannot interleave inside ours.
    std::ostringstream line;
    if(colors_) line << kDim;
    line << "[";
    if(colors_) line << kReset;
    line << "placecell";
    if(colors_) line << kDim;
    line << "]";
    if(colors_) line << kReset;

    if(colors_) line << level_color(level);
    line << "[" << std::left << std::setw(5) << name(level) << "]";
    if(colors_) line << kReset;

    if(show_elapsed_)
    {
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        if(colors_) line << kDim;
        line << "[" << std::right << std::fixed << std::setprecision(3) << std::setw(9) << seconds << "s]";
        if(colors_) line << kReset;
    }
    line << " ";
    if(!component.empty())
        line << component << ": ";
    line << message << "\n";

    const std::string text = line.str();
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fflush(stderr);
}

const char* Logger::name(const LogLevel level)
{
    switch(level)
    {
        case LogLevel::off:   return "OFF";
        case LogLevel::error: return "ERROR";
        case LogLevel::warn:  return "WARN";
        case LogLevel::info:  return "INFO";
        case LogLevel::debug: return "DEBUG";
        case LogLevel::trace: return "TRACE";
    }
    return "?";
}

std::optional<LogLevel> Logger::parse(const std::string_view text)
{
    std::string lower;
    lower.reserve(text.size());
    for(const char c : text)
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if(lower == "off" || lower == "0") return LogLevel::off;
    if(lower == "error" || lower == "1") return LogLevel::error;
    if(lower == "warn" || lower == "warning" || lower == "2") return LogLevel::warn;
    if(lower == "info" || lower == "3") return LogLevel::info;
    if(lower == "debug" || lower == "4") return LogLevel::debug;
    if(lower == "trace" || lower == "5") return LogLevel::trace;
    return std::nullopt;
}

} // namespace placecell
