/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 *
 * Recorder: the dependency-free data side of placecell's visualization manager. One
 * instance per PlaceCell collects what the plots and post-mortems need — every
 * unexplained_information query, every cull_keyframes call and each cull inside it,
 * plus what only the host knows and tells us: its keyframe decisions and the
 * thresholds in force (record_decision / set_thresholds). Rendering lives in the
 * optional placecell::viz module (OpenCV); dump_csv() writes the same data for offline
 * plotting (tools/plot_placecell.py).
 *
 * Time stamps are seconds since the recorder was created (steady clock). Everything
 * is kept (a 20k-frame run is a few MB); clear() drops it all.
 *
 * Thread-safe.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace placecell
{

class Recorder
{
public:
    using ExternalId = std::uint64_t;

    struct Query
    {
        std::uint64_t index{0};          // 0-based query counter
        double time_s{0.0};
        float unexplained{1.0f};
        int explainers{0};
        ExternalId best_explainer{0};
        float best_similarity{0.0f};
        bool centred{true};
        int stored{0};                   // views in the store at query time
        int window{-1};                  // size of the host's window (-1: whole map)
    };

    struct Cull
    {
        std::uint64_t call_index{0};     // the cull_keyframes call this belongs to
        double time_s{0.0};
        ExternalId id{0};
        float unique_information{0.0f};
        float worst_unexplained_after{0.0f};
        int alive_after{0};
        float tau{0.0f};
    };

    struct CullCall
    {
        std::uint64_t index{0};
        double time_s{0.0};
        float tau{0.0f};
        bool centred{true};
        bool local{false};
        int views_total{0};
        int candidates{0};
        int culled{0};
        int alive_after{0};
        int history_over_budget{0};
        float worst_history{0.0f};
        double ms{0.0};
        // Unique information of every alive view in scope after the call (kept for the
        // latest call only — earlier calls carry empty vectors)
        std::vector<ExternalId> alive_ids;
        std::vector<float> alive_unique_information;
    };

    struct Decision
    {
        std::uint64_t query_index{0};    // the query count when the host decided (x position in plots)
        double time_s{0.0};
        ExternalId id{0};
        bool inserted{false};
        float unexplained{std::numeric_limits<float>::quiet_NaN()};
        std::string reason;
    };

    struct Thresholds
    {
        double time_s{0.0};
        float tau{std::numeric_limits<float>::quiet_NaN()};
        float min_information{std::numeric_limits<float>::quiet_NaN()};
    };

    Recorder();

    void set_enabled(bool enabled);
    bool enabled() const;

    // ---- Host-facing ------------------------------------------------------------------
    // The host's keyframe decision for a view (drawn as markers on the information plot)
    void record_decision(ExternalId id, bool inserted,
                         float unexplained = std::numeric_limits<float>::quiet_NaN(),
                         std::string reason = {});
    // Thresholds in force (drawn as horizontal lines); every change is kept as history
    void set_thresholds(float tau, float min_information);
    Thresholds thresholds() const;

    // ---- Library-facing (called by PlaceCell) ----------------------------------------
    void record_query(const Query& query);       // index/time_s filled in here
    void record_cull_call(CullCall call, const std::vector<Cull>& culls);   // index/time_s filled in here

    // ---- Access (snapshots) ------------------------------------------------------------
    std::size_t query_count() const;
    std::size_t cull_count() const;
    std::size_t cull_call_count() const;
    std::vector<Query> queries() const;
    std::vector<Cull> culls() const;
    std::vector<CullCall> cull_calls() const;
    std::optional<CullCall> latest_cull_call() const;
    std::vector<Decision> decisions() const;
    std::vector<Thresholds> threshold_history() const;
    double elapsed_s() const;

    // queries.csv, culls.csv, cull_calls.csv, decisions.csv, thresholds.csv into `directory`
    void dump_csv(const std::string& directory) const;
    void clear();

private:
    double now_s() const;

    mutable std::mutex mutex_;
    bool enabled_{true};
    std::chrono::steady_clock::time_point start_;
    std::vector<Query> queries_;
    std::vector<Cull> culls_;
    std::vector<CullCall> cull_calls_;
    std::vector<Decision> decisions_;
    std::vector<Thresholds> thresholds_;
};

} // namespace placecell

// save_npy (kernel dumps) lives in kernel_io.h; included here so existing callers of
// recorder.h keep seeing it
#include "placecell/kernel_io.h"
