/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Opus 5.5)
 * Created: 2026-09-29
 * License: Apache-2.0
 *
 * Internal header (not installed): what PlaceCell::cull_keyframes hands a culling
 * method. The shell (placecell_cull.cpp) parses the method and the objective, builds a
 * CullScope from its kernel snapshot and a CullExecutor around the host callback, and
 * calls one method (placecell_gram_greedy.h is the only one). A method sees neither the
 * store nor its lock: it reads the scope, proposes culls through the executor and fills
 * the report.
 */
#pragma once

#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "placecell/placecell.h"
#include "placecell/profiler.h"

namespace placecell
{

enum class CullMethod { gram_greedy };
// CullParameters::method -> CullMethod; throws std::invalid_argument on an unknown name
CullMethod parse_cull_method(const std::string& name);

enum class CullObjective { unique, minimax, total_loss };
// CullParameters::objective -> CullObjective; throws std::invalid_argument on an unknown name
CullObjective parse_cull_objective(const std::string& name);

// What the shell hands a method. Indices are kernel rows of `similarity` (raw or
// centred, taken under the lock); alive/history are the usable rows in scope;
// candidate[a] says whether alive[a] may be proposed (not protected).
struct CullScope
{
    Eigen::MatrixXf similarity;
    std::vector<PlaceCell::ExternalId> row_ids;
    std::vector<int> alive;
    std::vector<int> history;
    std::vector<char> candidate;
    double tau{0.0};          // +inf in count-driven mode
    int stop_at{0};           // stop once this many alive views remain in scope
    int max_per_call{0};      // 0 = unlimited
    bool centred{false};
    bool local{false};
    CullObjective objective{CullObjective::unique};
};

// Executes a cull for a method: proposes a kernel row to the host (the shell guarantees
// the store lock is NOT held, the host typically takes its own map mutex in the
// callback), on acceptance calls `mark_culled(row)`, which the shell supplies to set the
// row's culled flag under the store lock, and accumulates the callback time, which the
// shell subtracts from the call's own timing.
class CullExecutor
{
public:
    using MarkCulled = std::function<void(int row)>;

    CullExecutor(const PlaceCell::CullCallback& try_cull, const std::vector<PlaceCell::ExternalId>& row_ids,
                 MarkCulled mark_culled)
        : try_cull_(try_cull), row_ids_(row_ids), mark_culled_(std::move(mark_culled)) {}

    // The host's answer; true = the view is gone on the host side and is history here.
    bool operator()(const int row)
    {
        const Profiler::Stopwatch watch;
        const bool culled_by_host = try_cull_(row_ids_[row]);
        ms_ += watch.ms();
        if(culled_by_host)
        {
            mark_culled_(row);
            count_++;
        }
        return culled_by_host;
    }
    double ms() const { return ms_; }
    int count() const { return count_; }

private:
    const PlaceCell::CullCallback& try_cull_;
    const std::vector<PlaceCell::ExternalId>& row_ids_;
    MarkCulled mark_culled_;
    double ms_{0.0};
    int count_{0};
};

} // namespace placecell
