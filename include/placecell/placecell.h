/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-02
 * License: Apache-2.0
 *
 * PlaceCell: the core store of global descriptors, keyed by the host system's own ids
 * (e.g. a SLAM keyframe's frame id), plus the similarity kernel over them. Each
 * external id is mapped to a contiguous internal id — the row index of the kernel,
 * which is grown incrementally on every add() (one row of dot products, double
 * accumulation; unit descriptors make that the cosine). Entries are immutable and
 * append-only — a culled view keeps its row and acts as history, which is exactly
 * what information-based culling needs.
 *
 * This core is Eigen-only by design: descriptors come in already computed. Image
 * frontends (MegaLocPlaceCell) live in optional modules.
 *
 * Contracts:
 * - add() is idempotent: an id that already exists keeps its stored descriptor and
 *   returns its existing internal id.
 * - descriptor() returns a stable pointer (entries are never relocated or mutated),
 *   nullptr for an unknown id.
 * - kernel()/external_ids() return snapshots (consistent with each other only when
 *   taken together by the caller in the absence of concurrent adds; row counts can
 *   only grow, so a kernel snapshot is always a leading principal submatrix of any
 *   later one).
 * - A kernel entry is NaN when the two descriptors' sizes mismatch.
 * - clear() drops everything (store and kernel) — for a host system reset.
 * - Thread-safe: all methods serialise internally.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace placecell
{

class PlaceCell
{
public:
    using ExternalId = std::uint64_t;
    using InternalId = std::size_t;
    static constexpr InternalId invalid_id = static_cast<InternalId>(-1);

    PlaceCell();
    virtual ~PlaceCell();
    PlaceCell(const PlaceCell&) = delete;
    PlaceCell& operator=(const PlaceCell&) = delete;

    // Store a descriptor under the host's id and grow the kernel by its row (O(n)
    // dot products); returns its internal id. Idempotent: a known id returns its
    // existing internal id and keeps the stored descriptor and kernel row.
    InternalId add(ExternalId id, Eigen::VectorXf descriptor);

    bool has(ExternalId id) const;

    // Stable pointer to the stored descriptor (append-only storage: never relocated,
    // never mutated); nullptr for an unknown id.
    const Eigen::VectorXf* descriptor(ExternalId id) const;

    // Internal (contiguous) id of a stored view; invalid_id for an unknown id.
    InternalId internal_id(ExternalId id) const;

    // Number of stored views.
    std::size_t size() const;

    // Snapshot of the n x n similarity kernel (row/col i = internal id i; dot of the
    // unit descriptors == cosine; NaN on descriptor-size mismatch).
    Eigen::MatrixXf kernel() const;

    // Snapshot of the row -> external id mapping (index = internal id).
    std::vector<ExternalId> external_ids() const;

    // Drop every stored view and the kernel (host system reset).
    void clear();

    // ---- Culling -----------------------------------------------------------------

    // Per-view protection: a protected view is never proposed for culling but still
    // acts as an explainer. Unknown ids are ignored (set) / not protected (get).
    void set_protected(ExternalId id, bool value = true);
    bool is_protected(ExternalId id) const;

    // Record a view as removed OUTSIDE cull_keyframes (e.g. the host culled it by
    // another rule): it keeps its kernel row and becomes culling history. Views culled
    // through cull_keyframes' callback are recorded automatically.
    void set_culled(ExternalId id);
    bool is_culled(ExternalId id) const;

    struct CullParameters
    {
        // Culling method; "gram-greedy" (the joint-information greedy rule on the
        // Gram kernel) is the only one implemented.
        std::string method{"gram-greedy"};
        // tau: max unexplained information any view (alive or history) may be left with
        float max_unexplained{0.1f};
        // Double-centre the kernel over the usable views (Pearson correlation of the
        // mean-centred descriptors; removes a common-mode similarity floor)
        bool centred{true};
        // Never cull below this many alive views (in scope)
        int min_keyframes{10};
        // Never cull the last n inserted views (insertion order; n >= 1 also protects
        // the newest view)
        int protect_last{1};
        // Cap on culls per call (0 = unlimited)
        int max_per_call{0};
        // Never cull the first inserted view (it anchors the host's map)
        bool protect_first{true};
    };

    struct CullReport
    {
        struct CulledView
        {
            ExternalId id;
            float unique_information;         // v_i at the time it was culled
            float worst_unexplained_after;    // max unexplained view right after the cull
            int alive_after;                  // alive views in scope right after the cull
        };
        std::vector<CulledView> culled;
        int views_total{0};                   // rows ever inserted (usable or not)
        int candidates{0};                    // unprotected alive views in scope
        int alive_after{0};                   // alive views in scope after the call
        float worst_history{0.0f};            // max unexplained over the history rows
        int history_over_budget{0};           // history rows above tau (tau was lowered)
        bool reached_max_per_call{false};
    };

    // The host executes each cull and reports back: return true when the view was
    // actually removed (it becomes history here), false to leave it alive and skip
    // it for the rest of this call (e.g. the host deferred the erase).
    using CullCallback = std::function<bool(ExternalId)>;

    // Greedy joint-information culling on the kernel (see the .cpp for the maths).
    // Alive views = stored, not culled; candidates = alive, unprotected, in scope.
    // `local_window` (optional) restricts the marginalisation to those external ids
    // (the host's covisibility window): candidates and explainers come from the
    // window, and history is reduced to the rows whose best alive explainer (over the
    // whole map) lies in it. The callback is invoked WITHOUT the internal lock held.
    // Throws std::invalid_argument for an unknown method.
    CullReport cull_keyframes(const CullParameters& parameters, const CullCallback& try_cull,
                              const std::vector<ExternalId>* local_window = nullptr);

private:
    mutable std::mutex mutex_;
    std::deque<Eigen::VectorXf> descriptors_;           // indexed by InternalId, append-only
    std::deque<ExternalId> external_ids_;               // InternalId -> ExternalId
    std::unordered_map<ExternalId, InternalId> ids_;
    Eigen::MatrixXf kernel_;                            // n x n, grown on add()
    std::deque<char> culled_;                           // InternalId -> removed (history row)
    std::deque<char> protected_;                        // InternalId -> never cull
};

} // namespace placecell
