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
#include <mutex>
#include <unordered_map>

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

private:
    mutable std::mutex mutex_;
    std::deque<Eigen::VectorXf> descriptors_;           // indexed by InternalId, append-only
    std::deque<ExternalId> external_ids_;               // InternalId -> ExternalId
    std::unordered_map<ExternalId, InternalId> ids_;
    Eigen::MatrixXf kernel_;                            // n x n, grown on add()
};

} // namespace placecell
