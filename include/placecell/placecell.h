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
 * Besides the store and the culler, unexplained_information() answers the dual
 * question for a view that is NOT stored: how much of it the alive views cannot
 * explain (the information a keyframe made from it would add) — a read-only query
 * on the same kernel the culler marginalises.
 *
 * Two ways to fill a store, never mixed (the first call decides; clear() resets):
 * - descriptor-backed: add(id, descriptor) grows the kernel by the new row of dots
 *   and keeps the descriptor, so unexplained_information(descriptor) can compare a
 *   new view against the store;
 * - kernel-only: set_kernel(similarity, ids) initialises an EMPTY store from a host-
 *   supplied n x n similarity (e.g. a precomputed VPR matrix loaded with kernel_io.h's
 *   load_npy + similarity_from_distance). There are no descriptors: descriptor() is
 *   nullptr for every id, add() is refused (ERROR log, invalid_id) and the descriptor
 *   query returns NaN — the kernel side (cull_keyframes, kernel(), centred_kernel(),
 *   snapshot(), dump()) works exactly as for a descriptor-backed store.
 *
 * Contracts:
 * - add() is idempotent: an id that already exists keeps its stored descriptor and
 *   returns its existing internal id.
 * - descriptor() returns a stable pointer (entries are never relocated or mutated),
 *   nullptr for an unknown id (and for every id of a kernel-only store).
 * - kernel()/external_ids() return snapshots (consistent with each other only when
 *   taken together by the caller in the absence of concurrent adds; row counts can
 *   only grow, so a kernel snapshot is always a leading principal submatrix of any
 *   later one).
 * - A kernel entry is NaN when the two descriptors' sizes mismatch.
 * - clear() drops everything (store and kernel) — for a host system reset.
 * - Thread-safe: all methods serialise internally.
 *
 * Diagnostics (see log.h / profiler.h / recorder.h): every store owns a Profiler (timing
 * of the main entry points) and a Recorder (query / cull / decision history for the
 * plots); the Logger is process-wide. Options selects what is on; dump() writes the
 * kernel, the recorder's CSVs and the profile samples into a directory for offline
 * inspection (tools/plot_placecell.py).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "placecell/log.h"
#include "placecell/profiler.h"
#include "placecell/recorder.h"

namespace placecell
{

class PlaceCell
{
public:
    using ExternalId = std::uint64_t;
    using InternalId = std::size_t;
    static constexpr InternalId invalid_id = static_cast<InternalId>(-1);

    struct Options
    {
        // Process-wide log verbosity to apply at construction; PLACECELL_VERBOSITY in
        // the environment wins over this, an explicit Logger::set_level() over both
        std::optional<LogLevel> verbosity{};
        bool profile{true};                 // time the main entry points (Profiler)
        bool record{true};                  // keep the query / cull history (Recorder)
        bool report_on_destruction{false};  // print the profile table when the store dies
        std::string name{"PlaceCell"};      // label in the profile report
    };

    PlaceCell();
    explicit PlaceCell(const Options& options);
    virtual ~PlaceCell();
    PlaceCell(const PlaceCell&) = delete;
    PlaceCell& operator=(const PlaceCell&) = delete;

    const Options& options() const { return options_; }
    Profiler& profiler() { return profiler_; }
    const Profiler& profiler() const { return profiler_; }
    Recorder& recorder() { return recorder_; }
    const Recorder& recorder() const { return recorder_; }
    // Print the profile table through the Logger (bypasses the verbosity gate)
    void print_profile() const;
    // kernel.npy, kernel_centred.npy, views.csv (internal id, external id, culled,
    // protected) + Recorder::dump_csv + Profiler::dump_csv into `directory`
    void dump(const std::string& directory) const;

    // Store a descriptor under the host's id and grow the kernel by its row (O(n)
    // dot products); returns its internal id. Idempotent: a known id returns its
    // existing internal id and keeps the stored descriptor and kernel row. Refused on
    // a kernel-only store (set_kernel was called): ERROR log, returns invalid_id.
    InternalId add(ExternalId id, Eigen::VectorXf descriptor);

    // ---- Kernel-only initialisation ----------------------------------------------

    struct KernelOptions
    {
        // Replace S by (S + S^T)/2. A matrix built as the min over image rotations
        // (VPR-LAB's D.npy) is not symmetric; both entries estimate the same quantity.
        bool symmetrise{true};
        // WARN (instead of INFO) when max|S_ij - S_ji| exceeds this before symmetrising
        float asymmetry_warn{0.05f};
        // Set every diagonal entry to exactly 1 (a distance matrix's diagonal is ~0 up
        // to rounding, so the converted similarity is ~1 up to rounding)
        bool unit_diagonal{true};
        // Compute the smallest eigenvalue (O(n^3): ~0.5 s at n = 1000, minutes at
        // n = 10000) and WARN when it is negative — the kernel is then not a valid
        // covariance and cull_keyframes' scores degrade softly (issue #3)
        bool psd_check{true};
        // Project onto the PSD cone (clip negative eigenvalues at 0) and renormalise to
        // unit diagonal; implies the eigen-decomposition whatever psd_check says
        bool clip_to_psd{false};
    };

    struct KernelReport
    {
        int views{0};
        float max_asymmetry{0.0f};           // max |S_ij - S_ji| of the input
        float max_diagonal_deviation{0.0f};  // max |S_ii - 1| of the input
        // Smallest / largest eigenvalue of the (symmetrised, unit-diagonal) kernel
        // BEFORE clipping; NaN when neither psd_check nor clip_to_psd asked for them
        float min_eigenvalue{std::numeric_limits<float>::quiet_NaN()};
        float max_eigenvalue{std::numeric_limits<float>::quiet_NaN()};
        int negative_eigenvalues{0};
        bool clipped{false};
        double ms{0.0};
    };

    // Initialise an EMPTY store from a host-supplied n x n similarity kernel (row/col i
    // -> ids[i], or 0..n-1 when `ids` is empty): the views are stored without
    // descriptors, all alive and unprotected, and the centring statistics are rebuilt
    // from the matrix. The store becomes kernel-only (see the class comment). What
    // the options fixed is returned and logged (INFO; WARN for a negative eigenvalue
    // or a large asymmetry).
    // Throws std::logic_error when the store is not empty (call clear() first), and
    // std::invalid_argument for a non-square matrix, an ids size mismatch, duplicate
    // ids, or a NaN entry.
    KernelReport set_kernel(const Eigen::MatrixXf& similarity, const std::vector<ExternalId>& ids,
                            const KernelOptions& options);
    // Default options (an overload rather than a default argument: GCC cannot use a
    // nested struct's member initialisers as a default argument of the enclosing class)
    KernelReport set_kernel(const Eigen::MatrixXf& similarity, const std::vector<ExternalId>& ids = {});

    // True once set_kernel has initialised this store (until clear())
    bool kernel_only() const;

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

    // The kernel double-centred over the usable views and renormalised to unit diagonal
    // — exactly what cull_keyframes marginalises with parameters.centred (identity to
    // kernel() below 3 usable views). Snapshot; O(n^2).
    Eigen::MatrixXf centred_kernel() const;

    // Everything a plot needs, taken under one lock so the pieces agree
    struct Snapshot
    {
        Eigen::MatrixXf kernel;               // raw, or centred when requested
        std::vector<ExternalId> ids;          // index = internal id
        std::vector<char> culled;             // history rows
        std::vector<char> protected_views;
        bool centred{false};
    };
    Snapshot snapshot(bool centred = false) const;

    // Drop every stored view and the kernel (host system reset).
    void clear();

    // ---- Information of a view not in the store ----------------------------------

    struct Information
    {
        // v_x = K_xx - k_xA K_AA^-1 k_Ax in [0,1]: 1 = nothing alive resembles the view,
        // 0 = the alive views explain it completely. NaN when the query cannot be
        // compared (descriptor-size mismatch with the store).
        float unexplained{1.0f};
        // Alive views the query was marginalised over (0 -> unexplained is 1)
        int explainers{0};
        // Most similar explainer and its similarity on the kernel used (centred or raw)
        ExternalId best_explainer{0};
        float best_similarity{0.0f};
    };

    // Unexplained information of `descriptor` given the alive views (or, with
    // `window`, the alive views among those ids) — the information a keyframe made
    // from this view would add to them. Read-only: nothing is stored, the kernel is
    // untouched. NaN on a kernel-only store (no descriptors to compare against). Same kernel as cull_keyframes: with `centred` the stored views are
    // double-centred over every stored view (alive and history) exactly as the culler
    // does and the query is centred out-of-sample against that same mean, so on a
    // raw kernel the value equals the unique information v_i the view would have
    // right after add() (Schur identity; on a centred kernel up to the mean shift
    // its own insertion causes). Cost: one dot product per stored view plus a solve
    // of the |explainers| x |explainers| system.
    Information unexplained_information(const Eigen::VectorXf& descriptor,
                                        const std::vector<ExternalId>* window = nullptr,
                                        bool centred = true) const;

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
        // Count-driven mode (0 = off): cull the least unique alive view, one at a time,
        // until this many alive views remain in scope, ignoring max_unexplained (the
        // history constraint is not enforced either) — the offline "best N keyframes"
        // selection of a sequence. min_keyframes and the protections still apply, so
        // fewer culls than requested can happen (report.alive_after tells).
        int target_alive{0};
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
        // Unique information v_i = 1/(K_AA^-1)_ii of every alive view in scope after the
        // call (NaN where the inverse is not positive) — what the next call would score
        std::vector<ExternalId> alive_ids;
        std::vector<float> alive_unique_information;
    };

    // The host executes each cull and reports back: return true when the view was
    // actually removed (it becomes history here), false to leave it alive and skip
    // it for the rest of this call (e.g. the host deferred the erase).
    using CullCallback = std::function<bool(ExternalId)>;

    // Greedy joint-information culling on the kernel (see the .cpp for the maths).
    // Alive views = stored, not culled; candidates = alive, unprotected, in scope.
    // With parameters.target_alive > 0 the same greedy order runs count-driven instead
    // of tau-driven: it stops when that many views are alive (see CullParameters).
    // `local_window` (optional) restricts the marginalisation to those external ids
    // (the host's covisibility window): candidates and explainers come from the
    // window, and history is reduced to the rows whose best alive explainer (over the
    // whole map) lies in it. The callback is invoked WITHOUT the internal lock held.
    // Throws std::invalid_argument for an unknown method.
    CullReport cull_keyframes(const CullParameters& parameters, const CullCallback& try_cull,
                              const std::vector<ExternalId>* local_window = nullptr);

protected:
    // For const entry points in subclasses that still time themselves (the profiler is
    // mutable state by design: timing a const query does not change the store)
    Profiler& mutable_profiler() const { return profiler_; }

private:
    // The maths (unexplained_information / cull_keyframes) call these once per event;
    // they fan out to the Recorder and the Logger (src/placecell_events.cpp)
    void on_add(ExternalId id, InternalId internal, bool size_mismatch) const;
    void on_set_kernel(const KernelReport& report, const KernelOptions& options) const;
    void on_query(const Information& information, int stored, int window_size, bool centred, double ms) const;
    void on_cull_call(const CullParameters& parameters, const CullReport& report, bool local, double ms) const;

    Information compute_unexplained_information(const Eigen::VectorXf& descriptor,
                                                const std::vector<ExternalId>* window, bool centred,
                                                int& stored_out) const;
    // Double-centre `kernel` over the rows flagged usable and renormalise to unit
    // diagonal (no-op below 3 usable rows)
    static void centre_kernel(Eigen::MatrixXf& kernel, const std::vector<char>& usable);

    Options options_;
    mutable Profiler profiler_;
    mutable Recorder recorder_;
    mutable int last_history_over_budget_{-1};   // for the once-per-change over-budget line

    mutable std::mutex mutex_;
    std::deque<Eigen::VectorXf> descriptors_;           // indexed by InternalId, append-only
    std::deque<ExternalId> external_ids_;               // InternalId -> ExternalId
    std::unordered_map<ExternalId, InternalId> ids_;
    Eigen::MatrixXf kernel_;                            // n x n, grown on add()
    // Centring statistics of the kernel, maintained by add() so unexplained_information()
    // never has to sweep the n x n kernel: per-row sums and the total sum (double).
    std::deque<double> kernel_row_sums_;
    double kernel_total_sum_{0.0};
    // Descriptor size of the first stored view and whether a different size ever
    // arrived (NaN kernel entries): then no query can be compared.
    Eigen::Index descriptor_size_{0};
    bool size_mismatch_{false};
    // set_kernel() filled this store: no descriptors, add() refused, descriptor query NaN
    bool kernel_only_{false};
    std::deque<char> culled_;                           // InternalId -> removed (history row)
    std::deque<char> protected_;                        // InternalId -> never cull
};

} // namespace placecell
