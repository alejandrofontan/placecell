# `src/placecell_gram_greedy.cpp`

The gram-greedy culling method. The shell, [`cull_keyframes`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_cull.cpp#L197 "case CullMethod::gram_greedy:"),
builds a `CullScope` (the kernel snapshot, the alive and history rows in scope, the candidates,
tau, the stop count, the cap and the objective) and a `CullExecutor`, and hands both to
[`gram_greedy::cull`](#cull), with its profiler. The driver seeds the linear-algebra state once, then
culls one view at a time: [`propose`](#propose) picks the feasible
candidate, the executor asks the host to erase it, and [`downdate`](#downdate)
applies an accepted cull by rank-one updates, so the only O(|A|³) step of a call is the seed's
inverse. [`report`](#report) fills the report from the final state. All five are
free functions in `namespace placecell::gram_greedy` that see neither the store nor its lock;
the executor is the only way a cull reaches the store. The
source carries no comments, so this page is the description of the file; the derivation is
`paper/sec/03_methodology.tex` (§ Unexplained information, § Greedy joint-information culling).

Sources: `src/placecell_gram_greedy.cpp` and its internal header `src/placecell_gram_greedy.h`
(`namespace placecell::gram_greedy`: `State`, `Proposal`, the two constants and the five
declarations); `CullScope`, `CullExecutor`, `CullObjective` in the internal header
`src/placecell_cull_method.h`, shared by every method. Neither header is installed; of the culler, the public
`include/placecell/placecell.h` declares `cull_keyframes` and the types the host passes and gets back
(`CullParameters`, `CullCallback`, `CullReport`). The shell is on
[`placecell.md`](placecell.md). Reading notes:
[`docs/review/placecell_gram_greedy.md`](../review/placecell_gram_greedy.md).

## State

`State` belongs to one call, indexed like `scope.alive`; it is built by the seed and
discarded when the driver returns.

| Member | Holds | Written by |
|---|---|---|
| `M` | $\mathbf{K}_{AA}^{-1}$ over `scope.alive`, jittered; the row and column of a view culled in this call are exactly zero | [`seed`](#seed), [`downdate`](#downdate) |
| `W_rows` | one vector per history row, $\mathbf{w}_h = \mathbf{M}\mathbf{k}_{hA}^{\!\top}$ over the alive positions | seed (the history in scope), downdate (every row updated, one appended per accepted cull) |
| `v_h` | the unexplained information of each history row, same order as `W_rows` | seed, downdate |
| `removed` | one flag per alive position, set for the views culled in this call | seed (all zero), downdate |

`Proposal` is what [`propose`](#propose) returns:

| Field | Holds | Read by |
|---|---|---|
| `index` | position in `scope.alive`, −1 = no feasible candidate | driver (stop test, executor, refusal, report id), downdate |
| `unique_information` | $v_i = 1/M_{ii}$ | driver (refusal DEBUG line, `CulledView`), downdate (the culled view's $v_h$) |
| `worst_after` | the largest unexplained view right after the cull | driver (`CulledView::worst_unexplained_after`) |
| `score` | the value of `scope.objective` that won | propose only (comparison with the running best) |

## Call graph

- **[`cull`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L145 "void cull(const CullScope& scope")** → [`seed`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L23 "State seed(const CullScope& scope)"); per iteration [`propose`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L49 "Proposal propose(const CullScope& scope"), [`CullExecutor::operator()`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_cull_method.h#L71 "bool operator()(const int row)"), [`downdate`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L95 "void downdate(const CullScope& scope"); after the loop [`report`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L118 "void report(const CullScope& scope")
- called from the shell's switch in [`cull_keyframes`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_cull.cpp#L197 "case CullMethod::gram_greedy:")

## Flow

```mermaid
%%{init: {"themeVariables": {"fontSize": "19px"}, "flowchart": {"nodeSpacing": 25, "rankSpacing": 30}}}%%
flowchart LR
    SHELL([cull_keyframes]) --> SEED["<b>seed</b><br/>M = K_AA⁻¹"]
    SEED --> PROP["<b>propose</b><br/>best feasible candidate"]
    PROP -- proposal --> EXEC{host culls it?}
    EXEC -- yes --> DOWN["<b>downdate</b><br/>rank-one update"] --> PROP
    EXEC -- no --> PROP
    PROP -- "none, or stop / cap reached" --> REP["<b>report</b>"]
    REP --> OUT([back to the shell])

    click SEED "#seed"
    click PROP "#propose"
    click DOWN "#downdate"
    click REP "#report"

    %% VSLAM-LAB logo squares: cyan #b5f3f9, periwinkle #8195fb, lavender #a59ddf
    classDef entry fill:#8195fb,stroke:#5f74d6,color:#fff
    classDef step fill:#b5f3f9,stroke:#7fcfd8,color:#1b2a4a
    classDef check fill:#a59ddf,stroke:#7e75c4,color:#1b2a4a

    class SHELL,OUT entry
    class SEED,PROP,DOWN,REP step
    class EXEC check
```

## Driver

### `cull`

```cpp
void cull(const CullScope& scope, CullExecutor& execute, PlaceCell::CullReport& report, Profiler& profiler)
```

Runs one gram-greedy call. It seeds the linear-algebra state once, then culls one view at a time, always the feasible candidate that minimises `scope.objective`, and stops when no candidate is feasible, when `scope.stop_at` views are alive in scope, or when `scope.max_per_call` culls were accepted. Finally it fills the report.

Each iteration asks [`propose`](#propose) for a candidate and hands it to the host through the executor. An accepted cull appends a `CullReport::CulledView` (`id`, `unique_information` $= v_i$, `worst_unexplained_after` from the proposal's `worst_after`, `alive_after`) and runs [`downdate`](#downdate). A refused cull (e.g. a deferred erase) leaves the state untouched: the view stays alive and keeps explaining the others, but it is cleared in the driver's copy of the candidates, so it is not proposed again in this call. A refusal counts towards neither `max_per_call` nor the alive count; since each one removes a candidate, the loop still ends. The next call starts again from the shell's candidates.

`max_per_call` exists for online threshold changes: raising tau makes many views feasible at once, and the cap spreads their culls over successive calls. After the loop, [`report`](#report) fills the rest of the report from the final state.

`scope` is the shell's snapshot (kernel, alive and history rows, candidates, budget, objective). `execute` runs a cull on the host and marks the row culled on acceptance. `report` arrives with `views_total` and `candidates` set by the shell; `cull` appends to `report.culled` and fills the rest through [`report`](#report). `profiler` is the store's; `cull` records `cull_keyframes/inverse` (the seed, size $|A|$) and `cull_keyframes/greedy` (the loop and the report minus the executor's time, sizes $|A|$ and the history count at the end of the call).

!!! note
    The executor calls the host's callback with `mutex_` not held (the shell never holds it here; the host typically takes its own map lock in the callback). On acceptance it calls the mark-culled function the shell gave it, which sets the row's culled flag under `mutex_`. That is the only way a cull reaches the store.

!!! warning
    When the history in scope is empty, the kernel is centred and the scope is the whole map, the centring set equals the alive set: $\mathbf{K}_{AA}$ is rank-deficient and every $v_i$ of the call is jitter-scale (issue #5). `cull` logs this once per process, after the seed and before the loop.

**Cost** O(|A|³ + |H| |A|²) for the seed, then per iteration one propose (O(|A| |H|), plus O(|A|²) for `total-loss`), per accepted cull one downdate (O(|H| |A| + |A|²)), then one report (O(|H| + |A|)); $|H|$ grows by one per accepted cull. Timed as `cull_keyframes/inverse` (seed) and `cull_keyframes/greedy` (the rest, minus the executor's time, which the shell records as `cull_keyframes/host_callback`)

## Steps

### `seed`

```cpp
State seed(const CullScope& scope)
```

Builds the state a gram-greedy call starts from: the inverse of the alive block of the kernel and, for every history row in scope, its explanation weights and its unexplained information. It is the only step of a call that is not incremental.

$$
\mathbf{M} = (\mathbf{K}_{AA} + \varepsilon \mathbf{I})^{-1}, \qquad \mathbf{w}_h = \mathbf{M}\,\mathbf{k}_{hA}^{\!\top}, \qquad v_h = K_{hh} - \mathbf{w}_h \cdot \mathbf{k}_{hA} \quad \forall h \in H
$$

$\mathbf{K}_{AA}$ is the alive block of the kernel (`scope.similarity`, raw or centred): the rows and columns indexed by `scope.alive`. The jitter $\varepsilon$ = `jitter` is added to its diagonal while it is copied. A centred kernel has rank $m-1$ over its $m$ usable views, so its alive block is singular or close to it, and near-duplicate views make a raw one badly conditioned; the jitter makes $\mathbf{K}_{AA}$ positive definite for any PSD kernel, at the cost of a bias of order $\varepsilon$ in every score. $\mathbf{M}$ comes from an LDLT factorisation solved against the identity.

The rows $\mathbf{w}_h$ form $\mathbf{W} = \mathbf{K}_{HA}\mathbf{M}$ (paper eq. `downdate_W`), one per history view, computed as $\mathbf{M}\mathbf{k}_{hA}^{\!\top}$ since $\mathbf{M}$ is symmetric. $v_h$ is paper eq. `unexplained` with $x = h$, clamped at $0$ against rounding. The history rows are seeded in count-driven mode too: the price test cannot fail there, but $v_h$ still feeds `worst_after` in [`propose`](#propose), hence the `minimax` score and `CulledView::worst_unexplained_after`, and `worst_history` in the report.

`scope` is the shell's snapshot; `seed` reads its kernel, its alive rows and its history rows.

Returns the `State`: `M` as above, `W_rows` and `v_h` with one entry per history row in `scope.history` order, and `removed` with one zero flag per alive position.

!!! warning
    The LDLT pivots are not checked. An indefinite kernel (an unclipped `set_kernel` matrix) can leave non-positive $M_{ii}$, which [`propose`](#propose) then skips without a word (issue #3).

!!! note
    $K_{hh}$ and $\mathbf{k}_{hA}$ carry no jitter while $\mathbf{M}$ does, so a seeded $v_h$ sits $\varepsilon$ below the Schur identity $1/M_{ii}$ that [`downdate`](#downdate) assigns to views culled in the same call (exactly $\varepsilon$ in exact arithmetic, when the kernel and the scope are unchanged). Harmless at $10^{-6}$.

**Cost** O(|A|³) for the inverse + O(|H| |A|²) for the history rows; space O(|A|² + |H| |A|). Timed as `cull_keyframes/inverse`

### `propose`

```cpp
Proposal propose(const CullScope& scope, const State& state, const std::vector<char>& candidate)
```

Picks the next view to cull. A candidate is feasible when removing it keeps every view within budget: its own unique information, and the unexplained information of every history row after the cull. Among the feasible candidates, `propose` returns the one with the smallest `scope.objective` score.

$$
v_i = \frac{1}{M_{ii}} \le \tau, \qquad
\frac{W_{hi}^2}{M_{ii}} \le \begin{cases} \tau - v_h, & v_h \le \tau \\ \delta, & v_h > \tau \end{cases}
\quad \forall h \in H
$$

The price $W_{hi}^2/M_{ii}$ is exactly the rise of $v_h$ that [`downdate`](#downdate) would apply (paper eq. `price`), so the test is exact. A row within budget is never raised past $\tau$. A row already above it (tau was lowered between calls) may deteriorate by at most $\delta$ = `over_budget_slack` per cull, so it vetoes only the candidates that really explain it. The paper's eq. `greedy_rule` uses the same split bound.

The score depends on `scope.objective`:

| Objective | Score |
|---|---|
| `unique` (default) | $v_i$ |
| `minimax` | the worst unexplained view right after the cull, $\max\big(v_i, \max_h (v_h + W_{hi}^2/M_{ii})\big)$ |
| `total-loss` | $v_i + \sum_h W_{hi}^2/M_{ii} + \sum_{j} \big(1/(M_{jj} - M_{ji}^2/M_{ii}) - 1/M_{jj}\big)$ over the other alive $j$ |

Ties go to the smaller $v_i$, then to the lowest kernel row (insertion order). In count-driven mode ($\tau = +\infty$) both tests pass and the result is a plain argmin of the score. Why there are three objectives, what separates them and which to use: [`2026-09-29_cull_objectives.md`](../notes/2026-09-29_cull_objectives.md).

`scope` is the shell's snapshot. `state` holds the current inverse `M`, the history rows `W_rows` and `v_h`, and the `removed` flags. `candidate` has one flag per alive position: the driver's copy of `scope.candidate`, with refused views cleared.

Returns a `Proposal`: `index` is the position in `scope.alive`, or −1 when no candidate is feasible; `unique_information` is $v_i$; `worst_after` is the worst unexplained view right after the cull; `score` is the objective value that won.

!!! warning
    A candidate with $M_{ii} \le 0$ (an indefinite kernel, issue #3, or rounding drift) is skipped without a word: its $v_i$ would be negative and win the argmin. A feasible `total-loss` candidate whose downdate leaves some pivot $\le 0$ scores $+\infty$, but it still beats the initial empty proposal, so it is returned when no candidate has a finite score.

**Implementation.** One scan over the alive positions. With `unique`, a candidate whose $v_i$ is at or above the running best is skipped before the history scan, since it cannot win. The history scan of a candidate stops at its first veto, and accumulates the worst view and the summed prices on the way. `total-loss` then adds the rise of each other alive view, read off the diagonal of the downdated $\mathbf{M}$ (paper eq. `downdate_M`), without forming it.

**Cost** O(|A| |H|), plus O(|A|²) for `total-loss`; space O(1). Once per loop iteration, inside `cull_keyframes/greedy`

### `downdate`

```cpp
void downdate(const CullScope& scope, State& state, const Proposal& proposal)
```

Applies an accepted cull of alive position $i$ = `proposal.index` to the state: it removes the view from the alive block of $\mathbf{M}$, raises every history row by its price, and appends the culled view to the history. No solve is repeated; everything is a rank-one update.

$$
v_h \mathrel{+}= \frac{W_{hi}^2}{M_{ii}}, \qquad
\mathbf{w}_h \mathrel{-}= \frac{W_{hi}}{M_{ii}}\,\mathbf{m}, \qquad
\mathbf{M} \mathrel{-}= \frac{\mathbf{m}\mathbf{m}^{\!\top}}{M_{ii}}, \qquad \mathbf{m} = \mathbf{M}_{:,i}
$$

These are paper eqs. `price`, `downdate_W` and `downdate_M`, applied in that order with $M_{ii}$ and $\mathbf{m}$ taken before anything changes. The price is the one [`propose`](#propose) tested. The update of $\mathbf{M}$ is the Schur downdate that leaves the inverse of the alive block without $i$ on the other positions; it also zeroes row and column $i$ in exact arithmetic, and the code sets them to exactly $0$ afterwards to remove the rounding residue. Then `state.removed[i] = 1`.

The culled view joins the history. Its row is $\mathbf{w}_i = \mathbf{M}\mathbf{k}_{iA}^{\!\top}$ against the updated $\mathbf{M}$, with $\mathbf{k}_{iA}$ read over every alive position; its own column and the removed ones contribute nothing because they are exactly zero in $\mathbf{M}$. Its $v_h$ is `proposal.unique_information` $= 1/M_{ii}$, by the Schur identity, with no fresh solve.

`scope` supplies the kernel for the new history row. `state` is updated in place. `proposal` is the accepted proposal from [`propose`](#propose); the state has not changed since it was computed, because a refusal never touches it.

!!! note
    Entry $i$ of each existing $\mathbf{w}_h$ is $0$ in exact arithmetic but keeps rounding dust, which is never read again: [`propose`](#propose) skips removed positions, and a later downdate reads only its own column. Nothing in a call re-factorises $\mathbf{M}$, so rounding accumulates over the culls of one call.

!!! note
    $1/M_{ii}$ is the unexplained information under the jittered kernel, so the culled view's $v_h$ sits $\varepsilon$ above the $K_{hh} - \mathbf{w}_h \cdot \mathbf{k}_{hA}$ that [`seed`](#seed) computes for the same view in the next call (exactly $\varepsilon$ in exact arithmetic, when the kernel and the scope are unchanged).

**Cost** O(|H| |A|) for the history rows + O(|A|²) for $\mathbf{M}$ and the new row; space O(|A|). Once per accepted cull, inside `cull_keyframes/greedy`

### `report`

```cpp
void report(const CullScope& scope, const State& state, PlaceCell::CullReport& report)
```

Fills the report from the state the loop ended with: the alive count, the history against tau, whether the cap was hit, and the unique information of every view that stays alive. It is output only: nothing here feeds a culling decision. The fields go to the Recorder and the log through `on_cull_call`, to the viz alive strip, and to the host.

The counts come from the report itself: `report.culled` holds one `CulledView` per accepted cull (the shell never appends to it), so the number culled is its size and the alive count is $|A|$ minus that.

| Field | Value |
|---|---|
| `alive_after` | the alive count in scope after the call |
| `worst_history` | the largest $v_h$ over every history row, including the views culled in this call; $0$ with no history |
| `history_over_budget` | the rows with $v_h > \tau$; always $0$ in count-driven mode |
| `reached_max_per_call` | `max_per_call > 0` and that many culls were accepted; set whenever the cap was hit, even if the loop would have stopped there anyway |
| `alive_ids`, `alive_unique_information` | every alive position not removed, in `scope.alive` order, with $1/M_{ii}$, or NaN where $M_{ii} \le 0$ |

`scope` supplies the ids, tau and the cap. `state` is the final state of the call. `report` must already hold every accepted cull in `culled`.

!!! note
    `alive_unique_information` holds the scores at the end of this call. The next call re-seeds from a fresh snapshot, so its scores differ once the kernel, the scope or the centring set changed.

**Cost** O(|H| + |A|); space O(|A|) for the two alive vectors. Once per call, inside `cull_keyframes/greedy`

## Parameters read by this file

Everything comes through `CullScope`, filled by the shell from `CullParameters` and the store,
plus two constants in `src/placecell_gram_greedy.h`.

| Source | Field | Value | Used in | Effect |
|---|---|---|---|---|
| `CullScope` | `similarity` | the kernel snapshot, raw or centred | [`seed`](#seed), [`downdate`](#downdate) | the matrix marginalised |
| `CullScope` | `row_ids` | the external id of every kernel row | [`cull`](#cull), [`report`](#report) | ids for the DEBUG line, `CulledView` and `alive_ids` |
| `CullScope` | `alive` | usable, not-culled rows in scope | [`seed`](#seed), [`downdate`](#downdate), [`cull`](#cull), [`report`](#report) | the columns of `M` / `W`; kernel rows handed to the executor and reported |
| `CullScope` | `history` | usable, culled rows in scope | [`seed`](#seed), [`cull`](#cull) | the seeded rows of `W` and `v_h`; empty history triggers the issue-#5 warning |
| `CullScope` | `candidate` | alive positions that are not protected | [`propose`](#propose) (through the driver's copy) | which views may be proposed |
| `CullScope` | `tau` | `max_unexplained`, or +inf in count-driven mode | propose, report | the budget of every view; `history_over_budget` counts rows above it |
| `CullScope` | `stop_at` | `min_keyframes`, or `max(min_keyframes, target_alive)` | driver | the loop stops at this many alive views in scope |
| `CullScope` | `max_per_call` | `CullParameters::max_per_call` | driver, report | cap on accepted culls (0 = unlimited) |
| `CullScope` | `objective` | `CullParameters::objective` parsed to `CullObjective` (`unique`) | [`propose`](#propose) | what the feasible candidates are ranked by: `v_i`, the worst view after the cull, or the total loss |
| `CullScope` | `centred`, `local` | `CullParameters::centred`, window given | driver | the issue-#5 warning fires only when centred, map scope, no history |
| constant | [`jitter`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.h#L45 "inline constexpr double jitter") | `1e-6` | seed | added to the diagonal of `K_AA`; the query's `solve_information` uses the same value as a separate literal (issue #6) |
| constant | [`over_budget_slack`](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.h#L46 "inline constexpr double over_budget_slack") | `0.01` | propose | the most a history row already above tau may deteriorate per cull; rows within budget are bounded by `tau − v_h` instead |
