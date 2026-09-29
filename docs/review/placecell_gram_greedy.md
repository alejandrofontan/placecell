# Review — gram-greedy

- **Sources:** `src/placecell_gram_greedy.cpp`, `src/placecell_gram_greedy.h` (`namespace placecell::gram_greedy`: `State`, `Proposal`, the two constants, the declarations), `src/placecell_cull_method.h` (`CullScope`, `CullExecutor`, shared by every method)
- **Reviewer:** Alejandro Fontan
- **Last reviewed at:** `74acd51` (2026-09-26) — 1 of 4 functions read; the file is the culler split out of `src/placecell.cpp`

## Checklist

| Function | Source | State | Notes |
|---|---|---|---|
| `gram_greedy::cull` | [placecell_gram_greedy.cpp#L145](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L145 "void cull(const CullScope& scope") | unread |  |
| `gram_greedy::seed` | [placecell_gram_greedy.cpp#L23](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L23 "State seed(const CullScope& scope)") | question | unchecked LDLT (#3), jitter gap, see 2026-09-26 |
| `gram_greedy::propose` | [placecell_gram_greedy.cpp#L49](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L49 "Proposal propose(const CullScope& scope") | unread |  |
| `gram_greedy::downdate` | [placecell_gram_greedy.cpp#L95](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L95 "void downdate(const CullScope& scope") | unread |  |
| `gram_greedy::report` | [placecell_gram_greedy.cpp#L118](https://github.com/alejandrofontan/placecell/blob/main/src/placecell_gram_greedy.cpp#L118 "void report(const CullScope& scope") | unread |  |

`CullExecutor::operator()` (header-only, `src/placecell_cull_method.h`) is not listed; note it under the driver if it matters. The functions were renamed on 2026-09-29 (`PlaceCell::gram_greedy_<step>` → `gram_greedy::<step>`, `cull_gram_greedy` → `gram_greedy::cull`); the dated notes below keep the old names. `report` is new (split out of the driver).

## Notes

### 2026-09-26 — gram_greedy_seed (at `74acd51`)

- `M` is the LDLT solve of the jittered alive block against the identity and the result is never
  inspected: on an indefinite kernel (an unclipped `set_kernel` matrix) some `M_ii` come out
  non-positive and `gram_greedy_propose` drops those views without a word, so the culler goes
  quiet with an empty report. The only signal is `set_kernel`'s PSD WARN at load time, far from
  the symptom. Already the proposal of the kernel-contract issue (a check on the pivots here). → #3
- The seeded `v_h = K_hh − w_h · k_hA` uses the unjittered `K_hh` and `k_hA` against the jittered
  `M`, so it differs from the Schur identity `1/M_ii` that `gram_greedy_downdate` assigns to views
  culled in the same call by a gap of order ε. Harmless at 1e-6; the reference entry states it. → reference
- The history rows are seeded in count-driven mode too, where tau is infinite and the price test
  cannot fail, so their only consumer is `worst_history` in the report: O(|H||A|²) for a diagnostic.
  Costs nothing today (offline runs start with no history); skip them when `tau` is infinite if an
  online count-driven use ever appears. → reference
- Two spellings of one operation: the jitter is added inside the copy loop here and to the
  diagonal after the copy in `solve_information`, and the two constants are separate
  (`gram_greedy_jitter` vs a literal in `solve_information`) with a comment promising they are
  equal. One named constant in the header would make the promise a fact. `<cmath>` in this file is
  unused. → #6

### 2026-09-26 — checklist created (at `74acd51`)

No findings yet.
