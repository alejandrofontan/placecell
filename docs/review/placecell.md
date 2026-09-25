# Review — PlaceCell

- **Sources:** `src/placecell.cpp`, `src/placecell_events.cpp`, `include/placecell/placecell.h`
- **Reviewer:** Alejandro Fontan
- **Last reviewed at:** `ccc5c8c` (2026-09-25) — 3 of 41 functions read

## Checklist

| Function | Source | State | Notes |
|---|---|---|---|
| `PlaceCell::centre_kernel` | [placecell.cpp#L287](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L287 "PlaceCell::centre_kernel(") | question | m vs n threshold, silent floor, see 2026-09-25 |
| `PlaceCell::usable_rows` | [placecell.cpp#L314](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L314 "PlaceCell::usable_rows(") | ok | second pass defensive, see 2026-09-25 |
| `PlaceCell::materialise_kernel_locked` | [placecell.cpp#L735](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L735 "PlaceCell::materialise_kernel_locked(") | question | incremental rebuild, see 2026-09-25  |
| `PlaceCell::cull_keyframes` | [placecell.cpp#L854](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L854 "PlaceCell::cull_keyframes(") | unread |  |

## Notes

### 2026-09-25 — centre_kernel (at `ccc5c8c`)

- The centring is entry-for-entry the one `query_system_locked` applies out of sample, floors
  included (lines 302–311 against 554–571), so the culler and the insertion query centre
  identically. → reference
- The "at least 3 views" rule counts different things on the two sides: here usable views
  (`m < 3` returns unchanged), in the query stored views (`n >= 3`, then the sums run over the
  `m` finite rows). A store with 3 stored views of which one is unusable is centred by the query
  over 2 views and not centred by the culler. Rare, but the two should count the same thing.
- The diagonal floor `1e-9` makes a view identical to the mean descriptor, or an indefinite host
  kernel, numerically safe but silent: its centred row becomes very large instead of a
  correlation, and nothing reports it. Either warn once or mark such a view unusable.
- Unusable rows are left raw inside the centred matrix; the culler never reads them, but
  `snapshot(true)`, `dump` and the viz module return the mixed matrix. → reference
- Centring the covisibility kernel brings nothing (no common mode, exact zeros) and costs
  sparsity plus the rank-deficiency degeneracies; `false` is the right default. The density bias
  of the item kernel is a normalisation question, not a centring one; note it under `set_items`.
  
### 2026-09-25 — usable_rows (at `ccc5c8c`)

- The rule is "unusable when NaN against every other view", not "any NaN in the row": a culprit
  puts a NaN into every other row's column, so the stronger rule would exclude everyone as soon
  as one culprit exists, which was the bug before 2026-09-24. → reference
- The second pass (a survivor still NaN against another survivor) is unreachable with the two
  NaN sources that exist (`add` size mismatch, empty item set: both a whole row and column, and
  `set_kernel` rejects NaN). It guards the invariant the callers need, finite among the usable
  rows, at one extra O(n²) sweep; on a hypothetical stray pair it drops the lower index only.
  Keep it; add a comment in the code saying it is defensive.
- The item query does not use it and drops empty views by `item_norms_ == 0`; the two rules
  coincide in item mode. → reference

### 2026-09-25 — materialise_kernel_locked (at `ccc5c8c`)

- Full O(n²) rebuild where the counts already allow an O(n) refresh of row and column i;
  a cheaper first step folds the second n² pass (the sums) into the pair loop. Acceptable while
  `set_items` bursts are batched by the flag; revisit if `snapshot+centring` dominates a cull.

### 2026-09-25 — checklist created (at `ccc5c8c`)

No findings yet.
