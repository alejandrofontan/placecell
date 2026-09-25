# Review — PlaceCell

- **Sources:** `src/placecell.cpp`, `src/placecell_events.cpp`, `include/placecell/placecell.h`
- **Reviewer:** Alejandro Fontan
- **Last reviewed at:** `ccc5c8c` (2026-09-25) — 2 of 41 functions read

## Checklist

| Function | Source | State | Notes |
|---|---|---|---|
| `PlaceCell::usable_rows` | [placecell.cpp#L314](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L314 "PlaceCell::usable_rows(") | ok | second pass defensive, see 2026-09-25 |
| `PlaceCell::materialise_kernel_locked` | [placecell.cpp#L735](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L735 "PlaceCell::materialise_kernel_locked(") | question | incremental rebuild, see 2026-09-25  |
| `PlaceCell::cull_keyframes` | [placecell.cpp#L854](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L854 "PlaceCell::cull_keyframes(") | unread |  |

## Notes

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
