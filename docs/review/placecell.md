# Review — PlaceCell

- **Sources:** `src/placecell.cpp`, `src/placecell_events.cpp`, `include/placecell/placecell.h`
- **Reviewer:** Alejandro Fontan
- **Last reviewed at:** `ccc5c8c` (2026-09-25) — materialise_kernel_locked, 1 of 41 functions read

## Checklist

| Function | Source | State | Notes |
|---|---|---|---|
| `PlaceCell::materialise_kernel_locked` | [placecell.cpp#L735](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L735 "PlaceCell::materialise_kernel_locked(") | question | incremental rebuild, see 2026-09-25  |
| `PlaceCell::cull_keyframes` | [placecell.cpp#L854](https://github.com/alejandrofontan/placecell/blob/main/src/placecell.cpp#L854 "PlaceCell::cull_keyframes(") | unread |  |

## Notes

### 2026-09-25 — materialise_kernel_locked (at `ccc5c8c`)

- Full O(n²) rebuild where the counts already allow an O(n) refresh of row and column i;
  a cheaper first step folds the second n² pass (the sums) into the pair loop. Acceptable while
  `set_items` bursts are batched by the flag; revisit if `snapshot+centring` dominates a cull.

### 2026-09-25 — checklist created (at `ccc5c8c`)

No findings yet.
