# Cull objectives: what each one optimises, measured effect, which to use

`CullParameters::objective` selects how gram-greedy ranks the feasible candidates of each cull:
`"unique"` (the default), `"minimax"` or `"total-loss"`. This note records why there are three,
what separates them and what was measured when they were added. The exact computation is in the
[`propose`](../reference/placecell_gram_greedy.md#propose) entry of the reference page, and the
one-line contract is the header comment on `CullParameters::objective`. Append, do not rewrite:
a later measurement gets its own dated section.

## What the objective does and does not change (2026-09-29)

The **feasibility test is the same for all three.** A candidate $i$ may be culled only if its
unique information $v_i = 1/M_{ii}$ is at most $\tau$ and no history row is pushed past its budget
(price $W_{hi}^2/M_{ii} \le \tau - v_h$ for a row within budget, $\le \delta = 0.01$ for a row
already above $\tau$). So every objective respects the same budget; they differ only in **which
feasible view goes first.** The rule is greedy and irreversible, so the order changes what stays
feasible afterwards: the objectives end with different survivor sets and, in general, a different
number of survivors.

| Objective | Score of a feasible candidate $i$ | Culls first |
|---|---|---|
| `unique` | $v_i$ | the view the others explain best: the most redundant one |
| `minimax` | $\max\big(v_i,\ \max_h (v_h + W_{hi}^2/M_{ii})\big)$, the worst unexplained view right after the cull | the view whose removal leaves the worst-off view least bad |
| `total-loss` | $v_i + \sum_h W_{hi}^2/M_{ii} + \sum_{j \ne i} \big(1/M'_{jj} - 1/M_{jj}\big)$ over the other alive views | the view whose removal raises everything least, summed |

- **`unique`** is the rule in the paper (eq. `greedy_rule`) and the only one with a stated
  property: on a raw kernel in count-driven mode it is a greedy maximisation of the joint
  information of the survivors, and the score at which each view is culled rises monotonically.
  It looks at the candidate alone; the history enters only through the feasibility test.
- **`minimax`** optimises the quantity tau bounds. Under a budget, it prefers culls that leave
  headroom, so later culls are less likely to be vetoed. It costs nothing extra: `worst_after` is
  already computed by the feasibility scan.
- **`total-loss`** adds up two different kinds of rise. The first two terms are information
  *lost*: the candidate and the history rows are views that are no longer alive, and their
  unexplained information goes up. The third term is not a loss: an alive view $j$ is still
  there, and a rise in its unique information only means it has become less redundant, so harder
  to cull next time. Summing them treats "harder to cull later" as equal to "lost now". That is a
  heuristic, not something that follows from the Gaussian model. It also costs an extra O(|A|) per
  candidate, so O(|A|²) per proposal.

In count-driven mode ($\tau = +\infty$) the feasibility test cannot fail, so the objective alone
decides the order, and the difference between the three is at its largest.

## Measured when the objectives were added (2026-09-26)

From the commits that introduced them (`1942b4a`) and fixed the history bound (`7a4005a`), not
re-run for this note. Offline runs: `kernel_demo` on YOUTUBE `acinipo` `D.npy`, `--clip --raw`,
tau 0.3.

| Objective | Survivors at `1942b4a` | Survivors at `7a4005a` | Worst view |
|---|---|---|---|
| `unique` | 19 | 19 (unchanged) | 0.295 after any cull |
| `minimax` | 19 | 19 (unchanged) | 0.283 after any cull |
| `total-loss` | 16 | 23 | worst history row 0.303 → 0.29999 |

- The three survivor sets overlapped by at most 3 views at `1942b4a`: the objectives do not
  re-rank a few ties, they pick substantially different frames.
- `minimax` kept the same number of views as `unique` with a lower worst view (0.283 vs 0.295),
  the headroom effect described above.
- `total-loss`'s 16 survivors at `1942b4a` came with a history row at 0.303, above tau: it
  relied on the old bound, $\max(\tau - v_h, \delta)$, which let a row near its budget be pushed
  past tau by up to the slack. With the bound fixed it keeps 23 views, more than the other two.
  On this sequence it is the least aggressive objective once the budget is actually enforced.
- The default output is unchanged by both commits: `synthetic_demo` and `kernel_demo` (acinipo,
  `--clip --raw`) with `unique` are row-identical to the builds before them.

This is one sequence at one tau. No SLAM run with a non-default objective exists yet.

## ETH `table_3` and the near-tau veto (2026-09-29)

`kernel_demo` on ETH `table_3` `vpr-lab/D.npy` (1180 views, `--clip --raw --tau 0.3`), with a temporary counter in `propose` (not committed) that logged every veto by a history row within budget, and two temporary switches for the bound of those rows: a tolerance $\varepsilon$ (`v_h + price \le \tau + \varepsilon`) and the old rule $\max(\tau - v_h, \delta)$ from before `7a4005a`.

| Objective | Current rule | $\varepsilon$ = 1e-6 … 1e-3 | Old rule | Vetoes by rows within 1e-3 of tau |
|---|---|---|---|---|
| `unique` | 91 kept, worst history 0.297 | identical | identical | 0 of 5 |
| `minimax` | 90 kept, worst 0.299 | identical | identical | 0 of 325 |
| `total-loss` | 142 kept, worst 0.300 | 139–140 kept, worst 0.300–0.301 | 84 kept, worst 0.326 | 734 of 1260 |

- `unique` and `minimax` never meet a near-tau row: the bound of rows within budget does not matter for them.
- `total-loss` minimises the summed rise, so it spreads small rises over many history rows and pushes them just under tau; those rows then veto. The prices they veto are real, not rounding: median 1.6e-4, 10th percentile 5.5e-6, 90th percentile 7e-4. A tolerance does not change the outcome (140 views even at 1e-3), and the old rule kept fewer views only by letting rows drift past the budget (0.326 at tau 0.3), the violation `7a4005a` fixed.
- Decision: keep the strict bound for rows within budget (no tolerance). On `table_3`, `total-loss` keeps 142 views where `unique` keeps 91, which supports keeping it experimental.

**Over-budget rows after lowering tau.** `synthetic_demo --lower-tau T2` (400 views, tau 0.3 lowered halfway): at T2 = 0.25 and 0.2 no row goes over budget; at 0.15 six rows do, at 0.1 sixteen. In neither case does the worst history row rise afterwards (rise +0.0000 over the 3–4 later calls), and every over-budget row is back within budget by the last call, because newly inserted views explain them. So the per-cull slack $\delta$ did not accumulate here. This is weak evidence: few calls follow the change, and longer runs (1600, 4000 views) insert nothing after the change (the trajectory only revisits known places), so they make no cull calls. A host that keeps culling after lowering tau without inserting near the old history is the untested case.

## Which to use (2026-09-29)

- **`unique` stays the default.** It is the rule the paper derives and the cheapest.
- **`minimax`** when the worst-covered view matters more than the count, e.g. offline selection
  where a single poorly explained frame is the failure mode. On acinipo it gave the same count
  with more headroom.
- **`total-loss`: experimental.** It mixes lost information with a rise in redundancy, and on
  the one measured sequence it culled least. Keep it out of experiments until that is settled.

## Open questions (2026-09-29)

- **Infinite loss can still win.** A feasible `total-loss` candidate whose downdate makes some
  pivot $M'_{jj} \le 0$ gets loss $+\infty$, but it beats the initial empty proposal (score
  $+\infty$, tie broken on $v_i$), so it is returned when no candidate has a finite loss. Should
  that count as infeasible?
- **The early skip could apply to all three.** Every score is at least $v_i$ (exactly for
  `minimax`, up to rounding for `total-loss`), so a candidate with $v_i$ at or above the running
  best cannot win under any objective. Today the skip is only applied for `unique`.
- **The winning score is not reported.** `Proposal::score` never leaves `propose`, so for
  `total-loss` the report cannot show the loss of each cull; that would need a `CulledView`
  field.
- **More sequences.** The comparison above is acinipo only, and raw only; the centred kernel and
  the count-driven mode are unmeasured.
- **The paper** describes only `unique`. If `minimax` or `total-loss` goes into the experiments,
  eq. `greedy_rule` needs the objective, and `total-loss` needs a justification for its third
  term.

## The objectives in the paper (2026-09-29)

`paper/sec/03_methodology.tex` now presents all three. Eq. `greedy_rule` is an argmin of a score $s_i$ ($v_i$ by default), and a new paragraph, *Ranking the feasible candidates*, defines the three scores in eq. `objectives` with the same caveats as this note: the constraints decide which views may go and the score only the order; minimax is the quantity tau bounds and costs nothing; the third term of total-loss is a rise in redundancy, not a loss, so the sum is a heuristic that pushes history rows towards tau. This answers the last open question above except for the experiments: no numbers are in the paper yet. The count-driven paragraph now restricts the joint-information property to the unique score, and states it through $\log\det\mathbf{K}_{A\setminus\{i\}} = \log\det\mathbf{K}_{AA} - \log v_i$.
