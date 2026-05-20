# Phase 7 sub-step 3 — D6 real-rater replay harness: design audit

Per `UNIFIED_REPO_MERGE_PLAN.md` D6 + §"Phase 7" sub-step 3, this is
the **v1.0 blocker** — the harness must drive both the deployment
engine (`simulate_test.py`) and the Mode-A engine (`core_mcmc.py`)
with held-out *real* rater response sequences. User-locked decisions
(2026-05-19):

- **Design A** — strict constrained-bank: engine selects EV-optimal
  from the rater's scored-segs subset; Y from `labels.csv`;
  bank-exhaustion before decision → REFER.
- **Coverage iii** — BOTH per-task and per-candidate.

Audit date: 2026-05-19. Suite at start: **250 passed / 1 xfailed**.

## 1. The attach-point in each engine

### Deployment engine (`deployment/simulate_test.py`)

`simulate_candidate(true_theta, Σ_prior, ℓ*, bank_by_task, cfg, rng)`
inner loop (lines 389-468):

```python
sel = select_next_case(state, bank_by_task, pending, used_segids, ...)
k, seg, s_val = sel
t_true = true_theta[_slot(k, "t")]; l_true = true_theta[_slot(k, "l")]
eta_true = np.exp(l_true) * (s_val + t_true)
p_true = LAPSE_RATE + _ONE_MINUS_2LAMBDA * float(norm.cdf(eta_true))
Y = int(rng.random() < p_true)              # ← BERNOULLI DRAW
update_state(state, k, s_val, Y)
```

`select_next_case` consumes `bank_by_task[k]` (a pandas DataFrame
per task with at least `seg_id, s_mean` columns; `s_sd` for the
Phase-3.5 marginalised path). It picks EV-optimal from the unseen
rows. Output: PASS/FAIL/REFER per task + n_per_task (E[n] surrogate).

### Mode-A engine (`engine/core_mcmc.run_session_mcmc_auroc`)

Inner loop (lines 727-734):

```python
k, s = choose_item(state, bank_signals, n_subsample=n_subsample)
t_true = true_params[k * 2]
l_true = true_params[k * 2 + 1]
y = simulate_response(s, t_true, l_true, rng)    # ← BERNOULLI DRAW
update(state, k, s, y, s_sd=s_sd)
```

`bank_signals` is a list-of-K arrays of probit signals **without
seg_ids** — Mode-A's bank format is signal-only (the methodology
abstraction). For strict-A replay, we need to extend it so we can
look up the rater's Y at the chosen item — requires either passing a
seg_id alongside each signal or re-architecting `choose_item` to
return both.

## 2. The real-rater observation source

`data/labels/labels.csv` (2,115,793 rows, 5,246 raters; columns
`seg_id, rater_id, label_type, value, source_dataset`). With the
erratum-correct `{bipd,birds,other}→other` mapping (Phase-3 D5 / §6
audit), real binary `(rater_id, seg_id, task) → y ∈ {0,1}` triples
are derivable:

- `spike` task: rows with `label_type='spike'`, `value ∈ {0,1}`.
- IIIC 6 tasks (`sz`/`seizure`, `lpd`, `gpd`, `lrda`, `grda`,
  `other`): rows with `label_type='pattern_class'`, with the
  `{bipd,birds,other}→other` mapping, where
  `y_task = int(value == task)`.

## 3. The bank source

`data/deployment_prior/case_bank.csv` (230,599 rows, 7 tasks, cols
`task, seg_id, s_mean, s_sd, n_raters, pos_rate`). This is the
production K=7 frozen item bank (produced by
`freeze_deployment_prior.py:138-146`).

Per-task bank sizes:

| Task | Bank size |
|---|---|
| spike | 17,346 |
| seizure | 40,984 |
| lpd | 30,476 |
| gpd | 30,476 |
| lrda | 39,747 |
| grda | 40,701 |
| other | 30,869 |

## 4. The strict-A constrained bank per (rater, task)

For each (rater r, task k):

```
scored_segs(r, k) = { seg_id : labels[(rater_id=r, seg_id, task=k)].value defined }
strict_A_bank(r, k) = case_bank.csv[task=k, seg_id ∈ scored_segs(r, k)]
   columns: seg_id, s_mean, s_sd  (the engine inputs)
Y_lookup(r, seg_id, k) = labels[(rater_id=r, seg_id, task=k)].value
```

For each engine inner-loop step, the engine picks `(k, seg, s_val)`
from `strict_A_bank(r, k)`; replay substitutes
`Y = Y_lookup(r, seg_id, k)` for the Bernoulli draw.

## 5. The per-rater per-task density (re-confirmed for the build)

From sub-step 7-audit (`docs/PHASE7_AUDIT.md` §sub-3), at the
deployment contract's `N_min_per_task = 10` floor:

| Threshold (≥n/task on ALL 7 tasks) | # raters | tier breakdown |
|---|---|---|
| ≥10 | (re-check below) | mostly expert |
| ≥20 | 21 | expert=19, experienced=2 |
| ≥100 | 20 | expert=18, experienced=2 |

(Below: re-run at threshold ≥10 to confirm the per-candidate cohort.)

**Per-task coverage** (no all-7 requirement) is much broader: 16,765
(rater, task) cells with non-zero responses; mean 402 segs/task per
cell; median 59.

## 6. Output schema (proposed)

Strictly mirroring the Bernoulli-sim output schema for direct
comparison (= `data/deployment_prior/sim/candidates.csv` for
deployment; per-rater per-task AUROC posterior summary for Mode-A):

### Deployment replay output (`results/replay/deployment_replay.csv`)

Same column set as the Bernoulli sim (already present at
`data/deployment_prior/sim/candidates.csv`), with one row per
(rater, task) for per-task replay or per (rater) for per-candidate
replay. Key columns:

- `rater_id` (replaces simulated `candidate_id`)
- `tier` (from `raters.csv.expertise_level`)
- `task`, `decision`, `n_per_task`, `total_trials`
- For per-candidate: full 7-task verdict roll-up
- Replay-specific: `bank_exhausted` (bool — Design A exhaustion flag),
  `n_scored_segs_total` (the rater's seg budget for that cell)

### Mode-A replay output (`results/replay/mode_a_replay.csv`)

One row per (rater, task) cell:

- `rater_id`, `tier`, `task`
- `auroc_mean`, `auroc_ci_low_95`, `auroc_ci_high_95`, `auroc_hw`
- `stop_d0_025`, `stop_d0_05`, `stop_d0_1`, `reached_d*` flags
- `n_q_total`, `bank_exhausted`

### Bernoulli comparator

For each replay cell, run the same engine on the same (rater,
task)'s personal bank but with Y drawn from the Bernoulli at the
rater's *fitted* `(σ_k, θ_k) → (ℓ_k, t_k)` from
`data/engine_inputs/sdt_fits.csv`. Same (seg, s) selection sequence
the engine produces; only the Y source differs. This is the
**paired** comparator the plan asks for ("Report PASS/FAIL/REFER +
E[n] under replay vs Bernoulli").

## 7. Architectural decisions still open (user input needed)

- **Q1 — engines:** drive deployment only (the plan's headline:
  PASS/FAIL/REFER + E[n]), or deployment + Mode-A both (the plan's
  exact wording: "drives `simulate_test.py` (and the Mode-A
  engine)")?

  Recommended: **both** — the plan calls out both engines
  explicitly. Mode-A produces AUROC CI replay (a separate scientific
  story); deployment produces the PASS/FAIL/REFER headline.

- **Q2 — N_min floor for per-task cells:** include only cells with
  ≥`N_min_per_task` (=10 by `cfg`) scored segs per task (filters
  out marginal cells, keeps verdicts interpretable), or include all
  cells honestly (every cell, including bank-exhaust-at-1) and
  report the distribution?

  Recommended: **report at N_min ∈ {1, 10, 20}** stratification —
  the headline analysis uses `≥N_min_per_task` (the deployment
  contract's own minimum), with finer-grained strata as supplementary.

- **Q3 — Bernoulli comparator:** for the paired-comparator arm,
  draw Y from Bernoulli at the rater's *fitted* θ (from
  `sdt_fits.csv` — quantifies "how well does the SDT-fit Bernoulli
  predict the actual responses"), or at a *prior-drawn* θ (the
  current Bernoulli sim approach — quantifies "is real-rater
  behaviour worse than a random draw from the deployment prior")?

  Recommended: **fitted-θ Bernoulli** — the SDT fits are the
  reference-correct rater models the deployment was validated
  against; the rater's fitted (σ_k, θ_k) is the closest "synthetic
  twin"; the paired comparison cleanly isolates "real Y vs SDT-fit
  Bernoulli" on identical engine (seg, s) selections.

- **Q4 — sub-sub-step pacing inside 7.3:** the build naturally
  splits into three components — bank-builder, deployment replay
  driver, Mode-A replay driver. Pause for review after each, or
  treat 7.3 as one cohesive commit?

  Recommended: **three sub-sub-steps** with pause after each. The
  bank-builder is load-bearing for both engines; surfacing it
  first lets us validate the per-(rater, task) inventory before
  building the engine wrappers. Matches the established
  sub-step-by-sub-step pattern.

## 8. The build plan (proposed, conditional on Q1-Q4)

- **7.3-A — bank-builder** (`pipeline/replay/build_rater_replay_bank.py`):
  produces per-(rater, task) replay bundles (scored-segs subset of
  `case_bank.csv` + Y lookup table) with the erratum-correct
  task-mapping. Output: `data/replay/rater_replay_bank.parquet` (or
  CSV). Tests gate the inventory + erratum-correctness.

- **7.3-B — deployment replay driver**
  (`deployment/replay/run_deployment_replay.py`): wraps
  `simulate_candidate` with strict-A bank filtering + Y lookup;
  produces per-task + per-candidate PASS/FAIL/REFER + E[n] tables.
  Tests gate the strict-A contract (Y from labels not Bernoulli;
  bank-exhaust → REFER).

- **7.3-C — Mode-A replay driver**
  (`bridge/run_multi_auroc_replay.py`): wraps
  `run_session_mcmc_auroc` with the same strict-A pattern. Mode-A's
  `bank_signals` API requires extending to carry seg_ids — minimal
  surface: add an optional `bank_segids` kwarg that, when present,
  returns `(k, seg_id, s)` from `choose_item` instead of `(k, s)`.

- **7.3-D — paired Bernoulli comparator**: runs each engine on the
  same personal banks with Bernoulli draws at fitted θ for direct
  comparison; persists alongside.

- **7.3-E — close-out doc + headline numbers**.

---

## 9. Sub-step 7.3-A — bank-builder CLOSE-OUT (2026-05-19)

`pipeline/replay/build_rater_replay_bank.py` ships the per-rater
per-task strict-A bank. Outputs (gitignored — regenerable):

  * `data/replay/rater_replay_bank.csv.gz` (102 MB; long-form
    rows of `(rater_id, task, seg_id, y, s_mean, s_sd)` ready
    for the engine drivers in 7.3-B/7.3-C)
  * `data/replay/rater_replay_summary.csv` (801 KB; per-(rater,
    task) summary with `n_segs`, `n_pos`, `expertise_level`)

Build run (single-thread): **27.8 s**. Headline inventory:

| Quantity | Count | Note |
|---|---|---|
| Total (rater, seg, task) observations | 5,502,146 | After erratum-correct mapping + case_bank join |
| Dropped (no s_mean in K=7 bank) | 421,084 (7.1 %) | Segs the rater scored but not in the production frozen bank |
| (rater, task) cells with ≥1 response | 16,705 | per-task universe |
| (rater, task) cells at floor=10 | **14,823** | per-task headline cohort |
| (rater, task) cells at floor=20 | 13,573 | per-task strata |
| Per-candidate cohort at floor=10 | **21 raters** | full-7 headline cohort |
| Per-candidate cohort by-tier | 19 expert + 2 experienced | almost entirely expert |

**Pre-build audit vs post-build:** the pre-build inventory in
`PHASE7_AUDIT.md` §sub-3 reported 22 raters at floor=10 across all
7 tasks; after joining with `case_bank.csv` (= the production K=7
frozen item bank), **21** survive. The drop is honest data hygiene
— one rater had a task on which all scored segs lacked an `s_mean`
signal in the production bank (legitimately filtered, not a bug).

**Per-task cells**: pre-build audit found 14,955 (rater, task)
cells at floor=10; post-build join reports 14,823 — same hygiene
reason, 132-cell drop (0.9 %).

**Gate (`tests/test_phase7_replay_bank.py`, 9 tests, 2.25 s):**

  - `test_outputs_exist`, `test_bank_schema`
  - `test_all_seven_tasks_populated`
  - `test_erratum_correct_iiic_mapping` (no `bipd`/`birds` task)
  - `test_centaur_gold_experts_present` (the 4-expert panel —
    cal/matt/tianyu at exactly 5,000 segs/IIIC task; mbw=97 on
    all 7 tasks)
  - `test_summary_schema`
  - `test_per_task_cohort_at_n_min_10` (≥10,000 cells, per-task
    ≥1,000 raters)
  - `test_per_candidate_cohort_at_n_min_10` (cohort within
    20–23 raters; mostly expert)
  - `test_per_task_y_distributions_sensible` (spike pos rate
    0.3–0.9; each IIIC task 0.05–0.40)

The Centaur 4-expert gold cohort verifies as a strong-typed
sanity check: cal/matt/tianyu each have **exactly** 5,000 segs
on every IIIC task (no spike — Centaur-IIIC is IIIC-only); mbw
has responses across all 7 tasks (general expert pool member).

**Idempotence**: re-running the builder produces byte-identical
output (no nondeterminism in the join). The gitignore policy
treats the bank + summary as regenerable build artifacts (matches
the precedent for Phase-3 `pipeline/_calib_work/`).

---

## 10. Sub-step 7.3-B — deployment replay driver CLOSE-OUT (2026-05-19)

`deployment/replay/run_deployment_replay.py` ships the strict-A
deployment replay driver. The engine attach-point uses **two
minimal-surface optional kwargs** added to
`deployment/simulate_test.py::simulate_candidate` (defaults
byte-identical to pre-edit, gated by
`tests/test_phase7_replay_engine_drift.py`):

  * `y_source: Callable[[k, seg, s_val], int] | None = None`
    — when callable, replaces the default Bernoulli draw; the
    strict-A Y-lookup attach.
  * `initial_decision: list[str] | None = None` — when a length-K_
    list, overrides the default `["pending"] * K_`; lets per-task
    replay pre-mark the 6 non-target tasks as `"refer"`.

### 10.1 The per-task interpretation-(i) fix discovered in smoke

The user-locked per-task interpretation (i) — K=7 engine, 6 non-
target task banks empty — initially produced REFER on every
target task at exactly `n=10`. Root cause (from the smoke trace):
`select_next_case`'s `n_min_per_task` constraint
(`simulate_test.py:276-279`) restricts the candidate set to
under-min tasks when any pending task is under-min. The 6
empty-bank tasks always have `n_per_task=0 < N_min_per_task=10`,
so they're always under-min; the target task hits `n=10` (no
longer under-min) and is excluded from the restricted set. The
engine then tries to select from the 6 empty-bank tasks → returns
`None` → outer loop breaks → target task REFER.

**Fix**: pre-mark the 6 non-target tasks as `"refer"` so they're
never in `pending`. The engine's `select_next_case` then sees only
the target task as pending, gives it up to `N_max_per_task=120`
trials, and reaches a real verdict (PASS / FAIL / REFER).

The `initial_decision` kwarg is the minimal-surface implementation
of this fix — same auditable-edit pattern as `y_source`.

### 10.2 Smoke result

`--limit 5 --out-dir /tmp/replay_smoke2`: 5 per-task cells + 5
per-candidate raters complete in **19 s wall** (single-thread).
Sample (rater 97 = mbw, the general-pool expert with broadest
coverage):

| Task | Decision | n_per_task | pos_rate (real) | n_engine_calls |
|---|---|---|---|---|
| gpd | pass | 14 | 0.643 | 14 |
| grda | pass | 34 | 0.676 | 34 |
| lpd | pass | 41 | 0.683 | 41 |
| lrda | pass | 22 | 0.636 | 22 |
| other | refer | 120 | 0.775 | 120 |

The engine reaches PASS verdicts at variable trial counts driven
by the actual posterior dynamics — *not* the forced `n=10` floor
that interpretation (i) was exhibiting pre-fix.

### 10.3 Test coverage

`tests/test_phase7_replay_engine_drift.py` — 2 tests (5 s):

  - `test_simulate_candidate_default_y_source_is_byte_identical`
    (the SLOW drift-guard contract: default branch behaviour
    unchanged from pre-edit Phase-4)
  - `test_simulate_candidate_y_source_callable_is_consumed`
    (sanity: the new path actually invokes the callable)

`tests/test_phase7_deployment_replay.py` — 8 tests (24 s):

  - `test_y_lookup_returns_recorded_response` — `_RaterYLookup`
    returns labels.csv Y on every recorded (k, seg) for rater 97
  - `test_y_lookup_raises_on_unrecorded_seg` — contract guard
  - `test_bank_for_rater_task_returns_strict_subset` — the
    per-(rater, task) bank exactly equals the rater's scored-seg
    subset with engine-input cols
  - `test_per_task_engine_y_matches_labels` — every Y the engine
    records via state.history matches the rater's labels.csv Y
    for the seg the engine selected (strict-A contract; uses
    call-logging y_source since `state.history` records signal,
    not seg_id)
  - `test_per_task_replay_runs_end_to_end` — representative cell
    produces a valid verdict + sensible accounting
  - `test_per_candidate_replay_runs_end_to_end` — representative
    rater produces decisions on all 7 tasks
  - `test_per_task_replay_other_tasks_skipped` — interpretation
    (i): the 6 non-target tasks receive zero engine calls
  - `test_per_candidate_replay_y_matches_labels` — strict-A
    contract on the full 7-task replay (call-logging method)

### 10.4 What 7.3-B does NOT yet do

- Does not run the full **14,823 per-task + 21 per-candidate**
  cohort headline. At ~3 s/cell single-thread, the per-task
  cohort would be ~12 hours. Parallelization (`parallel_map`
  integration like `run_tier2_oc_simstudy.py`) lands in 7.3-C
  alongside the Bernoulli comparator + Mode-A replay.
- Does not run the **fitted-θ Bernoulli paired comparator** (Q3
  arm). That is 7.3-C's headline.
- Does not drive the Mode-A engine. 7.3-C extends Mode-A's
  `bank_signals` API with the optional `bank_segids` for Y
  lookup.

### 10.5 What 7.3-B DOES ship

The full strict-A deployment replay harness:

  - `simulate_candidate(y_source=..., initial_decision=...)` —
    engine hooks (2 optional kwargs, defaults byte-identical to
    pre-edit; gated by drift-guard)
  - `deployment/replay/run_deployment_replay.py` — CLI driver
    (`--floor`, `--limit`, `--out-dir`, `--seed`); produces
    `deployment_replay_per_task.csv`,
    `deployment_replay_per_candidate.csv`, and
    `deployment_replay_run_summary.json`
  - The strict-A bank construction + Y-lookup callable
    (`_bank_for_rater_task`, `_RaterYLookup`)
  - 10 tests gating the harness end-to-end
  - Documentation: this section + design audit §1–§9
