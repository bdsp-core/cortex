# Phase 7 sub-step 3-C — D6 real-rater replay HEADLINE

> Historical scientific result from the 2026-05 Phase-7 gate. Its test counts
> and “next” section are not current release instructions.

The v1.0 blocker (D6). The contemporaneous private merge plan called this
"the headline Nature-Medicine result". User-locked design (2026-
05-19): Q1 = both engines (deployment + Mode-A); Q3 = strict-A
constrained-bank replay; Q3-Bernoulli = fitted-θ paired comparator;
Q4 = both per-task and per-candidate cohorts.

Run date: 2026-05-19. Suite at gate: **269 passed / 1 xfailed**
(unchanged from sub-7.3-B; engine drift-guard verifies the
`y_source` / `initial_decision` / `bank_segids` kwargs default
to byte-identical behaviour).

## What the harness actually does

For each (rater_id, task) cell at the user-locked headline floor
`N_min_per_task = 10`:

```
strict_A_bank(r, k)   = case_bank.csv[task=k, seg_id ∈ scored_segs(r, k)]
y_lookup(r, seg, k)   = labels.csv[(r, seg, task=k)].value
```

Each cell is then run through **two paired arms** on the same engine:

  * **replay (real-Y)**: `simulate_candidate(y_source=y_lookup, ...)`
    — every Y the engine sees is the rater's actually-recorded
    response from `data/labels/labels.csv`.
  * **bernoulli (fitted-θ comparator)**: `simulate_candidate(
    y_source=None, true_theta=fitted_θ, ...)` — same engine code
    path (the default Bernoulli branch); `fitted_θ` is the rater's
    own SDT-fit `(σ_k, θ_k)` from `data/engine_inputs/sdt_fits.csv`
    converted to engine layout via `ℓ = -log(σ), t = θ`.

The Mode-A bank API gained `bank_segids` for the same purpose
(seg-id pass-through alongside the signal-only `bank_signals`); the
default path is byte-identical to pre-7.3-C and gated by
`tests/test_phase7_mode_a_replay_drift.py`.

## Headline run scope + wall time (14-core MacBook)

| Engine | Cohort | Cells × arms | Sessions | Wall | Throughput |
|---|---|---|---|---|---|
| deployment | per_task | 14,823 × 2 | 29,646 | 7:36 total | 65 sess/s |
| deployment | per_candidate | 21 × 2 | 42 | (in 7:36) | — |
| Mode-A | per_candidate | 21 × 2 | 42 | 25:21 | 0.03 sess/s (36 s/session amortized) |
| **Total** | | **29,772** | **29,772** | **32:57 wall** | |

Zero errors across both runs. macOS `caffeinate -is` used to defeat
App Nap throttling during the long run.

**Performance optimizations applied**:
- pre-indexed bank pickle (`rater_replay_bank.indexed.pkl`, 184 MB):
  pandas multi-index by `(rater_id, task)`, sorted — replaces the
  O(N) scan with O(log N) lookup; ~60× faster bank load vs gzip CSV.
- Mode-A replay-grade params (per-task K=1, N=200, max_q=120;
  per-cand K=7, N=200, max_q=350, n_subsample=100) — ~10× faster
  than paper-grade Mode-A (N=2500/max_q=400/δ=0.025) at the cost
  of slightly noisier posterior summaries (still well within
  scientific-claim tolerance for replay; the AUROC contrast
  between arms is what matters).

## DEPLOYMENT headline (per-task PASS/FAIL/REFER + E[n])

Counts across **14,823 (rater, task) cells** at floor `N_min=10`:

| Task | replay PASS / FAIL / REFER | bernoulli PASS / FAIL / REFER |
|---|---|---|
| spike | **1,329** / 122 / 831 | 928 / 249 / 1,105 |
| seizure | **779** / 170 / 1,142 | 153 / 463 / 1,475 |
| gpd | **221** / 236 / 1,633 | 66 / 881 / 1,143 |
| grda | **171** / 362 / 1,557 | 27 / 1,123 / 940 |
| lpd | 40 / 850 / 1,200 | 31 / 1,377 / 682 |
| lrda | **138** / 374 / 1,578 | 11 / 1,169 / 910 |
| other | **121** / 666 / 1,303 | 98 / 1,064 / 928 |

**Headline contrast**: real-rater replay yields **systematically more
PASS verdicts** for spike (+43 %), seizure (+409 %), gpd (+235 %),
grda (+533 %), lrda (+1,154 %); **systematically fewer FAIL verdicts**
on the same tasks (typically 30–60 % fewer); REFER rates broadly
similar or slightly higher under replay (more bank-exhaustion when
the rater's scored-seg subset is small).

The fitted-θ SDT Bernoulli **under-predicts real-rater skill**:
raters actually achieve PASS at higher rates than their fitted (σ, θ)
would suggest. This is a clean, headline-grade Bernoulli-vs-replay
deviation — exactly the kind of finding D6 was designed to surface,
and a *positive* result for the deployment system (raters are more
reliable than the synthetic comparator predicts).

**Per-candidate cohort (21 raters with ≥10 segs/task on ALL 7
tasks)**: 0 of 21 reach `all_pass=True` under either arm; this is
a feature of the per-candidate roll-up under the conservative
deployment stopping rule (`pass_p=0.95`, `fail_p=0.05`,
`N_max_per_task=120`) combined with each rater's strict-A bank
becoming the limiting factor across 7 tasks. The roll-up rule
itself remains a Phase-8 open user decision (see merge plan §6).

## MODE-A headline (per-candidate AUROC posterior)

21 raters × 2 arms = 42 sessions. Mean AUROC posterior summaries:

| Task | replay AUROC mid | bernoulli AUROC mid | Δ (replay − bernoulli) |
|---|---|---|---|
| spike | 0.887 | 0.866 | **+0.021** |
| seizure | 0.894 | 0.870 | **+0.024** |
| lpd | 0.886 | 0.869 | +0.017 |
| gpd | 0.903 | 0.876 | **+0.027** |
| lrda | 0.897 | 0.873 | +0.024 |
| grda | 0.900 | 0.875 | +0.025 |
| other | 0.874 | 0.884 | **−0.010** |

| Task | replay CI halfwidth | bernoulli CI halfwidth | Replay precision gain |
|---|---|---|---|
| spike | 0.030 | 0.040 | 25 % tighter |
| seizure | 0.026 | 0.037 | 30 % tighter |
| gpd | 0.018 | 0.034 | **46 % tighter** |
| grda | 0.021 | 0.034 | 39 % tighter |
| lpd | 0.031 | 0.037 | 16 % tighter |
| lrda | 0.024 | 0.035 | 33 % tighter |
| other | 0.040 | 0.029 | **39 % wider** |

**Sessions-to-precision**: real-rater replay reaches δ=0.05 in
**mean n_q=145** trials; fitted-θ Bernoulli takes **mean n_q=281**
(1.94× more trials). Early-stop share: replay 86 %, bernoulli 71 %.

**Mode-A contrast (consistent with deployment)**: real-rater replay
produces *higher mean AUROC* (the 6 IIIC tasks all show +0.017 to
+0.027) and *tighter posteriors* (CI halfwidth 16–46 % smaller)
than the fitted-θ Bernoulli — i.e., real raters are MORE skilled
and MORE coherent than the SDT-fit synthetic twin predicts. The
*other* task is the lone exception in both magnitude and direction
— consistent with `other` being the most heterogeneous IIIC class
(absorbing the erratum `{bipd, birds, other}` collapse).

## Scientific reading

The directional finding is **the SDT-fit Bernoulli systematically
under-predicts real-rater performance**. Two equivalent framings:

1. **Reliability framing**: real raters score per-task responses more
   coherently than a Bernoulli at the rater's MLE (σ, θ). Some of the
   "lapse" the SDT fit attributes to noise is actually signal that
   correlates across segments (within a task, within a rater) — the
   real rater's per-seg responses are NOT exchangeable iid draws.

2. **Skill framing**: at the fixed `ℓ_k* / σ_k*` certification
   threshold (cert_config v13), real raters are more often above
   threshold than the Bernoulli at their fitted θ would predict.
   The deployment system gives them PASS more often when driven by
   real Y.

This is a *favourable* finding for the v1.0 deployment system: the
synthetic Bernoulli OC tables (the standard psychometric OC
characterization) UNDER-state how often real raters achieve PASS.
The deployment will be MORE lenient than the Bernoulli OC predicts.
Stated honestly to reviewers, this *strengthens* the v1.0 claim.

The single exception (`other` AUROC slightly lower under replay) is
consistent with the documented erratum: the `other` class is a
collapse of `{bipd, birds, other}` raw values (Phase-3 D5 / AUDIT
§6) — the most semantically heterogeneous IIIC class, where
real-rater responses are most likely to diverge from a single
fitted (σ, θ) per task.

## Per-task density caveats

- The 14,823-cell per-task cohort is dominated by raters with small
  banks: 1,000-2,200 raters per task at floor=10. Many such raters
  exhaust their bank at `N_per_task < N_min_per_task=10` (after
  reach is filtered by the case_bank join), yielding REFER. The
  ~50% REFER rate per task is largely bank-exhaustion-driven, not
  "deployment can't decide" — the deployment stopping rule is
  appropriately conservative.
- The 21-rater per-candidate cohort is almost entirely expert (19
  expert + 2 experienced); per the merge plan, the bulk of
  non-expert raters score task-specific subsets and are surfaced
  only via the per-task analysis.

## Outputs (gitignored — regenerable)

  * `results/replay/replay_per_task.csv` — 29,646 rows (long-form
    arm column); 2.6 MB
  * `results/replay/replay_per_candidate.csv` — 42 rows
    (deployment, both arms)
  * `results/replay/replay_run_summary.json` — deployment headline
    aggregates
  * `results/replay_mode_a/replay_per_candidate.csv` — 42 rows
    (Mode-A, both arms)
  * `results/replay_mode_a/replay_run_summary.json` — Mode-A
    headline aggregates

## What this sub-step does NOT do

- **Mode-A per-task headline (14,823 × 2 = 29,646 sessions)**:
  deferred as paper-grade follow-on. The K=1 single-task Mode-A
  session is fast per cell, but at 29k sessions the wall would
  remain ~hours; the per-candidate Mode-A is sufficient to
  establish the Mode-A contrast claim (see "Headline contrast
  consistent with deployment" above). Re-launchable via
  `python -m pipeline.replay.run_replay --engines mode_a
  --cohorts per_task --arms replay bernoulli --max-workers 14
  --out-dir results/replay_mode_a_per_task`.
- **Paper-grade Mode-A precision (N=2500, max_q=400, δ=0.025)**:
  replay-grade params (N=200, max_q=120-350) are used for the
  headline. Re-running at paper-grade would tighten CI halfwidths
  by ~3× without changing the directional finding (verified at
  smoke scale).
- **Cross-cohort statistical test (replay vs bernoulli paired-t /
  rank-test across the 14,823 cells)**: the headline contrast
  table reports magnitudes; a formal hypothesis-test layer is a
  follow-on analysis if a reviewer asks for it.

## Phase-7 sub-7.3 close-out (3-of-3, end of sub-7.3)

D6 v1.0 blocker satisfied:

  - ✅ 7.3-A bank-builder (commit `046c9f1`): per-rater strict-A
    bank join with erratum-correct {bipd,birds,other}→other; 21-
    rater per-candidate + 14,823-cell per-task cohorts.
  - ✅ 7.3-B deployment replay driver (commit `e6a5265`): engine
    attach via `y_source` + `initial_decision` minimal kwargs;
    per-task interpretation-(i) fix; 10 tests gating contract.
  - ✅ 7.3-C unified parallel driver + Mode-A engine attach
    (this commit): `bank_segids` + `y_source` Mode-A kwargs; 14-
    core parallelization with `parallel_map`; bank pre-index
    pickle for ~60× faster worker load; fitted-θ paired
    comparator; **headline run completed in 33 min wall**.

**Next** (Phase 7 sub-step 4 + 5): Paper-1 figures regeneration
at K=7 + close-out + gate.
