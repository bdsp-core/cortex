# Phase 7 — Audit of plan vs reality (pre-execution)

`UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7 — Validation re-run (the scientific
gate; incl. D6 replay)" has 5 sub-steps. This audit traces each to actual
code/data in the unified repo and flags the decisions/builds required.

Audit date: 2026-05-19. Reference suite state at start: **240 passed /
1 xfailed** (Phase 0–6 complete).

## Sub-step 1 — Re-run the Phase-2 validation suite at K=7

Plan: SBC, posterior coverage, lapse equivalence 1e-12, sigma sensitivity,
sparcnet test-retest, gold-chain on the unified corpus + v13 calibration,
at K=7.

**What exists in the unified repo (carried byte-identical from methodology
in Phase 2):**

| Script | Sub-step | K-handling |
|---|---|---|
| `scripts/run_sbc_validation.py` | SBC | K-parameter (synthetic) |
| `scripts/run_coverage_validation.py` | posterior coverage | K-parameter (synthetic) |
| `scripts/run_coverage_sweep.py` | coverage sweep | K-parameter |
| `scripts/run_lapse_sensitivity.py` | lapse equivalence | K-parameter (synthetic) |
| `scripts/run_sigma_sensitivity.py` | σ sensitivity | K-parameter |
| `scripts/run_sparcnet_test_retest.py` | SPARCNET test-retest | per-domain (K-agnostic by iteration) |
| `scripts/run_gold_chain_reference.py` | gold-chain | task["K"] from task spec; default fallback K=6 |
| `tests/test_sbc_skeleton.py` | SBC (fast skeleton, slow-marked) | K-parameter |
| `tests/test_posterior_coverage.py` | coverage (skeleton, slow-marked) | K-parameter |
| `tests/test_lapse_rate.py` | lapse algebra pin | K-independent |
| `tests/test_phase35_sbc_engine.py` | engine SBC under s_sd propagation | K-parameter |

**Reading the actual code:** the Phase-2 validation drivers are
*K-agnostic synthetic-data tests* that take `K` as a parameter,
construct `t_true`/`l_true` of length `K`, simulate histories, and
compare engine output. The methodology scripts can be re-invoked at
`K=7` by passing K to the harness.

**Phase 7 sub-1 scope decision (open question — see §Q1 below):**

- (α) **Synthetic-K=7**: re-run the carried K-agnostic scripts with
  `K=7` as a parameter. Re-derives the headline pins (ICC≥0.70 6/6 for
  test-retest; 35/36 gold-chain agreement) at K=7. The harnesses are
  small enough to invoke at K=7 with the same per-domain inputs (the
  6 IIIC SPARCNET fits are unchanged; the 7th "other" domain consumes
  `sparcnet_iic` for which a fit *does* exist).
- (β) **Production-artifact at K=7**: SBC/coverage of the *engine
  driven by the K=7 frozen `Sigma_prior` and v13 ℓ\*/σ\**. This
  exercises the operational system, not just the synthetic engine.
  Already partially gated (`tests/test_phase35_sbc_engine.py` runs SBC
  *under s_sd propagation*; passes — engine soundness gated). Production
  artifact gating would mostly add documentation, not new tests.

The risk register R6 explicitly asks "gate on **K=7 SBC/coverage** in
Phase 7" — that points at (α). (β) is documentation.

## Sub-step 2 — Tier-2 OC simulator (`validate.py`) for both engines

Plan: re-run Tier-2 OC simulator for both engines; regenerate operating
characteristic tables.

**Reality:**

- There is no `validate.py` in this repo. **`scripts/validate.py` does
  not exist.**
- The methodology repo's closest match is
  `/Users/elikeldsen/Documents/Research/spike-test-project/ilae-skill-certification-test-multi-main/scripts/run_phase4_simstudy.py` —
  "F4.2 Synthetic ℓ-grid simulation study (Paper-1 Results centerpiece)".
  It is a *synthetic ℓ-grid OC surface* over `ℓ ∈ {-1.5, …, 1.0}`,
  `K ∈ {2,4,6,8}`, Σ_l ∈ {empirical, independent, cs0.7}, methods
  ∈ {random, brute, hier}.
- It was **NOT carried into the unified repo at Phase 2** — Phase 2's
  port scope was deliberately minimal (the validation scripts ported
  were SBC/coverage/lapse/sigma/sparcnet-test-retest/gold-chain;
  `run_phase4_simstudy.py` was not among them).
- Mode-B OC tests (`tests/mode_b/test_oc_*.py`) exist and gate the
  Mode-B binary cert OC, including the one documented xfail
  (`test_oc_borderline_pass_rate`). Mode-A OC ≡ the Tier-2 simulator
  needs to be brought across.

**Phase 7 sub-2 scope decision** (see §Q2 below): port
`run_phase4_simstudy.py` into `scripts/`? It is a K-agnostic synthetic
study and our hardened engine is byte-identical to the methodology one
(Phase 2). Concretely:

- Single-file faithful port (path-only edits like Phase 4.4-B), gated
  on output equivalence at K=6 (the methodology-validated config).
- Re-run at K=7 + the operating point.
- Persist OC tables under `results/phase2_validation/`.

## Sub-step 3 — D6 real-rater replay (v1.0 blocker)

Plan: "build/validate a replay harness that drives `simulate_test.py`
(and the Mode-A engine) with held-out *real* rater response sequences
(seed: PI `data/deployment_prior/deployment_replay.csv` + `sim/`), not
Bernoulli draws."

**CRITICAL FINDING — `deployment_replay.csv` is NOT a real-rater
trajectory source.**

Inspecting the file:

```
$ head -3 data/deployment_prior/deployment_replay.csv
tier,task,decision,n_per_task,ell_hat,ell_true,total_trials,candidate_id

shape: (480, 8)
80 candidates × 6 tasks (K=6 PI era)
tier ∈ {expert, experienced, crowd, novice}
decision ∈ {pass, fail, refer}
ell_hat = posterior estimate (sim output)
ell_true = prior draw (sim input/output)
```

This is a **PER-CANDIDATE Bernoulli-simulation OUTPUT** — `ell_hat`/
`ell_true` are simulated latent skills; `decision` is the simulator's
per-task verdict; tiers are the SIM tier labels. It is the
`deployment_prior/sim/candidates.csv` content reshaped per
(candidate, task), **not held-out real-rater response sequences.**
The plan's wording is misleading; both `deployment_replay.csv` and
`sim/` are Bernoulli-simulator outputs.

**No "replay" code or `RealRater` harness exists** in the unified
repo (grep verified). D6 is a genuine *build*, not a *port*.

**The actual real-rater observation source** = `data/labels/labels.csv`
(2,115,793 rows; 5,246 unique raters; columns
`seg_id, rater_id, label_type, value, source_dataset`). With the
erratum-correct `{bipd,birds,other}→other` mapping, real `(rater_id,
seg_id, task) → y ∈ {0, 1}` triples are derivable for all 7 Mode-A
tasks: `spike` from `label_type='spike'`; the 6 IIIC tasks from
`label_type='pattern_class'` with `value==task` (1) vs other (0).

**Per-rater per-task density inventory** (the v1.0 feasibility floor):

| n/task threshold (ALL 7 tasks) | # raters meeting | tier breakdown |
|---|---|---|
| ≥20 | 21 | expert=19, experienced=2 |
| ≥50 | 21 | expert=19, experienced=2 |
| ≥100 | 20 | expert=18, experienced=2 |
| ≥200 | 20 | expert=18, experienced=2 |

⇒ **Only ~20 raters have full-7 density**, almost entirely
expert-tier. Crowd/novice raters scored task-specific subsets (a
novice rated spike on Centaur but no IIIC, etc.); they exist but
not as full-7 candidates.

Per-task bank size (distinct seg_id, post-erratum): spike 19,332;
each of the 6 IIIC tasks 69,806. Plenty of bank, scarce full-7
raters.

**Replay-harness design decision** (see §Q3 below):

- **Design A — strict constrained-bank replay**: the engine's
  `select_next_case` operates on `bank_by_task[k]` restricted to the
  rater's *scored-segs subset* for task k; Y comes from `labels.csv`;
  if the rater-subset exhausts before a decision is reached → REFER.
  This is what "real-rater replay" usually means in the literature.
  Most rigorous; cleanest headline; most defensible to reviewers.
  Constrains the bank tightly (rater-specific banks ≈ tens to hundreds
  per task).
- **Design B — hybrid shadow replay**: engine picks EV-optimal from
  FULL bank; if the rater scored that seg, use real Y; else fall back
  to Bernoulli draw at the rater's *fitted* latent θ from `sdt_fits.csv`.
  Maximizes bank exploration but muddies the headline ("real-rater
  replay supplemented by Bernoulli for unseen items").

(Recommended: A — D6 was framed in the plan as **the** scientific gate
for v1.0; the headline claim must be unambiguously real-rater.)

**Coverage policy** (see §Q4 below): real-rater replay can be run

- **(i) per-task** for every (rater, task) with ≥N_min responses on
  that task — maximizes data, reports per-task PASS/FAIL/REFER +
  E[n] vs the per-task Bernoulli OC; **does NOT roll up to candidate**;
- **(ii) per-candidate** restricted to the ~20 raters with ≥N_min
  on ALL 7 tasks — produces the candidate-level OC table the plan
  describes;
- **(iii) both** — separable analyses, separately documented.

(Recommended: iii — the per-task analysis is the broad coverage gate,
the per-candidate analysis is the headline real-rater claim. Both are
defensible; both are separately valid.)

## Sub-step 4 — Regenerate Paper-1 + PI deployment figures

Plan: regenerate `results/phase1_figures`, `phase2_validation`, and
`deployment_prior/figures` from the unified pipeline at K=7.

**Reality:**

- `data/deployment_prior/figures/` — **already regenerated at Phase 4.7**
  (5 figures at K=7: fig1_concept, fig2_single_candidate,
  fig3_skill_by_tier, fig4_recovery, fig5_verdicts). Sub-step 4
  deployment-figures part is **DONE**.
- `results/phase1_figures/` — **does not exist**. The script that
  produces it (`scripts/run_phase1_experiments.py`) is carried; the
  output dir is to be created at run time.
- `results/phase2_validation/` — sparcnet_test_retest.csv/.md
  outputs go here per the docstring; not yet generated in the unified
  repo.

**Phase 7 sub-4 scope**: run carried producers
(`run_phase1_experiments.py` and the validation harnesses) at K=7 (or
their natural K), persist outputs, generate figures via
`plot_phase1_figure.py` / `plot_validation_figure.py`.

## Sub-step 5 — Gate

Plan: coverage within target band; real-rater replay OC computed and
within pre-registered bounds; no regression vs the last validated run
OR every delta explained by intended corpus/likelihood/calibration/K=7
changes and signed off.

**Reality:** this is the close-out aggregation. Pre-registered bounds
are not yet documented for the real-rater replay; will be set as part
of sub-step 3 (Phase-7.3 build) before running the headline analysis.

## Proposed sub-step decomposition (incremental, regression-gated)

Per the established merge pattern
([[feedback_incremental_regression_gated]]):

- **7.1** — Phase-2 validation suite at K=7 (per §Q1 scope)
- **7.2** — Tier-2 OC simulator port + K=7 re-run (per §Q2)
- **7.3** — D6 real-rater replay harness BUILD (per §Q3, §Q4) — the
  v1.0 blocker
- **7.4** — Paper-1 figures regeneration at K=7
- **7.5** — close-out + gate

Each sub-step: dedicated tests, full-suite gate, commit, pause for
review (the established pattern from Phases 3–6).

## Open questions (user input required before Phase-7 execution)

- **Q1 (sub-1 scope):** synthetic K=7 (α), production-artifact K=7
  (β), or both?
- **Q2 (sub-2 Tier-2 OC):** port `run_phase4_simstudy.py` from sibling?
- **Q3 (sub-3 replay design):** A (strict constrained-bank), or
  B (hybrid shadow)?
- **Q4 (sub-3 replay coverage):** per-task (i), per-candidate (ii),
  or both (iii)?
- **Q5 (overall pace):** proceed sub-step by sub-step (pause for review
  after each), or batch 7.1+7.2+7.4 (the lighter "re-run carried code"
  trio) and reserve a separate sit-down for 7.3 (the real build)?
