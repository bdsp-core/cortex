# CLAUDE.md — reviewer-grade context

Authoritative reviewer-facing context for the unified ILAE Skill
Certification repo. **READ THIS BEFORE EDITING.** For end-user install
+ entry-point usage, see `README.md`.

Status: **v1.0.0-rc1 candidate** (Phase 8 packaging + docs in
progress; Phases 0–7 SHIPPED). Suite **282 passed / 1 xfailed**.

## What this repo is

A unified merge of two sibling repos:

  * Methodology repo (SMC + MCMC validated engine; Paper-1 Mode-A:
    Multi-AUROC Precision Protocol).
  * PI deployment repo (Laplace + EKF clinical-deployment runtime;
    `simulate_test.py` + `freeze_deployment_prior.py` + 5 ingest
    scripts).

Merged 2026-05-18 via the plan at `../UNIFIED_REPO_MERGE_PLAN.md` and
9 numbered decisions D1–D9.

## The 9 D-decisions (locked, source of truth)

| # | Decision | Choice | Where applied |
|---|---|---|---|
| D1 | Production runtime | **Two engines, cleanly separated.** SMC+MCMC = Paper-1; Laplace/EKF = deployment. Shared calibration + data layer. | `engine/` and `deployment/` never import each other; both read `data/` + `calibration/` only |
| D2 | ℓ\*/σ\* calibration authority | Recompute unified non-circular CV Youden on PI superset corpus | `pipeline/reference_calibration/` (byte-verbatim reference fitters) + `pipeline/run_unified_calibration.py` orchestrator |
| D3 | Data layout | Single source of truth = PI `data/labels/`; `engine_inputs/` + `deployment_prior/` are derived | `data/DATA_PROVENANCE.md` records the sha256 chain |
| D4 | Ship scope | Deployment + Paper-1 reproducibility + PI methodological variants ported | `engine/variants/` (learn-r / beta-prior / K-scaling / AUROC-native) |
| D5 | Per-candidate verdict policy | **Full-7**: spike + {seizure, lpd, gpd, lrda, grda, other/iic}. K=7, Σ 14×14 | `deployment/freeze_deployment_prior.py:32`, `deployment/simulate_test.py:53`, all use TASKS = 7-list with `other = (pattern_class=='other')` |
| D6 | Real-rater replay vs Bernoulli sim | Real-rater replay required for v1.0 (strict-A constrained-bank) | Phase 7.3 `deployment/replay/` + `pipeline/replay/`; `docs/PHASE7_REPLAY_HEADLINE.md` |
| D7 | ℓ\* independent-panel reproducibility | Required for v1.0 | Phase 3 D7 ordinal-only; see `calibration/CALIBRATION_PROVENANCE.md` |
| D8 | Repo name & history | Squash to clean root commit; provenance in `docs/MERGE_SOURCE_MANIFEST.md` | Done at Phase 0 |
| D9 | EEG source (`SN1_combined_v2.h5`) | Stays external; never vendored | Retrieval path documented in `data/SENSITIVE.md`; in-repo curated banks vendored at Phase 7.4-A as derived signals (not raw EEG) |

## Reference-truth invariants (Phase 6 audit; HARD pins)

Per `docs/INVARIANT_AUDIT.md` — 5 PASS, 4 signed-off deviations.

  1. `LOGIT_TO_PROBIT = 1.0 / 1.7` everywhere `c_j → probit signal`.
     Single definition at `pipeline/reference_calibration/
     fit_sdt_per_domain.py:38`; runtime-asserted in
     `pipeline/run_unified_calibration.py:201`.
  2. `λ = 0.025` fixed in every lapse model. Single source at
     `engine/core.py:21` (`LAPSE_RATE = 0.025`); reference fitter
     uses the same value with variable name `LAMBDA = 0.025`. The
     spike paper's Eq. 2 mixture `λ + (1 − 2λ)·Φ(z)` is consumed by
     the Mode-A SMC, Mode-B engine, deployment runtime, and the
     calibration fitter — **one likelihood definition**.
  3. Two `fit_sdt_per_domain.py` copies stay byte-equivalent (md5
     `6b90d59dcd0aaa878f9a52802254044b`); gated by
     `tests/test_phase6_invariants.py::test_fit_sdt_two_copies_byte_equivalent`.
  4. σ\*/ℓ\* on **TRAIN-only**, no validation leakage:
     `EXPERT_TRAIN_FRAC = 0.70` at
     `pipeline/reference_calibration/youden_sigma_star_ref.py:19`.
  5. **Expert split = 70/30** (NOT 50/50; the legacy CLAUDE.md note
     was stale). Non-expert split = 50/50.

Four signed-off deviations from the merge plan §"Phase 6" checklist
(all documented in `docs/INVARIANT_AUDIT.md` §§6-9):

  * `GRAY_ZONE_DELTA` and `T_TOL` are Mode-B/Paper-2 constants not
    carried into the unified Paper-1 scope (N/A).
  * `EXPERTS` set / `gold_standard_raters.yaml` named non-existent
    sources; **actual** authoritative expert membership is
    `data/labels/raters.csv:expertise_level` (used in
    `pipeline/run_unified_calibration.py:450` +
    `deployment/freeze_deployment_prior.py`).
  * `r_ℓ = 0.326` is the retired K=6 PI-era value; current K=7
    shipped value is **0.36656350316581954** (derived from the frozen
    Σ ℓ-block at `deployment/freeze_deployment_prior.py:126`).

## Phase-by-phase index (where things live)

| Phase | Closed | Key artifacts | Close-out doc |
|---|---|---|---|
| 0 | yes | repo skeleton; `pyproject.toml`; entry-point stubs | `docs/MERGE_SOURCE_MANIFEST.md` |
| 1 | yes | PI corpus adopted; rater_id collision check; data provenance | `data/DATA_PROVENANCE.md` |
| 2 | yes | hardened engine adopted byte-identical; variants ported; FIX-T1.9 IUT resolved (joint default, Berger toggle) | `CHANGELOG.md` "Phase 2 — engine consolidation" |
| 3 | yes | reference-faithful Rasch + per-rater probit-lapse + CV-top-14 two-stage Youden; `cert_config` v13 | `calibration/CALIBRATION_PROVENANCE.md` |
| 3.5 | yes | joint hierarchical s_j unification + engine s_sd propagation; UNCERT restores SBC coverage 0.88→0.93 | `docs/DATA_UNIFICATION_ANALYSIS.md` §8 |
| 4 (.1–.7) | yes | deployment integration; K=7 re-freeze; RNG decoupled; λ-lapse leniency shift signed off; `ilae-deploy` CLI | `docs/DEPLOYMENT_INTEGRATION.md` |
| 5 | yes | `engine_inputs` provenance (D3) | `data/engine_inputs/README.md` + `data/DATA_PROVENANCE.md` §6 |
| 6 | yes | invariant audit | `docs/INVARIANT_AUDIT.md` |
| 7 (.1–.5) | yes | Phase-2 validation K=7 + Tier-2 OC + D6 replay + Paper-1 figures K=7 + Phase-7 gate | `docs/PHASE7_CLOSEOUT.md` (the consolidated scientific gate) |
| 8 | in progress | README + CLAUDE merge + open-decisions doc + clean-env install gate + `v1.0.0-rc1` tag | (this commit + next) |

## Two-engine architecture: what NOT to do

  * **Never** import `engine/` from `deployment/` or vice versa.
    The module boundary IS the D1 architecture; if you find yourself
    wanting to share code, the right place is `engine/core.py` (for
    the likelihood) or a new module in `pipeline/` (for data
    transforms). Both engines have direct copies of `LAPSE_RATE`
    sourcing because of this; that is correct.
  * **Never** add `np.clip(norm.cdf(z), 1e-9, 1-1e-9)` anywhere it
    will be consumed by the certification likelihood. The hardened
    engine uses `scipy.special.log_ndtr` + `logsumexp` everywhere
    (FIX-T0.7). The clipped form collapses to 1.0 at |z|≳6 and
    biases the posteriors of confident raters. If you see clipped
    `norm.cdf` in a variant or a port, replace it with the helpers
    in `engine/core.py` (`_log_p_response` / `_p_response_yes`).
  * **Never** change `LAPSE_RATE` from 0.025 without updating
    Phase-6's invariant audit. The constant is shared across 5 code
    paths; changing it requires a re-run of Phase 2 lapse-equivalence
    (1e-12 tolerance) and Phase 3 calibration recompute.
  * **Never** vendor `SN1_combined_v2.h5` (D9). It stays in the
    external sibling repo. The unified repo's `data/curated_banks/`
    holds *derived* per-segment signal banks (s_probit), not raw EEG.
  * **Never** sweep `trainer_rd/` into a repo-wide invariant scan,
    tree-walk test, or grep/license audit. It is a frozen, verbatim
    R&D subproject (relocated 2026-07-06) that deliberately vendors its
    own copies of the engine, `auroc.py`, and the `*_general` data
    banks; convergence with the production engines is the future port,
    not a defect to flag. Its script-style suite runs from inside the
    directory with the system interpreter, NOT pytest — a `conftest.py`
    guard makes `pytest trainer_rd/` collect 0 items. See
    `trainer_rd/README.md`. (Same compartmentalization contract as
    `methodology_rd/` and `discrimination_rd/`.)

## Engine reproducibility contract

Bit-exact SMC + MCMC reproducibility requires single-thread BLAS:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

These are set in `conftest.py` for pytest, in
`scripts/_parallel.configure_blas_single_thread()` for the parallel
runners, and in worker-pool initializers throughout. Gated by
`tests/test_parallel_determinism.py` (serial == parallel bitwise).

Phase 3.5's NumPyro/JAX path is **calibration-stage-only** —
`engine/core_mcmc.py` runtime never imports JAX. This isolation is
deliberate: JAX uses XLA threading internally and would break the
engine's BLAS contract if it crossed module boundaries.

## Contributor conventions

Established via feedback memory and prior incidents:

  * **Incremental + regression-gated edits.** Strict component-by-
    component port onto PI baseline; gate on PI-output-equivalence +
    the 282-test suite. **No big-bang rebuilds.** This is how Phase
    4.1–4.7 was done and is the documented contributor convention.
  * **Pause for review after each non-trivial step.** Used throughout
    Phases 4, 7, and 8 sub-step pacing; matches the user's stated
    preference.
  * **Reference the actual code.** Memory + state logs can be stale;
    when claims conflict with code, code wins. The Phase-6 invariant
    audit is the worked example — multiple plan/CLAUDE.md notes were
    stale (50/50 split, `EXPERTS` set, r_ℓ=0.326); the actual code at
    the cited lines was the source of truth.
  * **Drift-guard tests for any new kwarg or path.** When sub-7.3-B/C
    added optional kwargs to `deployment/simulate_test.py` and
    `engine/core_mcmc.py`, the defaults were pinned BYTE-IDENTICAL
    by drift-guard tests (`test_phase7_*_drift.py`). Do the same for
    any future engine surface extension.
  * **Document the "what NOT to do"** alongside the "what to do" in
    close-out docs. Reviewer-grade transparency.

## Where to look for what

Same table as `README.md` — kept canonical there. Quick pointers for
common reviewer / contributor questions:

  * Reproducing Paper-1 figures: `docs/PHASE7_PAPER1_FIGURES.md`
  * Reproducing the deployment K=7 simulation: `ilae-deploy all`
  * Reproducing the real-rater replay headline: see
    `docs/PHASE7_REPLAY_HEADLINE.md` "Outputs" section + the CLI
    invocations therein.
  * Re-running calibration on a new corpus: `ilae-calibrate` (Phase 3
    orchestrator) — see `calibration/CALIBRATION_PROVENANCE.md` for
    inputs/outputs/sha256 chain.

## Open decisions (v1.0 ship review)

See `docs/OPEN_DECISIONS.md`. The five open items from the merge plan
§"Phase 8" sub-4 (per-candidate roll-up policy, real-rater replay
choice, ℓ\* independent-panel reproducibility, bias-warning channel,
split-half reliability + external-cohort validation) are recorded
there with current dispositions; per-task certificates (no single
roll-up) is the v1.0 working policy.
