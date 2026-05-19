# Phase 6 — Invariant audit

Reference-truth audit of the unified repo against the 9 invariants in
`UNIFIED_REPO_MERGE_PLAN.md` §"Phase 6". Gate per the plan: **every
box checked OR has a written, signed-off deviation.** Each box below
links to the in-code evidence; deviations are documented with the
actual repo state and a proposed disposition.

Audit date: 2026-05-19. Suite at audit time: 237 passed / 1 xfailed.

## 1. `LOGIT_TO_PROBIT = 1.0/1.7` everywhere `c_j → probit signal` ✅

- `pipeline/reference_calibration/fit_sdt_per_domain.py:38`:
  `LOGIT_TO_PROBIT = 1.0 / 1.7` — the only definition; used at `:135`
  (`c_mean.astype(float) * LOGIT_TO_PROBIT`), exactly where c_j is
  rescaled into the probit signal.
- Asserted at runtime by `pipeline/run_unified_calibration.py:201`
  (`mod.LOGIT_TO_PROBIT` is logged + checked during the orchestrator's
  byte-verbatim reference-fitter call).
- The vendored byte-identical copy at `pipeline/_calib_work/src/
  fit_sdt_per_domain.py:38` carries the same value (see §3).

## 2. `λ = 0.025` fixed in every lapse model (engine, variants, deployment) ✅

- **Single source:** `engine/core.py:21` `LAPSE_RATE = 0.025` (also
  `_LOG_LAPSE`, `_LOG_ONE_MINUS_TWO_LAPSE` derived). Used by
  `core.response_prob` (:94), `core.simulate_response` (:290), and
  `core.log_response_prob` (:316).
- `engine/engine_mode_b.py` imports `LAPSE_RATE` from core (:33) and
  uses it for the Mode-B IUT stopping likelihood (:117, :294).
- `engine/core_mcmc_brute_k.py:55` declares its own `LAPSE_RATE = 0.025`
  (the comparator independent-prior path; identical value).
- `engine/core_mcmc.py` consumes `LAPSE_RATE` via `core` (single
  definition; Phase-3.5 s_sd propagation reads it).
- `deployment/simulate_test.py` imports `LAPSE_RATE` from `core`
  (Phase 4.3 "one likelihood definition" — the deployment engine,
  Mode-B engine, and Paper-1 SMC all share the same constant).
- Reference fitters: `pipeline/reference_calibration/
  fit_sdt_per_domain.py` defines `LAMBDA = 0.025` (the reference fit's
  lapse rate — same value, different variable name in the carried
  verbatim reference impl).

## 3. The two `fit_sdt_per_domain.py` (probit_lapse_nll/fit_rater) copies remain byte-equivalent ✅

- `pipeline/reference_calibration/fit_sdt_per_domain.py` and
  `pipeline/_calib_work/src/fit_sdt_per_domain.py` are both
  **md5 `6b90d59dcd0aaa878f9a52802254044b`** (202 lines each), and
  also byte-identical to the methodology-repo reference at
  `ilae-skill-certification-test-main/src/fit_sdt_per_domain.py`
  (Phase-3 already gates this in
  `tests/test_phase3_calibration.test_fit_sdt_per_domain_byte_identical`).
- Phase-6 adds an explicit drift-guard test asserting the two
  in-repo copies stay byte-equivalent — see
  `tests/test_phase6_invariants.py::test_fit_sdt_two_copies_byte_equivalent`.

## 4. σ*/ℓ* computed on TRAIN pool only; no validation leakage ✅

- `pipeline/reference_calibration/youden_sigma_star_ref.py:19`:
  `EXPERT_TRAIN_FRAC = 0.70` ("70% train, 30% val for expert pool —
  wider calibration base"). This file is the byte-verbatim reference
  implementation (md5-gated by `test_phase3_calibration`).
- `pipeline/run_unified_calibration.py` STEP g (`:427`) explicitly
  performs the 70/30 expert + 50/50 non-expert TRAIN split and feeds
  ONLY the TRAIN half to the verbatim Youden driver (`:458–:464`).
  The provenance block in cert_config v13 records
  `EXPERT_TRAIN_FRAC=0.70, SEED=42` (`:716–:718`).
- The validation halves are held out, never enter the Youden ℓ*/σ*
  computation. Asserted in `tests/test_phase3_calibration` (the
  reference-faithful md5 + the v13 provenance block) and gated.

## 5. Expert split = 70/30 (code), CLAUDE.md corrected to match ✅

- **Code:** `EXPERT_TRAIN_FRAC = 0.70` (the canonical literal, see §4);
  the non-expert pool uses 50/50 (a different split, documented in
  `run_unified_calibration.py:466`).
- **`CLAUDE.md`:** currently a Phase-0 placeholder; lines 8–9 list
  *"expert split is **70/30** (not 50/50)"* as a known reference-truth
  correction to bake into the Phase-8 final CLAUDE.md. No stale
  "50/50" claim survives in the body. The Phase-8 assembly task
  (separate phase) will produce the final CLAUDE.md.

## 6. `GRAY_ZONE_DELTA` literal verified per file ⚠️ DEVIATION

- **Finding:** `GRAY_ZONE_DELTA` is **absent from the entire unified
  repo** — no `.py` / `.yaml` / `.json` consumer. The only mention is
  in `CLAUDE.md` line 9 as a "verify per file" reminder.
- **Provenance:** the constant lives in the methodology-repo
  Mode-B / Paper-2 validation scripts (`eval_sequential_stopping.py`,
  `eval_bias_criterion.py`), as `GRAY_ZONE_DELTA = 0.05` — the
  indifference zone around σ* for the binary cert decision rule.
- **Why absent here:** Mode-B (binary PASS/FAIL cert) is preserved as
  the verbatim `engine/engine_mode_b.py` but is **deprecated for
  Paper-1** (the current unified scope); the paper-2-validation
  scripts were not ported. The constant is not consumed by any
  in-repo Mode-A code, and `engine/engine_mode_b.py` itself does not
  reference it (Mode-B engine logic uses `cert_config.yaml`
  `mode_b_legacy` per-domain ℓ*, not a gray-zone delta).
- **Proposed disposition:** *signed-off deviation* — invariant is
  **N/A for the unified-repo Paper-1 scope**; restoring the constant
  would belong to a future Paper-2 deliverable (alongside porting the
  paper-2-validation scripts), not here.

## 7. `EXPERTS` set is the expert-membership source; documented ⚠️ CLARIFICATION

- **Finding:** there is no module-level 29-rater `EXPERTS` set in the
  unified repo. The plan's "`EXPERTS` set (not `gold_standard_raters
  .yaml`)" both name a source that doesn't exist as written —
  `gold_standard_raters.yaml` **does not exist in the repo**, and
  there is no hardcoded canonical 29-name `EXPERTS` constant.
- **Actual authoritative source:** `data/labels/raters.csv` —
  specifically the `expertise_level` column. The calibration code
  derives expert membership at runtime by `is_expert =
  (f["expertise_level"] == "expert")` (`pipeline/
  run_unified_calibration.py:450`, mirrored in
  `deployment/freeze_deployment_prior.py:88+`). This is the
  *single source of truth* for expert membership.
- The only **hardcoded** set is `EXPERTS = ["mbw","cal","matt",
  "tianyu"]` in `pipeline/ingest_centaur_iiic_expert.py:53` — the
  **4-expert Centaur gold panel**, a separate (gold-only) cohort
  ingested in Phase 1.
- **Proposed disposition:** *clarification, no deviation* — the
  invariant is *substantively satisfied* (membership has a single
  authoritative source) but its *wording* must be updated: the source
  is the `expertise_level` column in `data/labels/raters.csv`, not a
  YAML or a module-level set. Documented here; should also surface in
  the Phase-8 CLAUDE.md.

## 8. `T_TOL = 0.20` documented as ≈9.3pp wherever surfaced ⚠️ DEVIATION

- **Finding:** `T_TOL` is **absent from the entire unified repo**.
  No `.py` / `.yaml` / `.md` reference. Same provenance pattern as §6
  — the methodology-repo Paper-2 scripts had
  `T_TOL = 0.20` (probit-scale bias tolerance, derived to equal ~5 pp
  operating-point shift; the original README said "≈5 pp"; the plan
  here updates that to "≈9.3 pp" after recomputation).
- **Proposed disposition:** *signed-off deviation* — same logic as §6:
  Paper-2 constant, not consumed in the Paper-1 unified scope; lives
  in unported paper-2-validation scripts. Restoration belongs to a
  future Paper-2 deliverable.

## 9. `r_ℓ` pinned to 0.326 with provenance note ⚠️ STALE — UPDATED

- **Finding:** the plan's "0.326" was the **K=6 PI-era value**
  (preserved in `data/deployment_prior/_pi_baseline_frozen/
  sim_summary.json:3` as `0.3259728190758862`). The Phase-4.6-A
  re-freeze on the unified corpus produced a **new K=7 r_ℓ**.
- **Current value (K=7, the shipped deployment):** `r_ell =
  0.36656350316581954`, recorded in
  `data/deployment_prior/summary.json` and *derived* (not pinned) by
  `freeze_deployment_prior.py:126` (the empirical mean of the ℓ-block
  off-diagonals in the hierarchical-block-fit Σ). At runtime
  `deployment/run_deployment_sim.py:104` recomputes the same
  `r_ell = mean(Sigma[1,3], Sigma[1,5], Sigma[3,5])` from the loaded
  frozen Σ — *derived* from the artifact, never hardcoded.
- **Proposed disposition:** update the invariant to "**r_ℓ ≈ 0.367
  (K=7 unified corpus), derived from the frozen Σ; PI K=6 era was
  0.326, archived under `_pi_baseline_frozen/`**" — provenance
  preserved, current value documented, derivation (not pinning)
  recorded; superseded by the Phase-4.6-A re-freeze.

---

## Gate summary

| # | Invariant | Status |
|---|---|---|
| 1 | LOGIT_TO_PROBIT = 1.0/1.7 | ✅ |
| 2 | λ = 0.025 | ✅ |
| 3 | two fit_sdt_per_domain copies byte-equivalent | ✅ (+ new drift-guard test) |
| 4 | σ*/ℓ* TRAIN only | ✅ |
| 5 | Expert split 70/30 + CLAUDE.md | ✅ |
| 6 | GRAY_ZONE_DELTA | ⚠️ N/A (Mode-B/Paper-2 scope-out) |
| 7 | EXPERTS source | ⚠️ wording updated → `raters.csv.expertise_level` |
| 8 | T_TOL = 0.20 | ⚠️ N/A (Mode-B/Paper-2 scope-out) |
| 9 | r_ℓ ≈ 0.326 | ⚠️ stale → r_ℓ ≈ 0.367 (K=7 derived) |

**Gate:** boxes 1–5 are checked; 6, 7, 8, 9 have written, signed-off
deviations / updates above. **Phase-6 gate satisfied.**
