# Phase 7 sub-step 1 — Phase-2 validation suite at K=7

Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7" sub-step 1, scope α
(synthetic K=7 only, user-confirmed 2026-05-19): re-validate the
engine soundness pins at the production deployment dim **K=7** =
6 IIIC + `combined_spike` + `other`.

Audit date: 2026-05-19. Suite at gate: **243 passed / 1 xfailed**
(+3 from Phase-6's 240).

## Why synthetic-only is the right scope

The engine code paths under test (`core_mcmc.sample_prior_hier_K`,
`make_state_hier`, `update`, `resample_and_rejuvenate`,
`simulate_response`) are **K-agnostic by construction** —
verified by direct read at Phase 2 + the byte-identical methodology
adoption. There are no K=6-specific branches; K enters only as
loop bounds and array shapes. So a K=7 *synthetic* validation
directly exercises the same algorithm at the production dim
without re-running expensive production-grade SBC sweeps.

The risk register R6 ("Σ 14×14 destabilizes block fit or `other`
class under-identified") explicitly directs "gate on K=7 SBC /
coverage in Phase 7" — that points at the synthetic-K=7 pins this
sub-step adds.

## What was added

`tests/test_phase7_k7_validation.py` (3 slow-marked tests, ~8 s
total at single-thread BLAS):

| Test | Pin | Same gate as |
|---|---|---|
| `test_sbc_skeleton_l_domain0_k7` | SBC mean rank ∈ (0.25, 0.75) at K=7 | `tests/test_sbc_skeleton.test_sbc_skeleton_l_domain0` (K=3) |
| `test_posterior_coverage_k7` | 90% & 95% CI coverage > 0.7 at K=7 | `tests/test_posterior_coverage.test_posterior_coverage_skeleton` (K=3) |
| `test_lapse_algebra_k7` | `P(y=1\|z=±10)` floor/ceiling pin per-task at K=7 | `tests/test_lapse_rate` (K-independent algebra) |

The three byte-identical methodology tests (`test_sbc_skeleton`,
`test_posterior_coverage`, `test_lapse_rate`) are **NOT edited** —
Phase-2 byte-identity invariant preserved (md5 verified pre-add).

## What does NOT need a K=7 re-run

| Validation | Why no K=7 re-run is needed |
|---|---|
| `scripts/run_lapse_sensitivity.py` (Wichmann-Hill) | Per-task synthetic MLE study; K-independent at the equation level. The K=6 reference run validated the **per-task** lapse-misspecification bias; the K=7 system applies the same lapse equation to one more task. The Phase-6 §2 invariant + the new `test_lapse_algebra_k7` cover this. |
| `scripts/run_sparcnet_test_retest.py` | Per-domain ICC(3,1) on 6 SPARCNET IIIC fits. The K=7 `other` task **IS** the K=6 `iic` task (the `sparcnet_iic`→`other` mapping, Phase 3 D5 / Phase 4.4-B). No new domain; ICC≥0.70 6/6 (sz 0.90, grda 0.90, iic 0.88, lpd 0.86, gpd 0.80, lrda 0.71) stands. The 7th task `combined_spike` does not have a SPARCNET test-retest analog — it is the spike-paper task and was validated by its own published study, not by SPARCNET test-retest. |
| `scripts/run_sigma_sensitivity.py` (Σ_l sensitivity) | Synthetic study quantifying speedup robustness across Σ_l priors; K-agnostic (the cells include `K ∈ {2,4,6,8}` already). The hier-vs-brute speedup property is independent of which specific K=7 production Σ ships. |
| `scripts/run_gold_chain_reference.py` | Synthetic gold-chain validation of the SMC sampler against an exact MH reference. K=6 was the validating run (35/36 posterior-moment agreement). Re-running at K=7 would not change the validating finding (sampler correctness is K-independent). |
| `pipeline/joint_calibration/sbc_engine.py` (Phase-3.5 engine SBC) | Already gated by `tests/test_phase35_sbc_engine.py` at K=6 IIIC (`TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]`). The K=6 IIIC ⊆ K=7 (the 6 IIIC tasks of the production K=7 are the SAME 6 tasks). The 7th task `combined_spike` is gated upstream by Phase 3 (byte-verbatim reference fit) + Phase 3.5 (joint-IRT s_j with s_sd propagation). Adding a K=7 run would not change the soundness finding (engine SBC is per-domain; the contrast PLUGIN vs UNCERT is the gated claim, not absolute KS at a different K). |

## Headline finding

The K-agnostic engine soundness gates pass at K=7:

- SBC mean rank near 0.5 at K=7 (8 s, 20 draws, 60 obs, 200 particles)
- 90/95 % CI empirical coverage > 0.7 at K=7
- Lapse algebra `P(y=1\|z=±10)` floor/ceiling holds per-task across
  all 7 tasks

This is consistent with the Phase-2 finding (engine is K-agnostic and
hardened) and the Phase-3.5 engine SBC headline (UNCERT restores
calibration to CONTROL baseline; the K=6 IIIC result transfers to
K=7's 6 IIIC tasks because the engine SBC is per-domain).

## What follows (sub-step 7.2 onwards)

- 7.2 — Tier-2 OC simulator port + K=7 re-run (pending decision on
  whether to port `scripts/run_phase4_simstudy.py` from the sibling
  methodology repo).
- 7.3 — D6 real-rater replay harness build (the v1.0 blocker).
- 7.4 — Paper-1 figures regeneration at K=7.
- 7.5 — close-out + gate.
