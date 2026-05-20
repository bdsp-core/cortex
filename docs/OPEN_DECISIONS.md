# Open shipping decisions — v1.0 acceptance review

Per `../UNIFIED_REPO_MERGE_PLAN.md` §"Phase 8" sub-4: the open
shipping decisions are explicitly unresolved in the PI decks and are
carried here as **a documented decision register** rather than
silently locked in code. Each item records the engineering state,
the scientific stake, the current working policy (where one exists),
and what remains for v1.0 acceptance review to confirm.

Date: 2026-05-20 (Phase 8 sub-8.2).
Status: 5 open items — 1 working policy adopted, 1 resolved by
Phase-7 evidence, 3 carry-as-open for acceptance review.

## 1. Per-candidate overall verdict policy

**Status**: 🟢 WORKING POLICY — *per-task certificates, no single
roll-up* (user scope 2026-05-20).

The K=7 deployment produces a per-task verdict (PASS / FAIL / REFER)
for each of `{spike, seizure, lpd, gpd, lrda, grda, other}`. The
question is whether to aggregate these into a single candidate-level
verdict, and if so, how.

**Three candidate roll-up rules considered:**

  1. *Per-task certificates with no single roll-up* (working v1.0
     policy). Each task gets its own PASS / FAIL / REFER; no
     aggregated `all_pass` verdict.
  2. *spike + ≥k-IIIC composite*: candidate PASSes iff spike PASSes
     AND at least `k` of the 6 IIIC tasks PASS, for `k ∈ {1, …, 6}`.
     PI decks reference this style ("spike + ≥3-IIIC") but `k` is
     left unspecified.
  3. *Full-7 conjunction* (`all_pass`): candidate PASSes iff every
     one of the 7 tasks PASSes. Most conservative; matches the
     deployment engine's existing `all_pass` field (which is what's
     currently reported in `results/replay/replay_per_candidate.csv`
     but is empirically degenerate — see below).

**Evidence supporting the working policy (per-task certs)**:

From sub-7.3-C (`docs/PHASE7_REPLAY_HEADLINE.md`) on the 21-rater
full-7 cohort under the conservative deployment stopping rule
(`pass_p=0.95`, `fail_p=0.05`, `N_max_per_task=120`):

  * **0 of 21** raters reach `all_pass = True` under EITHER the
    real-rater replay arm OR the fitted-θ Bernoulli arm.
  * This is NOT a sign the raters are unskilled — most achieve
    PASS on 4 / 5 / 6 of the 7 tasks. It is a sign that conjunctive
    aggregation across 7 conservative-PASS gates is empirically
    unreachable with the current deployment stopping rule + bank
    sizes.
  * The per-task PASS rate is informative and varies by task
    (sub-7.3-C: spike +43 %, seizure +409 %, gpd +235 %, grda +533 %,
    lrda +1154 % more PASS under replay than Bernoulli predicts).

**Trade-offs**:

  * *Per-task certs* (working policy): maximally informative, matches
    the per-task headline of sub-7.4, no arbitrary `k`. Downstream
    consumers (credentialing boards, study sites) can apply their own
    roll-up rule. Honest to reviewers that the engine reports per-task
    skill, not a single "credentialed Y/N" bit.
  * *spike + ≥k-IIIC*: pragmatic; aligns with PI's deck terminology;
    needs an explicit `k` choice. No evidence base in this repo for
    picking `k`.
  * *Full-7 conjunction*: scientifically clean but empirically
    degenerate (0 / 21 PASS). Would force every candidate to REFER →
    not a usable v1.0 policy.

**For v1.0 acceptance review**: confirm the per-task certs policy
is what the credentialing board / paper-1 deliverable expects.
Engineering implication: `replay_per_candidate.csv`'s `all_pass`
column should be regarded as informational, NOT the certification
verdict. The certification verdicts are the 7 per-task
`decision_<task>` columns.

## 2. Real-rater replay vs Bernoulli sim (D6)

**Status**: ✅ RESOLVED in Phase 7 (replay is the v1.0 path).

The merge plan D6 (locked 2026-05-18): "Real-rater replay is
required for the v1.0 Nature-Medicine claim. Bernoulli sim retained
only as a supplementary sanity check."

Phase 7 sub-7.3 built the real-rater replay harness and ran the
v1.0-blocker headline (`docs/PHASE7_REPLAY_HEADLINE.md`). The
fitted-θ Bernoulli comparator runs on the same engine + same v13
calibration as the replay arm (same default Bernoulli branch in
`deployment/simulate_test.py:simulate_candidate`), so a Bernoulli
sanity arm is always available without additional engineering.

**Documented finding**: the fitted-θ SDT Bernoulli **under-predicts**
real-rater skill (+43 % to +1154 % more PASS verdicts under replay
across the 7 tasks; Mode-A AUROC +0.017 to +0.027 higher and 25–46 %
tighter CI halfwidths across 6 IIIC tasks). The deployment will be
MORE lenient than the Bernoulli OC tables predict — favourable for
the v1.0 claim, honest to reviewers.

No open engineering work. Carrying this entry so reviewers see the
explicit resolution.

## 3. ℓ\* reproducibility on an independent panel (D7)

**Status**: 🟡 PARTIAL — ordinal-only verification done in Phase 3
(see `calibration/CALIBRATION_PROVENANCE.md`); a fully independent
panel re-fit is documented as a Phase-3 D7 carry.

The merge plan D7 (locked 2026-05-18): "ℓ\* must be shown
reproducible on a held-out independent rater panel, not just the
calibration pool."

**Current state**:

  * The Phase-3 unified calibration recompute on the PI superset
    corpus (`pipeline/run_unified_calibration.py`,
    `calibration/cert_config.yaml` v13) used the CV-top-14 two-stage
    Youden methodology; CV-fold construction was done on the joined
    calibration pool.
  * D7 ordinal-only verification was reported in
    `calibration/CALIBRATION_PROVENANCE.md` as "min J 0.372, mean J
    0.521 (5× stronger than legacy ALL_7)".
  * A full **held-out independent panel re-fit** (e.g. partition raters
    by site/cohort, fit on one, validate on the other; report per-task
    ℓ\* delta + verdict-concordance) has not been run in this repo.
    The Centaur IIIC 4-expert gold panel (`docs/PHASE7_REPLAY_DESIGN.
    md`; cal/matt/tianyu/mbw, exactly 5,000 IIIC segs each) is a
    candidate independent panel for this — fully crossed,
    expert-anchored, n=4 (small).

**For v1.0 acceptance review**: decide whether the Phase-3 ordinal
verification suffices, or whether a hold-out-panel reproduction
(e.g. on the n=4 Centaur gold panel) is a v1.0 gate. The compute is
small (re-run `pipeline/run_unified_calibration.py` with a different
expertise-level partition); the question is scientific scope.

## 4. Bias-warning channel (`|t_k| > tol`)

**Status**: 🔴 OPEN — no `t_k` bias-warning channel exists in the
deployment runtime; documented as a v1.0 acceptance-review item.

**The question**: should the deployment runtime emit a per-task
warning when the candidate's posterior median `t_k` (the SDT
criterion shift) exceeds a tolerance `tol`? Concretely, a candidate
with `t_k ≫ 0` is calling discharges too liberally (high false-pos),
`t_k ≪ 0` too conservatively (high false-neg). The current
deployment only certifies on `ℓ_k > ℓ\*_k`; a strongly-biased rater
may PASS on `ℓ` while still being clinically miscalibrated.

**Current state**:

  * The Phase-6 invariant audit (`docs/INVARIANT_AUDIT.md` §8)
    documented `T_TOL = 0.20` as a Mode-B / Paper-2 constant that
    was **NOT carried into the unified Paper-1 scope** (N/A).
  * `deployment/simulate_test.py` tracks the joint posterior over
    `(t_k, ℓ_k)` for each task; the per-task `theta_k` posterior is
    available at session end but is not surfaced as a warning channel.
  * The reference repo's `T_TOL = 0.20` ≈ 9.3 pp (Phase-6 §8
    notes the prior CLAUDE.md ±5 pp claim was wrong; corrected to
    9.3 pp).

**For v1.0 acceptance review**: decide whether to (a) add a bias
warning to the deployment output schema with a documented `T_TOL`,
(b) carry as a Paper-2 / post-ship Mode-B item, or (c) drop entirely
(per-task `ℓ_k > ℓ\*_k` is the sole certification criterion).
Engineering cost for (a) is small (one column in
`replay_per_task.csv` + a `Config.t_tol` constant); the question is
scope + tolerance value.

## 5. Split-half reliability + external-cohort validation

**Status**: 🔴 OPEN — split-half reliability has Phase-2 evidence
(sparcnet test-retest 6/6 ICC ≥ 0.70); external-cohort validation
has no in-repo evidence; documented as a v1.0 acceptance-review item.

**Two sub-questions**:

  1. *Split-half reliability* — within-task test-retest. The Phase-2
     validation suite (`scripts/run_sparcnet_test_retest.py`) computed
     ICC(3,1) on the SPARCNET 6-IIIC fits: sz 0.90, grda 0.90, iic
     0.88, lpd 0.86, gpd 0.80, lrda 0.71 (6 / 6 ≥ 0.70). The 7th task
     `combined_spike` has no SPARCNET test-retest analog — it is the
     spike-paper task and was validated by its own published study,
     not by SPARCNET test-retest. Per
     `docs/PHASE7_VALIDATION_K7.md`, the K=6 ICC carries to K=7's 6
     IIIC tasks because the engine SBC is per-domain.
  2. *External-cohort validation* — across-site / across-cohort
     generalization. No external-cohort validation has been run in
     this repo. The Centaur IIIC gold panel could serve as an
     external IIIC cohort (sub-7.3-A; n=4 experts × 5,000 segs each);
     spike has no obvious external cohort in the available data.

**For v1.0 acceptance review**: confirm split-half ICC ≥ 0.70 in 6
of 7 tasks suffices as the "split-half reliability" Phase-8 gate;
decide whether external-cohort validation is a v1.0 gate or a
Paper-2 / post-acceptance carry. The Centaur 4-expert n is small;
broader external validation would require additional data
acquisition.

## Decision tracking

| # | Decision | Status | Owner | Target |
|---|---|---|---|---|
| 1 | Per-candidate roll-up policy | 🟢 Working (per-task certs) | v1.0 review | acceptance |
| 2 | Real-rater replay vs Bernoulli | ✅ Resolved (replay) | Phase 7 (closed) | done |
| 3 | ℓ\* independent-panel reproducibility | 🟡 Partial (ordinal done) | Phase-3 carry | acceptance |
| 4 | Bias-warning channel | 🔴 Open | v1.0 review | acceptance / Paper 2 |
| 5 | Split-half reliability + external cohort | 🔴 Open (split-half OK; external TBD) | v1.0 review | acceptance / Paper 2 |

## Relation to the merge plan

Per `../UNIFIED_REPO_MERGE_PLAN.md` §"Phase 8" sub-4:

> Carry the **open shipping decisions** into an issue tracker /
> `docs/OPEN_DECISIONS.md` (these are explicitly unresolved in PI
> decks): per-candidate overall verdict policy (spike-only vs
> spike+≥3-IIIC vs full-6); real-rater replay (vs Bernoulli sim);
> ℓ\* reproducibility on an independent panel; bias-warning channel
> (`|t_k|>tol`); split-half reliability + external-cohort validation.

This document is that tracker. The five items above match the five
in the plan; the working policy for #1 was added at this Phase-8
sub-step (user scope 2026-05-20), and #2 is recorded as resolved
by Phase-7 evidence.
