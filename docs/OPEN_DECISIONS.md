# Open shipping decisions — v1.0 acceptance review

This is the documented decision register for unresolved scientific and
publication questions. The original private merge plan is no longer
distributed. Each item records the engineering state, scientific stake,
current working policy where one exists, and the evidence still required.

Opened 2026-05-20; refreshed 2026-07-21. The summary table at the end is
authoritative for current disposition; dated counts and version labels inside
individual findings describe the audit that raised them.

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
    notes the prior contributor-note ±5 pp claim was wrong; corrected to
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

## 6. Replay/calibration cohort overlap (Nature Medicine audit, 2026-05-28)

**Status**: 🔴 OPEN — flagged by the Nature Medicine readiness audit (Pipeline
G1; Biostat G1; EEG Gap 6 convergent). See `docs/NATURE_MEDICINE_AUDIT.md §4.G1`.

The Phase-7 D6 real-rater replay headline cohort (n=21) substantially
overlaps the calibration panels used to set ℓ\*: **16 of 21 replay raters
are in the 29-rater Q2-locked candidate pool**, and **8–9 of the 14 raters
in each domain's CV-Youden expert panel are also in the 21-rater replay
cohort**. The 70/30 split inside `step_g` is over the σ-pool, not over
rater identity — raters in TRAIN can appear in the replay cohort.

**Implication.** The +43 % to +1154 % replay-vs-Bernoulli headline contrast
uses overlapping individuals for calibration AND for replay measurement;
the magnitude of confounding is unquantified. Item 3 above (D7
independent-panel reproducibility) is related but does not address the
specific replay/calibration overlap.

**For v1.0 acceptance review.** Decide whether to (a) run a leakage-aware
sensitivity rerun (re-run `step_f` with the 21-rater replay cohort masked
out of the candidate pool, regenerate per-domain ℓ\*, re-evaluate the
replay headline, report the contrast next to the current headline; NATURE
MEDICINE AUDIT T1.2; 1–2 days), (b) accept the overlap with explicit
manuscript acknowledgement, or (c) defer to Paper 2.

## 7. cert_config v13 ℓ\* values not numerically pinned (engineering integrity)

**Status**: 🔴 OPEN — flagged by the Nature Medicine readiness audit
(Pipeline G4 unique). See `docs/NATURE_MEDICINE_AUDIT.md §4.G4`.

The seven `ell_star_unified_v13` values in `calibration/cert_config.yaml:58-112`
— the clinical decision boundary for every PASS/FAIL verdict — are not
pinned numerically in the 282-test audit-era suite. A hand-edit of the YAML would
pass CI silently.

**For v1.0 acceptance review.** Add `test_v13_ell_star_pinned()` to
`tests/test_phase3_calibration.py` pinning the 7 values to 1e-12 (NATURE
MEDICINE AUDIT T1.1; 30 min; no D-decision risk). Trivial close.

## 8. Engine vs deployment Σ are different priors

**Status**: 🟡 PARTIAL — flagged by the Nature Medicine readiness audit
(Pipeline G5 unique). See `docs/NATURE_MEDICINE_AUDIT.md §4.G5`.

The Mode-A Paper-1 engine reads `engine_paths.SIGMA_L` (symlink to the 6×6
frozen 15-rater-era artifact at repo root) and applies a matched-mean
compound-symmetry approximation to K=7 at runtime. The deployment Laplace +
EKF reads a fresh 14×14 PI block fit from `data/deployment_prior/Sigma.csv`.
The two engines use **different** priors.

Documented in code (`engine/core_mcmc.py:35-40`, `engine_paths.py:14-17`)
but not gated by any test — a reviewer asking "are the priors the same?"
gets "no."

**For v1.0 acceptance review.** Decide whether to (a) re-fit Sigma_l at K=7
from `data/labels/fits_hier_block`, ship as `Sigma_l_fitted_k7.npy`, update
engine paths, regenerate Paper-1 figures (NATURE MEDICINE AUDIT T2.7;
med effort; Paper-1 figures regenerate; re-run Phase-2 byte-equivalence
suite at new prior); or (b) accept the K=6→K=7 CS approximation and
document the cross-engine prior difference in the manuscript supplement.

## 9. No formal hypothesis test for the D6 replay headline

**Status**: 🔴 OPEN — flagged by the Nature Medicine readiness audit
(Biostat G2/R1). See `docs/NATURE_MEDICINE_AUDIT.md §5.B1`.
`docs/PHASE7_REPLAY_HEADLINE.md:209-212` explicitly defers a formal
hypothesis-test layer "if a reviewer asks for it" — Nature Medicine
reviewer 2 will ask.

**For v1.0 acceptance review.** Add `pipeline/replay/statistical_tests.py`:
mixed-effects `verdict ~ arm + (1|rater) + (1|task) + (arm|rater)`; paired
McNemar per task; Wilcoxon signed-rank on per-cell AUROC differences;
Hodges-Lehmann estimator; multiple-comparison adjustment table. (NATURE
MEDICINE AUDIT T1.5; 1–2 days; no D-decision risk.)

## 10. Calibration of probabilistic outputs absent

**Status**: 🔴 OPEN — TRIPOD+AI mandatory; flagged by the Nature Medicine
readiness audit (Biostat G7/R2; TRIPOD #15/Rec 2; Psychom Rec 7
convergent). See `docs/NATURE_MEDICINE_AUDIT.md §5.B2`.

No Brier score, calibration-in-the-large, expected calibration error,
calibration slope, or reliability diagrams anywhere in the codebase
(except SBC Bonferroni for Bayesian posterior coverage — a different
sense of calibration). TRIPOD+AI Item 15 currently MISSING.

**For v1.0 acceptance review.** Add `pipeline/replay/calibration_metrics.py`
computing per-task Brier + CITL + slope + reliability diagram + ECE +
Integrated Calibration Index on the 14,823 replay cells. (NATURE MEDICINE
AUDIT T1.4; 4–6 hours; no D-decision risk.)

## 11. Subgroup / fairness analyses absent

**Status**: 🟡 PARTIAL — feasible-now subgroup analyses exist on existing
data; full fairness analysis is data-gated. Flagged by the Nature Medicine
readiness audit (TRIPOD #19/Rec 3; EEG Rec 5/6 convergent). See
`docs/NATURE_MEDICINE_AUDIT.md §5.B4 + §6.E2`.

Feasible NOW on existing data (no acquisition): per-task PASS rate by
expertise tier (174 / 163 / 47 / 1,244 / 740 / 465 in raters.csv); per-task
PASS rate by self-reported `years_eeg` (n=289); spike PASS rate by patient
age band (n=15,670 with age) and sex (n=3,966 with sex).

Not feasible without new acquisition: race / ethnicity (n=0); comorbidity
/ etiology (n=0); ICU / EMU / outpatient setting (n=0); site-level
fairness (no `site_id` in segments.csv).

**For v1.0 acceptance review.** (a) Run the feasible subgroup analyses
(NATURE MEDICINE AUDIT T1.7; 2–3 hours); (b) backfill demographics from
BDSP source records under IRB amendment (T2.2; high effort; unlocks
race/etiology fairness); (c) explicit "data not collected" limitation
statement (T1.10) per TRIPOD+AI.

## 12. Decision-curve analysis absent

**Status**: 🔴 OPEN — Nature Medicine clinical-AI standard since 2019
(Vickers et al. BMJ); flagged by the Nature Medicine readiness audit
(TRIPOD #20/Rec 4; Psychom Rec 6 convergent). See
`docs/NATURE_MEDICINE_AUDIT.md §5.B5`.

Youden-J implies symmetric FP/FN cost; clinical EEG has highly asymmetric
costs (false-positive "seizure" → AED low-risk; false-negative seizure →
missed status high-risk). DCA net-benefit grid expected for any
clinical-AI / decision-support paper.

**For v1.0 acceptance review.** Run net-benefit grid at `pass_p × fail_p`
on the 21-rater replay arm with asymmetric cost; report cost-weighted
threshold `ℓ*_α` for `α ∈ {0.3, 0.5, 0.7}` alongside Youden-optimal
`ℓ*_J = 0.5`. (NATURE MEDICINE AUDIT T1.8; 6 hours; no D-decision risk —
the Youden-J ℓ\* remains the v1.0 ship value.)

## 13. Multiple-testing burden across 7 tasks × 21 raters

**Status**: 🔴 OPEN — flagged by the Nature Medicine readiness audit
(Biostat G8/R3 unique). See `docs/NATURE_MEDICINE_AUDIT.md §5.B3`.

Per-candidate FWER under the per-task `pass_p = 0.95` rule is
`1 − 0.95⁷ ≈ 0.30`. `mode_b_legacy.stop_thresh_sidak: 0.9915` exists for
the retired Mode-B path; the live deployment doesn't use it.

**For v1.0 acceptance review.** Document explicitly that the per-task
certificates policy (Item 1 above) is the FWER-resolving choice — each
task is its own certificate; downstream consumers (credentialing boards,
study sites) apply their own roll-up rule with their own α. Add
`docs/MULTIPLE_TESTING.md` formalising this. (NATURE MEDICINE AUDIT T1.6;
0.5 day; no D-decision risk.)

## 14. Pre-registration is retroactive

**Status**: 🔴 OPEN — flagged by the Nature Medicine readiness audit
(TRIPOD #4/Rec 1 unique; `docs/PHASE7_CLOSEOUT.md:243-271` already
admits). See `docs/NATURE_MEDICINE_AUDIT.md §5.B7`.

Phase-7 sub-5 carved out four "implicit bounds" retrospectively (engine
drift-guard, both engines run, honest pilot caveats, no regression). For
Nature Medicine, a registered protocol is expected.

**For v1.0 acceptance review.** Retroactively register the v1.0 acceptance
protocol on OSF using TRIPOD-AI + SPIRIT-AI templates, citing 2026-05-18
as the Phase-7 design lock and 2026-05-19 as the headline run. Be
explicit that this is retrospective. Prospectively register the CORTEX
live-deployment cohort on ClinicalTrials.gov BEFORE candidate enrollment.
(NATURE MEDICINE AUDIT T1.9; 1 day; no D-decision risk.)

## 15. Lapse-sensitivity sweep never run on unified corpus

**Status**: 🔴 OPEN — extremely cheap to close; flagged by the Nature
Medicine readiness audit (Psychom Rec 1; Biostat R8 convergent). See
`docs/NATURE_MEDICINE_AUDIT.md §5.B6`.

`scripts/run_lapse_sensitivity.py` exists with Wichmann–Hill λ_true ∈
{0.01, 0.025, 0.05, 0.10} grid and a coded acceptance criterion
(|AUROC excess bias| ≤ 0.02). `results/phase2_validation/` is empty —
the sweep has never been run on the unified corpus. The "λ = 0.025 is
robust" claim has no current empirical support in this repo.

**For v1.0 acceptance review.** Run `python scripts/run_lapse_sensitivity.py`
on the unified corpus; quote `worst_excess_bias` in Methods. (NATURE
MEDICINE AUDIT T1.3; minutes of compute; no D-decision risk.) This is
the single most likely reviewer-1-round ask.

## 16. AD6 variance-contraction gate (gate-2) is non-operative

**Status**: 🔴 OPEN — re-opens the deferred R\* recalibration (its premise is
void). Surfaced by an isolated estimator audit on 2026-06-19; that disposable
R&D directory is not part of the shareable repository, so the complete finding
is preserved here.

AD6's per-domain "relative info-gain" gate `R_k = 1 − Var_post(ℓ_k)/var_prior[k]
≥ 0.30` (`scripts/cortex_policy.py:287`) takes `var_prior = diag(Corr_l) = 1.0`
(`cortex_policy_k7.py:128`) — a **correlation** diagonal — while the particle ℓ_k
live on the fitted **covariance** scale `diag(Σ_l)=[.498,.838,1.426,1.696,1.811,
2.112,1.632]`. The threshold `(1−0.30)·1.0 = 0.70` sits **above 5 of 7 prior
variances**, so gate-2 never binds: in the baseline cohort **95/95 REFER are
BORDERLINE, 0 UNINFORMATIVE** → AD6 has effectively been `n_min ∧ π-escape` all
along (the R\* gate is structurally absent — preserving today's spike permissiveness
on the Σ scale would require R\* ≤ 0).

**Investigation**: correcting the reference to `diag(Σ_l)` at the shipped R\*=0.30
is **confounded** (scale and operating point change inseparably; spike stricter,
high-variance IIIC looser) and shows **no benefit** (Δhw < 0.0003, REFER reshuffled)
with a small non-significant **adverse** clear-case false-FAIL lean. Shipping the
reference fix *alone* is the worst option. No shipped verdict changes (gate inert);
verdicts kept byte-identical.

**For v1.0 acceptance review.** Bundle the reference fix (`var_prior → diag(Σ_l)`)
**with** R\* recalibration on the Σ scale into one pre-registered change; **trigger**
= a drift-guard pinning `REFER_UNINFORMATIVE == 0` going red (i.e. the day a cohort /
recalibration / real-rater replay makes gate-2 bind). Defer the *tuning*, not the
*decision to schedule it*. Inertness is **cohort-conditional** (binds if Var_post
stays above `(1−R*)·var_prior` for any resolved domain — plausible under
wider-variance/real raters); keep gate-2 (UNINFORMATIVE is a real verdict class).

## Decision tracking

| # | Decision | Status | Owner | Target |
|---|---|---|---|---|
| 1 | Per-candidate roll-up policy | 🟢 Working (per-task certs) | v1.0 review | acceptance |
| 2 | Real-rater replay vs Bernoulli | ✅ Resolved (replay) | Phase 7 (closed) | done |
| 3 | ℓ\* independent-panel reproducibility | 🟡 Partial (ordinal done) | Phase-3 carry | acceptance |
| 4 | Bias-warning channel | 🔴 Open | v1.0 review | acceptance / Paper 2 |
| 5 | Split-half reliability + external cohort | 🔴 Open (split-half OK; external TBD) | v1.0 review | acceptance / Paper 2 |
| **6** | **Replay/calibration cohort overlap** | 🔴 Open | NatMed audit (2026-05-28) | acceptance |
| **7** | **cert_config v13 ℓ\* numerically unpinned** | 🔴 Open (trivial close) | NatMed audit (2026-05-28) | acceptance |
| **8** | **Engine vs deployment Σ different priors** | 🟡 Partial (documented in code, not gated) | NatMed audit (2026-05-28) | acceptance |
| **9** | **No formal hypothesis test for D6 headline** | 🔴 Open | NatMed audit (2026-05-28) | acceptance |
| **10** | **Calibration of probabilistic outputs absent** | 🔴 Open (TRIPOD+AI mandatory) | NatMed audit (2026-05-28) | acceptance |
| **11** | **Subgroup / fairness analyses absent** | 🟡 Partial (feasible-now subset done; full data-gated) | NatMed audit (2026-05-28) | acceptance / Paper 2 |
| **12** | **Decision-curve analysis absent** | 🔴 Open | NatMed audit (2026-05-28) | acceptance |
| **13** | **Multiple-testing burden** | 🔴 Open (doc-only close) | NatMed audit (2026-05-28) | acceptance |
| **14** | **Pre-registration retroactive** | 🔴 Open (OSF retrospective + CORTEX prospective) | NatMed audit (2026-05-28) | acceptance |
| **15** | **Lapse-sensitivity sweep never run on unified corpus** | 🔴 Open (trivial close) | NatMed audit (2026-05-28) | acceptance |
| **16** | **AD6 gate-2 non-operative (var_prior corr-vs-cov scale)** | 🔴 Open (re-opens deferred R\* recalib) | estimator audit (2026-06-19) | acceptance |

## Register provenance

The register began with five merge-time shipping questions: candidate roll-up,
real-rater replay, independent-panel cut reproducibility, a bias-warning
channel, and split-half/external-cohort reliability. Later audits extended it
without silently rewriting the earlier decisions. Git history preserves the
private planning wording; this file is the maintained public record.
