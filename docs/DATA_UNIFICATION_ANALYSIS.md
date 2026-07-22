# Data Unification Analysis — 3 IIIC Sources (3-agent debate + adjudication)

> **Status (2026-07-21): historical decision record, not an active build
> plan.** The proposed joint/unified calibration was not promoted. Production
> continues to consume separately fitted `Corr_l` and `Corr_t` matrices. This
> report is retained because it records why linked cross-source inference was
> judged fragile; its proposed Phase 3.5 work and approval language below are
> superseded.

**Date:** 2026-05-18 · **Scope:** how to unify the 3 IIIC annotation sources
(SPaRCNet consortium · Kong crowd · Centaur gold+novice) to maximize the
statistical power of the Bayesian-hierarchical adaptive **skill (ℓ) + bias
(t)** test. Method: 3 independent PhD-level analyses (Bayesian hierarchical
inference · MCMC/computational · identifiability/Fisher-power) → neutral
cross-rebuttal + adjudication. This is the frozen decision record for a
proposed Phase 3.5 that was ultimately not promoted.

## 1. Verified facts (adjudicator re-checked against the repo)

- **`pipeline/fit_2pl_probit.py` does NOT exist** — the "PI joint-IRLS 2PL"
  alternative is a docstring phantom (`run_unified_calibration.py:24`
  only). There is currently **no joint fitter** in the repo; any joint
  model is net-new.
- **IIIC linkage graph = exactly 1 connected component (100% of 72,175
  rater+segment nodes).** Unification is *identifiable in principle*; the
  obstruction is statistical (weak welds), not topological.
- **Centaur is the binding fragility:** 5,000 Centaur segs connect to the
  65,160 non-Centaur segs by only **354 shared segments** and **28 shared
  raters**; the n=4 gold panel has **1** rater in any non-Centaur source
  and **0** rater overlap with the Centaur novice block — gold reaches the
  graph essentially only through its 354 shared *segments*.
- **IIIC `pattern_class` N = 962,467** (authoritative; the agents'
  829,292 / 984,084 were both wrong — discrepancy is the Centaur blocks +
  a CSV comma-in-JSON undercount). kong:crowd = 487,683 (**50.7%** of IIIC
  obs, 62 obs/seg), sparcnet50K = 290,268 (5.75 obs/seg), **kong:expert =
  8,534 (1.62 obs/seg → effectively single-rater; a weak anchor**, refuting
  the legacy "good expert anchor" claim). sparcnet50K∩kong:crowd = **6,691
  shared segs** (the dense weld).
- **The engine consumes `s_j` as a fixed scalar with zero uncertainty**
  (`core_mcmc.py:341` `z=exp(l)*(s+t)`, `:351` stores `float(s)`, `:459`
  `_expected_loss_vec` signal grid — no `Var(ŝ_j)` anywhere). The
  expected-posterior-variance item selector therefore **overstates
  per-item information → the headline power claim is inflated** until s_j
  uncertainty is propagated.
- Data-density corrections (all 3 agents, independently): SPaRCNet ≈5.8
  raters/seg (not 3), Kong-crowd ≈45–62 obs/seg (not "53 raters/seg"),
  kong:expert single-rater. The "37 dual-task experts / 29-rater matrix"
  is the cross-**task** set (incl. spike); the IIIC Centaur cut is **28**
  raters.

## 2. Consensus (all three experts agree)

1. The carried two-stage pipeline (per-task Rasch → fixed ×1/1.7 →
   per-rater probit-lapse) is **misspecified for cross-source
   unification**: hard-coded logit→probit constant, c↔l curvature
   discarded, point-estimate `c_mean` plugged forward as error-free, and
   **7 independent per-task anchors imposed on 1 connected component** →
   cross-task `s_j` not on a common scale.
2. Replace the 7 independent Rasch fits with **one joint cross-source
   hierarchical probit-lapse IRT**; keep the verbatim probit-lapse SDT
   likelihood + CV-top-N Youden **downstream untouched** (new upstream
   module emitting the identical `sdt_fits`/bank schema).
3. Plug-in `s_j` (no propagated variance) **inflates the adaptive-test
   power estimate**; honest power requires propagating `Var(ŝ_j)`.
4. **n=4 Centaur gold = a scale/location ANCHOR, not an expert
   population.** This mathematically explains the Phase-3 D7 failure (a
   4×4 near-collinear (ℓ,t) ridge cannot estimate an expert *distribution*
   for Youden) — D7 stays an ordinal/directional check only.

## 3. Adjudicated resolution of the live disagreements

- **Anchor location and scale SEPARATELY (identifiability lens wins):**
  - **Scale** → the dense **sparcnet50K∩kong:crowd weld (6,691 segs)**.
    Scale is the dangerous gauge (`I_k ∝ e^{2ℓ_k}`; a biased scale
    multiplies *all* power estimates). Anchoring scale on the n=4 gold
    (Position B) would propagate gold's demonstrated strictness bias
    through the 354-seg cut into every Centaur s_j and the Paper-1 block —
    rejected.
  - **Location** → the **n=4 gold on its 5,000 fully-crossed segs** (a
    full-rank dense design; a location shift is a harmless rigid
    re-origin and does not scale information). Salvages B's correct
    instinct (gold is the most informative single anchor) for the gauge it
    can safely fix.
  - A's per-task N(0,1) convention is the *existing misspecification* for
    cross-task comparability — rejected as the unification target (it
    remains fine strictly within a single task).
- **δ_Centaur:** weakly identified (354 segs/28 raters/~1 gold bridge) →
  partial-pool source effects `δ_d ~ N(0,τ²)`, non-centered;
  **fix-vs-estimate δ_Centaur sensitivity is a hard release gate**.
- **The ×1/1.7 bridge:** estimate a free **per-source dispersion γ_d** and
  report the posterior of the implied logit→probit ratio with an explicit
  **test of H0: ratio = 1.7** (don't hard-code a contested constant).
- **Kong tempering:** kong:crowd is 50.7% (not 59%) of IIIC obs — real but
  milder; a properly structured crossed hierarchy + source effect handles
  it. Tempering = a **sensitivity arm**, not the primary fit.
- **Polytomous vs one-vs-rest:** one-vs-rest forfeits ~20–50% Fisher
  information (the largest *intrinsic* lever) — but a polytomous
  reformulation breaks the carried-verbatim CV-top-N Youden contract.
  **Keep one-vs-rest for the Phase-4 release; scope polytomous as a
  quantified post-Phase-4 upgrade** (with Youden re-validation).

## 4. Single adjudicated recommendation

**Model** — one joint crossed-random-effects probit-lapse IRT over all
962,467 IIIC `pattern_class` obs, one-vs-rest per task, single component:

`P(Y=1) = λ_k/2 + (1−λ_k)·Φ( e^{ℓ_ik}·(s_jk + δ_d − t_ik) )`

- `s_jk` fully pooled across sources (no per-source rescale);
- `ℓ_ik,t_ik` partial-pooled with **expertise-tier means α_g + cross-task
  Σ** (preserves the engine's Σ_l prior);
- `δ_d ~ N(0,τ²)` source effects (non-centered) + optional per-source
  `γ_d` (tests the 1.7 ratio); `λ_k ~ Beta(2,40)`;
- **Gauge: scale ← sparcnet∩kong weld (6,691 segs); location ← n=4 gold
  (5,000 crossed segs)**, per connected component;
- Inference: **NumPyro NUTS, non-centered, GPU** (~1M obs, tractable);
- New module `pipeline/joint_calibration/fit_joint_iiic.py` emitting the
  **identical `sdt_fits`/bank schema** → engine + CV-top-N Youden untouched
  (carried-verbatim contract preserved);
- **Mandatory engine change:** propagate per-segment `(s_mean, s_sd)` and
  marginalize `s` in `core_mcmc.py` `update` (`:341`) and
  `_expected_loss_vec` (`:459`). NOTE: this modifies Phase-2
  byte-verbatim engine code → requires its own SBC/coverage/parity
  re-validation (flagged risk).

**Validation battery (release gate):** SBC; held-out posterior-predictive
coverage; **plug-in-vs-uncertainty held-out AUROC** (quantifies the power
overstatement); **fix-vs-estimate δ_Centaur sensitivity**; posterior of the
logit→probit ratio vs H0=1.7; D7 ordinal-only.

**Phased plan:**
- **Blocking before Phase 4** (prereq for any defensible power/E[N]
  claim): build `fit_joint_iiic.py` with the split anchoring; propagate
  `s_sd` into the engine; run validation 1–4; δ_Centaur gate.
- **Deferrable (scoped, quantified):** polytomous IRT (~20–50% Fisher
  recovery, needs Youden re-validation); active densification of the
  354-seg Centaur bridge (largest *real* power lever, gain ∝ √n_bridge);
  Kong ESS-tempering sensitivity arm.

**Honest net effect:** the unification **corrects an inflated power
estimate and de-biases cross-task s_j** — it is primarily a *correctness*
fix, not a power gain. Genuine power gains come from bridge densification
and the deferred polytomous upgrade, not from unification per se.

## 5. Documented dissents (user may override)

- **B (scale anchor):** anchor global scale on the n=4 gold, not the
  sparcnet∩kong weld — if SPaRCNet and Kong share a correlated platform
  bias the "dense weld" is not bias-free either. Override path: anchor on
  gold but mandate δ_Centaur + held-out-AUROC gates and report both.
- **C (polytomous):** put polytomous IRT in the Phase-4 model, not
  deferred — shipping a knowingly ≥20%-inefficient estimator into a
  Nature-Medicine power claim is hard to defend. Override = pull forward +
  re-validate Youden (schedule cost only).
- **A (per-task convention):** if the adaptive test never shares
  signal/segments across tasks, the cross-task misspecification is benign
  and the lighter per-task fit suffices — resolve by confirming whether
  the engine shares s_j across tasks.

## 6. Top risks & monitoring

1. **Global-scale ridge across the weak Centaur weld** — monitor NUTS
   divergences/R̂/BFMI on the scale param; δ_Centaur fix-vs-estimate shift
   as a hard gate; block Centaur s_j from release if unstable.
2. **Residual power overstatement if s_sd not propagated** — gate: no
   power/E[N] claim ships until the plug-in-vs-uncertainty arm passes.
3. **kong:crowd likelihood dominance (50.7%)** — monitor s_j on the 6,691
   sparcnet∩kong segs, full vs ESS-tempered; escalate tempering if they
   diverge.

## 7. Plan impact

This introduces a **proposed Phase 3.5** (joint hierarchical s_j model +
engine s_sd propagation + validation battery) as a prerequisite for a
defensible Phase-4 deployment power claim. It is a scope addition that also
touches the Phase-2 byte-verbatim engine (s_sd propagation). User decision
required before proceeding (see conversation).

## 8. Phase 3.5 close-out — RESULTS (2026-05-18)

Phase 3.5 was approved and executed. The validation battery (§ "release
gate") is **COMPLETE**. Results are reported here **honestly, including
where the effect is small or where a hard test is over-powered** — these
are not spun. Artifacts: `calibration/joint/{plugin_vs_uncertainty_auroc,
delta_centaur_sensitivity,sbc_engine}.json`; regression tests
`tests/test_phase35_*.py`; provenance mirrored in `cert_config.yaml`
→ `ell_star_unified_v13.provenance.phase35.release_gate`.

**Engine s_sd propagation.** Closed-form Gaussian-probit marginalisation
`z /= sqrt(1+(e^ℓ·s_sd)²)` in `core`/`core_mcmc`; `s_sd=0` ⇒ engine
default **bit-identical** to the prior engine (parity gated in the full
suite). Per-item Fisher deflation proven exact by
`test_phase35_engine_ssd`.

1. **Plug-in vs uncertainty held-out AUROC** — *real, correctly-signed,
   SMALL.* Fixed-budget metric (stop-at-δ was rejected as confounded:
   it forces both arms to CI≈δ and confounds n_q with the adaptive
   selector — owned and corrected). CI-halfwidth ratio uncert/plug-in
   = **1.019**; coverage 0.979 → 0.990. The plug-in engine was **not**
   badly overstating power in the favourable homogeneous-rater /
   rich-bank regime tested; s_sd propagation is a modest conservative
   refinement. The result **bounds** the plug-in cost; a stress regime
   (heterogeneous raters / thin bank) would show more — the ~2% is an
   honestly-framed optimistic lower bound.

2. **δ_Centaur sensitivity** — *robust* (cheap re-scoped probe, **not**
   a production refit, per the approved scope). δ_Centaur is partially
   non-identified against the deterministic Centaur-gold location
   anchor. Measured against a same-model/different-seed SVI **noise
   floor** (a first raw-magnitude cut was confounded by smoke-scale SVI
   noise — the gpd control with δ_Centaur≈0 proved it; owned and
   corrected): zeroing δ_Centaur perturbs s_j only **0.63× (gpd) /
   0.83× (iic)** the noise floor, min Pearson r **0.984**, even where
   δ_Centaur = −0.46 (iic). The s_j unification does not materially
   depend on this flagged modelling choice. Honest caveat: smoke-scale
   bound; production refit would tighten.

3. **Engine SBC** — *the strongest Phase-3.5 result.* θ* drawn from the
   engine's own prior ⇒ rank non-uniformity is pure likelihood
   misspecification. 150 replicates × 6 IIIC. The harness mirrors the
   production session loop exactly incl. ESS-triggered
   resample+rejuvenation (a reweight-only first cut degenerated the
   cloud, CONTROL mean_rank 0.73 — owned and fixed → 0.52).
   - CONTROL (noise off, plug-in): mean_rank **0.521**, cov95 **0.934**
     → harness sound, no false alarm.
   - PLUGIN (noise on, plug-in): KS **0.123**, mean_rank **0.574**,
     cov95 **0.884** → genuinely **miscalibrated / overconfident**.
   - UNCERT (noise on, s_sd): KS **0.031** (uniform not rejected),
     mean_rank **0.509**, cov95 **0.932** → calibration **restored to
     the CONTROL baseline**.
   ⇒ propagating the κ-calibrated s_sd is **not cosmetic**; it recovers
   posterior calibration the plug-in engine loses under realistic item
   noise. **Honest nuance (not spun):** CONTROL KS (0.0478) marginally
   exceeds crit (0.0453) at n=900 — the engine SMC's *known
   finite-particle approximation*, which KS over-detects at large SBC n
   even when correctly specified. It is the shared baseline of all
   three arms; mean_rank≈0.5 and coverage≈nominal prove soundness. The
   scientific claim is the **contrast** (PLUGIN degrades far beyond the
   baseline; UNCERT returns to it), which is unambiguous regardless of
   the absolute KS hard-test.

**Release-gate disposition.** Risk §6.2 ("no power/E[N] claim ships
until the plug-in-vs-uncertainty arm passes") is satisfied: the arm
passed, the magnitude is small and bounded, and the SBC contrast
demonstrates the propagation is material for *calibration* even where
it is modest for *power*. Risk §6.1 (δ_Centaur as a hard gate) is
satisfied at smoke scale (robust vs noise floor). The variance
calibration's documented limitation (SVI-vs-NUTS s_mean r≈0.72–0.78 on
an 80k NUTS subsample) stands as a known limitation, not a tight
equivalence claim. Phase 3.5 is **closed**; Phase 4 (deployment
integration) proceeds under the strict incremental, regression-gated,
PI-baseline discipline.
