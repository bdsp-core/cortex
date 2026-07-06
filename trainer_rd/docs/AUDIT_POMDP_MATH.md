# Mathematical audit — the POMDP trainer and its learning-parameter estimation

**Date:** 2026-07-01 (user-requested deep methodological audit)
**Scope:** `learning_algorithm_plan.md` (the POMDP formulation) and the code that
implements its estimation stack: `training/training_filter.py`,
`training/learner_sim.py`, `training/trainer_greedy.py`,
`training/trainer_policy.py`, `training/trainer_rollout.py`,
`training/bridge_conventions.py`, `engine/core_mcmc_general.py`, and the
benchmark/validation harnesses.
**Method:** independent re-derivation of every load-bearing formula, line-level
code-vs-math comparison, and targeted numerical experiments. The headline
experiment is reproducible: `python3 -m studies.study_exact_kernel_audit`
(`--smoke` for a 90-second version). Findings are numbered MA-1…MA-10 to avoid
colliding with the PROJECT_MEMORY F-series; each states severity, evidence,
and the concrete fix.

---

## 0. Executive summary

The architecture is sound and unusually well-verified for a prototype: the
sign/coordinate bridge, the s_sd-attenuated likelihood, the GH smearing, the
resampling scheme, the R–W/gradient-descent correspondence, the placement
constants, and the reward telescoping all check out exactly (§2). The known
weak spots in PROJECT_MEMORY (F28/F29 graduation self-assessment, F33
θ-undercoverage) are honestly documented.

The audit's central new result: **two of those "known, documented,
irreducible" residuals are in fact the same fixable mathematical
approximation.** The filter's propagate step marginalizes the stimulus
realization s_real independently in the likelihood and the transition, but the
learner's response y and its state update are driven by the *same* s_real
draw. Conditioning the transition kernel on the observed y — mean *and*
variance, using the 11-node Gauss–Hermite machinery already in the file —
restores essentially nominal coverage on both state coordinates and improves
RMSE (MA-1). The shipped filter's bias-intervals are ~40 % too narrow under
real-bank stimulus noise; that overconfidence feeds directly into the
premature-graduation pathology (F31/F32) and the θ-undercoverage (F33).

| # | Severity | Finding (one line) |
|---|---|---|
| MA-1 | **HIGH** | Soft-rule propagate ignores the y↔s_real coupling; exact conditional kernel (same GH nodes, O(11N)) takes cov_t@90 from 0.63→0.91, cov_ℓ 0.77→0.91, at *better* RMSE. Subsumes smear_w (F30); largely explains F33. |
| MA-2 | **HIGH** (structural) | The belief state carries no uncertainty over the dynamics parameters (α_t, α_σ, σ_∞ fixed at EB point values, F4/D7) — this, not filter quality, is the root of the F28/F29 flat graduation OC. `p_static` is a 2-point special case of the principled fix (a per-learner mixture over σ_∞ / learning-activity). |
| MA-3 | MEDIUM | F22 sharpened: the criterion update is not feedback-gated in *either* `learner_sim` or the filter (only σ is). Informationally impossible (uses y\* without feedback) and it already bit on real data (F34-i). One-line fix in two files; do it now, not Phase 3. |
| MA-4 | MEDIUM-LOW | The locked λ=0.025 is 3–10× the EXTSET-fitted lapse (F43/F51). The 85 %-placement constant is λ-conditioned: shipped 1.0772 trains at ~85.5 % instead of 84.1 % under the fitted λ; correct multiplier would be ≈1.01–1.03. Recompute constants from the instrument λ at port. |
| MA-5 | LOW | Greedy/tier-3 `E[w]` closed form is fold-then-smear; it under-counts the true folded weight by ~2× for near-boundary items. Harmless for skill-mode ranking (optimum is off-boundary); matters only if Q is ever read as a calibrated value. Exact two-bump closed form costs the same. |
| MA-6 | LOW | Config drift: `ModeThresholds.sd_floor = 0.23` encodes the s_sd=0 floor (1.5×0.154); the real-bank floor is 0.179, so the shipped margin is 1.28×, not the designed 1.5×. Also `pass_mass` MCSE uses ESS that resets to N right after resampling (optimistic). |
| MA-7 | LOW (improvement) | The D16 mean-skill gate is a repeated-look test with a heuristic AR(1) correction that clips to n_eff=1; replace with an anytime-valid confidence sequence (e-process) — same inputs, honest type-I control at any stopping time. |
| MA-8 | LOW | Tier-3 rollouts always use the *shipped* kernel: they ignore `p_static`/`smear_w` hardening (and would ignore MA-1's exact kernel), so a hardened filter plans with a different model than it believes. |
| MA-9 | LOW | `_expected_reward` (tier-1 Q, `bias_correction_score`) ignores `filt.learn`: under `p_static>0` the non-learning particle mass still contributes full expected reward. Multiply the deterministic terms by `learn` per particle. |
| MA-10 | INFO | Plan-text errata (exactness claims, unguarded API preconditions, Jensen conventions) — list in §3.10. |

**If only two things are acted on:** implement the exact conditional kernel
(MA-1) and re-run the M12 verification ladder with it; and treat per-learner
σ_∞ as an inferred quantity rather than a known constant (MA-2). Together they
attack the two honest limitations the current publication framing has to
disclaim (calibration of bias CIs; graduation self-assessment).

---

## 1. The estimation problem, restated

Per task, the learner's hidden state is θ_k = (σ_k, t_k) (plan coords; engine
coords ℓ=−log σ, θ=−t). The POMDP components as implemented:

- **Observation** Z: y ~ Bernoulli(λ + (1−2λ)Φ((s_real − t)/σ)), where the
  served item's true signal s_real ~ N(s_mean, s_sd²) is *never observed* —
  only (s_mean, s_sd) are known (F20).
- **Transition** T: soft-rule criterion t' = t + α_t(p(s_real) − y\*) + ξ_t;
  log-skill relaxation log σ' = log σ + α_σ·f·w(s_real)·(log σ_∞ − log σ) + ξ_σ,
  with w the Gaussian 85 %-difficulty window. Both the response *and* the
  update depend on the same s_real.
- **Belief**: per-task bootstrap SMC (reweight → systematic resample → propagate),
  correctly ordered against the generative timing, no MH (F1/D5 — correct call).
- **Parameters** (α_t, α_σ, σ_∞, q_t, q_σ, ρ): fixed at empirical-Bayes point
  values (D7/D10); *not* part of the belief state.
- **Policy tiers**: greedy Q (tier 1), mode-conditional (tier 2, shipped),
  H-step CRN rollouts (tier 3).

The audit examined (a) whether the belief update is the correct Bayes filter
for this generative model, (b) whether the policy scores are correct
expectations under the belief, (c) whether the parameter-estimation strategy
is sound, and (d) whether the gates built on the posterior are calibrated.

---

## 2. Verified and correct (checked independently, holds)

- **Coordinate bridge** σ=exp(−ℓ), t=−θ: z = e^ℓ(s+θ) ≡ (s−t)/σ exactly; the
  round-trip and the sign-flip hazard are correctly centralized (Step 0).
- **s_sd attenuation** E[Φ(e^ℓ(s_real+θ))] = Φ(e^ℓ(s+θ)/√(1+(e^ℓ s_sd)²)):
  verified by Monte Carlo to ≤1e-4 across representative states (scratch
  check 1). The lapse mixture passes through the expectation linearly. Same
  formula verified in engine `update`, `_expected_loss_vec`, filter `_p_yes`,
  `trainer_greedy`, `trainer_rollout`.
- **GH smearing** (F30): 11-node Gauss–Hermite E[w(s_real)] is implemented
  correctly (nodes s+√2·s_sd·x, weights w/√π), and the closed-form
  `expected_skill_weight` = ρ/√(ρ²+τ²)·exp(−(μ−m)²/2(ρ²+τ²)) is the exact
  Gaussian convolution of the *signed* window.
- **R–W ↔ cross-entropy gradient** (plan Eq. cegradient): re-derived; the
  sign/zero correspondence and the state-dependent multiplier
  m_k=(1−2λ)φ(z)/[σ·ŷ(1−ŷ)] are stated correctly, and the plan is honest that
  R–W-with-constant-α is a behavioural postulate, not algebra.
- **Placement constants** (F2/D6): m(a)=Φ⁻¹((a−λ)/(1−2λ)); 1.0772256 at
  λ=0.025 for the Wilson optimum Φ(1) — re-verified. (But see MA-4 for the λ
  it is conditioned on.)
- **Hard-rule conditional kernel** (F21): δ = y_obs − y\* for every particle is
  the exact conditional mean given the observed response. Correct.
- **Bias-probe score** ≈ Fisher information: the φ(z̃)/att proxy attains 99.9 %
  (mean; min 94.8 % over 300 random pools) of the exact Fisher information
  about t of the best item — the proxy is fine as-is (scratch check 5).
- **Reward telescoping and Jensen**: E|t'|−|t̄'| is monotone in |t̄'| (ranking
  preserved); the lognormal correction to the σ-term is ×exp(q_σ²/2)≈1.0002
  (immaterial — but the plan's word "exactly" is wrong, see MA-10).
- **Systematic resampling** (one uniform offset, searchsorted, fp guard):
  correct, and genuinely lower-variance than the multinomial it replaced.
- **Adaptive-design ignorability**: item selection depends only on the
  observed history, so the likelihood-based filter remains valid under
  adaptive placement — no correction needed. The plan's ordering
  (measurement step, then prediction step) matches the code.
- **Benchmark statistics**: CRN seed-pairing + paired bootstrap contrasts are
  the right design; censoring is reported alongside means.

---

## 3. Findings

### MA-1 (HIGH) — the propagate kernel drops the y ↔ s_real coupling; the exact conditional kernel restores calibration

**The math.** The one-step generative truth is

    p(y, θ' | θ, s) = ∫ N(s_real; s, s_sd²) · Z(y | θ, s_real) · T(θ' | θ, s_real, y*) ds_real.

The correct Bayes filter therefore propagates with the **y-conditioned**
kernel p(θ'|θ, y) whose moments are taken under the node posterior
p(s_real | θ, y) ∝ N(s_real; s, s_sd²)·Z(y|θ, s_real):

- mean: E[p(s_real)|θ, y] (soft-rule δ), E[w(s_real)|θ, y] (skill channel) —
  *for the hard rule too, the w-channel needs this*;
- variance: q_t² + α_t²·Var[p(s_real)|θ, y] and
  q_σ² + (α_σ·(log σ−log σ_∞))²·Var[w(s_real)|θ, y].

The shipped filter uses the *unconditional* moments E[p], and (only with
`smear_w=True`) E[w]; the extra conditional variance is nowhere ("folded into
the process noise" — but q_t, q_σ are constants, so it isn't). At s_sd=0 all
corrections vanish; this is exactly why F30's SBC dip disappeared in the
s_sd=0 control.

**Size of the moment gap** (scratch check 4, realistic states, s_sd=0.85):
E[p|y=1]−E[p] ≈ +0.05…+0.10 and E[p|y=0]−E[p] ≈ −0.10…−0.21, i.e. a
per-trial kernel-mean error α_t·Δ ≈ 0.01–0.04 — the same order as the entire
process noise q_t = 0.04–0.05. This is not second-order.

**Effect on the filter** (open-loop paired experiment, identical data streams
across arms; 30 seeds × 300 trials, N=1000, s_sd=0.85, eval-seeded prior;
`studies/study_exact_kernel_audit.py`):

| arm | RMSE(t) | RMSE(ℓ) | cov_t@90 | cov_ℓ@90 | width_t |
|---|---|---|---|---|---|
| shipped | 0.2074 | 0.0887 | 0.631 | 0.771 | 0.386 |
| smear_w (F30) | 0.2073 | 0.0815 | 0.630 | 0.893 | 0.386 |
| cond (exact mean) | 0.2037 | 0.0815 | 0.757 | 0.899 | 0.488 |
| **cond+var (exact kernel)** | **0.1989** | **0.0816** | **0.908** | **0.909** | 0.674 |

Paired contrasts vs shipped: Δcov_t +0.277±0.012, ΔRMSE(t) −0.0085±0.0022,
ΔRMSE(ℓ) −0.0071±0.0027. Three readings:

1. **The shipped t-posterior is overconfident, not just noisy**: intervals are
   ~40 % too narrow for the error they carry. Coverage lands *at* nominal
   (0.908/0.909 vs 0.90) with *better* RMSE — the signature of the correct
   kernel, not of generic widening.
2. **F33 ("persistent 3–8 pp θ-undercoverage, SMC approximation error,
   document don't oversell") is largely this kernel error, and it is fixable.**
   The open-loop gap here is bigger than F33's closed-loop 3–8 pp because
   tier-2's noise-aware selection prefers low-s_sd items; the mechanism is the
   same.
3. **smear_w (F30) is the ℓ-half of this fix**; the exact kernel subsumes it
   and adds the t-half that had no mitigation.

**Downstream consequences to expect when adopting:**
- The honest sd_t is larger ⇒ the D16 bias condition |t̂|+0.5·sd_t ≤ t\* and
  the F5 variance floor both shift; graduation gets later and more honest —
  the same direction D19 already pays for. Re-measure `steady_state_sd`,
  re-run the M12 SBC/OC studies, re-pin benchmark claims (same protocol as the
  planned smear_w default-flip).
- **The plan's §6.1 y-sum collapse (and F25's boundary) weakens.** Under
  stimulus noise the exact next-state distribution depends on y even for the
  soft rule, so "soft ⇒ lookahead adds nothing" is an artifact of the
  approximate kernel. Tier-3's cost-benefit (F25) should be re-checked with
  the exact kernel before Phase-3 conclusions are locked.

**Cost:** O(11N) per trial with the GH arrays already imported; ~2× the
propagate cost, negligible against reweight+selection. Implementation is
~25 lines (a variant of `propagate`; see the study file's `CondVarTaskFilter`
for a reference implementation that reduces bit-identically to shipped at
s_sd=0 or `rule="static"`).

### MA-2 (HIGH, structural) — the learning parameters are outside the belief state, and the graduation pathology is the bill

The plan (§7.1) says "estimate τ_σ, σ_∞, α_t per learner from their
accumulated trial history"; what is implemented (F4/D7, correctly arguing
path degeneracy) is: *fix them at cohort point values*. The POMDP being
solved is therefore conditional on (α_t, α_σ, σ_∞) known — the belief
b_k(σ,t | α, σ_∞) rather than b_k(σ, t, α, σ_∞). Every posterior summary the
gates consume understates uncertainty by exactly the parameter uncertainty,
and the F28 mechanism (deterministic pull of every particle toward the
*population* σ_∞, so the cloud contains no "this learner can't reach the cut"
trajectories) is the visible symptom. F29's flat OC and the F31 mitigations
(p_static, evidence gate) treat this symptom.

**The principled fix is smaller than it sounds.** The parameter that decides
graduatability is σ_∞ (is this learner's ceiling above the cut?); α's and q's
mainly affect timing. A per-learner **discrete mixture over σ_∞** (5–9 grid
points spanning the §2B cohort spread, e.g. ℓ_∞ ∈ expert_ℓ_mean ± 2·√Σ_l_kk)
run as parallel filter strata with prequential model-evidence weights

    P(cell j | data) ∝ P(cell j) · Π_k p(y_k | cell j, history)

avoids path degeneracy entirely (no static parameter lives on a particle; the
incremental likelihoods are already computed in `reweight` — they just need
to be accumulated per stratum before normalization). Cost: ×J filters of
N/J particles each, i.e. constant total cost. Notes:

- `p_static` is precisely the 2-point special case {σ_∞ = population value,
  no learning}; the F31 evidence (careless 0/30 declared, coverage restored)
  is direct evidence the mixture direction works.
- The graduation output upgrades from "believed ℓ crossed ℓ\*" to a
  **trainability posterior** P(ℓ_∞ > ℓ\* | data) — the quantity F29's oracle
  arm shows is what a correct filter needs to reproduce the step-function OC.
- Same machinery extends to a coarse α_σ grid if pilot data shows rate
  heterogeneity matters (F17/D14 says lean low until then).
- This does **not** replace the D18 re-cert layer; it makes the trainer's
  *recommendation* honest instead of prophecy-driven (F28's phrase).

For Phase-3 offline fitting, the plan's hierarchical intent is right; the
audit adds: fit with the **exact kernel** (MA-1) or the dynamics estimates
inherit the kernel's bias; expect strong (α_σ, q_σ, σ_∞) confounding on
short sessions (an exponential approach with noise is notoriously
weakly identified at n ≪ τ_σ — the EXTSET median 22 reads/user is far below
τ_σ ≈ 33–50, so only the pilot's longer feedback sessions will identify it);
SMC²/particle-MH over the dynamics parameters with the per-task filters as
inner loops is the natural fitter, and the F52 lesson (over-dispersed inits,
R̂-gated SBC) transfers verbatim.

### MA-3 (MEDIUM) — the criterion update is not feedback-gated (F22, sharpened: fix now)

`learner_sim.Learner.step` and `TaskFilter.propagate` gate only the σ-channel
by f; the t-channel applies α_t·δ with δ = p−y\* (or y−y\*) **on every trial,
feedback or not**. That requires the learner to know y\* without feedback —
informationally impossible — and it is not hypothetical: the F34-i replay
anomaly (+0.256 ℓ̂ inflation on real *no-feedback* eval data with dynamics on)
is partly this term steering t during trials that carried no information to
steer with. Phase 2 hides it (all training trials give feedback), but every
mixed-feedback use — eval replays, the F15 lapse probes if ever served
feedback-free, Phase-3 fitting on mixed data — inherits it silently. The fix
is one line in each file (multiply the δ-term by f, in learner and filter
symmetrically) plus re-running test_step2/3 expectations. There is no reason
to wait for Phase 3.

### MA-4 (MEDIUM-LOW) — placement and likelihood constants are conditioned on a lapse rate the data reject

EXTSET fitting (F43, Bayesian F51) puts the real lapses at λ_fa≈0.0024,
λ_miss≈0.0079–0.0095 — the engine's locked 0.025 is 3–10× too high where the
instrument actually operates, with P(both λ < 0.025 | data)=0.9999. Trainer
consequences (scratch check 3):

- `SKILL_MODE_MULTIPLIER` = m(Φ(1); λ): 1.0772 at λ=0.025, but ≈1.007–1.028
  at the fitted λ. The shipped placement trains at ~85.2–85.8 % accuracy
  instead of the targeted 84.1 % — a small, systematic too-easy bias in the
  one constant the whole skill mode is built around.
- The likelihood's floor/ceiling at 0.025 mildly discounts confident errors
  (an error on an easy item reads as "lapse" rather than "signal"), which
  slightly slows criterion/skill inference; the F15 lapse-probe channel
  (P(error) ≈ λ on very-easy items) is mis-centered by the same factor.

Action at port: make λ an instrument parameter (v14/v15 style), recompute all
placement constants from it via the existing `difficulty_multiplier(a, λ)`
(the code is already parameterized — this is a constants-refresh, not a
redesign), and consider the asymmetric-lapse form the EXTSET fits prefer
(M2: λ_fa ≠ λ_miss breaks the (1−2λ) symmetry but the closed forms all
generalize trivially).

### MA-5 (LOW) — fold-then-smear E[w] under-counts near-boundary items ~2×

`trainer_greedy._expected_reward` and `expected_skill_weight` smear the
*signed* window after folding: E[w] ≈ ρ/√v·exp(−(|μ|−m)²/2v). The true
expectation of the folded window exp(−(|d|−m)²/2ρ²) has a second bump at
d=−m; at μ=0 the closed form returns ~51–53 % of the true value for
τ ∈ [0.5, 1.5] (scratch check 2). The docstring's "fine for ranking" holds
for skill mode (optimum is near ±m, where the error ≤ a few %), and the
under-count only *further* deprioritizes near-boundary items, which is the
direction F7/F23 want. But if Q is ever consumed as a calibrated value
(tier-3 comparisons across modes, Phase-3 reward fitting), use the exact
two-bump form — it is the sum of two of the existing closed forms
(ρ/√v)·[e^{−(μ−m)²/2v} + e^{−(μ+m)²/2v}] up to the folding of the tails,
or one GH pass. Same cost.

### MA-6 (LOW) — gate-constant drift and MCSE optimism

- `ModeThresholds.sd_floor = 0.23` hard-codes 1.5× the *synthetic* floor
  (0.154). The measured real-bank floor is 0.179 (F5/M10), so the shipped
  margin is 1.28× under the physics the trainer actually runs in — tighter
  than the design rule, and it will shift again if MA-1 is adopted (the exact
  kernel injects honest extra variance). Recommendation: compute sd_floor at
  construction from `steady_state_sd(s_sd=median_bank_sd)` × 1.5 instead of a
  constant.
- `pass_mass` MCSE = √(π(1−π)/ESS) is correct under weight independence, but
  ESS resets to N at every resample while distinct-particle count doesn't;
  right after resampling the MCSE is optimistic. Cheap honest alternative:
  use min(ESS, #unique particles) in the denominator. The Z=2 buffer eats
  most of this in practice; worth one line all the same.

### MA-7 (LOW, improvement) — replace the mean-skill gate's ad-hoc repeated-look correction with an anytime-valid test

`_meanskill_ok` clips the AR(1) n_eff at 1 (any r₁ ≥ 0.9 → SE = full
posterior SD), then tests every trial. It conflates posterior SD with the
sampling variability of a time-averaged, strongly autocorrelated estimator,
and its type-I control under continuous monitoring is only sandbox-verified
(F19: 0 observed false fires at −0.15 margin). Since D16 acceptance semantics
still await PI ratification, propose the ratified version be an
**anytime-valid confidence sequence** on ℓ (e.g. a sub-Gaussian e-process /
mixture-martingale lower bound on the trailing posterior means): identical
inputs, a lower confidence bound valid at *every* stopping time, so
"gate fires the first time LCB ≥ ℓ\*" has provable repeated-look control —
the property the current z=1.645 constant only approximates.

### MA-8 (LOW) — tier-3 plans with a different model than the filter believes

`trainer_rollout.tier3_select` hard-codes the shipped kernel inside rollouts:
no `p_static` stratification, no `smear_w`, (and no MA-1 conditioning) even
when the outer filter is hardened. A hardened trainer would therefore
*evaluate* candidates under dynamics it has already partially rejected —
lookahead value estimates inherit the σ_∞-attractor optimism F28 describes.
Bounded today (tier-3 is not the ship default), but fix before any Phase-3
tier-3 revisit: route the rollout transition through one shared kernel
function used by `learner_sim`/`TaskFilter`/`trainer_rollout` so the three
cannot drift (they are already three hand-synchronized copies — the CRN
correctness of the file is good, but kernel unification is overdue).
Also worth noting: the per-rollout belief-resampling consumes RNG
data-dependently; the step-noise CRN alignment survives (draws are tiled
before that point), so this is only a variance nit, not a bug.

### MA-9 (LOW) — hardened filter and policy scoring disagree about who is learning

`_expected_reward` (tier-1 Q, and `bias_correction_score` which tier-2's
corrective bias picks use) computes the deterministic next state for **every**
particle, ignoring `filt.learn`. With `p_static=0.3` (D19 recommended
config), 30 % of the belief mass is asserted non-learning, yet items are
scored as if that mass learns — expected rewards are uniformly inflated and,
near the static/learning disagreement region, mis-ranked. One-line fix:
multiply the δ- and g_σ-terms by `filt.learn[None, :]` inside
`_expected_reward` (it already receives the filter).

### MA-10 (INFO) — plan/documentation errata

1. Plan §6.1: process noise does **not** integrate out "exactly" for the
   σ-term (lognormal ⇒ ×exp(q_σ²/2); ≈1.0002 — immaterial, but say
   "to O(q²)").
2. Plan §6.1's y-sum collapse for the soft rule is only true under the
   approximate kernel (see MA-1); the general Eq. Qgreedy form becomes the
   correct one once s_sd physics is exact.
3. Plan Eq. (eq:eightyfive) still prints the 1.04 constant that F2 corrected;
   the memo should carry the correction (readers of the memo alone will
   re-introduce it — the constant is load-bearing).
4. `_coarse_to_fine_argmin` requires candidates sorted by signal; nothing
   asserts it. Current callers sort (`pipeline_demo` line 81); one
   `np.all(np.diff(sigs)>=0)` debug-assert would make the precondition
   explicit.
5. `plan_mean()`/placement use σ̂ = exp(−E[ℓ]) (the posterior geometric mean,
   ~2.7 % below E[σ] at SD 0.23) — fine, but worth a one-line docstring note
   since mastery compares E-side quantities.
6. The trainer's t-channel reward is in t-units while mastery/cuts live in
   ℓ-units; a Δℓ-based skill reward (β_σ·(ℓ_{k+1}−ℓ_k)) would be
   scale-invariant, align the greedy objective with the gate geometry, and
   make β_t/β_σ comparable across the σ range — cosmetic today (tier-2
   doesn't read the reward), relevant the moment tier-1/3 matter.

---

## 4. Improvement opportunities beyond the fixes (ranked by value ÷ cost)

1. **Exact conditional kernel** (MA-1) → then re-run V4 SBC, F5 floor, D16
   thresholds, and re-pin benchmarks. Expected: F33 closed, honest (later)
   graduation, fewer wasted re-certs — the same direction as D19 but from
   correctness rather than conservatism.
2. **σ_∞-mixture belief** (MA-2): the smallest change that makes graduation a
   statement about the learner instead of the prior. Replaces/keeps p_static
   as its 2-point case; reports P(ℓ_∞ > ℓ\*) as the recommendation statistic.
   Re-run the F29 OC grid; the oracle arm's step function is the target.
3. **Certification probes inside training** (refines D18/F50): the structural
   reason graduation self-assessment fails is that training-optimal placement
   is information-poor for the ℓ-vs-ℓ\* contrast under bank s_sd (F31). The
   information-optimal probe for that contrast is computable with the tools
   already present: maximize the attenuated Fisher information about ℓ
   *evaluated at the cut state* (σ=σ\*, t=t̂) — i.e. low-s_sd items at
   |s−t̂| ≈ σ\* (z≈1 under H₀ "at the bar"), which is *not* where skill mode
   places (z≈1.077 under the *believed* σ̂ — the difference is exactly the
   self-confirmation loop). A 1-in-10 probe schedule gives the evidence gate
   (D19) real discriminating data instead of asking it to milk training
   trials that cannot contain the contrast.
4. **Anytime-valid gates** (MA-7) — pairs naturally with 3: probes feed a
   clean e-process on the raw outcomes, closing the F31 "raw-evidence gate is
   blind to latent shortfalls" gap at the source.
5. **Tier-3 variance reduction, then re-adjudicate F25 under the exact
   kernel**: CRN is in; add the free control variate (subtract each rollout's
   one-step Q̂ from its return — the difference estimates the *lookahead
   increment* directly), and the soft-rule "MC-noise tax" that made tier-3
   lose by 2–4 trials may vanish. With MA-1 making even soft dynamics
   y-conditional, the F25 boundary ("tier-3 pays only for hard rule") needs
   one fresh 30-seed campaign before it hardens into port lore.
6. **Cross-task strength within a session** (post-D2 middle path): the K
   independent filters discard the Σ_l positive manifold W4 measured
   empirically (ρ≈0.29–0.67). A cheap EB recoupling — after each task's
   update, shrink the *dynamics-parameter strata weights* (not the state)
   across tasks — recovers most of the pooling benefit with none of the F1
   trajectory-replay problems. Phase 3, but the ledger already stores what it
   needs.
7. **Phase-3 fitting protocol** (when pilot data exists): fit the dynamics
   hierarchically with the exact kernel via SMC²/PMMH; pre-register the
   soft-vs-hard rule comparison (F12) as a model-comparison endpoint —
   the trainer's tier choice (F25) hinges on it; apply the F52 SBC hygiene
   (over-dispersed inits, R̂ gating) to the fitter before trusting it.

---

## 5. Suggested action order

| step | action | effort | verifies/re-pins |
|---|---|---|---|
| 1 | MA-3 feedback-gate fix (2 lines) | trivial | test_step2/3; re-run F34 replay arm |
| 2 | MA-1 exact kernel as `TaskFilter(exact_kernel=True)` opt-in | ~25 lines (reference impl. in the audit study) | study_train_sbc, steady_state_sd, gate_oc, benchmark re-pin |
| 3 | MA-9 + MA-6 one-liners | trivial | test_step6/7 |
| 4 | MA-2 σ_∞-mixture prototype | ~1 day | F29 OC grid vs oracle arm |
| 5 | Probe schedule + e-process gate (improvements 3–4) | ~2 days | F31 static_below residual (23/30 → target ≈0) |
| 6 | Re-run F25 tier-3 campaign under exact kernel (improvement 5) | hours (CRN harness exists) | F25 conclusion |
| 7 | MA-4 λ-parameterization at port | port-time | placement constants, F15 probe channel |

Items 1–3 are safe now (opt-in, bit-preserving defaults, same protocol as
smear_w). Items 4–6 are the pre-pilot work that most changes what the pilot
can claim. Item 7 rides the port.

---

## Appendix — evidence trail

- `studies/study_exact_kernel_audit.py` — MA-1 headline experiment
  (reproduces the §3.1 table exactly; `--smoke` ≈ 90 s).
- Scratch checks (session scratchpad `audit_checks.py`, results quoted
  inline): (1) attenuation ≤1e-4; (2) fold-then-smear worst-case 0.51× at
  μ=0; (3) λ-sensitivity of the placement constant; (4) conditional-moment
  gaps Δ up to ±0.21; (5) probe-score Fisher efficiency 0.999;
  (6) lognormal/Jensen magnitudes.
- All other findings are line-cited to the audited files in §3 and checkable
  by reading the named functions.

---

# Part II — implementation log (2026-07-01, the seven steps executed)

Every step below was implemented the same day, surgically (shipped defaults
bit-preserved unless stated; full suite green — `tests/test_audit_fixes.py`
adds 15 pinning checks). Each entry records what changed, the mathematical
optimization applied, and the justification of every estimated/chosen
parameter.

## Step 1 — MA-3 feedback gating (DEFAULT-changing correctness fix)

`learner_sim.Learner.step` and `TaskFilter.propagate` now multiply the
criterion update by f ∈ {0,1}. Justification: δ = p−y\* (soft) or y−y\*
(hard) is a function of the ground-truth label, which the learner possesses
only when feedback is delivered; the ungated form (F22) was informationally
impossible and produced the F34-i replay artifact. All Phase-2 training
paths pass feedback=True (f=1.0, multiplication bit-exact), so every pinned
behavior is unchanged — verified by the full suite. Pinned by checks 1–2.

## Step 2 — MA-1 exact conditional kernel (`TaskFilter(exact_kernel=True)`)

Implemented as derived in §3.1: node posterior p(s_real|θ,y) ∝
N(s; s_mean, s_sd²)·P(y|s_real,θ); conditional means E[p|y], E[w|y] and the
conditional-variance injections q_t² + f·learn·α_t²·Var[p|y] and
q_σ² + f·learn·(α_σ·gap)²·Var[w|y]. The variance rides the same f/learn
gates as the mean — a non-learning or no-feedback particle has no α·δ term,
hence no Var[δ] inflation.

**Node-count estimation (revised under audit-of-the-audit):** the audit ran
at 11 GH nodes; checking 11 vs an 81-node reference exposed worst-case
conditional-moment error 2.2e-2 in the realistic regime s_sd/σ ≤ 2 (the
likelihood factor concentrates node mass in one tail — a harder integrand
than F30's unconditional bump). 21 nodes bring it to ≤ 2.0e-3, i.e.
per-trial kernel-mean error ≤ α_t·2e-3 ≈ 4e-4 — two orders below q_t —
at 2× a trivial cost. The kernel therefore uses a dedicated 21-node rule
(`_GHE_*`); smear_w keeps its 11-node rule (bit-pinned by the M12 SBC).
Error table (worst |E[p^m|y]−ref| over 8k random states):

    s_sd/σ bin   11-node max   21-node max   31-node max
      [0,1)        1.5e-4        2.7e-7        7.6e-10
      [1,2)        2.2e-2        2.0e-3        2.1e-4
      [2,3)        7.3e-2        1.7e-2        4.0e-3

**Measured consequences:** headline table re-pinned at 21 nodes (Part I
§3.1 — unchanged to MC noise; `repo-flag` arm proves
`TaskFilter(exact_kernel=True)` ≡ the study reference bitwise). The F5
information floor rises as predicted — like-for-like at s_sd=0.85:
θ-SD 0.153→0.222, ℓ-SD 0.114→0.155 — so the D16 `sd_floor` MUST be
recomputed via `recommended_sd_floor(params, s_sd=…)` when the kernel is
adopted (that helper is the Step-3 deliverable). Pinned by checks 3–4.

## Step 3 — MA-9 + MA-6 one-liners

- `_expected_reward` multiplies both deterministic reward terms by
  `filt.learn` (check 5: an all-static belief now earns exactly zero Q;
  p_static=0 ⇒ learn=1 ⇒ bit-identical).
- `pass_mass` MCSE denominator = min(ESS, n_anc), n_anc = distinct-lineage
  count at the most recent resample. Estimation rationale: ESS reads N
  right after every resample although the support holds n_anc < N atoms;
  compounding lineage counts across resamples would over-correct because
  the process noise re-mixes the cloud over ~1/α trials, so the single-
  resample count is the honest leading term. Conservative direction only
  (check 6: mcse 0.052 vs ESS-only 0.028 at n_anc=87/300).
- `recommended_sd_floor(params, s_sd, mult=1.5)`: computes the gate floor
  from the measured steady state instead of the 0.23 constant (which
  encodes the s_sd=0 floor; real-bank is 0.179 ⇒ 0.27, exact kernel higher
  still).

## Step 4 — MA-2 σ_∞-mixture filter (`training/mixture_filter.py`)

Bayesian model averaging over J ceiling hypotheses; stratum weights update
by prequential evidence log ω_j += log p_j(y_k|H_k) (the reweight
normalizer, now returned by `TaskFilter.reweight`). Exact over the grid;
valid under adaptive selection by the same ignorability argument as the
filter itself; immune to static-parameter path degeneracy because the
hypothesis weights live outside the particle system (F4's objection).

**Parameter estimation justifications** (full derivations in the module
docstring): grid center = D10 expert ceiling (the only data-grounded
ceiling); τ = 0.30 bracketed by [F29-envelope resolvability, EXTSET
population σ_ℓ = 0.434 as the upper bound since Var(skill) ≥ Var(ceiling)],
giving P(ceiling < cut) ≈ 0.22 for domain3 — consistent with the F28
heterogeneity draws; J = 7 over ±2.2τ because the spacing 0.73τ ≈ 0.22 is
already below the filter's own ℓ-resolution (the F5 floor), so finer grids
buy nothing; grid extended downward when needed so the decision boundary
ℓ\* is straddled; static stratum (rule="static", prior 0.15) subsumes
F31(a)'s p_static — prior mass is second-order because posterior odds move
exponentially in the log-evidence.

**Output upgrade:** `trainability(ℓ*) = P(ceiling > ℓ* | data)` — the
honest D18 recommendation statistic. Results: see the Step-5 table.

## Step 5 — certification probes + anytime-valid e-gate

- `cert_probe_score`: Fisher information about ℓ evaluated at the CUT state
  (σ = σ\*, t = t̂) — the anti-circularity placement (training placement
  scores difficulty against the believed σ̂, which is exactly what the F28
  attractor corrupts). Derivation surprise worth recording: because ℓ is a
  SCALE-type parameter (∂z̃/∂ℓ = z̃/att²), the information optimum is
  z̃ ≈ ±1.35 under λ=0.025 (between the location-parameter 0 and the
  no-lapse scale optimum 1.57), not the z̃ ≈ 1 the initial draft assumed —
  caught by the pinning test (check 12), docstring corrected.
- `EProcessGate`: mixture of Hoeffding e-processes on probe outcomes vs the
  at-bar accuracy a0_i, e_λ = exp(λΣ(a0−c) − kλ²/8). Increments are
  1/2-sub-Gaussian (range 1) ⇒ each e_λ is a supermartingale under every
  "at-or-above bar" H0, and Ville gives P(ever fire | H0) ≤ α at ANY
  stopping time — the honest repeated-look control MA-7 asked for.
  λ-grid {0.05,…,0.8} covers accuracy deficits δ ∈ [0.0125, 0.2]
  (optimal bet λ\*=4δ, growth 2δ²/probe): careless-grade deficits fire in
  ~30–60 probes; near-bar deficits need ~δ⁻² probes — a Bernoulli
  information bound, not test inefficiency; the mixture covers that case
  through the full likelihood instead. Measured OC (check 10–11): type-I
  0/200 at H0, power 200/200 against δ=0.2 within 200 probes.
  probe_every=5 in the study (20% overhead, pilot-stage setting).

**Validation** (`studies/study_audit_hardening.py`, domain3 real-bank pool,
D16 hybrid gate, 20 seeds × 400 trials):

    learner        arm        declared  med n   FG   med trainability  egate
    wellspec       shipped      20/20     49    10        —              0
    wellspec       exact        20/20     62     8        —              0
    wellspec       mix          20/20    109     2       0.95            0
    wellspec       mix+probe    19/20    119     3       0.96            0
    static_below   shipped      20/20     47    20        —              0
    static_below   exact        20/20     62    20        —              0
    static_below   mix           7/20    125     7       0.60            0
    static_below   mix+probe     6/20    130     6       0.38            0
    careless       shipped      20/20     65    20        —              0
    careless       exact        20/20     85    20        —              0
    careless       mix           0/20      —     0       0.00            0
    careless       mix+probe     0/20      —     0       0.00           20

Readings: (1) shipped reproduces F28 in full; (2) the exact kernel alone
does NOT fix graduation — calibration fix, not ceiling-knowledge fix
(attribution confirmed; it does cut wellspec premature-FG 10→8 and delay
honestly); (3) the mixture kills the careless false graduations outright
(0/20, e-gate independently 20/20) and cuts static_below from 20/20 to
6–7/20 — better than the M12 hardened stack's 23/30 — while wellspec still
declares 20/20 with FG 10→2 at a ~2.2× later median (the honest-timing
cost D19 already accepted). (4) A trainability-conjunct does NOT remove
the static_below residual: the 7 declaring seeds had trainability
0.77–0.99 at declaration — the mixture itself was fooled on those seeds,
so the residual is F31's structural information poverty (400 trials at
bank s_sd cannot always resolve a ceiling 0.22 below the cut), not a gate
artifact. Certification-grade discrimination remains the D18 re-cert's
job; the mixture's contribution is an honest per-learner recommendation
where the shipped filter had a prophecy. (5) Secondary finding: mixture
discrimination speed depends on served-item PRECISION (at uniformly high
s_sd the ceiling hypotheses stay near-indistinguishable for hundreds of
trials) — an independent argument for the probe schedule beyond the
e-gate.

## Step 6 — F25 re-adjudicated under the exact kernel

Paired CRN campaign (20 seeds, domain3, pool 600, `run_one(filter_kwargs=)`
passthrough added):

    rule  kernel   tier2  tier3   paired t3−t2
    soft  shipped   47.5   49.1   +1.6 [−1.6, +4.8]
    soft  exact     47.5   48.0   +0.5 [−1.8, +2.7]
    hard  shipped   48.3   46.8   −1.6 [−6.4, +2.6]
    hard  exact     48.3   48.5   +0.1 [−1.1, +1.3]

The hard-rule tier-3 advantage (F25's −2.9 at 30 seeds) vanishes under the
exact kernel: the lookahead was partly compensating for belief mis-tracking
that the kernel fix removes, and the paired-contrast variance collapses
(±1.2 vs ±4.5) — honest beliefs also de-noise the policy comparison.
Tier-2-as-default is REINFORCED; tier-3's Phase-3 case must be re-argued
from fitted dynamics, not from F25. (Caveats: 20 seeds vs F25's 30; pool
600 vs 1000; CIs overlap 0 in all cells.)

## Step 7 — λ scaffolding + consolidation

`bridge_conventions.skill_mode_multiplier(lapse, target_acc)` added (port
scaffolding; frozen constant untouched — λ=0.025 remains a locked engine
invariant until the port decision). Checks 14–15 pin the equivalence at
λ=0.025 and the fitted-λ direction (1.0173 at λ=0.006).

## Regression status

`test_step0/1/2/2_5/3/5/6/7`, `test_misspec`, `test_tier3`, `test_v15` and
the new `test_audit_fixes` (15 checks) — all green post-change; the three
default-affecting fixes (MA-3 gating, MA-9 learn factor, MA-6 MCSE) altered
no pinned number (f=1/learn=1 paths are bit-exact; the MCSE change is
conservative and did not move any pinned graduation timing).

---

# Part III — M16 continuation (2026-07-01, same-day iteration)

The next-step items executed after Part II, each derived → tested → verified
(memory findings F57–F59):

## III.1 F33 closed in CLOSED loop (F57)

Part II's coverage evidence was open-loop. The 200-rep closed-loop SBC (the
F30/F33 harness — truth from the filter's own prior, live tier-2 selection,
real-bank pool) now shows: shipped kernel reproduces the archived record
exactly (ℓ dip at n=50: cov90 0.82, KS 0.22; θ persistently 0.82–0.88 @90);
exact kernel restores NOMINAL coverage at every checkpoint and both
coordinates (ℓ 0.91/0.90/0.95/0.92, θ 0.90/0.89/0.91/0.94 @90 at
n=10/50/150/300; all KS ≤ 0.08). `data_train_sbc{,_exact}.npz` re-pinned.
The D21 port-flip case is now complete: SBC ✓, floor ✓ (`recommended_
sd_floor`), F25 re-adjudication ✓, benchmark re-pin ✓ (see III.4).

## III.2 D23 wired into the trainer (opt-in)

`TrainerPolicy(probe_every=, egate_alpha=)`: cert-probe override on the
per-task served-trial cadence (never hijacks retention trials), e-gate
attached per task, declaration blocked while the gate is `elevated`.
Blocking uses the CURRENT mixture e-value rather than the sticky
sup-crossing: a learning learner who fires the gate during an early honest
deficit self-releases as increments turn negative, while a persistent
deficit keeps positive drift (effectively permanent block) — the sticky
Ville-valid `fired` flag is retained for diagnostics and upper-bounds the
block's false-positive persistence. Default `None` is bit-identical
(test_step5 re-verified); pinned by checks 16–19.

## III.3 Phase-3 offline dynamics fitter, train/test-validated (F58/F59)

`training/dynamics_fit.py`: per-learner penalized ML of
(log α_t, log α_σ, ℓ_∞) by Nelder–Mead on the CRN prequential filter
evidence (the reweight normalizer — the filter IS the likelihood
evaluator); q/ρ held fixed (F4/D7 — weakly identified from single
sessions); weak N(x₀, 1.5²) log-space penalty (×4.5 one-sigma band, wider
than F28's ±4× stress bound); exact-kernel filter so fits don't absorb
kernel bias. Validation: `studies/study_phase3_fit.py` (24 train / 16 test
learners, temporal 150/150 splits):

    A recovery      ℓ_∞ identifiable per learner (corr 0.71 @ n=150, RMSE
                    0.22 @ n=300 vs cohort SD 0.21); rates weakly identified
                    per learner even at n=300 (corr ≤ 0.35) — but the
                    POPULATION means move correctly on all three axes:
                    α_t 0.200→0.240 (true 0.320), α_σ 0.060→0.061 (0.075),
                    ℓ_∞ 0.733→0.829 (0.800).
    B held-out      fitted-pop ≈ 0, oracle +0.003/trial; PERSONALIZED fits
                    on 150 trials OVERFIT (−0.008±0.004/trial) ⇒ the plan's
                    hierarchical-fit prescription is now evidence-backed:
                    per-learner estimates need shrinkage before deployment.
    C rule-ID       15/20 correct (p ≈ 0.02), median margin 0.72 nats —
                    per-learner inconclusive at 300 trials, decisive at
                    cohort scale (evidence adds across learners).
    D closed loop   fitted population params HALVE declaration lateness
                    (median n_decl − n_true: 41 → 21.5 wasted trials;
                    oracle 7) at unchanged learning efficiency; FG pattern
                    (fitted-pop 2, oracle 0 of 16) ⇒ feed fitted ℓ_∞ into
                    the D22 mixture CENTER, not a point filter — a higher
                    point-ceiling strengthens the F28 attractor for the
                    rare non-trainable learner.

**Methodological finding worth its own line (F59):** one-step prequential
evidence is nearly FLAT in the dynamics parameters (~0.005 nats/trial even
for gross σ_∞ error) because the filter's state-tracking self-corrects
every step. Design consequences (all baked into the study): identification
lives in the TRANSIENT (biased sub-skill starts; post-convergence trials
dilute the signal); evidence comparisons need multi-seed CRN averaging;
pilot dynamics analyses must be powered on decision-level outcomes
(declaration lateness, FG rate, trainability classification) — never on
ΔELPD.

## III.4 D21 re-pin: benchmark under the exact kernel

Paired-CRN policy benchmark (20 seeds, domain3, pool 600), shipped vs exact
kernel: tier2 Δ +0.0 (identical trajectories — the mode-conditional argmax
picks are belief-robust, replicating M10's rate-misspec insensitivity),
tier1 −2.3 ± 0.8 (greedy Q benefits from honest beliefs), staircase85 and
measure_opt unchanged within noise; ordering preserved everywhere. As
predicted: the kernel's value is calibration and declaration honesty, not
raw placement speed.

---

# Part IV — M17 (2026-07-01): user-ratified decisions executed

User inputs: flip defaults; priorities FG ≥ lateness > speed > retention;
open-ended 40-trial sessions; measurement-based retention; v15 cuts;
OQ8 = feedback was given; goal = zero bias; λ studies permitted; 42 cores.

## IV.1 Defaults flipped (D26) and v15 targets (D25)

`exact_kernel=True` is now the TaskFilter default (historical arms pass
False explicitly); the full 14-file suite re-pinned with zero assertion
changes (all pins were relative). Floor documentation: at s_sd = 0.85 the
1.5× rule gives sd_floor 0.246 (default params) / 0.131 (benchmark params)
— the 0.23 constant stays, erring toward FG safety. Trainer targets moved
to the v15 cuts (`ELL_STAR_V15` etc.); every ceiling-to-cut margin widens.
`pipeline_demo` deliberately stays v14-coherent: trainer targets must match
the re-cert instrument or graduates would be systematically under-trained —
the pipeline flips together with the eval-side v15 switch.

## IV.2 The zero-bias theorem (F60, resolves OQ6)

For the soft R–W criterion under mirror-paired, label-alternating items
anchored at the true boundary (offset a, effective noise σ_e = √(σ²+s_sd²)):

    E[t_∞] = 0                            (anchoring sets the mean — F23),
    κ      = α_t(1−2λ)·φ(a/σ_e)/σ_e       (restoring gain),
    Var_∞  = (q_t² + α_t²·Var[p̂])/(κ(2−κ)) + (α_t·δ̄)²/(2−κ)².

Three structural corollaries, each verified numerically (≤3% formula-vs-
simulation error across the parameter grid; test_audit_fixes checks 20–26):
1. Placement that TRACKS the criterion (s̄ = t̂) has no restoring force —
   the attractor follows the learner and t random-walks (SD ≈ 3 vs ≈ 0.13).
   The M10 herding pathology (F23), now analytic.
2. Strict label ALTERNATION is load-bearing: iid-random balanced labels
   inject α_t²/4 of diffusion — an order-of-magnitude worse floor. F6b's
   balance constraint is control theory, not just fairness.
3. "Zero bias" is achievable exactly down to the band ±c·SD_∞ and no
   further; the D16 gate's bias condition now demands that derived,
   σ̂-adaptive band by default (`derived_t_star`, ≈ 0.25 for benchmark
   params vs the retired 0.30 assumption). Demanding less buys pure
   lateness. Zero-bias serving coincides with skill serving to within ~15%
   of the offset-optimum, so no dedicated mode is needed inside the band.

## IV.3 Measurement-based retention (F61, D27 — the user's design)

Rather than committing to a forgetting law, `GapAnchor` measures each gap:
session-open mixture over retention fractions r ∈ {1, .8, .55, .3, 0}
toward personal baselines (prior tilted by a one-scalar personal stability
S, sd 0.25 — evidence dominates), resolved by a 4–16-trial anchor block of
Fisher-optimal probes, then collapsed; S updates by through-origin LS of
−log r̂ on Δt. Validation against a POWER-LAW truth (off-grid, off-tilt;
8 × 40-trial sessions, gaps 0.5–21 d, 20 seeds):

    arm        |t̂−t| post-gap   |ℓ̂−ℓ| post-gap   stale-mastered   probes
    identity        0.235            0.330              71             0
    widen-only      0.228            0.328              11             0
    GapAnchor       0.111            0.067               0            62

The stale-mastered count is the priority-1 metric (a decayed learner the
gate still believes mastered). Update-cadence answer (D28): estimates
update every trial in-session; between sessions the anchor block IS the
update, its length personalized through S.

## IV.4 λ sensitivity (F63 part 1) and real-data anchoring (F62)

Open-loop calibration under the EXTSET-fitted asymmetric lapse
(λ_fa=0.0024, λ_miss=0.0095) with the filter assuming the locked 0.025:
coverage indistinguishable from well-specified (cov_t/cov_ℓ @90 =
0.921/0.919 vs 0.923/0.918) — the locked λ is calibration-benign; the
placement-constant refresh (D24) remains a port nicety. The zoo-OC arm and
the EXTSET population fits are recorded in the M17 §6 row of
PROJECT_MEMORY (campaign artifacts `data_m17_capstone.npz`,
`data_extset_dynamics.npz`).

---

# Part V — M18 (2026-07-01): the M17 queue executed

## V.1 The static_below residual was the mean-skill gate (F64/D30)

The M18 hardening ablation instrumented GATE-BRANCH ATTRIBUTION at every
declaration: in the M17 baseline, **all 11/11 static_below false
graduations fired via the D16 mean-skill branch** (pass-mass: 0). The
mechanism is structural: each high-ceiling stratum of the σ∞-mixture is a
conditional F28 filter whose cloud converges to its own attractor, so the
mixture's MEAN ℓ̂ is dragged upward even while the mixture WEIGHTS are
still resolving honestly — and the trailing-mean test reads that mean.
The branch exists for v14 domain1's sub-floor margin (F19); under v15
margins its rationale is gone. `meanskill_gate=False` for mixture
graduation: static FG 11→1/30, powerlaw FG 8→1, wellspec median
declaration 104→131 (8/30 pushed past the 400-trial window — later
sessions under the open-ended protocol). τ=0.45 arms reached FG 0/30 at
~2× lateness — recorded as the PI-visible frontier knob, not adopted
(D28 judgment: 3.3% residual FG feeding a re-cert layer that catches ~90%
does not justify doubling every genuine learner's declaration time).

## V.2 Hierarchical refit on real fits (F65) — a sim-to-real reversal

2,040 split-half fits over 170 users: per-user RATES are moderately
reliable on real data (split-half r: log α_t 0.47, log α_σ 0.48 — they
were unidentifiable in the F58 sim; real heterogeneity is simply much
larger, echoing F49). Held-out second-half evidence: raw personalized
params beat the population by +0.0081±0.0015/read — REVERSING the sim's
overfit result — while the shrinkage estimator under-performed raw
(+0.0053±0.0009) exactly for the pre-registered reason: first/second-half
splits inflate the noise estimate V̄ with real within-user drift, so B
over-shrinks (0.49/0.48/0.22). Queued refinement: odd/even interleaved
splits remove the drift term. Deployment stance: shrunk personalization
for robustness (smaller SE), population means confirmed
(α_t ≈ 0.12, α_σ ≈ 0.045 — F62 replicated on halves).

## V.3 The M18 operating point (F66) and coherence (D31)

Final capstone OC under the full M18 config: **adversarial FG 1/90**
(static_below 1/30, careless 0/30 with e-gate 29/30, anti 0/30 with
e-gate 26/30) versus the point comparator's 78/90 — 11× better than
M17's 11/90. Genuine learners all declare, later (the chosen trade);
powerlaw is the honest slow case (trainability 0.43 — unresolved, not
falsely declared). The pipeline is now v15-coherent end-to-end (one bar
for eval, trainer, and re-cert; anchored assumed rates): the demo
candidate FAILs all three tasks, trains 174 trials, and flips ALL THREE
to PASS at re-cert.
