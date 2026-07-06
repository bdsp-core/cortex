# M24 — Question selection by expected learning progress: the machinery, the full-scale evidence, and why both defaults stayed put (2026-07-04)

Scope: the item-selection mathematics — "give the learner the question most
likely to improve their pattern recognition, while still serving
estimation." Two queued successors executed as an autonomous PECR loop:
expected-progress-per-trial ALLOCATION (the F77 successor, F85 — validated
opt-in with two named residuals; sandbox default DEFERRED) and posterior
expected-progress PLACEMENT (the F75 successor, F84 — a mixed result after
six adversarially-tested variants; default-off experimental). The
checkpoint's shipped value is the stratum-aware selection machinery, the
delivered-value audit, and five sharp mechanism findings — with both
default flips correctly blocked by the guardrails (D28: FG ≥ lateness >
speed; conventions: no default flip without a dominating study).
Validation: `studies/study_m24_selection.py` →
`figures/data_m24_selection.npz`, all arms at the M23 shipping stack
(Gate 4 boundary hazards + sd_floor 0.33) over the real bank.

## 1. The mathematical structure being improved

The plan (docs/learning_algorithm_plan.md) defines per-trial reward as
state improvement (Eq. reward)

    R = β_t(|t|−|t′|) + β_σ(σ−σ′) + β_r·ret,

whose σ channel moves by E[Δ ln σ] = −α_σ·f·w(s,θ)·(ln σ − ln σ_∞) with the
85%-rule weight w (Eq. weight, Wilson-optimal at |s−t|/σ ≈ 1.077 under
λ=0.025), and prescribes tier-1 selection as argmax_s E[R | b, s]
(Eq. greedy). What the shipped tier-2 stack actually does:

- **bias mode** already implements the β_t term of Eq. greedy over the
  cloud (`bias_correction_score`, D15) in dual-control alternation with a
  t-information probe;
- **skill mode** ranks by E[w] at the PLUG-IN (σ̂, boundary) with the F75
  z-inflation — three stacked approximations: (i) point estimate instead
  of posterior, (ii) w instead of w·gap (no credit for how much the
  trained mass can still improve), (iii) no mixture structure (ceiling
  strata, weights, static stratum never enter);
- **allocation** is worst-first deficiency × the D36 trainability
  discount (an ad-hoc multiplicative patch on the F77 starvation trap);
- latent defect found in passing (**F84-ii**): the sandbox's bias mode
  feeds the MIXTURE through the pooled tier-1 scorer, which credits the
  static stratum (rule="static" ⇒ T = identity ⇒ R ≡ 0 exactly) with
  corrections it cannot make, and uses the base σ_∞ for every stratum.
  Fixed by the stratum-aware scorer for anything that opts in; the D15
  path's ranking distortion is mild (β_t's correction sign is common
  across particles) and left as shipped.

Delivered-value audit motivating all of this (seven profiles, per-session
behavioral probit fits, |t| bounded): the pipeline delivered a **mean
per-trial training value w_true = 0.48 of ideal**; the F80-lag sessions
fell to 0.15–0.17 (A-s3/E-s3 served over-easy against a stale-high σ̂;
D-s1/E-s1 over-hard pre-contact), 5/14 sessions below 0.5.

## 2. F85 — allocation by expected progress per trial (validated opt-in; sandbox deferred)

Task-level scoring (`expected_progress_rate`): per stratum j, particle i,

    EP_σ = α_σ · w̄_i · (ℓ_∞^(j) − ℓ_i)⁺ · 1[ℓ_i < ℓ*_k]
    EP_t = min(α_t, (|t_i| − t*_k)⁺)
    EP_k = Σ_j ω_j 1[rule_j ≠ static] Σ_i w_ij (β_σ·EP_σ + β_t·EP_t)

with w̄_i = ρ/√(ρ² + (s̄_sd/σ_i)²) the noise-smeared 85%-optimum weight
(ideal placement; the pool-edge feasibility caveat errs toward serving
struggling learners and is documented). EP_k reads "how much bar-ward
progress one trial of task k is expected to buy under the current
belief." The scheduler ranks unmastered tasks by EP_k; finish-first,
suspensions, retention, and the consec cap are untouched (measurement and
variety are separate layers by design — D28's division of labor).

Why this subsumes the D36 Gate-1 discount from first principles: the
static stratum and at-ceiling mass contribute EXACTLY zero (B's plateau
self-deprioritizes, no floor constant needed); a wide onboarding
posterior keeps EP high through its trainable tail; a near-mastery task's
above-bar mass earns nothing (finishing is measurement — the F70 phase's
job). Bar-referencing is PRINCIPLED at this level because tasks, not
difficulties, are being ranked (contrast F84-iii below).

Validation (arm Q, 20 seeds, K=2 production stack, real bank, 6×40
trials; three rosters × {legacy worst-first, D36 Gate-1, F85 EP};
first-declaration median (rate) for the trainable task0 / total trials
sunk into task1):

| roster | legacy | Gate-1 | **EP** |
|---|---|---|---|
| B-trap (near-bar + static plateau) | 142 (75%) / 169 | 134 (65%) / 156 | **86 (80%) / 72** |
| asymmetric rates (both trainable) | 154 (75%) / 146 | 150 (65%) / 146 | **72 (90%) / 86** |
| trainable + pre-contact jumper | 198 (65%) / 153 | 184 (65%) / 155 | **82 (80%) / 71** |

Arm S guardrails: careless FG 0/20 everywhere; well-spec pair lateness
IMPROVES under EP (median 99 → 72, rate 100% → 95%).

**F85-ii — the two residuals that block the sandbox default,** both
surfaced by the full run (the smoke missed them):

1. **Post-boundary re-polish loop.** After each Gate-4 boundary shift, a
   nearly-mastered task's re-widened below-bar mass reads to EP as a
   progress opportunity (small gap × very high trainability × freshly
   sub-bar mass), and EP out-ranks the LOW-trainability recovering task —
   worst-first, ranking by deficiency, serves the wide task instead. Net:
   in the trainable-pair rosters EP's task1 completion is censored at 241
   trials where Gate-1 completes at 188/189, and task1 receives only
   ~12 trials/session even after task0 first declares. For the E-shaped
   real pattern (belief writes a task off, learner recovers) this is the
   wrong side of the trade despite the M23 fixed-share partially
   re-opening trainability each boundary.
2. **Sharp-margin FG direction.** static_below (ℓ*−0.15): 20-seed arm S
   read 9/20 (EP) vs 6/20 (Gate-1); a 40-seed adjudication extension gave
   18/40 vs 16/40 — pooled 27/60 vs 22/60, z ≈ 0.9, not significant, but
   the direction is unfavorable and the guardrail is one-sided. (The
   pooled baseline also sharpens the M23 open item: the M23-shipped stack
   itself false-graduates ~37% [22/60] at this margin in 240 trials.)

Decision D40 (as taken): `DeficiencyScheduler(progress_alloc=,
progress_s_sd=)` ships OPT-IN, default off, bit-identical;
`sandbox PROGRESS_ALLOC = False` — the default flip is DEFERRED until the
queued successor lands: couple EP to the confirmation lifecycle (a task
that has ever been provisional ranks by finishing/measurement need, not
EP) plus an exploration floor for written-off tasks.

## 3. F84 — posterior-progress placement: six variants, three mechanisms, a mixed verdict

The candidate rule: skill-mode argmax of the stratum-aware posterior
expected σ-progress (`expected_progress_score`, the exact β_σ term of
Eq. greedy under the mixture — the same upgrade bias mode got at D15).
Six variants were built and adversarially measured (arm P: four learner
scenarios × K=1 production stack; arm H: live R–W criterion herding).
The intermediate-variant table below is from the 4-seed iteration smokes
(directional, mechanism-isolating); the FINAL variant's row is the
20-seed full run:

| variant | wellspec declared | jump w_true | herding \|t\| end | verdict |
|---|---|---|---|---|
| shipped z=1 baseline | 38 @100% | 0.31 | 0.115 | — |
| v1 pure progress argmax | 59 @75% | **0.40** | — | trains better, declares slower |
| v2 bar-referenced credit | 96 | 0.34 | — | REJECTED: zeroes the bulk's vote; argmax chases the below-bar high-σ tail (over-easy, info-poor: w_true 0.60→0.37 wellspec) |
| v3 + bar-anchored finishing info (σ*) | 108 | 0.38 | 0.217 | REJECTED: near-deterministic responses for an above-bar bulk |
| v4 + belief-anchored finishing info (σ̂) | 94 | 0.40 | 0.264 | finishing fixed; pre-finishing starvation remains |
| v5 + width gate (progress only while wide) | 102 | 0.36 | 0.192 | gate never anneals — M23 keeps sd_ℓ above the floor BY DESIGN |
| **v6 (final) + M10 mirror pairing, FULL RUN (20 seeds)** | **49 @90% (vs 46 @100%)** | 0.36 (vs 0.35) | **0.171 (vs 0.177 — parity)** | near-parity wellspec; jump declaration 134 vs 148 (FASTER); contact 35% vs 30%; slow 121 @70% vs 102 @85% (SLOWER) |

Three named mechanisms, each isolated by a variant pair:

1. **Tail-vote pathology** (v1/v2): the progress integrand w_i·gap_i·α is
   dominated by high-gap (high-σ) posterior mass; the argmax serves that
   tail's preferred big-signal items, which are simultaneously
   low-information and low-training-value for the bulk. Bar-referencing
   makes it strictly worse by silencing the bulk entirely.
2. **Calibration externality** (v5→v6, the sharpest finding): with
   accuracy and w_true statistically indistinguishable from baseline
   (0.85/0.48 vs 0.85/0.50), declaration still doubled — because a free
   per-trial argmax abandons the M10 magnitude-mirror invariant, the
   served stream's midpoint un-pins, and the Rescorla–Wagner attractor
   herds the learner's TRUE criterion to it (|t| end 0.115 → 0.19–0.26),
   stalling the declaration gate's bias condition. The failure was never
   difficulty; it was calibration. Restoring the mirror (v6) fixed
   herding but not the remaining information deficit.
3. **Model-injected width** (the root cause, and the reason no variant
   can win): under the M23 stack the posterior is wide MOST OF THE TIME
   BY DESIGN — boundary hazards and cross-strata spread are safety
   machinery (FG ≥ lateness > speed, D28), not calibrated beliefs about
   where the learner is. The myopic posterior-progress integral takes
   that width at face value and hedges against tails the hazard model
   itself will erase within a handful of trials. The plug-in 85%-rule at
   σ̂ is, under this belief architecture, simultaneously near-optimal for
   training AND for ℓ-information at the mass that matters.

Full-run verdict: the six-iteration rework converged v6 from a real 2×
declaration regression to NEAR-PARITY — herding fully fixed (arm H:
|t| end 0.171 vs 0.177, tracking error equal), well-specified declaration
within noise, the E-like jump 14 trials FASTER, the contact scenario
slightly better — but the slow-wide learner regresses (121 @70% vs
102 @85%) and delivered w_true no longer beats baseline after the
finishing switch (the training-value gain and the measurement fix trade
against each other). No dominance ⇒ no default flip (conventions);
`ModeThresholds.progress_placement` stays implemented, default-off,
experimental. The three mechanisms (tail vote, calibration externality,
model-injected hazard width) are the durable finding — any future
selection work must respect them. Successor (queued): the non-myopic
answer — tier-3 rollout (`trainer_rollout.py`) at the item level prices
the information externality myopic progress cannot see. Altitude lesson,
now twice-confirmed (F84 and F85-ii): **selection changes must be priced
end-to-end in trials-to-confirmed-mastery for ALL tasks, at the real
serving rate, under the real boundary hazards — per-trial training value
and single-task first-declarations both mislead.**

## 4. What changed where

- `training/trainer_greedy.py`: `_expected_reward(..., bar_ell=)` opt-in
  bar-referenced σ-credit (default None bit-identical).
- `training/trainer_policy.py`: `expected_progress_score` (stratum-aware
  tier-1 Q; F84-ii static-stratum fix), `expected_progress_rate` (F85),
  `ModeThresholds.progress_placement` (default False),
  `DeficiencyScheduler(progress_alloc=, progress_s_sd=)` (default off),
  progress skill-mode branch with finishing switch + mirror window.
- `sandbox/config.py` + `protocol.py`: passthrough wired;
  `PROGRESS_ALLOC = False` (F85-ii deferral), placement off.
  `tests/test_m24.py` (11 checks); suite 18 files / 258 checks.

## 5. Queued from this checkpoint

1. **EP v2 (the F85-ii successor):** couple allocation to the confirmation
   lifecycle — a task that has ever been provisional ranks by
   finishing/measurement need, never EP; plus an exploration floor for
   written-off tasks (the M23 fixed-share alone under-explores at the
   measured ~12 trials/session rate). Re-run arm Q's trainable-pair
   rosters with BOTH-DECLARED as the endpoint.
2. **Item-level tier-3 rollout** as the non-myopic placement successor
   (prices the information externality; the existing CRN machinery in
   `trainer_rollout.py` is the starting point).
3. **Sharp-margin FG study** (inherited from M23, now sharpened: the
   shipped stack false-graduates 22/60 static ℓ*−0.15 learners in 240
   trials on this harness; D33 + D18 sit above, but the gate itself needs
   work at that margin).
4. Field expectation meanwhile (sandbox unchanged in behavior):
   allocation and placement telemetry identical to M23 — any drift
   falsifies the bit-identity claims.
