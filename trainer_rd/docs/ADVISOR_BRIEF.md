# Advisor brief — the adaptive evaluation→training pipeline (M25, 2026-07-04)

> Audience: academic advisors reviewing the learning algorithm before wider
> sharing. Self-contained summary of what the system is, how it was
> validated, its measured operating characteristics, and the decisions we
> want your judgment on. Everything cited as F#/D#/M# is traceable in
> `docs/PROJECT_MEMORY.md` (the project's single source of truth), and every
> number is reproducible from a named script in this directory.

## 1. What the system is

A closed pipeline for perceptual-classification expertise:

1. **Certification eval** — an adaptive test measuring a rater's per-task
   skill/bias with a sequential Monte Carlo posterior over 2K traits
   (probit-lapse SDT observation model; A-optimal item selection; anytime
   verdict policy with pass/fail/refer semantics).
2. **Handoff** — the eval's final posterior seeds the trainer's belief
   (variance-inflated; no re-calibration block needed — D1).
3. **Adaptive trainer** — a POMDP-style training loop: per-task 2-D
   bootstrap particle filters track the LEARNING rater (skill σ, criterion
   t) under Rescorla–Wagner criterion dynamics and exponential log-σ
   relaxation; a mode-conditional policy (bias-correction / skill-building
   / retention) picks items from a real 89k-item bank; a deficiency/
   expected-progress scheduler interleaves K tasks; graduation is a
   STOPPING RECOMMENDATION (D18), never a certification.
4. **Re-certification** — the eval instrument re-certifies; the trainer's
   bar equals the re-cert bar by construction (v15-coherent, D31).

The delivery-vehicle prototype (`sandbox/`) runs the full stack on real
humans with production-shaped telemetry; seven tester profiles (USER-A…G)
have been through it and every protocol gate shipped since M21 traces to a
mechanism observed in their data.

## 2. Model and estimation choices worth your scrutiny

- **Observation model:** y ~ Bernoulli(λ + (1−2λ)Φ(exp(ℓ)(s + θ))), λ
  locked at 0.025 (fitted rater lapses are an order below it — F43/F51).
  Item signal s carries measured stimulus noise s_sd (medians 0.6–1.0);
  every scorer and the filter likelihood use the attenuated form (F20).
- **Dynamics:** criterion t moves by α_t·(p̂−y*) on feedback trials only
  (D20); log σ relaxes toward a per-learner ceiling ln σ_∞ with the
  85%-rule weight w. Rates are empirical-Bayes anchored to 1,158 real
  learning curves (α_t 0.097, α_σ 0.047 — F62), deliberately conservative
  (over-estimated rates break filter coverage, F17/D14).
- **Filter:** bootstrap SMC, no MH rejuvenation (replaying history against
  a moving state targets the wrong posterior — F1). The transition kernel
  is the EXACT conditional kernel (21-node Gauss–Hermite over s_real;
  restores nominal coverage, cov_t@90 0.63→0.91 — F53, closed-loop SBC
  nominal — F57).
- **Ceiling uncertainty:** static parameters cannot live on particles
  (path degeneracy, F4), so trainability is a Bayesian model average over
  a J=7 ceiling grid + a "does not learn" stratum, with prequential
  evidence weights (`SigmaInfMixtureFilter`, D22). This replaced a real
  failure: a filter that KNOWS the ceiling clears the bar eventually
  certifies its own prior (F28: 30/30 false declarations on a static
  sub-bar learner; now 0–7/30 depending on margin).
- **Session boundaries carry a regime-shift model (Gate 4, M23):** real
  acquisition arrived as DISCRETE jumps at sitting boundaries (USER-E:
  d′ 0.8→3.0 across a 64 s restart). Every session open applies a σ-dominant
  state shock + fixed-share re-mix of ceiling weights (exact Bayes for a
  hidden-Markov ceiling). The declaration variance floor was re-measured
  accordingly (0.23→0.33 — the F5 rule: a gate below the achievable floor
  means nobody ever graduates).
- **Anytime-valid monitoring:** certification probes at the cut state +
  a Hoeffding e-process gate (Ville α-control under continuous looking)
  block declarations while a learner performs below bar on probes (D23).
- **Zero-bias is derived, not assumed (F60):** boundary-anchored,
  label-alternating serving makes E[t_∞]=0 with a derived stationary band;
  the gate demands the band, and both the anchoring and the strict label
  alternation are load-bearing (tracking placement provably random-walks
  the criterion; iid labels give a ×10 worse floor).

## 3. The evidence chain (what we would defend in review)

Verification ladder, lowest to highest rung (§3D of PROJECT_MEMORY):

1. **Unit/derivation pins** — 19 script-style test files, 273 checks, all
   green; every calibrated constant pins to the study that produced it.
2. **Calibration** — closed-loop SBC of the training filter nominal at
   every checkpoint (F57); open-loop paired kernel audit (F53).
3. **Misspecification envelope** — a 12-member mis-specified learner zoo
   (power-law, plateau, drifting ceiling, heavy-tail link, asymmetric
   lapse, fatigue, careless, anti-learner, static-below-cut). Training
   EFFICACY is robust across the zoo; graduation self-assessment under
   misspec is exactly where the hardening effort went (mixture + probes +
   e-gate: adversarial FG 1/90 at the M18 capstone — F66).
4. **End-to-end safety** — the re-cert backstop catches 75/75 forced false
   graduations (F32); D18 makes the trainer's declaration a
   recommendation, certification stays with the eval instrument.
5. **Real-data validation (EXTSET)** — 125,860 real reads × 699 users:
   the engine likelihood beats/matches alternatives held-out (P1/P1b),
   filter skill estimates track real accuracy (ρ=0.87 binary task,
   ρ=0.48–0.83 multiclass — P2/P2b), the fitted real link sits INSIDE the
   stress envelope (P3), cross-domain skill manifold recovered (P4), and
   dynamics priors are fitted from real learning curves (F62, F65).
6. **Real humans on the full stack** — seven tester profiles through the
   sandbox; each anomaly became a mechanism finding and a shipped gate:
   identity keys state (F72), render physics bound achievable skill (F73),
   label-consistent stimuli (F74), onboarding placement (F75), starvation/
   trainability gates (F77), flag-pause (F78), fatigue guard (F79),
   boundary regime shifts (F80–F81), RT careless channel (F82), contact
   e-process (F83).

## 4. Honest operating characteristics and limitations

- **Delivered training value on real humans:** mean per-trial value 0.48
  of ideal across the 14 real sessions (M24 audit); the belief-lag
  sessions fell to 0.15–0.17. The M23–M25 selection work targets exactly
  this number; selection changes are priced end-to-end (trials to
  CONFIRMED mastery for ALL tasks under real serving rates and hazards) —
  per-trial training value alone misleads (twice-confirmed lesson, F84/F85).
- **Sharp-margin false graduation (the open OC edge, now resolved
  architecturally):** a static learner parked 0.15 BELOW the cut is
  DECLARED ~30% of the time within 240 two-task trials by the M23/M24
  stack. At that margin the per-probe accuracy contrast is ≈0.03, so
  refutation by probes needs O(10³) probes and the full-likelihood route
  ~600 trials — an information bound, not gate inefficiency. M25 measured
  three candidate declaration conjuncts and REJECTED all three (they
  cannot beat the bound; the mixture's trainability at these FGs is 0.92,
  so a ceiling conjunct is a no-op; a refractory buys 1/20 at −15pp
  honest completion). The effective filter is architectural and already
  shipped: the D33 confirmation lifecycle kills 5/6 of these FGs
  (declared 6/20 → CONFIRMED 1/20), and the D18 re-cert backstop sits
  above it (75/75 forced FGs caught). Consequence: FG must be priced at
  the CONFIRMED level in all claims, and the pilot's FG endpoint is
  confirmed-FG. Full record: `docs/M25_SELECTION_V2.md` §3.
- **Bias intervals** of the shipped (pre-M15) kernel were ~40%
  overconfident; fixed by the exact kernel, but any port must re-pin gates.
- **Open protocol reality:** with calendar-time forgetting and open-ended
  sessions, the median simulated learner certifies ~2 of 7 tasks in 40
  sessions; consolidation dynamics are load-bearing and are a named pilot
  endpoint (F70). Pilot scope is K=1–3 tasks.
- **What we cannot claim:** the trainer certifies nothing (D18); the
  definitive methodology test requires the feedback-driven pilot (no
  existing dataset contains closed-loop training — M11.2b).

## 5. What shipped at M25 (this checkpoint)

Full derivations and tables: `docs/M25_SELECTION_V2.md`. The headline of
this checkpoint is three HONEST verdicts, two of them negative — we ship
the machinery opt-in and keep the shipped defaults, on evidence:

1. **EP-v2 allocation (F86 — implemented; endpoint-decisive NO):**
   expected-progress allocation coupled to the confirmation lifecycle
   (ever-declared tasks route to a budgeted measurement lane; exploration
   floor for written-off tasks). Mechanically validated, but behaviorally
   inert at K=2: the post-declaration re-polish cost (~28 trials/session)
   is a property of the session-boundary HAZARD, not the allocator, and
   on the BOTH-declared endpoint the shipped Gate-1 allocation dominates
   every expected-progress variant. Queued successor is at the hazard
   level (evidence-adaptive boundary hazard for declared tasks).
2. **Item-level tier-3 rollout placement (F87 — no dominance; the F84
   placement line is now CLOSED):** H-step CRN Monte-Carlo rollout value
   at the item level, mixture-honest, inside the F84 safety devices.
   Herding-safe and at parity-or-worse across scenarios; the candidate
   ranking is sensitive to the interior pool model (a sign-flipping
   sensitivity we report rather than hide). Third and strongest
   confirmation that under the hazard-widened belief architecture the
   plug-in 85%-rule placement is the right rule — any future successor
   must change the objective's treatment of hazard width, not add
   lookahead.
3. **Sharp-margin gate (F88 — conjuncts rejected; architecture wins):**
   see §4 third bullet. The load-bearing result: FG must be priced at the
   CONFIRMED level (the D33 lifecycle kills 5/6 sharp-margin FGs).
4. **Tester-facing visualizations:** per-profile MP4s of the belief
   trajectory over the real sessions (`viz/make_user_videos.py`).

## 5b. What shipped at M26 (2026-07-05) — and why iteration now stops

The M25-queued hazard-level successor was executed and honestly
adjudicated (`docs/M26_HAZARD.md`): an **evidence-adaptive boundary
hazard for declared tasks** (F89 — Gate 4's re-widening decays by a Beta
posterior-mean factor in a task's accumulated post-boundary
confirmations; exact for the marginal update by linearity). Validation
showed the pair-completion gain it buys is COUPLED to a sharp-margin
true-regression blind spot across the entire pin range (fixed hazard: +0
false-mastery flicker; any decayed pin: +14 to +100 trials), while all
FG guardrails stayed unchanged. **D42: the machinery ships opt-in; the
sandbox default keeps the fixed hazard** — the same D28 ordering (FG ≥
lateness > speed) that governed M24/M25.

That makes three consecutive checkpoints in which every queued
optimization successor was priced end-to-end and produced NO default
change. We read this as convergence of the current evidence base and
have frozen algorithm iteration under an explicit stopping rule —
`docs/SUBMISSION_CRITERIA.md` (D43) defines the submission gate (S1–S8,
all MET) and what may reopen iteration (your requests, new human data, a
regression, or a gate unblocking). Suite at freeze: 20 files / 287
checks.

## 5c. What shipped at M27 (2026-07-05) — the pre-submission completion loop

The parked queue was completed or adjudicated (full record:
`docs/M27_TERMINAL.md`); the architecture is now formally
domain-pluggable (`training/domain.py` + `docs/ARCHITECTURE.md` — start
there for a first read of the codebase):

1. **Terminal confirmation (F90, opt-in; D44 default off):** a
   confirmed task leaves the training rotation for the retention layer
   (one consolidated review schedule + at-bar maintenance probes + an
   anytime-valid stale-mastery e-gate). Pair completion +20pp at a
   measured price: ~half of post-confirmation regressions go undetected
   within-protocol (the trainer's evidence stream on a terminal task is
   0.5–3.5 trials/session). Whether that trade is RIGHT is your call —
   decision ask #7 below.
2. **Contact-triggered onboarding (F91; D45 ships ON in the sandbox):**
   demonstration ramp until the contact e-process certifies + a
   trainability-guarded belief re-open at certification. The observed
   no-contact onboarding shape goes censored@30% → declared 148@70%;
   healthy learners unharmed (100% rate, +14 median); careless
   responders never certify and never declare.
3. **λ(time-on-task) closed by real data (F92):** within-sitting
   easy-read accuracy RISES (+8.7pp/h) on 125,860 real reads — learning
   dominates and self-paced quitting truncates the fatigued tail, so a
   lapse-growth term is not sign-identifiable observationally; deferred
   to pilot fixed-length sessions.
4. **Fourth consecutive default-decisive NO recorded** (F90 after
   F85/F86/F87/F88-conjuncts/F89): the shipped defaults keep winning
   their challengers end-to-end — the D43 convergence claim
   strengthened.

## 6. Decisions we want advisor/PI input on

1. **D18/D19 ratification** — trainer graduation as recommendation-only +
   the hardened port configuration (exact kernel default, mixture
   graduation, probes/e-gate).
2. **D16 acceptance semantics** — the trailing-mean-skill graduation
   branch (z=1.645, W=20) for narrow-margin domains: acceptable as an
   acceptance-testing semantic? (Disabled for mixture graduation, D30.)
3. **OQ3** — should post-training re-certification warm-start from the
   trainer posterior (efficient) or a fresh prior (psychometric
   integrity)? We lean fresh.
4. **F26/F27** — the v15 instrument's claimed resolution gain is mostly
   bar recalibration (anchoring convention), and particle-doubling does
   not fix posterior-ℓ coverage; the decisive K=7+corr_t coverage run is
   port-gated. Comfortable?
5. **Confirmed-FG semantics (F88/D41)** — we propose all false-graduation
   claims and the pilot's FG endpoint move from declared-FG to
   CONFIRMED-FG (the D33 lifecycle is part of the instrument; measured:
   declared 6/20 → confirmed 1/20 at ℓ*−0.15, with pre-declaration probe
   densification measured ineffective against the information bound).
   Please ratify this endpoint definition before the pilot SAP freezes.
6. **Pilot SAP** (`PILOT_SAP_M14.md`) — power target α=0.005/0.9 needs
   ~33/arm; consolidation-exponent endpoint from F70; decision-level
   outcomes only (F59: dynamics-parameter evidence is flat).
7. **Terminal-confirmation semantics (F90/D44)** — should a CONFIRMED
   task belong to the re-certification layer rather than the trainer's
   rotation? This is D18 ("the trainer certifies nothing") taken
   seriously: it buys +20pp pair completion by design, and it means
   post-confirmation regression is caught by SCHEDULED RE-CERTS, not by
   the trainer (measured: ~50% of regressions undetected within 5
   sessions by the thin terminal evidence stream, vs 0% when the
   trainer keeps re-polishing). If the pilot has scheduled re-certs —
   it does — we believe this is the coherent semantics; we did NOT ship
   it by default pending your ratification.

## 7. Reproducibility

- Everything runs from the scratch root with `python3 -m`:
  - test suite: `python3 -m tests.test_step0` … `tests.test_m27`
    (21 files, 299 checks);
  - studies: `python3 -m studies.study_m26_hazard [--smoke]`,
    `python3 -m studies.study_m27_terminal [--smoke]`,
    `python3 -m studies.study_m27_onboard [--smoke]`,
    `python3 -m studies.study_lambda_tot` (all cache to
    `figures/data_*.npz`);
  - architecture / code map: `docs/ARCHITECTURE.md`; domain plug-in:
    `training/domain.py`;
  - sandbox: `python3 -m sandbox.session --user NAME`;
  - videos: `python3 -m viz.make_user_videos`.
- Parameter conventions (the σ=exp(−ℓ), t=−θ bridge) live ONLY in
  `training/bridge_conventions.py`.
- All engine changes since M10 ship opt-in with defaults bit-identical to
  the previous checkpoint; defaults flip only on a dominating study, and
  every flip is a dated D# decision.
