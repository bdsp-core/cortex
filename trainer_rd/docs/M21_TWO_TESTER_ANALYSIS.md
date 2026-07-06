# M21 — Two-tester sandbox results: analysis and consequences (2026-07-02)

> Data: `sandbox/logs/trials.jsonl` (80 trials) + `sessions.jsonl`
> (2 × 40-trial sessions, 89.5 s apart, one shared belief state).
> Session 1 = tester A, session 2 = tester B (per the run report).
> Sandbox config at the time: M20 defaults (gain 0.55, unconditioned
> s_real draw, single machine-keyed state). Findings F72–F75, decisions
> D34–D35 recorded in PROJECT_MEMORY §3G/§4/§6.

## 1. What the telemetry shows

### Tester A (session 1: 40 trials, 6.7 min, median RT 4.4 s)

| task | n | acc vs y* | d′ vs y* | d′ vs percept (sign s_real) |
|---|---|---|---|---|
| domain2 | 6 | 0.50 | +0.00 | +0.00 |
| domain3 | 34 | 0.47 | −0.15 | −0.28 |

No measurable discrimination against either the bank label or the rendered
percept: tester A was **effectively guessing** (y-rate 0.45), with a
possible mild onboarding trend (acc 0.40 → 0.55 across halves, ≈1 SE, not
signal). First-trial RT 66.6 s = instruction reading; RT is heavy-tailed
exactly as F10 warned. The engine responded correctly given its model:
ℓ̂ fell 0.0 → −0.28/−0.43, π → 0.03–0.11, and the σ∞-mixture cut
trainability on domain3 from 0.74 → 0.37 ("ceiling likely below cut").

### Tester B (session 2: 40 trials, 9.3 min, median RT 7.7 s)

| task | n | acc vs y* | d′ vs y* | d′ vs percept | acc vs percept |
|---|---|---|---|---|---|
| domain2 | 26 | 0.38 | −0.73 | −0.23 | 0.46 |
| domain3 | 14 | 0.71 | +1.15 | +1.34 | 0.79 |

Tester B is **genuinely competent at domain3** — 0.79 accuracy against the
percept is near the ideal-observer ceiling for the served difficulties —
while at chance on domain2, where they were served predominantly
near-invisible bias-mode items (median |s_mean| 0.57 ⇒ rendered amplitude
≈ 0.31 vs per-sample noise 1.0) chosen to correct a criterion estimate
inherited from tester A. Their heavy "absent" response bias on those
invisible items (hit rate 0.15) drove t̂ +0.11 → +0.63 and σ̂ 1.32 → 1.86,
i.e. the filter chased an unmodelable (negative-d′) segment. Log-RT
distributions of A and B differ strongly (Δmean 0.58 nats, Mann-Whitney
p = 0.0009) — RT is a cheap identity/context signal.

### Learning evolution, net

* **Tester A**: no perceptual learning detectable in 40 trials; the session
  measured onboarding, not learning. Under the M20 render (see §3) even a
  motivated novice had little visible signal to learn from at the served
  difficulties.
* **Tester B**: arrived already able to do domain3 (d′ ≈ 1.3 within their
  first/only 14 domain3 trials — an inherited-state artifact kept them
  under-served there); no within-session trend measurable on domain2
  because the served items were largely below their perceptual threshold.
* **The engine**: every subsystem behaved per spec on the data it saw; the
  failures were **protocol-layer assumptions** (identity, stimulus
  contract, prior/placement for novices), not the math of the stack.

## 2. F72 — identity is a protocol assumption, not a measured quantity

The belief state was keyed to the machine. Consequences observed:

1. **Trainability pollution**: tester B inherited domain3 trainability
   0.365 from A's guessing; despite d′ +1.15 performance it recovered only
   to 0.48 in 14 trials — and the deficiency scheduler consequently
   under-served their strongest task (14 vs 26 trials).
2. **Criterion mis-attribution**: 14 of 26 domain2 trials went to
   bias-corrective serving against a criterion estimated on tester A.
3. **Invisibility**: e-gate quiet (it watches below-bar probe evidence),
   GapAnchor never opened (89.5 s < 4 h), prequential surprise flat
   (near-chance beliefs render everything unsurprising; a switch TOWARD
   competence *lowers* surprise).

**Fixes (D34)**: per-tester profiles (`--user`, state/logs namespaced;
production: state keyed by learner identity) + a shadow
`ConsistencyMonitor` (`training/consistency.py`): windowed GLR of the last
W = 20 trials against the belief's own running predictions, RT-shift
reported alongside, log-only. Validation (`study_learner_switch`, full run):
threshold pinned at the 99th pct of stationary two-session max-GLR
(**11.30**; 95th = 9.35). At that 1%-per-two-sessions false-alarm bar,
gross profile corruption (competent→guesser) is caught 27/40 within 40
single-task trials (median 19); the REAL two-tester switch peaks at GLR
7.0 on domain2 — **elevated (≈93rd stationary percentile) but not
decisive**, and guesser→competent is mostly absorbed by belief adaptation
(4/40). Read: 20–40 near-chance trials cannot decisively identify a
person; the monitor is a corruption tripwire, and PROFILES are the
primary identity mechanism. Raw GLR + rt_z are logged per trial, so
analysts can apply any percentile post hoc.

## 3. F73 — the render physics bound achievable skill

For the M20 display (48 samples, 12-sample window, per-sample noise 1.0,
gain 0.55), an ideal observer's effective sigma is
σ_eff = NOISE·√(1/12 + 1/36)/GAIN = **0.606** (verified empirically through
`render()`: 0.605). The v15 bars are σ* = 0.857 (domain2) / 0.736
(domain3): a human needed 71–82% of ideal efficiency just to reach the
bar, and the assumed expert ceilings σ_∞ ≈ 0.48–0.52 — the center of the
σ∞-mixture's grid — were **physically unreachable**. The trainability
collapse both testers produced is partly this ceiling, not only them.
**Fix (D35)**: RENDER_GAIN 0.85 ⇒ σ_eff(ideal) = 0.392; bars need 46–53%
ideal efficiency, ceilings 76–81%. Resume across a gain change is blocked
(state stamps its render params).

## 4. F74 — the stimulus contract was broken at the render layer

`s_real ~ N(s_mean, s_sd²)` unconditioned on y* ⇒ on 8/80 served trials
(10%; 4/26 in tester B's domain2) the rendered evidence OPPOSED the
feedback label — feedback then trains the wrong mapping. In production
this cannot happen for feedback items (experts labeled the actual trace;
D11 margin-filtering enforces it); the sandbox must honor the same
contract. **Fix (D35)**: label-consistent truncated draw (deterministic
per (seg, session), preserves the s_sd physics the filter models).

## 5. F75 — onboarding placement runs hot for novices

Skill placement targets |s − t̂| = 1.077·σ̂ against the belief-MEAN σ̂,
which starts at the optimistic prior (σ̂ = 1). Both testers spent whole
sessions at 47–50% served accuracy (target: 84%) while the filter
descended toward their true σ. Fix implemented as opt-in
`ModeThresholds.skill_sigma_z`: place against
σ̂·exp(z·max(sd_ℓ − sd_floor, 0)) — conservative while the posterior is
wide, annealing to exactly σ̂ at the variance floor; z = 0 is bit-identical
to M20. Sim (arm E): no cost in learning speed or final σ, small σ̂-RMSE
improvement. The human case (motivation, visible-signal onboarding) is the
stronger argument; the sandbox opts in at z = 1.

## 6. What did NOT need fixing

* The exact-kernel mixture filter's POINT estimates adapt within-task in
  ~10–15 trials after a switch (placement/served accuracy indistinguishable
  from fresh by the end of 40 trials: 0.84 vs 0.83, arm D). But the
  σ∞-mixture's TRAINABILITY is stickier — 0.54 on a guesser-polluted state
  vs 0.76 fresh after the same 40 competent trials — because strata
  weights carry the full prequential history. That stickiness is correct
  Bayesian behavior against the wrong (one-learner) assumption; with
  profiles enforcing the assumption it is a feature (resistance to
  transient careless stretches), not a bug.
* The trainability statistic did its job on the data it was given — both
  collapses were honest inferences under the (violated) one-learner and
  (mis-calibrated) render assumptions.
* D33/GapAnchor/e-gate: not exercised by this pair of sessions (no ≥4 h
  gap, no declarations); nothing to conclude.

## 7. Consequences for the pilot / delivery vehicle

1. Learner identity is a first-class key of belief state (profiles now;
   authenticated identity in production).
2. Ship the consistency monitor as shadow telemetry in the pilot; its
   flags mean "verify who is at the keyboard / what changed", not
   "reset beliefs".
3. Any rendered-stimulus deployment must state its ideal-observer bound
   and place the bars well inside it (≤60% of σ_eff(ideal) per the new
   test_sandbox check), and must guarantee feedback-percept consistency.
4. Novice onboarding should serve visibly-doable items while the belief
   is wide (skill_sigma_z), and the first session should be read as
   onboarding, not learning (tester A's entire session was procedural).
5. Telemetry schema addition: per-trial `consistency` {glr, rt_z, flag} +
   `consistency_flag` session events + profile stamp (`user`) in state.
