# M26 — Evidence-adaptive declared-task boundary hazard (F89) (2026-07-05)

Scope: the queued M25 successor #1 (docs/M25_SELECTION_V2.md §6), executed
as one autonomous PECR loop. Validation:
`studies/study_m26_hazard.py` → `figures/data_m26_hazard.npz`, all arms at
the M23 shipping stack (Gate 4 + sd_floor 0.33, z1 placement, Gate-1
allocation) over the real bank with a D33-style confirmation lifecycle at
session opens.

## 1. F89 — the mechanism

### 1.1 The residual being fixed

M25's endpoint-decisive finding: the Gate-4 boundary hazard re-widens a
DECLARED task at every session open, and re-confirming it consumes service
forever (~28 trials/session under any lane that serves measurement need to
completion) — the binding constraint that made every allocator-level fix
(F85/F86) inert on the BOTH-declared endpoint. The hazard is applied at the
PINNED rate (ε_b = 0.10, w_share = 0.10, arm G of study_m23_regime) no
matter how many boundaries the task has already survived: the model never
learns that this learner's mastery of this task is stable.

### 1.2 The rule

The F81 fixed-share update is exact Bayes for a hidden-Markov ceiling that
re-draws from the prior with a KNOWN per-boundary hazard ρ. For a task that
has demonstrated the declaration gate, each subsequent boundary followed by
a fresh demonstration is an (approximate) observation that no downward
re-draw happened. With ρ ~ Beta(a0, b0) (mean ρ0, prior strength
c0 = a0 + b0) and n such observations, the posterior-mean hazard is
ρ0·c0/(c0+n) — and because BOTH Gate-4 components are LINEAR in their
hazard (the fixed-share re-mix in ρ; the two-component state-jump mixture
in ε_b), plugging the posterior mean in is exact for the marginal
one-boundary update. Λ (shock size) and the θ-scale are not hazard
probabilities and do not decay.

Implementation (`training/mixture_filter.py`): `declared_hazard_scale`
(Beta form, `floor`, `geometric` ablation) + `AdaptiveBoundaryHazard`
(per-task ever/n/pending lifecycle, json-persistable). Semantics:

- **Never-declared tasks always pay the full pinned hazard** — the F80
  breakout machinery is untouched by construction (test_m26 check 4).
- **Once per boundary:** each Gate-4 application creates a pending debt;
  the next gate demonstration pays it (n += 1) at most once.
- **Credit rule** (`credit_from_pi`): the full declaration gate CANNOT be
  the per-boundary evidence unit — its sd-floor conjunct is exactly what
  the hazard injects, and under the full pinned hazard a declared task
  re-fires the strict gate ~never (measured while building the study:
  ZERO re-demonstrations in 5 post-declaration sessions at the real
  2-task service rate; the decay would be unreachable). The
  decision-relevant "no downward re-draw" observation is the MASS
  condition alone at the session open — π ≥ 1−α over the evidence the
  previous session accumulated against that session's re-widened prior (a
  satisfied gate implies it). π < 0.5 at an open (indifference) is
  positive evidence FOR a re-draw and RESETS n — the evidence-driven
  analog of a D33 revocation; an actual D33 REVOKED also resets (sandbox).
  Credit is never granted for the shock merely being small (a gate that
  held straight through a boundary earns nothing until post-boundary
  evidence arrives).
- **Floor:** the scale never drops below `floor`×1 — the re-draw process
  is not claimed stationary-forever; a permanent fraction of the safety
  machinery survives any evidence.

Approximations recorded honestly: (i) a re-draw that lands ABOVE the cut
also re-confirms, so counting every re-confirmation as "no re-draw"
over-credits by ≈ the prior above-cut mass — priced in arm G instead of
corrected; (ii) the π-credit reads the belief's own pass mass, which is
evidence-driven (it must survive the previous boundary's widening) but
shares the mixture's blind spots at information-bounded margins (F88) —
also priced in arm G.

This is a BELIEF-MODEL change (it weakens M23 safety machinery on the
declared side), so per the M25 queue note it carries its own FG guardrail
campaign; the new load-bearing scenario is TRUE REGRESSION after repeated
confirmation — exactly the event a decayed hazard is slower to re-open.

### 1.3 Candidate pins

{c5f25: Beta c0=5 + floor 0.25 (mild), c2f25: c0=2 + floor (fast),
c2f0: c0=2 no floor (floor priced), g5f25: geometric λ=0.5 + floor
(aggressive ablation), c5: c0=5 no floor}. Sandbox constants:
`HAZARD_ADAPT` / `HAZARD_STRENGTH` / `HAZARD_FLOOR`.

## 2. Validation (study_m26_hazard, 20 seeds/cell)

### 2.1 Arm A — BOTH-declared endpoint (M25 arm-Q2 rosters, 8 sessions)

Fixed-hazard baselines replicate M25 arm Q2 EXACTLY (same seeds, same
physics: rates 150/188/BOTH 300@55%, svc1 22.4; jumper 184/189/240@70%;
b-trap 134, svc1 26.9) — the lifecycle layer added for M26 is
measurement-only.

| roster | hazard | task0 first | task1 first | BOTH (rate) | end-scale |
|---|---|---|---|---|---|
| rates | fixed | 150 | 188 | 300 (55%) | 1.00 |
| rates | c5 / c5f25 | 150 | 188 | **266 (65%)** | 0.79 |
| rates | c2f25 | 150 | 188 | 275 (60%) | 0.67 |
| rates | g5f25 | 150 | 188 | **255 (75%)** | 0.56 |
| jumper | fixed | 184 | 189 | 240 (70%) | 1.00 |
| jumper | c5/c5f25/c2f25/g5f25 | 184 | 189 | 244 (65–70%) | 0.62–0.84 |
| b-trap | fixed / c2f25 / g5f25 | 134 | cens. | cens. (5%) | ≤1.00 |

Three facts:

1. **First declarations are bit-unchanged** (decay activates only after a
   declaration — by construction) and the plateau roster is unchanged
   (b-trap censoring, sunk trials, service split all within noise): the
   mechanism cannot make anything declare EARLIER or pour trials into a
   plateau.
2. **The gain is in the completion tail, where the endpoint lives.** On
   rates (early declarer + slow second task — the F85-ii shape), pair
   completion rises 55% → 65% (Beta pins) → 75% (geometric) and median
   BOTH falls 300 → 266/255. Medians of the per-task firsts are unchanged,
   so the whole effect is censored-tail seeds getting task1 over the line:
   a decayed task0 re-satisfies its gate sooner after each boundary,
   leaves the rotation, and the late-session service flows to the slow
   task exactly when it needs it.
3. **No gain where the shape is absent:** jumper's task0 declares at 184,
   one session before task1 (189) — there is almost no post-declaration
   window for decay to free, and the endpoint stays 240–244 (differences
   within ±1 seed). The mechanism is a TAIL-SHAPE fix, not a universal
   speedup — same altitude lesson as F84–F87.

### 2.2 Arm G — true-regression guardrail (10 sessions, regression at the
session-5 boundary; task1 well-spec competitor)

| margin | hazard | wrongly-satisfied opens /4 | last-true after reg. | end gate on | n_conf@end |
|---|---|---|---|---|---|
| deep (ℓ*−0.405) | fixed | 0.00 | +0 | 0% | 0.0 |
| deep | c5f25 | 0.05 | +0 | 0% | 1.7 |
| deep | c2f25 / c2f0 | 0.00 | +0 | 0% | 1.4 |
| deep | g5f25 | 0.05 | +0 | 0% | 1.6 |
| sharp (ℓ*−0.15) | fixed | **0.00** | **+0** | 0% | 0.0 |
| sharp | c5f25 | 0.10 | **+71** | 0% | 2.5 |
| sharp | c2f25 / c2f0 | 0.10 | **+14** | 0% | 2.8 |
| sharp | g5f25 | 0.15 | **+100** | 0% | 3.6 |

- **Deep regression detection is unchanged** under every pin: the gate is
  lost immediately and stays lost; wrongful re-confirmations ≤ 0.05/4
  opens (1 seed-open in 400).
- **Sharp regression prices the decay exactly where the F88 information
  bound predicts — and EVERY pin pays.** Under the FIXED hazard the
  boundary shock fails the gate's sd conjunct immediately (+0 flicker);
  under any decayed pin the re-widening is small enough for the gate to
  keep flickering on while stale above-bar mass drains at the
  information-bounded evidence rate: median +14 trials (c2), +71 (c5),
  +100 (geometric), with wrongly-satisfied opens 0.10–0.15/4. The
  flicker is NON-MONOTONE in the scale (c5 worse than c2): two channels
  compete — a larger residual shock re-fails the sd conjunct (pushes
  toward fixed's +0) while a larger residual re-mix re-arms above-cut
  strata (the F88 FG channel) and a SMALLER one lets the concentrated
  belief persist (pushes toward geometric's +100) — and the median of a
  bimodal 20-seed statistic is unstable in between. The robust facts:
  fixed = 0 everywhere; every decayed pin > 0; every seed ends
  un-declared within the run; the D18 re-cert backstop sits above this
  in production.
- **The floor never binds at the n reached here** (c2f0 ≡ c2f25 on every
  metric; c2's scale at n=3 is 0.40 > 0.25) — the floor is long-run
  insurance (it bounds what unlimited confirmations can ever remove), not
  a short-run guardrail.
- The π-collapse reset does NOT fire on sharp regression (n_conf stays
  ~2.8 at end): π settles in the 0.5–0.95 band, the gate is already lost,
  and the hazard stays partially decayed while the task re-enters
  training. Recorded honestly; the D33 lifecycle still gates any future
  confirmation, and a revocation there resets in the sandbox.

### 2.3 Arm F — FG zoo (6 sessions)

| zoo | hazard | FG declared | FG CONFIRMED | post-FG gate at opens |
|---|---|---|---|---|
| static ℓ*−0.15 | fixed / c2f25 / g5f25 | 6/20 (identical) | **0/20** | 0.00 |
| careless | fixed / c2f25 / g5f25 | 0/20 | 0/20 | 0.00 |

Sharp-margin declared-FG replicates the F88 baseline (6/20) BIT-IDENTICALLY
across hazard arms: a static learner's false declaration does not survive
to the next open (π < 0.95), so it never earns credit and never decays its
own hazard — **no FG lock-in channel exists** at this margin. Careless
0/20 everywhere.

## 3. Verdict and D42

**F89 verdict.** Implemented, mechanically validated (test_m26, 14
checks), and honestly effective WHERE THE SHAPE IT TARGETS EXISTS: the
early-declarer + slow-second-task pair (the F85-ii/F86 motivating shape)
completes +5pp at the best-priced Beta pin, +10pp at c5, +20pp under the
geometric ablation, with first declarations, plateau behavior,
careless/static FG, and deep-regression detection all
bit-unchanged-or-noise. But the endpoint gain and the sharp-regression
blind spot are COUPLED: the pins that buy real completion (c5f25 +10pp,
g5f25 +20pp) measure +71/+100 trials of post-regression false-mastery
flicker, and the pin that bounds the flicker at +14 (c2f25) buys only
+5pp / BOTH 300→275. There is no studied point that gets the tail gain
without opening the F88 corner.

**D42 (the default).** Sandbox `HAZARD_ADAPT` STAYS **False**; the
machinery ships opt-in with the best-priced studied pin recorded in the
config (`HAZARD_STRENGTH = 2.0`, `HAZARD_FLOOR = 0.25` — sharp flicker
+14 trials at 0.10/4 wrongly-satisfied opens) for field ablations. The
D28 ordering decides: the sharp-regression flicker is an FG-adjacent
false-mastery-persistence channel that the FIXED hazard provably does not
have (+0 across 200 seed-margins), and the +5pp completion the safe pin
buys does not clear the bar M24/M25 set for default flips on weaker
costs. This is the M25 warning measured: weakening the hazard on the
declared side re-opens exactly the corner it was built to close. The
principled way to remove the re-polish tax WITHOUT a blind spot is the
alternative successor M25 already named — terminal-confirmation semantics
(a CONFIRMED task leaves the training rotation for the retention layer,
whose forgetting model owns staleness) — queued as the real next step.

## 4. What changed where

- `training/mixture_filter.py`: `declared_hazard_scale` (Beta
  posterior-mean, floor, geometric ablation) + `AdaptiveBoundaryHazard`
  (lifecycle state, π-credit/collapse rule, json round-trip).
- `sandbox/config.py`: `HAZARD_ADAPT = False` (D42) + the recorded
  ablation pin (`HAZARD_STRENGTH = 2.0`, `HAZARD_FLOOR = 0.25`);
  `sandbox/protocol.py`: lifecycle
  construction from meta (F86 `ever_declared` seeds `ever`), confirm/
  revoke hooks, π-credit at open, scaled Gate 4 + `hazard_scale` events,
  mid-session re-demonstration credit, meta persistence
  (`hazard_lifecycle`).
- `studies/study_m26_hazard.py` (arms A/G/F, `--smoke`, npz merge-save);
  `tests/test_m26.py` (14 checks). Suite 20 files / 287 checks.

## 5. Results and decisions (the PECR record)

| line | proposed | executed | verdict |
|---|---|---|---|
| F89 adaptive hazard | Beta posterior-mean decay of Gate 4's (ε_b, w_share) in post-boundary confirmations, floor, π-credit lifecycle | arms A (endpoint, 3 rosters × 5 pins), G (regression × {deep, sharp} × 5 pins), F (FG zoo) | **implemented + validated opt-in; DEFAULT-DECISIVE NO for the sandbox (D42)** — the endpoint gain and the sharp-regression blind spot are coupled across the whole studied pin range; fixed hazard is the only point with +0 flicker |

PECR rejections recorded this checkpoint: the sandbox default flip
(above); the geometric λ=0.5 pin for any serving use (+100-trial sharp
blind spot); the c5-strength pin (non-monotone middle: +71 flicker for
+10pp); the strict declaration gate as the per-boundary credit event
(unreachable under the full hazard — measured zero re-demonstrations in 5
sessions; replaced by the π mass condition). Machinery retained opt-in
with tests: 20 files / 287 checks green.

## 6. Queued from this checkpoint

1. **Terminal-confirmation semantics** (now the primary F86-line
   successor): a CONFIRMED task leaves the training rotation for the
   retention layer, whose forgetting model owns staleness — removes the
   re-polish tax with NO hazard weakening; needs a retention-side FG
   story (stale-mastery detection cadence) before any flag.
2. If a field ablation ever wants the adaptive hazard: use the recorded
   c2f25 pin, watch `hazard_scale` events with climbing n_conf against
   sagging accuracy (the sharp-regression signature).
3. Confirmed-FG endpoint ratification + contact-triggered onboarding
   demonstration block + λ(time-on-task) (carried from M25 §6).
