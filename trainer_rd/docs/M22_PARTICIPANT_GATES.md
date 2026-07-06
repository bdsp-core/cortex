# M22 — Four-participant pilot: analysis and the three gates (2026-07-02)

> Data: `sandbox/logs-USER-{A,B,C,D}/` (M21 profiles working as designed —
> four clean per-person states; 241 trials total). Findings F77–F79,
> decisions D36–D37 in PROJECT_MEMORY. Validation:
> `studies/study_participant_gates.py` → `figures/data_participant_gates.npz`.

## 1. Per-participant evaluation

**M21 fixes verified in the wild**: d′-vs-label ≡ d′-vs-percept for every
participant×task (label-consistent draws — zero contradiction trials, vs
10% in M20), profiles kept four independent belief states, and the
consistency monitor produced its first real flags (USER-D).

| participant | story | engine response |
|---|---|---|
| **A** | Genuine cross-session learning: d′ 1.3→2.1 (domain2) / 1.3→2.7 (domain3), acc 0.72→0.85, RT median 4.0→1.6 s | Tracked cleanly: π→0.81 both tasks, trainability 0.87/0.91. The success path works. |
| **B** | Near-perfect on domain2 (6/6, then 4/6) but **starved** — 6 trials/session; domain3 plateaued at d′≈0.9 through 68 trials | domain2 π stuck 0.63–0.65 just under the 0.70 finishing bar; domain3 trainability crashed 0.56→0.15 (honest, but it kept absorbing trials) |
| **C** | Mirror of B in one session: weak task (domain2, d′ −0.33) got 25 trials; strong task (domain3, d′ +1.87, liberal bias t̂ −0.48) got 15 | trainability 0.39 vs 0.73; GLR peaked 9.6 on the weak task (elevated, sub-threshold) |
| **D** | Strong on domain2 (d′ +2.7) but ~chance on domain3 **including easy items** (0.44 acc at \|s\|≥1) — behavioral, not perceptual; quit next session after one slow wrong trial | ℓ̂ crashed to −0.89, trainability 0.05; **consistency flag fired 5×** (GLR 12.9) — and, being shadow, changed nothing: 10 more trials went in post-flag (trainability 0.22→0.04) |

Late-session collapse (fatigue): session-quarter accuracies fall to 0.4–0.5
for B (0.90→0.50 twice), C (0.90→0.40), D (0.70→0.40/0.30); A is stable.
The engine writes this into the persistent skill belief right before
state-save (B's end-of-s2 ℓ̂ slide from +0.23 to −0.09 over the last ~10
trials).

## 2. The three failure mechanisms → three gates

### F77 / Gate 1 — allocation conflates "far from cut" with "worth training"

Worst-first deficiency (D3) poured 60–85% of trials into each participant's
weakest task WHILE the mixture's own trainability statistic was collapsing,
and starved near-mastery tasks. The engine already computes the fix's
ingredient: **effective deficiency = deficiency × (floor + (1−floor)·
trainability)** — `DeficiencyScheduler(trainability_floor=0.25)`, opt-in,
None = bit-identical. On the participants' logged final states this
reverses B/C/D's allocation and leaves A's balanced case unchanged.

**F77-ii (found by the validation sim, not the participants):** finishing
eligibility `sd_ℓ ≤ sd_floor` deadlocks when the mixture's cross-strata
variance floor sits a hair above `sd_floor` (observed: 0.232 vs 0.23) — the
task can't enter finishing, so it never gets the on-task trials that would
concentrate the strata. This is the residual crack in the F70 deadlock fix.
`finish_sd_tol=1.25` relaxes ELIGIBILITY only; declaration keeps the strict
floor (no false-graduation channel — verified: FG 0 in the static arm).

### F78 / Gate 2 — a shadow flag must at least stop the bleeding

USER-D's five flags changed nothing; the session kept serving the broken
task. Protocol response (sandbox `CONSISTENCY_PAUSE=True`): a fired flag
**suspends that task for the rest of the session**
(`TrainerPolicy.suspended` / `pick(exclude=)`), logs `consistency_pause`,
tells the participant to re-read the instructions, and gives the task a
fresh monitor window next session. Beliefs are never mutated (D34's shadow
principle: the flag gates SERVING, not inference). Validation (G2): pause
cuts post-flag broken-task trials ~10 → 0 and halves the belief damage.

### F79 / Gate 3 — fatigue is a decline, not a level

Late-session collapse is real (3/4 participants) and un-modeled (λ is
constant in the observation model), so it lands in the skill posterior.
Protocol guard: end the session when the trailing-12
realized-vs-belief-predicted accuracy deficit rises more than δ = 0.35
above the **session's own first-12 baseline** (differenced; active from
trial 24). The differencing matters: the raw deficit false-fires on
onboarding belief-optimism (would have cut USER-A's fine first session at
trial 17); the differenced form fires exactly on the two real late-session
collapses (B s2 +0.41 at pair 29, D s1 +0.37 at pair 29) and stays silent
on A and C. δ pinned at 0.35: stationary false-break 5/100 sessions
(9/100 at 0.30) while both real events still clear it — USER-D's margin is
thin (+0.02), recorded honestly. Phase-3 note: the principled fix is a
λ(time-on-task) channel (F15's probes feed it); the guard is the
protocol-level containment.

## 3. Validation results (study_participant_gates, full run)

*(arm summaries — see `figures/data_participant_gates.npz`)*

- **G1** (30 reps × 5 sessions, trainable good task + weak partner):
  *guesser partner (the participant regime — trainability collapses fast)*:
  discount declares the good task earlier (median 116 vs 141) and more
  often (18/30 vs 16/30) with 10.5 fewer trials sunk into the guesser;
  *static-1.6σ\* partner (ambiguous evidence, trainability collapses
  slowly)*: statistically neutral (19–23/30 declared, medians 107–116) —
  the discount is correctly gentle when the mixture hasn't concluded
  anything; *both-trainable control*: unchanged (25/30 both arms).
  **FG = 0 in every arm** (finish_sd_tol touches eligibility only).
  Honest scope note: the sim's good task races past the mid-π crossover
  where the discount matters most; USER-B's good task SAT at π 0.63 for
  two sessions — the unit-test pin (π .55/tr .85 vs π .10/tr .05 flips
  allocation) is the participants' actual configuration. Residual: the
  worst-first geometry still gives a floored hopeless task (score 0.25)
  priority over a π>0.9 task between finishing bursts — an
  expected-progress-per-trial allocation is the principled successor
  (open item, not this gate).
- **G2** (30 reps): pause on → post-flag broken-task trials 12.9 → 0.0;
  end trainability of the broken task 0.14 → 0.36 (damage halved), good
  task unaffected (0.83 → 0.85). Flag rate 22/30 within 40 trials at the
  M21-pinned threshold.
- **G3a** (100 stationary sessions): false-break rate 32/20/9/5 per 100 at
  δ = 0.20/0.25/0.30/0.35 ⇒ pinned δ = 0.35.
- **G3b** (FatigueLearner, λ ramps 0.025→0.30 by trial 25): caught 15/40 at
  δ=0.25, 9/40 at 0.30, median at trial ~30 — power on MODEST ramps is
  weak by design; the guard targets big collapses (the real ones measured
  −0.37/−0.41), and misses are recoverable (D33 + anchors).
- **G3c**: real-participant replay — the differenced guard fires only on
  USER-B s2 (pair 29) and USER-D s1 (pair 29); silent on USER-A (both
  sessions), USER-B s1, USER-C s1.

## 4. What was left alone, deliberately

- **B's domain3 plateau** (d′ 0.9 through 68 trials): the trainability
  collapse is plausibly CORRECT — Gate 1 redirects effort, and the F67
  trainability+plateau handoff (re-cert owns the verdict) is the designed
  path. No observation-model change.
- **C's liberal bias on domain3** (t̂ −0.48): bias mode will engage next
  session; machinery exists (D15 dual control), nothing to fix.
- **The trainability statistic itself**: every collapse it produced was an
  honest inference on the data it saw. The gates change what the protocol
  DOES with it, not how it is computed.
- **Session length (D28 ≈40)**: the fatigue guard personalizes effective
  length; no constant change.
