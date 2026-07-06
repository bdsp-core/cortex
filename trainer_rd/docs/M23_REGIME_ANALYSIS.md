# M23 — Regime shifts in the seven-profile pilot: analysis, derivations, and the boundary-shift stack (2026-07-04)

Data: sandbox sessions of 2026-07-02/03 — the M22 four (USER-A..D) plus a new
session each for A and C and three fresh profiles (USER-E ×3 sessions,
USER-F ×1, USER-G ×2). 557 trials across 14 (user, task) sequences, all under
the M21 render calibration (gain 0.85) and the M22 gates. Analysis scripts:
this doc's §1 tables from the raw JSONL; everything from §3 on is reproduced
by `python3 -m studies.study_m23_regime` (cache
`figures/data_m23_regime.npz`).

## 1. What the new sessions show

Per-session accuracy, sensitivity (probit fit to the served (s_real, y)
pairs; d′ equivalently), and RT:

| user-sess | task | n | acc | d′ | RT med (s) | note |
|---|---|---|---|---|---|---|
| E-s1 | both | 40 | 0.45 | −0.4…0.0 | 10–15 | **pre-contact: pure guessing** (LR vs guessing ≈ 0) |
| E-s2 | both | 40 | 0.67 | +0.8/+1.0 | 14–18 | first contact, weak |
| E-s3 (64 s later) | dom3 | 26 | **0.96** | **+3.04** | 4.6 | **regime jump**: Δlnσ ≈ 1.3 across a 64-second restart |
| C-s1 | dom2 | 25 | 0.44 | −0.31 | 1.6 | pre-contact guessing |
| C-s2 (28 h later) | dom2 | 23 | 0.87 | **+2.01** | 1.4 | regime jump overnight |
| A-s3 (25 h later) | both | 36 | 0.92 | +2.4/+2.6 | 1.7 | mastery path: both tasks provisional, session self-ends `all_mastered_or_empty` |
| F-s1 | both | 40 | 0.85 | +1.6/+2.8 | 5.2 | strong from trial 1 |
| G-s1..2 | both | 80 | 0.79→0.88 | +1.5→+2.4 | 3.4 | textbook learner; dom2 provisional at global trial 56 |

Three quantitative patterns, each with a mechanism the M22 stack lacks:

**(i) Acquisition is a discrete regime shift, localized at sitting
boundaries.** E's d′ went 0.8 → 3.0 between two sessions separated by 64
seconds; C's went −0.3 → 2.0 overnight. In the transition kernel's terms
these are Δlog σ ≈ 1.3–1.6 moves; the kernel's per-trial innovation is
q_σ = 0.02 with drift ≤ α_σ·w·gap, so the between-session prior places
essentially zero mass on the observed move — the filter must grind it out of
likelihood. Measured lag: the live belief crossed π = 0.5 at trial 12/26 of
E-s3 (final σ̂ 0.61 while the fitted behavioral σ was ≈ 0.45), at trial 21/23
of C-s2. Placement inherits the lag in both directions: served-accuracy vs
the 0.8413 target was 0.45 (E-s1, under-placed for a pre-contact learner,
pool-limited) and 0.93–0.97 (E-s3/A-s3, over-easy against a stale-high σ̂,
information wasted exactly in the sessions that should certify).

**(ii) The onboarding failure mode is "no contact", not "reversed mapping"
and not "slow learning".** Constrained probit fits (slope sign ± vs the
guessing model): C-s1-dom2, D-s1-dom3, E-s1-both are all best described as
GUESSING (LR vs guessing ≤ 0.5 nats; the negative-slope alternative gains
≤ 1.5 nats — no reversal evidence). The belief meanwhile predicts
onboarding accuracy 0.73–0.77 (the F79 optimism), so the plug-in
belief-predictive log-loss loses to the climatological baseline in 11/14
sessions — the belief is systematically miscalibrated in BOTH phases: over-
optimistic pre-contact (+0.22…+0.29 predicted-minus-realized), pessimistic
post-jump (−0.13).

**(iii) Trainability is held hostage by stale evidence.** The whole-history
BMA weights drove trainability to 0.34 (E-dom3) and 0.36 (C-dom2) during the
guessing phase; the breakout session then OPENED at those values —
interacting with the M22 Gate-1 allocation discount (multiplier
0.25 + 0.75·P ≈ 0.5) precisely when the learner had become maximally
trainable. Recovery took 11 (E) and 23 (C) trials of the breakout evidence.
B-dom3 is the correct contrast: genuinely plateaued (d′ 0.86 → 0.90 across
two sessions), trainability 0.20 — the statistic should stay low there, and
does under every M23 variant (arm C).

Also in the new data: the D15 bias mode field-worked on USER-C (criterion
−0.48 → −0.13 over one session with d′ preserved — the first live
confirmation of the dual-control loop); the M22 gates fired nowhere they
shouldn't (0 consistency pauses, 0 fatigue breaks in the 8 new sessions);
USER-D's careless collapse remains the only RT-floor event in the corpus
(trailing-6 median 206 ms vs ≥ 1059 ms in every engaged window of all 14
sessions — the (F82) margin is 5×).

## 2. Derivations (F80–F83)

### F80 — the transition kernel needs a regime-shift component

Model the latent per-task state x = (t, log σ) between sittings as
subject to a shift hazard: at each session boundary, with probability ε_b
the observer re-organizes,

    x⁺ = x + J·Z,   Z ~ N(0, diag(Λ_θ², Λ²)),   J ~ Bernoulli(ε_b),

with Λ_θ = θ-scale·Λ (the observed jumps are σ-dominant: E moved Δlnσ ≈ 1.3
vs Δt ≈ 0.1; C's criterion moved 0.35 — hence a scaled, not zero, θ shock).
SMC implements this exactly by sampling J per particle (`boundary_jump`);
no static parameter lives on a particle, so the F4 path-degeneracy argument
is untouched. The (1−ε_b) bulk of the belief is unmoved — for a stable
learner the posterior pays only ε_b·Λ² added variance (≈ +0.056 in ℓ-var at
the pinned values, re-contracted by ~20 trials of serving, arm D) — while
the ε_b tail gives one session of likelihood a bridge to the new regime.
A per-trial contaminated-innovation kernel (η ~ (1−ε)N(0,q²) + εN(0,(κq)²),
the Masreliez/West robust-filter form) is implemented as well
(`jump_eps/jump_kappa`) and was fit head-to-head; the boundary variant wins
on the real logs (arm A) and costs nothing at the stationary floor, which
the per-trial variant inflates by construction (var factor 1 + ε(κ²−1)).

### F81 — the ceiling posterior must be a hidden-Markov, not a static, BMA

The σ∞-mixture treats the ceiling hypothesis as static, so its weights
integrate ALL history. Under a regime shift that is wrong in a specific,
fixable way: let the ceiling hypothesis itself re-draw from the prior with
probability ρ at each sitting boundary (a strategy epiphany changes what is
reachable). Exact Bayes for that hidden-Markov ceiling is FIXED-SHARE
(Herbster & Warmuth 1998) applied at boundaries:

    ω ← (1 − ρ)·ω + ρ·ω_prior,

implemented in weight space (`boundary_shift(w_share=ρ)`). Two properties
matter and were verified:

- Within-session BMA concentration DYNAMICS are untouched (unlike
  per-trial forgetting, which flattens on every trial): the tempting
  alternative — per-trial DMA forgetting ω^γ (Raftery et al. 2010), also
  implemented (`forget=`) — is REJECTED by the guardrails: at γ = 0.98
  the well-specified declaration rate collapses 100% → 30% (median 34 →
  200+ trials) because uniform flattening keeps the cross-strata variance
  above the sd_ℓ ≤ 0.23 declaration floor forever. This is a new
  quantified failure mode of evidence-forgetting in mixture gates
  (recorded as the F81-ii caveat; the flag stays for ablations only).
  Boundary fixed-share still re-injects cross-strata variance ONCE per
  sitting, which raises the achievable end-of-session floor — handled by
  the F5 floor re-measurement (arm G), not by weakening the mechanism.
- The static/careless protection (F54) survives fixed-share because the
  strata CLOUDS still track the data — π stays ≈ 0 for a sub-bar learner in
  every stratum; only the ceiling weights are re-mixed (arm C: FG not
  degraded vs base under the production gate stack).

Protocol ordering is load-bearing (and implemented): anchors → D33
confirm → boundary_shift → serve. A confirmation must reflect evidence at
close, not hazard widening; the widened tail then meets the new session's
first trials.

### F82 — non-perceptual response runs are RT-detectable with a 5× margin

Scanning the 48-sample trace takes ≳ 1 s; across all 14 human sessions every
engaged trailing-6-median RT stayed ≥ 1059 ms, while USER-D's careless
collapse ran at 206 ms. Gate 3b: k = 3 consecutive RTs < 500 ms ends the
session (union with the F79 accuracy guard). Calibration (arm E): 0 false
alarms on every engaged session at any floor ≤ 1000 ms; on D-s1 the RT
channel fires at trial ~36 vs the accuracy channel's 39 — and unlike F79's
differenced deficit (D's margin was +0.02 over δ), the RT margin is 5×.
Serving-layer only (D37 principle); beliefs never touched. The two robot
responders' simulated RT constants were raised to engaged-plausible values
(they predate any RT semantics).

### F83 — "is there perceptual contact at all?" is anytime-decidable

Null (no contact): the conditional probability of a correct response stays
in the chance band, P(c_t = 1 | F_{t−1}) ∈ [1−p0, p0], p0 = 0.55 (the
serving policy's measured label balance is 0.45–0.55). Statistic:

    E_T = Π_{t≤T} q_t(c_t)/p0,

with q_t the Krichevsky–Trofimov universal Bernoulli predictor
(q_t(1) = (n₁+½)/(t+1)). Because q_t is a predictable probability forecast,
E[q_t(c_t)/p0 | F_{t−1}] ≤ p0·(q_t(1)+q_t(0))/p0 = 1 under any law in the
null, so E_T is a nonnegative supermartingale and Ville's inequality gives
anytime-valid level α = 0.05 at threshold 20 — same machinery family as the
D23 e-gate. Design choices, each deliberate:

- KT numerator, not the belief predictive: immune to the F80 belief lag
  (the belief-predictive version never certifies E's breakout — its own
  pessimism throttles it) and to response bias (a μ-biased stimulus-blind
  guesser has marginal correctness ≈ ½ under label balance, which KT
  tracks; validated at μ = 0.65: 0/200 false contacts).
- Per-session restart: the monitor answers "is there contact NOW"; a
  cumulative version never recovers from E's 40-trial guessing deficit.
- Semantics: contact = correctness departs the chance band in the
  direction of information. An anti-correlated responder departs the band
  too and would certify — correctly, for protocol purposes (both cases
  mean "the learner extracts structure; instructions/mapping are the
  issue"). Power: an at-target (84%) responder certifies in ~15–20 trials;
  E-s3's 0.96 run certifies in 8.
- SHADOW-first (D34): logged per trial (`contact.log_e`), surfaced as a
  `contact` event, gates nothing until a study justifies action. The
  onboarding intervention it should eventually drive (a demonstration/
  comprehension block when no contact by ~trial 20) is queued.

## 3. Validation (study_m23_regime, full run 2026-07-04)

Pinning rule, pre-stated: (1) evidence-admissibility — pooled prequential
log-evidence on the 14 real sequences within 2×seed-SD of base or better;
(2) among per-family best admissible variants, minimize post-jump lag(π)
on the synthetic E-like jump; (3) guardrails — well-specified control keeps
cov90 ≥ 0.85 and declaration (rate ≥ 0.9, lateness ≤ 1.4× base); FG not
degraded on the production harness. STOP rule: FG degradation ⇒ do not
enable.

### Arm A — evidence on the real logs (557 trials, CRN, 3 seeds)

- σ-dominant boundary jumps (θ-scale 0.33) are evidence-neutral-to-positive
  (−0.04…+0.04 nats); FULL-θ shocks lose up to −0.83 — the data themselves
  say the jumps live in σ, not the criterion.
- Per-trial contaminated innovations (tj) lose at every setting (−0.09 to
  −0.31): the jumps are boundary events, not within-session events.
- Combos of boundary jump + fixed-share are the best family (+0.14…+0.36);
  per-trial DMA forgetting alone +0.21 (γ=0.98).
- Leave-one-user-out: combo_bj+fs selected for 4/7 held-out users;
  held-out Δs −0.15…+0.11 nats. Honest reading: **557 trials cannot
  distinguish these kernels prequentially** (admissibility tolerance
  2.53 nats; all 29 variants admissible). The evidence screen certifies
  "the real data do not disfavor the mechanism"; the selection weight
  falls on the operating characteristics below — exactly what the
  pre-stated rule provides for.

### Arm B — synthetic E-like jump (σ 3.5 → 0.55 at a boundary) + control

| variant | lag σ̂ | lag π>.5 | declare (post-jump) | cov90 post | control: declares |
|---|---|---|---|---|---|
| base (M22) | 113.6 | 93.5 | 116.4 | 0.11 | 100% @ 34 |
| forgetting γ=.98 | 57.9 | 43.2 | 120 (censored) | 0.81 | **30% @ 201** |
| bj(.2,1,.33) only | 47.7 | 38.6 | 117.9 | 0.85 | 100% @ 34 |
| per-trial tj(.01,6) | 111.2 | 89.1 | 117.5 | 0.15 | 100% @ 38 |
| bj+fs .2 (stage-1 pin) | 22.6 | 12.4 | 86.5 | 0.97 | 100% @ 34 |
| bj+fg | 31.1 | 22.4 | 120 (censored) | 0.97 | **25% @ 201** |

Two decisive facts: (i) the M22 kernel is refuted in exactly the way the
real data suggested — 93 trials of lag and 11% post-jump coverage on a
jump the pinned combo handles in 12 with 0.97 coverage; (ii) **per-trial
DMA forgetting is REJECTED by the declaration guardrail** (F81-ii): ω^γ
flattening keeps the cross-strata variance above the sd_ℓ ≤ 0.23 floor
forever, so a well-specified learner declares 30% of the time at 5×
lateness. The forgetting flag stays in the engine for ablations only.

### Arm G — the PECR rework: harness-realism re-pin

The stage-1 guardrail was measured at the WRONG serving rate: arm B's
control serves ONE task 40 trials per boundary; real sessions serve two
tasks (~20 trials/task/session). Re-measured on the 2-task production
harness (M22 Harness2, well-specified learner, 8 sessions):

| config | declaration (floor .23) |
|---|---|
| M22 base | median trial 126, rate 100% |
| stage-1 pin bj(.2,1)+fs.2 | **never (0/8 seeds)** |
| any hazard down to bj(.05,.75)+fs.1 | 0–38% rate |

Mechanism: each boundary re-injects eps·Λ² state variance AND fixed-share
re-injects cross-strata weight variance; at ~20 trials/task/session the
evidence cannot re-concentrate below 0.23 before the next shock — the
achievable end-of-session sd_ℓ cycle-minimum under the final hazard is
0.226–0.263, sitting AT/ABOVE the M22 floor. This is the F5 rule violated
by construction ("if the mastery gate's sd floor is set below the
achievable steady-state floor, no learner ever graduates") in its F77-ii
mixture form. The F5-sanctioned fix: re-measure and re-set the sandbox
declaration floor, 0.23 → 0.33 (≈1.3× the measured cycle-max). FG safety
is carried by π − 2·mcse ≥ 0.95 + the D23 e-gate, which a WIDER posterior
makes strictly harder, not easier, to satisfy; verified below.

FINAL Gate-4 config: **bj(ε_b=0.10, Λ=0.75, θ-scale=0.33) + fixed-share
ρ=0.10, sandbox sd_floor 0.33** — the gentlest hazard that retains most of
the tracking value, needing the smallest floor raise:

- declaration (2-task harness): **100% @ median 94** — faster than M22's
  126 at its own floor (the strict floor was itself costing lateness);
- E-like jump at the final config: lag(π) **15.2** (vs 93.5), declaration
  **51** post-jump trials (vs 116);
- real-log breakouts (arm A2, replay): E-t2 π-crossing 10 → 8.3 trials,
  C-t1 21 → 16.7; trainability at breakout-session open 0.37 → 0.42 (E)
  and 0.42 → 0.46 (C), recovery-to-0.7 8.7 → 7.3 and 16 → 13 trials;
- boundary transient at the final constants (arm D): sd_ℓ 0.201 → 0.298
  (analytic 0.311), re-contracting in 19 single-task trials.

### Arm C — FG safety at the shipping config (production gate stack)

| responder | M22 shipped (floor .23) | bj only (floor .33) | fs only (floor .33) | M23 final (floor .33) |
|---|---|---|---|---|
| static_below (ℓ*−0.15) | **11/20** | 6/20 | 9/20 | **6/20** |
| careless (λ=0.35 at bar) | 0/20 | 0/20 | 0/20 | 0/20 |

The headline surprise: the M22 sandbox stack ITSELF false-graduates 11/20
static-below learners at the sharp −0.15 margin on this harness (the
F63/F66 capstone measured a different config — K=7, larger bank s_sd; the
sandbox regime with s_sd ≈ 0.06 and the optimistic fresh prior is more
FG-prone; metric = ≥1 provisional declaration in 240 trials, D33
confirmation layer not included). Gate 4 IMPROVES it to 6/20, and the
attribution arms show the state jump is what does it — the boundary
shocks let likelihood repeatedly break the σ∞-relaxation attractor's pull
(the F28 prophecy mechanism) — while the floor raise plus fixed-share
alone still sits at 9/20 ≤ base (the sd gate was not the FG protection;
π + e-gate are). An auxiliary spot-check (floor raise alone, no hazard)
measured 8/20. The residual 6/20 at this margin is the open FG item
below.

### Arm D — width costs, measured

Boundary transient at the stage-1 constants: sd_ℓ 0.201 → 0.503 (analytic
√(sd² + εΛ²) = 0.490), re-contracting in ~40 single-task trials — the
number that killed the stage-1 pin at the 2-task rate. At the final
constants the per-boundary ℓ-variance injection is εΛ² = 0.056 (open
sd_ℓ ≈ 0.31 from 0.23), consistent with the measured cycle-min 0.23–0.26.

### Arm E — RT floor (F82)

0 false alarms across all 451 engaged trailing windows (14 sessions) at
every floor ≤ 800 ms, under both rules; D-s1 fires at trial 36
(3-consecutive @ 500 ms) vs the accuracy guard's 39. Pinned: 500 ms / k=3
— 2.4× above D's collapse level, 2.1× below the fastest engaged window.

### Arm F — contact e-process (F83)

Real logs (per session, threshold E ≥ 20): contact certified at trial 8
(F both tasks; E-t2-s3 — the breakout; G-t2-s1), 14–23 (A-s3, G-s2,
C-t1-s2), and NEVER in any pre-contact or plateau session (E-s1/s2, C-s1,
D, B-t2 twice). Null validity: 3/200 false contacts at μ=0.5 and 1/200 at
μ=0.65 (bound: 10/200) — the KT numerator's bias-robustness is real.
Power: median 8 trials for a σ=0.7 learner; the statistic is
under-powered below ~0.72 realized accuracy (B-t2's honest 0.68-acc
plateau never certifies — acceptable for a shadow onboarding monitor;
recorded as a power limitation, not a defect).

## 4. What ships, what stays shadow, what was left alone

Shipped opt-in (defaults bit-identical to M22, engine):
- `TaskFilter(jump_eps=, jump_kappa=)` per-trial contaminated innovations;
  `TaskFilter.boundary_jump(eps, sd, theta_scale)`.
- `SigmaInfMixtureFilter(forget=)` per-trial DMA forgetting (ablation-only
  per F81-ii); `boundary_shift(eps, sd, theta_scale, w_share)` fixed-share.

Sandbox (opts in, constants in `sandbox/config.py`):
- Gate 4: `boundary_shift(*BOUNDARY_JUMP, w_share=BOUNDARY_WSHARE)` =
  bj(0.10, 0.75, 0.33) + ρ=0.10 at every session boundary, after the D33
  confirm phase; declaration `sd_floor` re-measured 0.23 → 0.33 (arm G,
  F5 logic — the hazard raises the achievable cycle floor; π + e-gate
  carry FG protection and are unchanged).
- Gate 3b: RT floor 500 ms / k = 3, union with the F79 fatigue guard.
- Shadow contact e-process per (task, session), telemetry + event only.

Left alone, deliberately:
- Placement (F75 residual): E-s1's 0.45 served accuracy is pool-limited
  (max |s_mean| ≈ 1.3 — no item is easy enough for a pre-contact learner);
  the post-jump over-easy serving is a lag artifact that Gate 4 fixes
  upstream. A posterior-predictive placement rule (choose s so that
  E_post[P(correct)] = 0.8413, replacing the F75 z-heuristic) is queued as
  the principled successor.
- The consistency monitor, the F79 accuracy guard, and D33 semantics are
  unchanged; Gate 4's ordering keeps confirmations evidence-based.
- B's plateau (trainability 0.20) is treated as signal, not pathology —
  fixed-share re-tests it each sitting at ρ·prior weight, which is the
  designed "benefit of the doubt" and is bounded.

Open items queued from this checkpoint:
- **Sharp-margin FG at the sandbox config** (arm C): the M22-shipped stack
  false-graduates a static learner sitting 0.15 below the cut 11/20 times
  in 240 trials on the production harness (Gate 4 improves it to 6/20;
  D33 confirmation and D18 re-cert sit above this gate and were not
  simulated). Needs its own study: the sandbox regime (bank s_sd ≈ 0.06,
  optimistic fresh prior) is materially more FG-prone than the F63/F66
  capstone config, and nobody had measured it.
- Posterior-predictive placement (E_post[P(correct)] = 0.8413) replacing
  the F75 z-heuristic.
- Contact-triggered onboarding intervention (demonstration block when no
  contact by ~trial 20) once a session's worth of shadow contact
  telemetry exists.
- Per-session α-spending policy for the contact monitor if it ever gates
  serving (currently shadow; per-session restart = per-session α).

## 5. Session-2 predictions (falsifiable, for the next data drop)

1. E and C continue above d′ 2 ⇒ their beliefs should confirm within the
   first half of their next sessions (Gate 4 removes the re-descent lag).
2. Any new fresh profile that guesses for a full session should show
   `contact.log_e` pinned ≤ 0 throughout, then certify contact within
   ~10–20 trials of first real contact.
3. No engaged session should ever log `rt_floor_break` (the calibration
   margin is 5×); any careless collapse should end within 3 trials of
   sustained sub-500 ms responding.
4. Confirmed-mastery revocations should not increase vs M22 (the
   boundary_shift ordering protects the confirm phase; watch `revoked`
   events).
