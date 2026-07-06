# M25 — Selection successors: EP-v2 lifecycle allocation, item-level rollout placement, and the sharp-margin gate (2026-07-04)

Scope: the three queued M24 successors (docs/M24_SELECTION.md §5), executed
as one autonomous PECR loop. Validation:
`studies/study_m25_selection.py` → `figures/data_m25_selection.npz` and
`studies/study_sharp_margin.py` → `figures/data_sharp_margin.npz`, all
arms at the M23 shipping stack (Gate 4 boundary hazards + sd_floor 0.33)
over the real bank. Companion advisor-facing summary:
`docs/ADVISOR_BRIEF.md`.

## 1. F86 — EP-v2: allocation coupled to the confirmation lifecycle

### 1.1 The residual being fixed

F85's expected-progress allocation wins first-declarations decisively
(134–184 → 72–86 trials) but F85-ii found the post-boundary RE-POLISH
loop: after each Gate-4 boundary shift, a nearly-mastered task's
re-widened, freshly sub-bar posterior mass reads to EP as a progress
opportunity — small gap × very high trainability — and EP out-ranks the
LOW-trainability recovering task. The starved task got ~12 trials/session
and pair completion was censored at 241 trials where Gate-1 completes at
188/189.

The mechanism diagnosis: EP prices TRAINING need, but a task that has
already demonstrated the declaration gate does not have a training need —
it has a MEASUREMENT need (re-concentrate the hazard-injected variance and
re-confirm). Serving it by EP is a category error that the D28 division of
labor (training vs measurement layers) already names.

### 1.2 The rule

`DeficiencyScheduler(progress_lifecycle=True)` — active only with
`progress_alloc`:

- **ever_declared** (the lifecycle key): any task that has EVER satisfied
  the declaration gate, set by the serving loop at each mastery
  transition (`TrainerPolicy.record → mark_declared`) and seeded in the
  sandbox from the persistent provisional/confirmed/`ever_declared` meta
  (a REVOKED provisional still counts — revocation is what the re-polish
  loop looks like from the lifecycle's side).
- **Measurement lane:** ever-declared AND finishing-eligible (bias in
  band, sd within `finish_sd_tol`·floor) ⇒ the task never enters the EP
  ranking; it is served through the F70 finish-first machinery at the
  LOWER `refinish_threshold` 0.50 (it has already demonstrated the bar
  once; the finish budget + cooldown bound its absorption — the
  already-validated device). Below the refinish threshold it simply
  waits (the EP lane keeps the trials); if NO task has a training claim,
  measurement-lane tasks rank by legacy deficiency (no F70-style
  deadlock).
- **Regression escape:** ever-declared but NOT finishing-eligible (bias
  walked out of band / deep re-widening) ⇒ back in the EP lane. That is
  an honest re-training need, not measurement.
- **Exploration floor** (`explore_every=8`): once per 8 picks, an EP-lane
  task not served within the last 8 picks is served instead of the EP
  argmax (oldest first). The M23 fixed-share re-opens the PRIOR toward a
  written-off task, but without served trials the exonerating evidence
  never arrives; the floor guarantees the evidence stream that lets an
  E-shaped recovery re-open its own trainability. Cost bound: ≤5
  trials/session diverted, only when the floor actually binds.

Defaults off ⇒ bit-identical to M24 (test_m25 checks 1–2).

### 1.3 Validation (arm Q2) — endpoint BOTH-DECLARED

Three M24 rosters × {gate1, ep(v1), epv2nf (lifecycle only), epv2
(lifecycle + floor)} × 20 seeds × 8 sessions × 40 trials, K=2 production
stack, real bank. Primary endpoint: first trial by which BOTH tasks have
declared (the F85-ii censoring metric); plus per-task firsts, plateau-sunk
trials, and post-first-declaration service to the second task.

### 1.4 Results — the lifecycle is inert at K=2; the re-polish cost is HAZARD-level

20 seeds × 8 sessions × 40 trials (cap 320; "321" = censored):

| roster | alloc | task0 first | task1 first | BOTH (rate) | task1 trials | task1 svc/sess after task0 declares |
|---|---|---|---|---|---|---|
| b-trap | gate1 | 134 | cens. | cens. (5%) | 215 | 26.9 |
| b-trap | ep    | 86  | cens. | cens. (5%) | 90  | 6.8 |
| b-trap | epv2nf/epv2 | 86 | cens. | cens. (5%) | 91 | 7.0 |
| rates  | gate1 | 150 | 188 | **300 (55%)** | 197 | 22.4 |
| rates  | ep    | **72** | cens. | cens. (20%) | 120 | 11.5 |
| rates  | epv2nf/epv2 | **72** | cens. | cens. (20%) | 120 | 11.5 |
| jumper | gate1 | 184 | 189 | **240 (70%)** | 196 | 17.9 |
| jumper | ep    | **82** | cens. | cens. (5%) | 93 | 7.0 |
| jumper | epv2nf/epv2 | **82** | cens. | cens. (5%) | 94 | 7.1 |

(b-trap's task1 is a true plateau — "censored" is CORRECT behavior there
and its EP column reading is the plateau-protection win: 90 vs 215 sunk
trials.)

Three facts the table forces:

1. **EP-v2 ≈ EP-v1 on every metric** (epv2nf and epv2 identical to two
   decimals). The lifecycle lane routed the ever-declared task through
   finishing instead of the EP ranking — and spent the SAME ~28
   trials/session re-polishing it. The exploration floor never binds at
   K=2 (with task0 out of the EP lane, task1 IS the argmax; during
   finishing service the floor is bypassed by design).
2. **The re-polish cost is a property of the HAZARD, not the allocator.**
   Gate 4 re-widens a declared task at EVERY session open; re-confirming
   it costs ~28 trials/session under any lane that serves measurement
   need to completion. What differs across allocators is only who pays:
   EP variants pay with the second task's service rate (11.5→7
   trials/session — BELOW the ~20/task/session that the M23 arm-G
   analysis showed evidence needs to re-concentrate under the hazard, so
   the second task can never declare); Gate-1 pays with the first task's
   re-confirmation lateness (task1 keeps 18–27 trials/session and the
   pair completes).
3. **On the BOTH-declared endpoint, Gate-1 dominates every EP variant**
   (rates 300@55% vs cens.@20%; jumper 240@70% vs cens.@5%), while EP
   dominates FIRST declaration (72–86 vs 134–184) and plateau protection
   (90 vs 215 sunk). Under D28 (lateness of the full protocol > speed of
   the first win), the pair endpoint governs the sandbox default.

**F86 verdict.** The confirmation-lifecycle coupling + exploration floor
are implemented, mechanically validated (test_m25 checks 3–8), and
HONESTLY INERT at K=2 under the M23 hazard: they fix the allocator-level
symptom F85-ii named, but the binding constraint is one level down — the
boundary hazard taxes every declared task ~28 trials/session forever.
`PROGRESS_ALLOC` therefore STAYS False in the sandbox (D41), now on
endpoint-decisive grounds rather than a deferral. The real successor is
at the hazard level: an EVIDENCE-ADAPTIVE boundary hazard for declared
tasks (e.g., w_share/ε_b decaying in the task's accumulated post-boundary
confirmations, or terminal-confirmation semantics that hand a CONFIRMED
task to the retention layer instead of the training rotation). That is a
belief-model change (it needs its own FG guardrail campaign), queued.

## 2. F87 — item-level tier-3 rollout placement

### 2.1 What the rollout prices that myopic progress cannot

F84's mechanism-3 finding: under the M23 stack the posterior is wide BY
DESIGN (hazard machinery, not calibrated belief), and the myopic
posterior-progress integral hedges against tails the hazard model itself
will erase within trials. The three F84 mechanisms (tail vote, calibration
externality, model-injected width) are constraints any placement successor
must respect. The non-myopic answer: an item's value includes its
INFORMATION externality — a more concentrated belief makes every FUTURE
placement better. `rollout_item_q` prices exactly that channel: inside
each H-step rollout the interior belief updates on the simulated response
and the base policy π0 places all subsequent items against that belief, so
an item that sharpens the belief harvests its value as larger simulated
TRUE-state improvements downstream. Reward stays the plan's Eq. (reward)
on true rollout states; per F84-iii, NO bar-referenced credit at the item
level.

### 2.2 Construction

- Shortlist: stratum-aware `expected_progress_score` top-12 on the
  label side (tier-1-constrained shortlist, the tier3_select design),
  re-ranked by H=6-step, L=8-rollout CRN Monte-Carlo value; CRN across
  candidates is load-bearing (test_tier3 check 4 lesson).
- Mixture honesty (the F84-ii lesson): rollout TRUE states are drawn from
  the pooled posterior with per-particle (σ_∞, learn) — each ceiling
  stratum evolves under ITS OWN ceiling and static mass is frozen
  outright (including process noise). The interior belief keeps the
  base-dynamics pooled approximation (it only drives π0).
- F84-v6 safety devices retained: the mirror magnitude-window constrains
  the partner pick's CANDIDATES (calibration externality); the finishing
  switch hands declaration-imminent tasks to belief-anchored Fisher
  placement (information starvation). The rollout π0 pool is the
  PRE-mirror candidate view (future placements inside a rollout are not
  ring-constrained).

### 2.3 Validation (arm R + H2)

Scenarios {wellspec, jump, contact, slow} × {z1 shipped, roll} × 20
seeds; criterion-herding arm H2 with a live R–W criterion.

### 2.4 Results — no dominance; the F84 placement line CLOSES

20 seeds × 6 sessions (cap 240; "241" = censored), fixed-pool run:

| scenario | z1 declared (rate) | roll declared (rate) | w_true z1/roll | acc z1/roll |
|---|---|---|---|---|
| wellspec | 46 (100%) | 44 (95%) | 0.43 / 0.38 | 0.88 / 0.89 |
| jump | 148 (90%) | 170 (90%) | 0.35 / 0.36 | 0.79 / 0.80 |
| contact | cens. (30%) | cens. (35%) | 0.63 / 0.59 | 0.76 / 0.77 |
| slow | 102 (85%) | 105 (80%) | 0.61 / 0.58 | 0.84 / 0.86 |

Herding (H2, live R–W criterion): |t_true| end 0.173 (roll) vs 0.195
(z1) — the mirror-window device holds under rollout placement; t̂
tracking error 0.135 vs 0.106.

Sensitivity note (honest record): an intermediate code state constrained
the rollout's INTERIOR π0 pool to the mirror ring for partner picks and
measured jump at 118 — faster than z1's 148 — where the corrected
full-pool model measures 170. The candidate RANKING is that sensitive to
the interior pool model; a placement rule whose sign flips on an interior
modeling choice is not a shippable improvement.

**F87 verdict.** The item-level rollout is implemented, mixture-honest
(static mass provably earns zero — test_m25 check 9), herding-safe, and
runs at study speed — and it does NOT beat the plug-in placement anywhere
it matters (jump slower-or-noise, slow-wide parity-minus, wellspec
parity, contact +5pp rate). This is the third and strongest confirmation
of the F84 mechanism-3 conclusion: under the M23 belief architecture the
posterior is hazard-widened BY DESIGN, and any placement objective that
integrates that width — myopically (F84) or through lookahead (F87) —
hedges against tails the hazard will erase. The plug-in 85%-rule at σ̂
stands as the placement rule; `rollout_placement` stays default-off
experimental; **the F84→F87 placement line is CLOSED** (a future
successor would have to change the OBJECTIVE-side treatment of hazard
width, not add more lookahead).

## 3. F88 — the sharp-margin gate: information bound, mechanism, conjuncts

### 3.1 The information bound (why "more evidence" is not the fix)

Static learner parked at ℓ*−m; certification probes at the z̃≈1.35
optimum; λ=0.025 (s_sd≈0 approximation — real probes prefer low-s_sd
items):

| margin m | probe deficit δ=a0−p_true | e-gate probes to fire (~log20/2δ²) | per-trial KL static-vs-at-bar | trials for BF 20 |
|---|---|---|---|---|
| 0.10 | 0.021 | ~3,300 | 0.0021 nats | ~1,440 |
| 0.15 | 0.032 | ~1,430 | 0.0048 nats | ~630 |
| 0.25 | 0.055 | ~490 | 0.0140 nats | ~215 |

A 240-trial two-task run serves ~120 trials/task and ~24 probes/task: at
m=0.15 neither the e-gate route nor the full-likelihood BMA route can
settle the hypothesis. Consequence: within-run FG at this margin cannot be
driven to zero by evidence; the only honest lever is to stop declarations
that ride NON-EVIDENCE variance, and the D18 re-cert backstop remains the
certifier (it catches 75/75 forced false graduations — F32).

### 3.2 Instrumentation (arm X1) and the two conjuncts (arm X2/X3)

Hypothesis to adjudicate at X1: false declarations ride the Gate-4
boundary hazard — the fixed-share re-mix re-arms above-cut ceiling strata
at every session open and the state shock scatters particles across the
bar, so π transiently clears the gate before fresh evidence re-concentrates
the weights.

Conjunct candidates (both opt-in, `ModeThresholds`):

- `boundary_refractory=12`: no declaration within 12 served trials of a
  boundary shift (arm-D re-contraction takes ~19 single-task trials; 12
  at the 2-task rate ≈ half a session of that task's service).
- `min_trainability=0.50`: a declaration asserts the learner IS above the
  bar, which entails a clearing ceiling; a belief holding ≥50% mass on
  sub-bar/static ceiling hypotheses is not entitled to declare on a
  transient π.
- `probe_every 5→2` (densified e-gate evidence) is the "more evidence"
  comparator the bound predicts will NOT close the gap.

### 3.3 Results — the conjuncts are REJECTED; the lifecycle is the filter

**X1 instrumentation (20 seeds/margin, 6 sessions × 40, K=2)** refutes the
short-transient hypothesis:

| margin | FG (declared) | tsb median (≤12: %) | post-boundary session | trainability med (q10) | π med | e-gate med |
|---|---|---|---|---|---|---|
| 0.10 | 8/20 (14 decls) | 23 (14%) | 100% | 0.92 (0.90) | 0.959 | 0.75 |
| 0.15 | 6/20 (9 decls) | 22 (22%) | 100% | 0.92 (0.90) | 0.960 | 0.75 |
| 0.25 | 2/20 (3 decls) | 23 (33%) | 100% | 0.96 (0.94) | 0.961 | 0.93 |

Every false declaration lands in a post-boundary SESSION but at median
~22 session-trials in — NOT the immediate post-shock transient — with
HIGH trainability (0.92), a quiet e-gate, and π barely past the line.
The mixture is genuinely fooled, exactly as the §3.1 information bound
predicts: the margin sits below the evidence rate the run length can
resolve, and each boundary's fixed-share re-mix restores ~10% prior
weight to above-cut ceiling strata, so the weights never concentrate on
the sub-bar hypothesis within 240 trials.

**X2/X3 — all three conjuncts rejected (PECR record):**

| arm | static15 FG | careless FG | wellspec first (rate) | BOTH declared (rate) |
|---|---|---|---|---|
| base | 6/20 | 0/20 | 75 (100%) | 156 (75%) |
| refr12 | 5/20 | — | 75 (100%) | 180 (60%) |
| tr50 | 6/20 | — | 75 (100%) | 156 (75%) |
| both | 5/20 | 0/20 | 75 (100%) | 180 (60%) |
| probe2 | 6/20 | — | 100 (95%) | 296 (50%) |

- `min_trainability=0.5` is a NO-OP at this margin — X1 already showed
  trainability at FG is 0.92 (the conjunct would need τ≈0.95, which the
  boundary re-mix alone would then block for honest learners).
- `boundary_refractory=12` trims one FG in 20 while costing the honest
  pair −15pp completion and +24 trials (D28: not a trade).
- `probe_every=2` changes nothing on FG (the Bernoulli bound) and is
  catastrophic on lateness (BOTH 296@50%).

The knobs stay in `ModeThresholds` as opt-in ablation instruments;
NOTHING ships enabled (sandbox constants pinned at 0/0.0).

**X4 — the D33 confirmation lifecycle IS the sharp-margin filter:**

| scenario | declared | CONFIRMED (D33) | censored-provisional |
|---|---|---|---|
| static ℓ*−0.15 | 6/20 | **1/20** | 0 |
| careless | 0/20 | 0/20 | 0 |
| wellspec | 20/20 | 6/20 | 3 |

The marginal π≈0.96 crossings that produce sharp-margin FGs do not
survive the next session-open gate re-check: **5/6 false graduations die
at confirmation** — measured in a harness WITHOUT gap-anchor blocks, so
the honest-learner confirmation rate (6/20 within 240 trials, first-CONF
median censored) is a LOWER bound: the real protocol serves 4–16
info-optimal anchor probes immediately before the confirm check, which
re-sharpens an honestly-mastered belief (and adds evidence AGAINST a
static sub-bar one). The M20 robot campaign's lifecycle record
(provisional → revoked ×4 → CONFIRMED ×2 across 20 sessions) is the
same conservatism seen end-to-end.

**F88 verdict.** At sharp margins the trainer's DECLARATION is
information-bounded and cannot be hardened from inside the gate at
acceptable lateness (three rejected conjuncts). The product's protection
is architectural and already shipped: the D33 confirmation lifecycle
(kills 5/6 here) with the D18 re-cert backstop above it (F32: 75/75).
Publication language must therefore price FG at the CONFIRMED level, and
the pilot SAP's FG endpoint should be confirmed-FG, not declared-FG.

## 4. What changed where

- `training/trainer_policy.py`: `DeficiencyScheduler(progress_lifecycle=,
  refinish_threshold=, explore_every=)` + `mark_declared`/`ever_declared`
  (F86); `ModeThresholds.rollout_placement` + `rollout_H/L/nro/short` and
  the skill-mode rollout branch (F87); `ModeThresholds.boundary_refractory`
  + `min_trainability` + `TaskModePolicy.note_boundary` (F88);
  `TrainerPolicy` passthrough + lifecycle hook in `record`.
- `training/trainer_rollout.py`: `rollout_item_q` + `_pooled_truth_arrays`
  (F87).
- `sandbox/config.py` + `protocol.py` + `state_io.py`: EP-v2 constants
  (inert while `PROGRESS_ALLOC=False`), persistent `ever_declared` meta,
  `note_boundary` at Gate-4 application, F88 constants (0/0.0 no-ops).
- `viz/make_user_videos.py`: per-tester trajectory MP4s.
- `tests/test_m25.py` (15 checks); suite 19 files / 273 checks green.
- `docs/ADVISOR_BRIEF.md` (the C4 vehicle).

## 5. Results and decisions (the PECR record)

One checkpoint, three verdicts, all guardrail-priced end-to-end (the
twice-confirmed F84/F85 altitude lesson, applied a third time):

| line | proposed | executed | verdict |
|---|---|---|---|
| F86 EP-v2 allocation | lifecycle lane + refinish threshold + exploration floor | arm Q2, BOTH-declared endpoint, 4 allocators × 3 rosters | **implemented; endpoint-decisive NO** — inert at K=2; re-polish is hazard-level; Gate-1 dominates the pair endpoint; `PROGRESS_ALLOC` stays False (D41) |
| F87 rollout placement | item-level H-step CRN rollout, mixture-honest, F84-v6 devices | arm R (4 scenarios) + H2 herding | **no dominance; default-off; F84→F87 placement line CLOSED** — plug-in placement stands under the hazard-widened belief |
| F88 sharp-margin gate | information bound + instrumentation + 3 conjuncts | X1–X4 | **conjuncts REJECTED** (bound + measured no-ops/lateness); **the shipped D33 lifecycle is the filter** (5/6 FGs die at confirmation); FG claims move to CONFIRMED semantics (D41) |

Guardrails at the would-be shipping configs (arm S2): careless FG 0/20
everywhere; static_below m23 6/20 vs epv2 9/20 / epv2roll 8/20 (the EP
sharp-margin direction persists — one more reason D41 keeps the shipped
allocation); well-spec first declaration 99 (m23) vs 72 (EP variants) —
the speed EP buys is real but D28 ranks pair lateness above it.

PECR rejections recorded this checkpoint: boundary_refractory,
min_trainability, probe densification (all three F88 conjuncts); the EP-v2
sandbox default flip; the rollout placement default flip. Machinery
retained opt-in with tests: 19 files / 273 checks green.

## 6. Queued from this checkpoint

1. **Evidence-adaptive boundary hazard for DECLARED tasks** (the real
   F86 successor, one level down): decay Gate 4's (ε_b, w_share) for a
   task in its accumulated post-boundary confirmations — a declared task
   that keeps re-confirming should stop paying the full ~28-trial hazard
   tax every sitting, freeing exactly the service rate the second task
   needs. This is a BELIEF-MODEL change: it needs its own FG guardrail
   campaign before any flag ships (the hazard is the M23 safety
   machinery; weakening it on the wrong side re-opens F80).
2. **Confirmed-FG endpoint into the pilot SAP** (D41; advisor
   ratification requested — ADVISOR_BRIEF §6.5).
3. **Contact-triggered onboarding demonstration block** (M23 leftover):
   the delivered-value audit's pre-contact sessions (w_true 0.15–0.17)
   are the remaining known value sink; the F83 contact e-process already
   detects no-contact — the missing piece is the protocol response
   (very-easy demonstration ramp until contact certifies). Needs its own
   study + a human field check.
4. Field sessions under the M23/M24 stack (human-gated; M24 item 4's
   bit-identity expectation still holds — M25 changed no default
   behavior, re-verified by the suite).
5. λ(time-on-task) channel (Phase-3, M22 leftover).
6. CLOSED lines this checkpoint: F84→F87 placement (plug-in stands);
   allocation-level fixes for the re-polish loop (F86 — moved to the
   hazard level, item 1).
