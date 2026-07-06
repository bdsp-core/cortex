# M27 — Pre-submission completion loop: terminal confirmation (F90), contact-triggered onboarding (F91), λ(time-on-task) verdict (F92), the domain interface (F93) (2026-07-05)

Scope: the user-directed pre-submission pass (a legitimate D43 stopping-
rule reopen, category 1 — user request): complete the parked queue where
completable in-scratch, make the architecture committee-presentable and
domain-pluggable, keep the suite green. Validation:
`studies/study_m27_terminal.py` → `figures/data_m27_terminal.npz`,
`studies/study_m27_onboard.py` → `figures/data_m27_onboard.npz`,
`studies/study_lambda_tot.py` → `figures/data_lambda_tot.npz`;
`tests/test_m27.py` (12 checks).

HARNESS NOTE (applies to §1–§2). The M27 lifecycle harness serves an
8-trial info-optimal ANCHOR BLOCK to each provisional task before its
D33 confirm check — the sandbox's real confirm path, which the F88/M26
harnesses omitted (making honest confirmation artificially rare). All
arms including base run the anchored harness; baselines are re-pinned
here (e.g. rates BOTH-declared 256@65% vs 300@55% un-anchored).

## 1. F90 — terminal-confirmation semantics

### 1.1 Design

A D33-CONFIRMED task leaves the training rotation for good: the Gate-4
hazard keeps its FULL pinned rate on the belief (nothing in the M23
safety machinery is weakened — the D42 lesson), but the scheduler stops
converting the re-widening into the ~28-trials/session re-polish tax.
The retention layer owns the task — ONE consolidated SM-2 review
schedule + `maint_probes` at-bar certification probes per session — and
every terminal-task outcome feeds an anytime-valid stale-mastery e-gate
(Ville, α = 0.05: P(ever wrongly revoking an at/above-bar learner) ≤ α
per task); a fired gate returns the task to the training rotation.
Opt-in: `TrainerPolicy(terminal_confirmation=, maint_probes=,
stale_alpha=)`; sandbox `TERMINAL_CONFIRMATION` (D44: stays False).

**Mechanism finding recorded for the port (the campaign's first run
caught it):** difficulty-binned retention registration fires at every
false→true mastery TRANSITION, and the M10 gate flapping proliferates
bins; the live `is_mastered` pre-pick masks this in the shipped
protocol, but any semantics that bypasses the gate (terminal, or a
future port's "completed" state) lets EVERY stale bin fire every
session — measured ~20 retention trials/session, silently re-creating
the tax. Fix shipped with F90: `mark_terminal` consolidates the task to
the single (k, 0) schedule and terminal tasks are exempt from
transition registration.

### 1.2 Endpoint (arm T: anchored harness, 20 seeds, 8 sessions, K=2)

| roster | arm | BOTH declared (rate) | BOTH confirmed (rate) | post-conf tax/sess | svc→task1/sess |
|---|---|---|---|---|---|
| rates | base | 256 (65%) | cens. (15%) | 13.2 | 23.7 |
| rates | term0 | **212 (85%)** | cens. (20%) | **0.5** | 24.5 |
| rates | term3 | 234 (85%) | cens. (25%) | 3.5 | 25.0 |
| jumper | base | 314 (50%) | cens. (10%) | 15.2 | 20.5 |
| jumper | term0/term3 | 295/291 (60/65%) | cens. (15–25%) | 0.5/3.4 | 23.5 |
| b-trap | all | cens. (5%) | cens. (0–5%) | 0.4–10.4 | 27→34–37 |

The endpoint win is real: the tax collapses to the designed 0.5–3.5
trials/session and pair completion rises +20pp (rates) / +10–15pp
(jumper). The b-trap flip side is honest: the freed service pours into
the PLATEAU (svc1 27→37) — with the confirmed task gone, nothing
competes with an untrainable task except the session-end.

### 1.3 The staleness price (arm R: regression at the session-5 boundary)

| margin | arm | wrongly-on sessions /4 | revoke lag | still-on at end | evidence/sess |
|---|---|---|---|---|---|
| deep | base | 0.10 | +0 | 0% | 26.4 |
| deep | term0/term3 | 2.2/2.0 | ~182–185 (≈cens.) | **55/50%** | 15–17 |
| sharp | base | 0.40 | +10 | 0% | 22.4 |
| sharp | term0/term3 | 2.2/2.0 | ~178–185 (≈cens.) | **55/50%** | 13–16 |

The trainer's within-protocol self-correction is largely GONE for a
terminal task: 0.5–3.5 outcomes/session cannot fire the stale gate
within the run (the F88 information logic at one more remove), so
~half the regressed-after-confirmation learners are still marked
terminal five sessions later, at BOTH margins. Maintenance probes
barely help (their per-session count is the binding constraint).

### 1.4 FG zoo (arm Z) and the verdict

Declared FG 6/20 and CONFIRMED FG 2/20 are IDENTICAL base vs terminal
(decay-of-nothing: the lifecycle only engages post-confirmation);
careless 0/20 everywhere; wrongly-terminal lock-in 2/20 static — exactly
the 2 confirmed FGs, i.e. bounded by confirmed-FG itself.

**F90 verdict (D44).** Implemented, validated opt-in, endpoint-decisive
WIN on the pair-completion metric — and the same gain⇄blind-spot
coupling that rejected F89, at larger scale on both sides: the +20pp
completion is bought by deferring post-confirmation regression detection
almost entirely to the D18 re-cert layer (50%+ undetected within-run vs
0% base). Under the D42/D28 precedent the sandbox default STAYS False.
UNLIKE F89, though, the trade is architectural rather than a weakened
safety model: "a CONFIRMED task belongs to re-certification, not to the
trainer" is exactly the D18 'the trainer certifies nothing' semantics
taken seriously — IF re-certification exists at the deployment. That is
a PI/advisor call, not a scratch call: filed as ADVISOR_BRIEF decision
ask #7. If ratified for the pilot (which HAS scheduled re-certs), the
flag ships with the recorded pin (term3 — the maintenance probes buy
detection evidence at +3/session and slightly better confirmed rates).

## 2. F91 — contact-triggered onboarding (D45: ships ON in the sandbox)

Until a task has EVER certified perceptual contact (the F83 e-process;
persisted in `meta['contact_certified']`), serving goes through a
very-easy label-alternating DEMONSTRATION ramp; at first-ever
certification the belief takes one Gate-4 boundary_shift — contact IS
the regime shift the M23 hazard models, evidence-triggered instead of
boundary-scheduled — GUARDED by the poisoning signature (trainability
< 0.5 at certification; the unconditional shift measurably taxes
healthy learners ~+28 trials). Protocol-layer only (the D37 principle);
no engine/belief change.

Arm O (K=1, M23 stack, 20 seeds, 6 sessions; components priced
separately):

| scenario | base | shift-only | demo-only | demo+shift |
|---|---|---|---|---|
| contact60: declared (rate) | cens. (30%) | cens. (30%) | 174 (60%) | **148 (70%)** |
| contact60: certified @ | 98 | 98 | 83 | 83 |
| wellspec: declared (rate) | 46 (100%) | 46 (100%) | 60 (100%) | 60 (100%) |
| careless: declared / FG | cens. / 0 | cens. / 0 | cens. / 0 | cens. / 0 |

- The RAMP does the measurable work in-silico: easy items raise the
  correct-rate the KT e-process sees, certification comes ~15 trials
  earlier, and the pre-contact phase stops burning adaptive placement;
  the guarded SHIFT converts the remainder (60%→70%, −26 median).
  Shift-only does nothing (certification on the adaptive stream is late
  and the guard rarely binds well) — the components are complementary.
- No-harm: wellspec certifies at ~trial 7, pays ~10.6 demo trials
  (+14 median declaration) at an UNCHANGED 100% rate; the guard keeps
  shift-only ≡ base bit-identically (46 = 46).
- A careless responder never certifies, never leaves the ramp, never
  declares — and receives easy demonstrations instead of adaptive
  placement, which is also the right UX for a no-contact participant.
- The ramp's PEDAGOGICAL value (easy exposure teaching the concept) is
  deliberately NOT simulated (it would assume the conclusion); it is
  the field-check question. In-silico, the exogenous-contact model
  already favors shipping.

**D45.** `CONTACT_ONBOARD = True` in the sandbox (protocol-layer flip,
justified by the study; the observed USER-A/C/E onboarding shape is the
scenario it fixes). Resumed pre-M27 testers pay a one-time ~8-trial
re-certification ramp (meta starts empty) — noted, self-resolving.

## 3. F92 — λ(time-on-task): CLOSED by real data (no Phase-3 term)

125,860 EXTSET reads → per-user sittings (30-min gap rule), difficulty
via LOO consensus margins, per-user within-sitting accuracy slopes
(user-level bootstrap): EASY-read accuracy RISES +1.45e-3/min (95% CI
[+1.0e-4, +2.9e-3]; 0.542 → 0.747 across time bins) — within-sitting
LEARNING dominates any lapse growth at these horizons, and self-paced
volunteers quit when tired, truncating exactly the tail a λ(t) term
would fit. A lapse trend is not even sign-identifiable in this data; no
λ(t) term enters the Phase-3 model without pilot fixed-length sessions.
The operational risk stays owned by the F79 differenced fatigue guard
(which is differenced precisely because level trends are learning).

## 4. F93 — the domain interface (committee/port-facing)

`training/domain.py`: `ItemBank` (the candidates contract every
consumer already satisfies), `ArrayBank` (in-memory implementation with
BankAdapter's exact filter semantics — the plug-in path for a new
domain), `Domain` (task names + cuts ℓ*/σ* + assumed dynamics + bank +
belief-stack config; `fresh_filters()` / `trainer()` factories),
`v15_domain()` (the shipped instrument as the reference construction).
Proven end-to-end: a synthetic toy domain trains and graduates through
the UNCHANGED full stack (test_m27 check 11). The stimulus/render layer
is deliberately outside the interface (a delivery-vehicle concern —
sandbox owns the F73 contract). Companion: `docs/ARCHITECTURE.md`
(module map, math↔code correspondence, conventions, reading order).
Housekeeping: EXTSET release moved to `data/extset/` (single path
junction `extset_adapter._root`), README refreshed, `__pycache__`
removed.

## 5. Results and decisions (the PECR record)

| line | proposed | executed | verdict |
|---|---|---|---|
| F90 terminal confirmation | confirmed tasks → retention layer + stale e-gate | arms T/R/Z (anchored harness) | **endpoint win (+20pp), staleness blind spot (50% undetected regressions); default OFF (D44); ratification filed as advisor ask #7** |
| F91 contact onboarding | demo ramp + guarded contact handoff | arm O (4 components × 3 scenarios) | **ships ON in the sandbox (D45)** — pre-contact censored@30% → 148@70%, wellspec no-harm, careless FG 0 |
| F92 λ(time-on-task) | fit on real reads or defer | study_lambda_tot | **closed NO-TERM** — within-sitting learning dominates; not sign-identifiable observationally |
| F93 domain interface | name + prove the pluggability seam | domain.py + toy-domain graduation + ARCHITECTURE.md | **shipped** (additive; suite-pinned bit-identity) |

PECR rejections/findings recorded: the unconditional F91 handoff shift
(taxes healthy learners; replaced by the trainability guard); the
retention bin-proliferation mechanism under gate-bypassing semantics
(fixed, port-relevant); F90's first-run inflated tax was THIS bug, not
the design — re-run after the fix.

Suite at close: 21 files / 299 checks green.

## 6. Queued from this checkpoint

1. Advisor ask #7 (F90 semantics): if "confirmed ⇒ re-cert layer owns
   staleness" is ratified for the pilot, flip `TERMINAL_CONFIRMATION`
   ON at the term3 pin.
2. F91 field check: does the demonstration ramp shorten REAL no-contact
   onboarding (the pedagogical half the simulation cannot honestly
   claim)? Watch `contact_handoff` events + demo-trial counts in tester
   telemetry.
3. λ(t): revisit only with pilot fixed-length-session data (F92).
4. Port items unchanged (PHASE3_AND_PORT.md); the bin-consolidation
   lesson (§1.1) added to the port checklist considerations.
