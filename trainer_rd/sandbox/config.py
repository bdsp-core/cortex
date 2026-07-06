"""M20 — sandbox configuration (the manual-tester shadow protocol).

The sandbox is the LOCAL PROTOTYPE of the delivery vehicle: the user is the
learner, sessions run the full validated M15–M19 stack, and every trial is
logged in the production-shaped telemetry schema for analytics toward the
public release. Constants here are the D28/D25/D29/D30/D32/D33 decisions
made concrete; edit TASKS/TRIALS_PER_SESSION to re-scope (a fresh scope
requires `--reset`, which archives the previous state).

C1 note (bar coherence): the sandbox is v15-coherent end-to-end (D31) —
trainer targets, analytics bars, and any future re-cert all read the v15
instrument.
"""
from __future__ import annotations

import os

import numpy as np

from engine.instrument_v15 import instrument
from training.bank_adapter import (ANCHORED_ALPHA_SIGMA, ANCHORED_ALPHA_T,
                                   ELL_STAR_V15, SIGMA_STAR_V15, TASK_CODES)
from training.learner_sim import LearnerParams

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(ROOT, "state")
LOG_DIR = os.path.join(ROOT, "logs")
FIG_DIR = os.path.join(LOG_DIR, "figures")
TRIALS_JSONL = os.path.join(LOG_DIR, "trials.jsonl")
SESSIONS_JSONL = os.path.join(LOG_DIR, "sessions.jsonl")

# ── learner identity (M21/F72: belief state is PER PERSON) ──
# The M20 sandbox keyed state to the machine; two testers sharing one state
# produced cross-contaminated beliefs (trainability pollution, bias-mode
# escalation against the wrong person's criterion). `--user NAME` namespaces
# state+logs per tester; None keeps the legacy single-profile paths.
USER = None


def set_user(name):
    """Switch every state/log path to the named tester's namespace.
    Must be called before any sandbox.state_io I/O (session/report CLIs
    call it first thing). Production analog: state keyed by learner id."""
    global USER, STATE_DIR, LOG_DIR, FIG_DIR, TRIALS_JSONL, SESSIONS_JSONL
    USER = name
    suffix = "" if name is None else f"-{name}"
    STATE_DIR = os.path.join(ROOT, "state" + suffix)
    LOG_DIR = os.path.join(ROOT, "logs" + suffix)
    FIG_DIR = os.path.join(LOG_DIR, "figures")
    TRIALS_JSONL = os.path.join(LOG_DIR, "trials.jsonl")
    SESSIONS_JSONL = os.path.join(LOG_DIR, "sessions.jsonl")
    from sandbox import state_io
    state_io.STATE_NPZ = os.path.join(STATE_DIR, "state.npz")
    state_io.STATE_JSON = os.path.join(STATE_DIR, "state.json")

# ── protocol scope (C2: pilot-scoped K ≤ 3) ──
TASKS = (1, 2)                       # domain2, domain3 (bank task indices)
TRIALS_PER_SESSION = 40              # D28
PROBE_EVERY = 5                      # D23
MIN_GAP_S = 4 * 3600.0               # gaps shorter than 4 h: same "sitting"
MAX_ANCHOR_TRIALS = 16               # GapAnchor block cap per session

# ── M22 participant-pilot gates (F77–F79; study_participant_gates) ──
# Gate 1: allocation discounts deficiency by P(trainable) — stops the
# worst-first loop from pouring trials into a collapsing task while a
# near-mastery task starves (USER-B/C/D pattern).
TRAINABILITY_FLOOR = 0.25
# Gate 1b: finishing-eligibility sd tolerance (F77-ii) — the mixture's
# cross-strata variance floor can sit a hair above sd_floor until the
# ceiling strata concentrate, deadlocking finish-first exactly for the
# task that needs finishing trials. Declaration keeps the strict floor.
FINISH_SD_TOL = 1.25
# Gate 2: a fired consistency flag suspends that task for the session
# (USER-D: 10 post-flag trials drove trainability 0.22→0.04).
CONSISTENCY_PAUSE = True
# Gate 3: end the session early when the trailing realized-vs-predicted
# accuracy deficit rises this far above the session's own first-window
# baseline (DIFFERENCED — fatigue is a decline, not a level; the raw form
# false-fires on onboarding belief-optimism). Active from trial 2·W.
FATIGUE_W = 12
# pinned by study_participant_gates G3 (full run 2026-07-02): stationary
# false-break 5/100 sessions at 0.35 (9/100 at 0.30); both REAL collapses
# (USER-B s2 +0.41, USER-D s1 +0.37) exceed it — D's margin is thin (0.02),
# recorded honestly; a missed break is recoverable (D33 + anchors), a
# spurious one costs ≤16 trials.
FATIGUE_DELTA = 0.35

# ── M23 regime-shift gates (F80–F83; studies/study_m23_regime.py) ──
# Gate 4 (F80/F81): session-boundary regime-shift update. The 2026-07-03
# sessions showed discrete jumps at sitting boundaries (USER-E d′ 0.8→3.0
# across a 64 s restart; USER-C −0.3→2.0 overnight) that the smooth kernel
# tracked with a full-session lag while trainability sat poisoned at
# 0.34/0.36. At every session open (AFTER the D33 confirm phase — a
# confirmation must reflect evidence, not prior widening) each task's
# mixture takes boundary_shift(ε_b, Λ, θ-scale, w_share):
#   state: hazard-ε_b N(0,Λ²) shock on ℓ (θ scaled — jumps are σ-dominant),
#   weights: fixed-share (1−ρ)ω + ρ·prior — exact Bayes for a ceiling that
#   re-draws from the prior with hazard ρ at boundaries.
# Pinned by study_m23_regime 2026-07-04, arm G (the harness-realism
# re-pin): evidence-admissible on the real logs (+0.23 nats pooled);
# real-log breakout lag E 10→8.3 / C 21→16.7; E-like jump lag(π)
# 93.5→15.2 and declaration 116→51 post-jump trials; FG static_below at
# the sharp ℓ*−0.15 margin 11/20→6/20 (careless 0/20 throughout). The
# stage-1 pooled-argmax pin (0.2, 1.0, 0.33)/ρ=0.2 was REJECTED by arm G:
# at the real ~20 trials/task/session rate it re-shocks faster than
# evidence re-contracts and a clean learner never declares (0/8 seeds).
# θ-scale 0.33: full-θ shocks cost up to −0.83 nats on the logs — the
# observed jumps are σ-dominant (E: Δlnσ 1.3 vs Δt 0.1).
BOUNDARY_JUMP = (0.10, 0.75, 0.33)   # (ε_b, Λ, θ-scale)
BOUNDARY_WSHARE = 0.10               # fixed-share ρ
# Gate-4 corollary (F5/F77-ii logic): the boundary hazard raises the
# ACHIEVABLE end-of-session mixture sd_ℓ cycle-minimum to 0.226–0.263 —
# at/above the M22 declaration floor 0.23, which starves declarations
# (0% in 8 sessions). The floor must track the physics: re-measured and
# re-set with ~1.3× margin. π − 2·mcse ≥ 0.95 and the D23 e-gate carry
# the FG protection (a wider posterior makes π HARDER to satisfy);
# well-specified declaration returns to 100% @ median 94 (arm G3).
SD_FLOOR = 0.33
# Gate 3b (F82): RT-floor careless channel, union with the F79 accuracy
# guard. k consecutive RTs under the floor ⇒ non-perceptual responding
# (USER-D collapse: 206 ms trailing median vs ≥1059 ms across every
# engaged window of all 14 sessions). 0 false alarms at 500 ms/k=3.
RT_FLOOR_MS = 500.0
RT_FLOOR_K = 3
# F83: shadow contact e-process (sandbox/contact.py) — log-only telemetry.
CONTACT_ALPHA = 0.05

# ── M24 selection (F84/F85; studies/study_m24_selection.py) ──
# F85: allocation by EXPECTED PROGRESS PER TRIAL toward each task's bar
# (`expected_progress_rate`) — the principled F77 successor, VALIDATED
# OPT-IN but DEFERRED here (False): full-run arm Q shows decisive wins on
# the USER-B trap (near-bar declaration 134 → 86 trials, plateau-sunk
# 156 → 72) and on lateness (99 → 72), BUT two F85-ii residuals block the
# default flip per D28 (FG ≥ lateness > speed): (i) post-boundary
# RE-POLISH loop — after a Gate-4 shift, a nearly-mastered task's
# re-widened below-bar mass reads as a progress opportunity and EP
# out-ranks the low-trainability recovering task (worst-first serves the
# wide task instead), delaying trainable-pair completion (rates/jumper
# task1 censored at 241 vs gate1 188/189); (ii) sharp-margin FG direction
# unfavorable though not significant (60 seeds: 27/60 vs 22/60, z≈0.9).
# Successor queued: couple EP to the confirmation lifecycle (ever-
# provisional tasks rank by finishing need, not EP) + exploration floor.
PROGRESS_ALLOC = False
PROGRESS_S_SD = 0.057
# ── M25 selection successors (F86–F88; study_m25_selection,
# study_sharp_margin; docs/M25_SELECTION_V2.md) ──
# F86 EP-v2 (confirmation-lifecycle coupling + exploration floor): knobs
# INERT while PROGRESS_ALLOC is False. D41: the default STAYS off, now on
# endpoint-decisive grounds — on the BOTH-declared endpoint Gate-1
# dominates every EP variant (rates 300@55% / jumper 240@70% vs censored
# @20%/5%), because the Gate-4 hazard re-widens a declared task every
# boundary and its ~28 trials/session re-confirmation starves the second
# task below the ~20 trials/task/session evidence needs to re-concentrate
# (the arm-G rate) — an allocator cannot fix a hazard-level cost. Queued
# successor: evidence-adaptive boundary hazard for declared tasks.
PROGRESS_LIFECYCLE = True
REFINISH_THRESHOLD = 0.50
EXPLORE_EVERY = 8
# F88 sharp-margin declaration conjuncts: REJECTED by study_sharp_margin
# (X2/X3 — refractory buys 1/20 FG at −15pp honest completion;
# trainability conjunct is a no-op at trainability-0.92 FGs; denser
# probes lose to the Bernoulli bound). 0/0.0 = bit-identical no-ops; the
# effective sharp-margin filter is the shipped D33 confirmation lifecycle
# (X4: declared 6/20 → CONFIRMED 1/20) + the D18 re-cert backstop.
BOUNDARY_REFRACTORY = 0
MIN_TRAINABILITY = 0.0
# ── M26 (F89; studies/study_m26_hazard.py; docs/M26_HAZARD.md) ──
# Evidence-adaptive DECLARED-task boundary hazard — the queued F86
# successor one level down: Gate 4's (ε_b, w_share) for a task that has
# declared and re-confirmed across n boundaries decays by the Beta
# posterior-mean factor c0/(c0+n) (training.mixture_filter.
# declared_hazard_scale; linear-in-ρ updates make the plug-in exact).
# Never-declared tasks always pay the full pinned hazard (F80 breakout
# machinery untouched); a D33 REVOCATION or a π-collapse at a session
# open resets the count (full hazard returns). D42: the default STAYS
# False — arm A shows the pair-completion gain and the arm-G sharp
# (ℓ*−0.15) regression blind spot are COUPLED across the whole pin range
# (fixed +0 flicker; c2f25 +14 trials @ +5pp completion; c5f25 +71 @
# +10pp; geometric +100 @ +20pp), and D28 ranks that FG-adjacent
# persistence above the modest lateness win. Constants below record the
# best-priced studied pin for opt-in field ablations only. Careless/
# static FG and deep-regression detection are unchanged at every pin.
HAZARD_ADAPT = False
HAZARD_STRENGTH = 2.0
HAZARD_FLOOR = 0.25
# ── M27 (F90; studies/study_m27_terminal.py; docs/M27_TERMINAL.md) ──
# Terminal-confirmation semantics: a D33-CONFIRMED task leaves the
# training rotation for good — the Gate-4 hazard keeps its full pinned
# rate on the belief (nothing weakened, the D42 lesson), but the
# scheduler stops converting the re-widening into a re-polish tax; the
# retention layer owns the task (due-driven review + MAINT_PROBES at-bar
# probes/session), and an anytime-valid stale-mastery e-gate (Ville,
# α = STALE_ALPHA: P(wrongly revoking an at/above-bar learner) ≤ α per
# task) returns a regressed task to the rotation. Constants pinned by
# study_m27_terminal; see D44 for the default decision.
TERMINAL_CONFIRMATION = False
MAINT_PROBES = 3
STALE_ALPHA = 0.05
# ── M27 (F91; studies/study_m27_onboard.py; docs/M27_TERMINAL.md §3) ──
# Contact-triggered onboarding: until a task has EVER certified
# perceptual contact (the F83 e-process; certification persists in
# meta['contact_certified']), serving goes through a very-easy
# label-alternating DEMONSTRATION ramp instead of adaptive placement —
# the M24 audit's pre-contact sessions (w_true 0.15–0.17) burn adaptive
# placement on a belief that is meaningless before contact. At the
# certification moment the task's belief takes one Gate-4
# boundary_shift (DEMO_HANDOFF_SHIFT): contact IS the regime shift the
# M23 hazard models, with an anytime-valid evidence trigger instead of
# a sitting boundary — the guessing-poisoned trainability re-opens
# without waiting a session. A careless responder never certifies
# (arm F: 0 false contacts in 400 null sessions) and so never leaves
# the demonstration ramp. The demonstration's PEDAGOGICAL value (easy
# exposure teaching the concept) is a field-check question by design —
# the in-silico study prices only the measurable halves (no-harm on
# well-spec, post-contact recovery, FG safety).
CONTACT_ONBOARD = True
DEMO_HANDOFF_SHIFT = True
# the handoff shift exists to cure guessing-POISONED trainability (the
# F80 mechanism); an early certification with a healthy belief must not
# pay a widening tax (measured: wellspec certifies @~20 in-session-1 and
# the unconditional shift costs ~+28 declaration trials) — the shift
# fires only when trainability at certification shows the poisoning
DEMO_HANDOFF_MIN_TRAIN = 0.5
# D45 (2026-07-05): CONTACT_ONBOARD ships ON in the sandbox —
# study_m27_onboard: the pre-contact scenario (the M24 audit's remaining
# value sink; the observed USER-A/C/E onboarding shape) goes censored@30%
# → declared 148@70% under demo+guarded-handoff; wellspec pays +14 median
# trials at an unchanged 100% rate (certification ~trial 7 bounds the
# ramp); careless never certifies, never declares (FG 0), and gets easy
# items instead of adaptive placement — protocol-layer only (D37), no
# engine/belief change. NOTE for resumed pre-M27 testers: contact_
# certified meta starts empty, so their next session opens with ~8 demo
# trials until their (established) contact re-certifies — one-time,
# self-resolving.
# F84 (recorded MIXED/negative result): item-level posterior-progress
# placement (`ModeThresholds.progress_placement`) stays DEFAULT-OFF. At
# full scale the final (v6, mirror-paired) variant is near-parity on
# well-specified learners (49@90% vs 46@100%), FASTER on the E-like jump
# (134 vs 148), and slower on slow-wide learners (121@70% vs 102@85%) —
# no dominance; the intermediate variants document three named
# pathologies (tail vote, calibration externality, model-injected hazard
# width). Successor: tier-3 rollout at the item level.
# See docs/M24_SELECTION.md.

# ── belief stack (D22/D26/D29/D30; M21 F75: skill placement uses the
# conservative σ quantile z=1 while the belief is wide — fresh human
# testers otherwise sit at ~50% served accuracy for a whole session
# (target 84%) while the filter descends from the optimistic prior) ──
N_PARTICLES = 300                    # per mixture stratum
MIX_J = 7
MIX_TAU = 0.30
P_STATIC_STRATUM = 0.15
PRIOR_SD = 0.45                      # fresh-learner initial belief SD

V15 = instrument("v15")


def assumed_params(task):
    """Filter-side dynamics (D29 anchored rates, v15 expert ceiling)."""
    return LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                         alpha_sigma=ANCHORED_ALPHA_SIGMA,
                         sigma_inf=float(V15.sigma_inf[task]),
                         q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")


ELL_STAR = {t: float(ELL_STAR_V15[t]) for t in range(7)}
SIGMA_STAR = {t: float(SIGMA_STAR_V15[t]) for t in range(7)}
EXPERT_ELL = {t: float(V15.expert_ell[t]) for t in range(7)}

# ── stimulus rendering ──
# M21 calibration (F73): the render physics BOUND the achievable σ. An ideal
# observer of this display has
#     σ_eff(ideal) = RENDER_NOISE·√(1/n_win + 1/n_out) / RENDER_GAIN
# (n_win/n_out = samples in/outside the window). At the M20 gain 0.55 that
# bound was 0.606 — leaving only 18–29% headroom to the v15 mastery bars
# (σ* = 0.736/0.857) and putting the assumed expert ceilings (σ_∞ ≈ 0.48–0.52)
# physically OUT OF REACH, so no human could plausibly master and the
# σ∞-mixture was structurally forced toward "ceiling below cut". Gain 0.85
# ⇒ σ_eff(ideal) = 0.392: bars need ~46–53% ideal efficiency (attainable),
# ceilings ~76–81% (ambitious but meaningful). Changing gain rescales the
# human's effective σ units — resume across a gain change is blocked
# (state_io meta guard); use --reset.
RENDER_N = 48                        # samples per trace
RENDER_GAIN = 0.85                   # template amplitude per unit s_real
RENDER_NOISE = 1.0                   # per-sample noise SD
TEMPLATE_LO, TEMPLATE_HI = 18, 30    # target window (samples)

TASK_NAMES = {t: TASK_CODES[t] for t in range(7)}
