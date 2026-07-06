# Submission criteria — when the learning algorithm is DONE iterating (D43, 2026-07-05)

Purpose: a checkable definition of "ready to submit to the academic
advisors," so that PECR loops stop when the evidence says stop. The
trigger for writing this now is the project's own convergence record:
three consecutive checkpoints (M24, M25, M26) adjudicated every queued
optimization successor and shipped ZERO default changes — placement line
CLOSED (F84→F87), allocation-level fixes closed (F85/F86), hazard-level
fix default-decisive NO (F89/D42). Under the current evidence base the
shipped stack is at a measured local optimum; further unprompted
iteration has negative expected value (each new mechanism is now more
likely to be rejected than shipped, at real compute and review-surface
cost).

## 1. The submission gate (S1–S8)

Submission-ready ⇔ ALL of S1–S8 hold. Each criterion is checkable
against a named artifact — no judgment calls at evaluation time.

| # | criterion | checkable against | status 2026-07-05 |
|---|---|---|---|
| S1 | **Reproducible + green.** Full script-style suite passes from the scratch root; every study re-runs via `python3 -m studies.study_* [--smoke]` and caches to `figures/data_*.npz`. | suite 21 files / 299 checks (M27); §7 of `ADVISOR_BRIEF.md` | **MET** |
| S2 | **Calibration honesty.** Filter calibration nominal under the exact kernel (closed-loop SBC at every checkpoint); trainability is an honest posterior (σ∞-mixture), not a prior prophecy; known interval defects fixed and recorded. | F53/F57 (`data_train_sbc_exact.npz`), D22, brief §2/§4 | **MET** |
| S3 | **Safety OCs measured, priced at CONFIRMED semantics.** Careless/adversarial FG ~0; the sharp-margin FG edge quantified as an information bound (not gate inefficiency) with the shipped architectural filter measured (D33 kills 5/6; D18 backstop 75/75); TRUE-REGRESSION detection measured (M26 arm G: fixed hazard +0 flicker). | F88 (X1–X4), F32, F89 arm G; D41 confirmed-FG semantics | **MET** (ratification of the endpoint definition is advisor ask #5 — part of the submission agenda, not a precondition) |
| S4 | **Real-data grounding.** Dynamics priors fit on real reads (1,158 + 2,040 EXTSET fits, hierarchical shrinkage); real-session replay validated; 7 real tester profiles analyzed end-to-end and their mechanisms fed back as shipped gates. | F62/F65/F69; M21–M23 docs; per-tester MP4s | **MET** |
| S5 | **Optimization frontier closed (the anti-needless-iteration criterion).** Every queued selection/allocation/hazard/lifecycle successor has been adjudicated by an end-to-end priced study with an honest verdict; M24–M27 produced four consecutive default-decisive NOs on the training-side challengers (the one M27 flip, D45, is a protocol-layer onboarding gate, not a selector). New optimization lines require NEW DATA (see §2), not further search. | M24–M27 PECR records (D40–D42, D44) | **MET** |
| S6 | **Delivery vehicle operational.** The sandbox runs the full validated stack per-tester (profiles, telemetry, resumable state, analytics, trajectory videos); real tester data accumulates in the format the pilot will use. | M20–M23 rows; `sandbox/`; `viz/make_user_videos.py` | **MET** |
| S7 | **Advisor package current at the frozen checkpoint.** `ADVISOR_BRIEF.md` reflects the latest checkpoint (incl. M27), enumerates the decision asks, and its reproducibility section matches the tree. | brief §5b/§5c/§6/§7 (M27 addendum; ask #7) | **MET** (updated 2026-07-05, M27) |
| S8 | **Pilot-facing artifacts exist for the advisors to act on.** SAP + pre-registration freeze docs present with the named endpoints (consolidation exponent F70, confirmed-FG D41, decision-level outcomes F59) and the power target stated. | `PILOT_SAP_M14.md`, `PREREG_FREEZE_M14.md`, brief ask #6 | **MET** |

**Current verdict: the gate is MET — the algorithm is submission-ready
as of M27.** The seven ADVISOR_BRIEF §6 asks are the submission's AGENDA
(what we want from the advisors), not blockers; nothing on the remaining
queue is a precondition for their review.

## 2. The stopping rule (what may reopen iteration)

The algorithm is now SUBMISSION-FROZEN. A new PECR loop on the learning
algorithm opens ONLY on one of:

1. **Advisor-requested analysis or change** (the submission's purpose).
2. **New human data exposing a mechanism** — the M21/M22/M23 pattern
   (identity confound, starvation, regime shifts were all found in
   tester data, never in simulation). Field sessions remain human-gated
   and their analysis is in-scope.
3. **A regression**: suite failure, a guardrail OC drifting from its
   pinned value, or a reproducibility break.
4. **A gate unblocking**: port-gated items (D21 re-pin campaign, K=7 +
   corr_t coverage run) become runnable, or unscrubbed data arrives
   (real-domain trace rendering).

Explicitly NOT sufficient to reopen: an idea for a better
selector/allocator/hazard absent new data (S5 closed that frontier —
three consecutive honest NOs); polish of validated machinery; re-runs of
studies whose caches are current.

## 3. Parked queue (post-submission, recorded so it is not re-derived)

M27 (the user-directed pre-submission loop) completed or adjudicated the
former queue: terminal-confirmation semantics EXECUTED (F90/D44 —
opt-in, default off, advisor ask #7 decides the flip); the
contact-triggered onboarding block EXECUTED and SHIPPED (F91/D45);
λ(time-on-task) CLOSED by real data (F92). Remaining parked:

- F90 default flip at the term3 pin — gated on advisor ask #7.
- F91 field check (the ramp's pedagogical half; watch `contact_handoff`
  + demo-trial telemetry in tester sessions).
- λ(t) — revisit only with pilot fixed-length-session data.
- Production-port checklist (`PHASE3_AND_PORT.md`) — activates on C3;
  now includes the M27 bin-consolidation lesson (gate-bypassing
  semantics × transition-registered retention bins).

## 4. What submission does NOT claim (carried from the brief §4)

The trainer certifies nothing (D18); the definitive methodology test is
the feedback-driven pilot (no existing dataset contains closed-loop
training, M11.2b); pilot scope is K=1–3 with the consolidation exponent
as a named endpoint.
