# Sandbox — the manual-tester shadow protocol (M20)

You are the learner. Each session runs the **full validated training stack**
(M15–M19: exact-kernel mixtures, certification probes + e-gates, finish-first
scheduling, the derived 0-bias band, GapAnchor re-anchoring on real
wall-clock gaps, and the post-gap confirmation rule D33), and every trial is
logged in the production-shaped telemetry schema — this directory is the
local prototype of the delivery vehicle, and its logs are the analytics
feedstock for the public-release build.

## Daily use

```bash
python3 -m sandbox.session --user YOURNAME   # run one ~40-trial session (≈5–10 min)
python3 -m sandbox.session --user YOURNAME --trials 20   # shorter sitting
python3 -m sandbox.session --user YOURNAME --status      # where you stand
python3 -m sandbox.report  --user YOURNAME   # full analytics + trajectory figure
```

**Always pass `--user` with YOUR name (M21/F72).** The belief state is a
model of ONE person; the 2026-07-02 two-tester sessions showed that sharing
a state cross-contaminates it (the second tester inherited the first's
pessimistic trainability and was served bias-correction items against the
wrong person's criterion). Each profile gets its own `state-NAME/` +
`logs-NAME/`. A shadow **consistency monitor** (windowed GLR of recent
behavior against the belief's own predictions) is logged per trial and
raises a `consistency_flag` event when recent responses look like a
different person than the state describes — it never mutates beliefs.

Each trial shows a noisy trace with a marked window; answer whether the
window contains an **upward deflection** (1/y = present, 0/n = absent,
q = quit — quitting is always safe, state is saved every trial). You get
veridical feedback after each answer; that feedback IS the training signal.

Come back whenever you like — the protocol is open-ended (D28). If more
than 4 hours pass between sessions, the session opens with a short
re-anchoring block (up to 16 trials) that measures what you retained
(D27/F70: measure-or-leave-alone); your personal stability estimate S and
each gap's resolved retention r̂ appear in the report — the consolidation
trajectory is a primary analytics endpoint for the pilot (F70-ii).

M22 protocol gates (from the four-participant pilot, F77–F79): the
scheduler discounts a task's pull on your session by its measured
trainability (no more whole sessions sunk into a task the engine already
believes is not trainable while a nearly-finished task starves); a fired
consistency flag pauses that task for the rest of the session (re-read the
instructions — it resumes next time with a fresh window); and if your
accuracy drops well below your own earlier-session level, the session ends
early with progress saved (fatigue protection — the belief state should
not absorb your tired trials).

M23 regime-shift stack (from the seven-profile pilot, F80–F83): every
session opens with a small regime-shift hazard on the belief (state jump +
fixed-share ceiling re-mix — real testers jumped d′ 0.8→3.0 across a 64 s
restart, and the M22 belief needed half a session to catch up; now one
session of evidence suffices, and a task the engine had written off gets a
bounded benefit of the doubt each new sitting); sustained response times
under 500 ms end the session (non-perceptual responding — the trace takes
≳1 s to read; calibrated at 5× margin against every engaged tester window);
and a shadow per-session "contact" statistic (an anytime-valid e-process
against chance-band responding) records when your answers first carry real
signal — log-only, feeds the onboarding analytics.

Mastery lifecycle per task: `training → provisional → CONFIRMED`, where
confirmation requires the gate to hold **after** a gap and its re-anchor
(D33 — kills post-gap-transient false graduations; expect occasional
`revoked` events near the bar: that is the safety rule working). When all
tasks are CONFIRMED the report recommends the D18 hand-off to
re-certification.

## What is being measured (the perceptual task)

The scrubbed bank has latent signals, not renderable domain traces, so the
sandbox presents a synthetic signal-detection task whose difficulty maps
monotonically to each item's latent signal (`sandbox/stimulus.py`). Your
internal noise and criterion become genuine (σ, t) in the stack's latent
units — the estimation, placement, gating, and analytics are exactly the
production machinery; only the rendering is a stand-in.

M21 stimulus notes (from the first two tester sessions):
* the realized signal is now drawn **label-consistent** (F74) — the trace
  never shows evidence opposing the feedback you receive (10% of M20
  tester trials did, actively training the wrong mapping);
* `RENDER_GAIN` was recalibrated 0.55 → 0.85 (F73): the display physics
  bound the best achievable σ at `σ_eff(ideal) = NOISE·√(1/12+1/36)/GAIN`;
  at the old gain even a perfect observer sat only 18–29% above the v15
  bars and could never reach the assumed expert ceilings. σ estimates are
  NOT comparable across a gain change — state resume is blocked
  (`--reset` archives first).

## Files

| path | contents |
|---|---|
| `sandbox/logs/trials.jsonl` | per-trial telemetry: stimulus, response, RT, mode (skill/bias/retention/anchor/+cert probes), full belief snapshot (ℓ̂±SD, t̂, π, trainability, e-gate), mastery status |
| `sandbox/logs/sessions.jsonl` | session summaries: gaps, anchor results (r̂, S), lifecycle events, end-of-session beliefs |
| `sandbox/logs/figures/trajectory.png` | skill & bias trajectories vs the v15 bars over sessions |
| `sandbox/state/` | resumable belief state (npz + json; no pickle) |
| `sandbox/state/archive-*/` | archived runs (`--reset` never deletes) |

## Configuration & maintenance

* Scope (tasks, trials/session, probe cadence, gap threshold, stimulus
  gain): `sandbox/config.py`. Rescoping requires `--reset` (archives the
  current state + logs first).
* Robot self-test (also a demo): `python3 -m sandbox.session --auto
  sigma=1.3,t=0.6,seed=3 --fake-gap 86400`. Two full robot campaigns are
  archived under `sandbox/state/archive-2026*` (the second, post-fix one is
  the reference: 800 trials, serve-once clean, both tasks CONFIRMED with
  the D33 lifecycle visibly catching transients).
* Verification: `python3 -m tests.test_sandbox` (11 checks, isolated state —
  never touches your data).

## Toward the delivery vehicle / public release

The telemetry schema here is the proposed production schema. What the
sandbox adds to the readiness case (M19 verdict conditions): C2's post-gap
confirmation rule is implemented and exercised (D33); C1 is satisfied
inside the sandbox (v15-coherent end to end); C3's delivery-vehicle logic
(session flow, persistence, resumability, analytics) is prototyped here —
the remaining C3 work is packaging (UI/rendering of real domain traces,
which requires unscrubbed production data) and the production port.
