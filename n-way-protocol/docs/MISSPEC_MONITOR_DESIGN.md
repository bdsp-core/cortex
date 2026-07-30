# Distractor-misspecification monitor — design

Status: Python reference implementation plus the owner-authorized
TypeScript engine port (2026-07-30,
`cortex_web/apps/web/engine/misspec_monitor.ts` — parity-tested against
this reference). Serving it in production remains gated by the fail-closed
server rollout spec.

## Why this exists

The leakage-controlled fit pins the distractor lapse at the zero boundary:
real distractors are fully signal-following, so the deployed likelihood has
no natural uniform-noise margin. N3b
(research-policy/track-n/n3/N3_REPORT.md) shows that when distractor
identity stops following signal, the misspecified categorical engine
collapses catastrophically (skill coverage 0.26) while the binary reduction
is untouched by construction. The λ_d floor (artifact_floor.py) softens the
collapse; this monitor detects it and fails closed to binary updates.

## Statistic

For wrong-pick observation *i* (asked class `a`, pick `c ≠ a`, segment
response axes `z = s_mean`), the deployed artifact mixture predicts

    P_model(c | a, z) = Σ_d w_d · [ λ_d/5 + (1 − λ_d) · softmax_c(β_d · z) ]

over the five distractors. The monitor scores each wrong pick against the
uniform alternative — precisely the N3b collapse mode:

    s_i = log P_model(c_i | a_i, z_i) − log(1/5)

Under a signal-following world E[s_i] > 0; under uniform allocation
E[s_i] = −KL(uniform ‖ model) < 0. With a floored artifact the increment is
bounded below by log(floor), which keeps CUSUM excursions bounded and makes
threshold calibration stable.

Correct picks contribute nothing: the monitor watches only the channel the
categorical update adds on top of binary, so the binary fallback is always
a valid refuge.

## Decision rule (per-session guard, engine-side)

Page's CUSUM against the model:

    C_0 = 0;  C_i = max(0, C_{i−1} − s_i);  trip when C_i > h

- **Fixed-input discipline** (drift guards must be policy-layer fixed-input;
  see the MKL ulp note in the repo history): the statistic depends only on
  (asked, pick, frozen segment axes, frozen artifact). No session posterior,
  no RNG, double precision throughout.
- **Trip semantics — fail closed and replay**: on trip the session's
  categorical evidence so far is suspect, not just future evidence. The
  engine (a) marks the session `distractor_monitor_tripped`, (b) rebuilds
  the posterior from the prior by replaying the full history with every
  categorical observation reduced to its binary margin (asked-class
  hit/miss), and (c) continues binary-only for the rest of the session.
  Replay cost is one history pass — the same order of work as an MH
  rejuvenation sweep. The trip is one-way within a session.
- The raw picks stay in history and in the database (the additive migration
  already retains them), so tripped sessions remain fully auditable and
  re-scorable.

## Threshold calibration and operating characteristics

`misspec_monitor.py` calibrates `h` by Monte Carlo under the deployed
artifact as the true world (fixed seed grid), targeting a per-session false
trip rate; the OC harness then reports, per candidate threshold:

- false-trip rate under the model-true world (softmax, fitted β ensemble),
- detection latency (wrong-picks to trip) under uniform allocation
  (λ_true = 1) — the N3b world,
- detection latency under partial drift (λ_true = 0.5, and β_true = 0.5·β)
  — misspecification milder than total collapse.

Session-scale inputs use the qualification-profile burden range (own-cap 15,
six domains) with wrong-pick counts swept over the observed range.

## Population tier (ops-side, revalidation cadence)

The same increments, aggregated across sessions per population into rolling
means of `s̄`, give the drift monitor that the qualification contract's
"per population and over time" hard gate requires. In-control bands come
from the calibration simulation's percentiles. This tier is reporting-only
(no engine behavior change); alarms route to the same revalidation runbook
the deploy gates use. Population aggregation is out of scope for the
engine-side implementation and specified here so the field semantics are
frozen once.

## Interaction with the λ_d floor

Floor and monitor are complements, not substitutes: the floor bounds the
damage rate per update before detection; the monitor bounds the duration.
N3b replays (reports/floor_stress/) quantify the first; the OC table
(reports/misspec_monitor_oc.json) quantifies the second. Production promotion
requires both plus the preregistered campaign gates — nothing here relaxes
QUALIFICATION.md.
