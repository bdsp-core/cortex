# Phase 2 — floored-chain adoption, monitor port, powered campaign (2026-07-30)

Owner authorizations executed: 0.15 λ_d floor approved; n-way-protocol
stack committed and pushed to main; production (cortex_web) changes for the
floored chain and the monitor authorized. All work on `main`, one commit
per unit, every suite green at each step.

## What shipped

1. **Floored chain adopted end-to-end** (commit 0709363). The standard
   pointer now names the floor-0.15 ensemble
   (`nway_ensemble9_floor015_fisher_1200p30_v2`), with a new engine-profile
   artifact, schema const, and contract tests that reject sub-floor draws.
   The browser engine constants (`nway_profile.ts`), the server stamp
   (`services/api/nway_profile.py`), and the isolated-stack server contract
   were updated in lockstep, and the cross-language parity expectations
   were regenerated from the Python oracle — the generator was validated
   bit-for-bit against the old constants before switching draws.
2. **Distractor-misspecification monitor ported into the engine** (commit
   09632cc). Fit-frame CUSUM (threshold 5.213009866319716 from the OC
   calibration), one-way trip, fail-closed semantics: on trip the cloud is
   importance-reweighted onto the binary-reduced history and all later
   categorical responses update as their asked-class margin, raw picks
   retained. Monitor state rides `SessionCore` lazily — speculation
   clones, snapshots, and pre-monitor resumes work unchanged. Seven new
   tests including bit-for-bit Python parity and an in-engine trip.

**Byte-identical-off**: production sessions select binary via the
server-owned rollout (default off, fail-closed); the binary update path's
arithmetic is untouched (the monitor seam passes binary observations
through), and the full engine suite (124 passing) plus the fail-closed and
public-surface tests confirm behavior.

## Powered campaign — three 2,000-seed formal-SBC cells at 1200p/30MH

Floored (0.15) crossfit ensemble on the engine side throughout; binary arm
as CRN control (`reports/campaign_floor015/`).

| cell | truth | arm | skill RMSE | skill cov [Wilson 95] | bias cov |
|---|---|---|---|---|---|
| gate | fitted world | binary | 0.5691 | 0.9464 [0.9422, 0.9503] | 0.9460 |
| gate | fitted world | **floored F1** | **0.3457** | **0.9516 [0.9476, 0.9553]** | 0.9427 |
| tier-β | expert β = 1.746 | binary | 0.5713 | 0.9465 | 0.9464 |
| tier-β | expert β = 1.746 | floored F1 | 0.4433 | **0.8286 [0.8217, 0.8352]** | 0.9229 |
| uniform | distractor noise | binary | 0.5569 | 0.9517 | 0.9436 |
| uniform | distractor noise | floored F1 | 1.2218 | **0.4121** | 0.8439 |

### Finding 1 — the absolute skill-coverage gate CLOSES

Floored-categorical skill coverage 0.9516 with the Wilson interval
containing nominal, where the pre-floor powered runs sat at 0.9425–0.9454
with intervals excluding it. The floor's dispersion fixes precisely the
overconfident cross-domain updates behind the long-standing undercoverage
flag — while keeping a 39% skill-RMSE advantage and 40% tighter intervals
over binary at the same burden. Caveats recorded honestly: bias-side
absolute coverage is 0.9427 vs binary's own 0.9460 (a small undercoverage
shared by both arms — profile machinery, not an F1 defect; paired
noninferiority passes both blocks); skill SBC ranks are non-uniform
(KS 0.124) in the center-heavy/conservative direction.

### Finding 2 — expert-tier β misspecification is a real validity risk

With the world at the C1 expert point (β = 1.746) and the engine on the
pooled floored ensemble, skill coverage drops to 0.829. The Phase-1
intuition that an under-sharp pooled β is merely conservative is refuted:
the mis-shaped distractor conditional biases cross-domain updates. Every
certification sitting is a single reader, so an expert candidate
experiences this cell, not the population mixture. The monitor cannot
catch this direction — a sharper-than-model world makes picks *more*
model-favored, so the CUSUM never accumulates. Consequence: **do not
serve the categorical engine to expert-tier candidates until either a
tier-aware/skill-linked β artifact or a two-sided monitor lands.**
Novice-dominated populations match the fitted world (the gate cell).

### Finding 3 — collapse behavior is stable at power

The uniform world reproduces the 192-seed A3 numbers almost exactly
(coverage 0.412 vs 0.401; RMSE 1.222 vs 1.208): the floor softens but
never rescues collapse, and the binary arm is untouched. The monitor
(100% detection, median 14 wrong-picks, 1% false-trip) plus binary-replay
remains the load-bearing defense, now in the engine.

## Governance notes

- These are powered development campaigns executing the preregistered
  design (CRN seed grid, formal SBC, frozen floored artifact, declared
  cells). The **formally locked run still requires owner-set
  noninferiority margins**; the -0.03 house margin used here is inherited
  from the development gates and labeled as such.
- Serving categorical sessions in production remains gated by the
  fail-closed server rollout (`nway_response_rollout`, default off) — no
  rollout mode was changed in this phase.

## Owner decisions for Phase 3

1. **Expert-tier mitigation path**: tier-aware/skill-linked β artifact
   (preferred; C1 machinery exists) vs two-sided monitor (cheaper,
   detection-only). Until one lands, restrict any rollout to
   novice/trainee populations.
2. **Formal margins** for the locked qualification run.
3. **Rollout mode** when ready (email allowlist first, per the server
   contract), plus the ops runbook for monitor-trip telemetry and the
   per-population revalidation cadence.
