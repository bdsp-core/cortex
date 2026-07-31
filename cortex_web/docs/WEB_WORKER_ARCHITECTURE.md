# Certification Web Worker architecture

Status: deployed native n-way architecture. The initial public worker rollout
was promoted in July 2026; subsequent production commits added exact selector
and likelihood optimizations, ranked speculation/preemption, runtime load
protection, persistent MH-history caching, fixed-per-session calibrated pools,
and participant-visible wait-state feedback. New installations still fail
closed to serial computation unless the server enables the rollout.

## Non-negotiable invariants

- PrecisionPolicy, AD6, the native six-way likelihood, 1,200 particles, ESS
  threshold 0.5, 30 MH steps, selector objective, tie ordering, stopping,
  intervals, content eligibility, and cut classification are unchanged.
- AD6 and devices below the native n-way pool threshold use the exact serial
  path.
- The authoritative particle state and RNG live in one coordinator worker.
- Pool workers receive immutable indexed inputs and return numerical shards;
  they never mutate authoritative state or select the adopted response.
- Only the participant's actual raw response is committed.
- Startup calibration selects the pool size once. That worker count is fixed
  for the session; heartbeat telemetry cannot resize it.
- Worker startup, calibration, timeout, malformed-response, or runtime failure
  falls back to exact coordinator calculation from untouched state.
- Scheduling, device timing, worker counts, caching, and progress presentation
  never enter statistical state or policy decisions.
- The complete question-bank manifest remains in the UI `Bundle`; workers
  receive only numerical fields used by inference.

## Runtime flow

```text
authenticated API session
  └─ persisted computeMode (server-owned; default serial)
      └─ browser Bundle
          ├─ full manifest → selected segId → EEG/spectrogram rendering
          └─ compact numerical index → authoritative coordinator Web Worker
                ├─ exact posterior update and RNG ownership
                ├─ ranked response speculation during participant think time
                └─ calibrated persistent native n-way worker pool
                     ├─ Fisher/entropy screening shards
                     ├─ exact candidate-refinement shards
                     └─ per-particle MH-history likelihood shards
                              ↓
                    merge by original index/order
                              ↓
                    adopt actual response only
                              ↓
                    next segId + fresh policy status
```

Binary/AD6 work and ineligible devices use the established serial path. Native
n-way Precision sessions may use multiple same-origin workers, bounded by the
device profile and startup calibration. Probability-ranked expansion is also
bounded; optional speculative work is cancelled or deprioritized once the
observed response is known. Required rejuvenation is never skipped.

Candidate shards preserve frozen candidate ordering, and the coordinator
applies the existing `(loss, segId)` ordering. Particle shards preserve each
particle's categorical-history order. Persistent workers cache only validated
immutable history material; updates invalidate or extend that cache through
the typed protocol. Floating-point reductions and RNG acceptance remain in
their qualified ordering.

## Module boundaries

| Module | Responsibility |
|---|---|
| `engine/worker_protocol.ts` | typed main-thread/coordinator messages |
| `engine/compute_payload.ts` | validated structure-of-arrays wire format |
| `engine/worker.ts` | authoritative coordinator, calibration, and pool lifecycle |
| `engine/execution_profile.ts` | pure eligibility and conservative worker ceilings |
| `engine/nway_selector_protocol.ts` | selector/history worker messages |
| `engine/nway_selector_executor.ts` | persistent pool, sharding, validation, timeout, and cache lifecycle |
| `engine/nway_selector_worker.ts` | immutable screen, score, and history-likelihood jobs |
| `engine/nway_selector.ts` | exact native n-way selection calculations |
| `engine/nway_likelihood.ts` | allocation-aware exact categorical likelihood |
| `engine/particles.ts` | posterior update and exact MH mechanics |
| `engine/ranked_speculation.ts` | bounded response-rank scheduling |
| `engine/speculation_cancellation.ts` | observed-answer preemption rules |
| `engine/core_snapshot.ts` | exact RNG, particles, policy, and status snapshot |
| `engine/session.ts` | orchestration and atomic observed-result adoption |
| `src/components/TrajectoryOptimizationStatus.tsx` | non-statistical long-wait feedback |
| `services/api/compute_rollout.py` | authenticated server-side rollout decision |

The older binary branch-worker modules remain exact tested infrastructure. The
native six-way path uses the n-way executor because six categorical outcomes,
large candidate sets, and MH history make blind full-branch duplication
counterproductive.

## Device adaptation

`navigator.hardwareConcurrency` establishes a conservative ceiling, not the
final allocation:

- requested serial, non-Precision policy, unavailable workers, or unknown
  concurrency: serial;
- fewer than four reported logical cores: serial;
- four cores: at most two pool workers;
- five through seven: at most three;
- eight through eleven: at most five;
- twelve or more: at most six.

This reserves at least one reported core, and at least two on devices reporting
eight or more. At startup, a bounded 24-candidate real-bank probe compares
eligible pool sizes. It chooses the smallest pool within 5% of the fastest
responsive result, subject to a 50 ms heartbeat-delay limit, and terminates
unused workers. The selected count remains fixed until the session ends. A
failed probe selects exact serial execution; post-start heartbeat data is
observational only.

The tuning points are isolated in `execution_profile.ts`. Calibration
candidates come from the runtime domain arrays, while candidate, particle, and
history jobs derive their dimensions from runtime domain count, `N`, `K`, and
history length. Adding an approved domain requires statistical/content and
device requalification, but not a worker-topology rewrite.

SharedArrayBuffer, SharedWorker, ServiceWorker, WebGPU, cross-origin isolation,
and device-memory APIs are not required.

## Full-bank and media boundary

The full rendering manifest never leaves `Bundle`. After the engine selects a
`segId`, `Bundle.segment(segId)` resolves and lazily fetches that exact item's
EEG and spectrogram. Typed calculation buffers cross same-origin module-worker
boundaries. No optimization changes available questions, media, response
categories, or content/exposure eligibility.

## Failure, privacy, and telemetry posture

- Worker startup is bounded at 10 seconds and a shard job at 30 seconds.
- Messages are same-origin module-worker traffic under the existing CSP
  (`worker-src 'self'`). No blob workers, eval, shared memory, or external
  origins are introduced.
- Payload dimensions, job kinds, shard offsets, history versions, and output
  lengths are validated before results can be merged.
- Workers terminate on completion, abort, replacement, startup failure, and
  sign-out. Pending jobs reject and timers are cleared.
- Schema-v2 `_enginePerformance` stores bounded phase distributions, outcome
  ranks, branch lifecycle counts, heartbeat delay, selected worker count,
  calibration summary, `answerToItem`, `answerToMediaReady`, and a memory
  estimate. `answerToMediaReady` includes the selected segment's fetch/decode
  boundary and is the participant-facing transition measure; `answerToItem`
  ends at the earlier engine handoff. Telemetry records no new EEG or
  participant content, is excluded from statistical state, and is stripped
  from participant dashboard/history responses.
- The schema retains a runtime-adjustment array only for older summary
  compatibility. Fixed-pool sessions emit no adjustment events.
- Post-start heartbeat delay is observational telemetry only and does not feed
  worker-pool control.

## Rollout and rollback

The API persists `sessions.compute_mode`; resume retains that stamp. Supported
server settings are `off`, `email_allowlist`, and `all`. Unknown values and
non-Precision sessions select serial. A request body cannot opt in.

Configuration rollback sets the compute rollout to `off` for new sessions and
restarts the service. Existing stamped sessions remain reproducible. An
application rollback uses the health-gated immutable-release procedure in
`../deploy/README.md`. Never change the policy, bank, particle profile, or
stored participant rows to respond to a compute-placement incident.
