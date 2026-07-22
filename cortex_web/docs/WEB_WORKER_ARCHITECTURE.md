# Certification Web Worker architecture

Status: production architecture for the native n-way performance release,
qualified on 2026-07-20 and promoted on 2026-07-21. The server owns rollout;
the application default remains fail-closed `off`.

## Non-negotiable invariants

- The native six-way response model, 1,200 particles, ESS threshold 0.5,
  30-step MH rejuvenation, selector objective, tie ordering, stopping statuses,
  coverage rules, and downstream cut classification are unchanged.
- AD6 and devices below the n-way pool threshold use the established exact
  serial path.
- The authoritative session state and RNG exist in one coordinator worker.
  Pool workers receive immutable inputs and return indexed numerical shards;
  they never mutate authoritative state.
- Only the participant's actual raw outcome is adopted. Timing, calibration,
  and device information never enter statistical state or policy decisions.
- Startup calibration selects the pool size once. That worker count is fixed
  for the session; heartbeat telemetry cannot resize it.
- Worker initialization, calibration, timeout, malformed-response, or runtime
  failure falls back to the exact coordinator calculation from untouched
  authoritative state.
- The complete question-bank manifest remains in the UI `Bundle`. Compute
  workers receive only numerical fields used by the algorithm.

## Runtime flow

```text
authenticated API session
  `- persisted computeMode (server-owned)
      `- browser Bundle
          |- full manifest -> selected segId -> EEG/spectrogram rendering
          `- packed numerical index -> authoritative coordinator Web Worker
                |- conservative core ceiling
                |- bounded real-work startup calibration
                `- persistent same-origin n-way pool
                     |- domain Fisher-screen shards
                     |- exact candidate-score shards
                     `- per-particle MH history-likelihood shards
                            |
                 coordinator merges by original index/order
                            |
                 adopt actual raw response only
                            |
                    next segId + policy status
```

The coordinator speculates the most probable raw outcome during participant
think time. If that exact outcome is observed, its completed result is adopted.
Otherwise the coordinator computes the observed outcome from the untouched
state. Selection and MH history work may use the calibrated pool in either
case. The release deliberately does not start six simultaneous full outcome
branches: measurement showed that unused rejuvenation work creates contention
and worsens the common path.

Candidate shards preserve the frozen candidate order and return indexed loss
vectors. The coordinator applies the existing `(loss, segId)` ordering.
Particle shards preserve each particle's original categorical-history order;
there is no floating-point reduction across workers, and RNG generation and
acceptance remain centralized.

## Module boundaries

| Module | Responsibility |
|---|---|
| `engine/worker_protocol.ts` | Typed main-thread/coordinator messages |
| `engine/compute_payload.ts` | Validated structure-of-arrays wire format and transfer list |
| `engine/worker.ts` | Authoritative coordinator, calibration, pool ownership, and lifecycle |
| `engine/execution_profile.ts` | Pure eligibility, safe worker ceilings, and calibration choice |
| `engine/nway_selector_protocol.ts` | Typed selector and history-shard messages |
| `engine/nway_selector_executor.ts` | Persistent deterministic pool, sharding, validation, and timeouts |
| `engine/nway_selector_worker.ts` | Immutable screen, score, and history-likelihood jobs |
| `engine/nway_selector.ts` | Analytical Fisher screen and exact total-variance refinement |
| `engine/nway_likelihood.ts` | Allocation-free categorical response likelihood |
| `engine/particles.ts` | Packed history and exact coordinator/worker MH mechanics |
| `engine/core_snapshot.ts` | Exact RNG, particles, policy, status, and telemetry snapshot |
| `engine/session.ts` | Rank-one speculation and atomic observed-result adoption |
| `src/performanceSummary.ts` | Bounded schema-v2 non-policy timing aggregation |
| `services/api/compute_rollout.py` | Authenticated server-side rollout decision |

The earlier branch-worker modules remain available as exact, tested
infrastructure, but native n-way production does not blindly allocate helper
workers for every response outcome.

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
unused workers. The selected count then remains fixed until the session ends.
A failed probe selects exact serial execution.

These tuning points are intentionally isolated in `execution_profile.ts`.
Calibration candidates are assembled from the runtime domain arrays, while
candidate, particle, and history jobs derive their dimensions from runtime
domain count, `N`, `K`, and history length. Adding an approved domain therefore
requires statistical/content and device requalification, but not a worker
topology rewrite.

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
- Payload dimensions, job kinds, shard offsets, and output lengths are
  validated before results can be merged.
- Workers terminate on completion, abort, replacement, startup failure, and
  sign-out. Pending jobs reject and timers are cleared.
- Schema-v2 `_enginePerformance` stores bounded phase distributions, outcome
  ranks, branch lifecycle counts, heartbeat delay, selected worker count,
  calibration summary, and a memory estimate. It records no new EEG or
  participant content, is excluded from statistical state, and is stripped
  from participant dashboard/history responses.
- Post-start heartbeat delay is observational telemetry only and does not feed
  worker-pool control.

## Rollout and rollback

The API persists `sessions.compute_mode` through the existing additive
SQLite/PostgreSQL migration. Resume uses the persisted value, so an in-flight
sitting never changes its compute profile.

```text
CORTEX_PRECISION_COMPUTE_ROLLOUT=off|email_allowlist|all
CORTEX_PRECISION_COMPUTE_EMAILS=comma,separated,normalized@example.org
```

Unknown values, non-Precision policies, and non-allowlisted accounts select
serial. A request body cannot opt in. Configuration rollback affects newly
created sessions; application rollback uses the atomic release procedure in
`deploy/README.md`. The previous n-way release remains compatible with the
additive database state.
