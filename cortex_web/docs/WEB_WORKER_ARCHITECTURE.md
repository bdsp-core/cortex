# Certification Web Worker architecture

Status: implemented and qualification-gated in `cortex_web_optimized`; not
deployed. The server-side compute rollout defaults to `off`.

## Non-negotiable invariants

- PrecisionPolicy, AD6, particle count, MH steps, selector, tie ordering,
  stopping statuses, and downstream cut classification are unchanged.
- AD6 always uses the established serial path.
- The authoritative session state exists in one coordinator worker. Helper
  workers receive snapshots and can never mutate that state.
- Only the result for the participant's actual response is adopted.
- Any helper initialization, computation, timeout, deserialization, or runtime
  failure recomputes from the untouched authoritative state through the serial
  path.
- The complete question-bank manifest remains in the UI `Bundle`, including
  EEG and spectrogram locations. Compute workers receive only the numerical
  fields used by the algorithm.

## Runtime flow

```text
authenticated API session
  └─ persisted computeMode (server-owned; default serial)
      └─ browser Bundle
          ├─ full manifest → selected segId → exact EEG/spectrogram rendering
          └─ 4.16 MiB packed numerical index → coordinator Web Worker
                ├─ predicted response → unchanged advanceCore()
                └─ alternate response → one persistent helper Web Worker
                         ↓
                  adopt actual response only
                         ↓
                 next segId + fresh policy status
```

The split-branch arrangement uses the coordinator for the predicted branch and
one helper for the alternate branch. This preserves the common path's existing
latency and removes the serial misprediction tail without maintaining two
extra 35k-bank copies. There are at most two simultaneous compute threads.

## Module boundaries

| Module | Responsibility |
|---|---|
| `engine/worker_protocol.ts` | Typed main-thread/coordinator messages |
| `engine/compute_payload.ts` | Validated structure-of-arrays wire format and transfer list |
| `engine/worker.ts` | Authoritative coordinator lifecycle |
| `engine/execution_profile.ts` | Pure device/policy eligibility decision |
| `engine/branch_executor.ts` | Browser-independent scheduling interface |
| `engine/branch_worker_executor.ts` | Persistent helper, timeouts, failure propagation |
| `engine/branch_protocol.ts` | Typed helper request/result serialization |
| `engine/branch_worker.ts` | Alternate-branch calculation only |
| `engine/core_snapshot.ts` | Exact RNG, particles, policy, status, and telemetry snapshot |
| `engine/session.ts` | Predicted/alternate scheduling and atomic result adoption |
| `src/performanceSummary.ts` | Bounded non-policy timing aggregation |
| `services/api/compute_rollout.py` | Authenticated server-side rollout decision |

## Device adaptation

`navigator.hardwareConcurrency` is a hint, not a command to allocate that many
workers. The rule is deliberately small:

- requested mode `serial`: serial;
- AD6: serial;
- unavailable Worker API or unknown concurrency: serial;
- fewer than four reported logical cores: serial;
- four or more: coordinator plus one helper (`dual_branch`).

No worker pool scales with core count. SharedArrayBuffer, particle sharding,
SharedWorker, ServiceWorker, and cross-origin isolation are not required.

## Full-bank and media boundary

The 35,193-item production manifest is 34,865,653 bytes. Its compute payload is
4,363,932 bytes (4.16 MiB): Float64 segment ids, Uint32 task masks, and Float64
`sMean`/`sSd`. Typed-array buffers are transferred rather than cloned.

The rendering manifest never leaves `Bundle`. After the engine selects a
`segId`, `Bundle.segment(segId)` resolves and lazily fetches that exact item's
EEG and spectrogram. The optimization therefore changes neither available
questions nor participant media.

## Failure and security posture

- Helper startup is bounded at 10 seconds; a branch job is bounded at 30
  seconds. Either failure disables the helper and invokes exact serial
  recomputation.
- Worker messages are same-origin module-worker traffic under the existing CSP
  (`worker-src 'self'`). No blob workers, eval, shared memory, or new external
  origins are introduced.
- Packed payload dimensions and task indices are validated before execution.
- Workers are terminated on completion, abort, replacement, error, and idle
  sign-out. Pending jobs are rejected and timers cleared.
- Execution timing never enters policy state, selection, stopping, reporting
  intervals, or cuts.
- Bounded timing summaries are stored under `_enginePerformance` for admin
  review and stripped from participant dashboard/history responses.

## Rollout and rollback

The API persists `sessions.compute_mode` with an additive SQLite/PostgreSQL
migration. Resume uses the persisted value, so an in-flight sitting never
changes execution mode.

```text
CORTEX_PRECISION_COMPUTE_ROLLOUT=off|email_allowlist|all
CORTEX_PRECISION_COMPUTE_EMAILS=comma,separated,normalized@example.org
```

Unknown values, non-Precision policies, and non-allowlisted accounts select
`serial`. A request body cannot opt in. Rollback is an environment change plus
service restart and affects new sessions only; already-stamped sessions remain
reproducible.

