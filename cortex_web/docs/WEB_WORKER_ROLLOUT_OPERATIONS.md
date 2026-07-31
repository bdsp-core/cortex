# Precision browser-compute rollout operations

This is the authoritative operational record for the Web Worker architecture.
The numerical contract remains `WEB_WORKER_ARCHITECTURE.md`; qualification
evidence remains `WEB_WORKER_QUALIFICATION.md`.

## Production record

This table preserves the initial rollout event and the production state
verified on 2026-07-22. The running release must always be verified from the
release stamp and deep-health endpoint; do not assume a Markdown file proves
current production state. The complete handoff is in
`PRODUCTION_BASELINE.md`.

| Field | Value |
|---|---|
| Initial worker release | `dc2750853febe164716a8d1ba86ae982c755b64f` |
| Initial release activated | 2026-07-20 04:43 UTC |
| Public compute enabled | 2026-07-20 05:05 UTC |
| Fixed-pool source | `016c6d8` (startup-selected worker count remains fixed for the sitting) |
| Long-wait UX source | `c5d63567417b33a4d6a24e93de112d137a1862f8` (`cortex-web-prod-rejuvenation-loading-ux-2026-07-21`) |
| Accepted latency/numerical baseline | `d7efd5a26ccd64920a6a4064f345ffb6ce08a3e9` |
| Verified live source | `e68d59b528619e9ed809ac7033bb2fc79553251c` |
| Verified live release | `/opt/cortex/releases/20260722T212609Z-e68d59b52861` |
| Policy exposure | `CORTEX_PRECISION_POLICY_ROLLOUT=all` |
| Compute exposure | `CORTEX_PRECISION_COMPUTE_ROLLOUT=all` |
| Previous compute exposure | `email_allowlist` |
| Environment backup | `/etc/cortex/cortex.env.pre-compute-all-20260720T0505Z` |

The compute flag affects only new sessions. A session's server-owned
`compute_mode` stamp is immutable across resume. AD6, unsupported browsers,
devices reporting fewer than four logical cores, and failures during worker
startup or branch calculation retain the exact serial path.

The source range from `d7efd5a` through `e68d59b` does not change the
certification numerical implementation. Current live observations may
therefore be compared with the accepted `d7efd5a` engine reference while still
recording the actual live release SHA.

## Observation checklist

The historical initial snapshot below was captured at 2026-07-20 05:15 UTC.
Later live qualification confirmed a completed high-core native session with
all workers retained, zero serial fallback, and exact serving reconciliation;
future observations must still be evaluated independently.

Run the immediate checks after a configuration change, then repeat the data
checks after the first completed certification, after the first five, and
after at least 20 completed eligible-device certifications. Low traffic leaves
a gate `PENDING`; zero samples is never reported as a pass.

| Check | Pass condition | 2026-07-20 initial observation |
|---|---|---|
| Release identity | local `HEAD`, `origin/main`, release stamp, and health SHA agree | PASS |
| Runtime configuration | running process reads policy=`all`, compute=`all` | PASS |
| Service readiness | service active; deep health and DB checks succeed | PASS |
| Worker delivery | SPA, coordinator worker, and branch-worker assets return JavaScript with HTTP 200 | PASS |
| Error containment | no uncontained worker error, API warning, or `/api/client-error` report | PASS; zero observed |
| Session stamping | every new Precision sitting is `dual_branch_auto`; AD6 remains `serial` | PENDING; zero new sittings |
| Telemetry completeness | every completed Precision sitting contains the current schema-v2 `_enginePerformance` summary | PENDING at initial snapshot; zero completions |
| Device adaptation | eligible devices report `dual_branch`; low-core/unsupported devices report the documented serial reason | PENDING |
| Fallback behavior | any `serial_fallback` is contained and investigated; repeated fallbacks do not concentrate by device stratum | PENDING |
| User latency | eligible-device p95 improves without a material p50 regression; absolute timing is interpreted with device load | PENDING |
| Numerical behavior | no parity complaint or result/trajectory discrepancy | PASS for qualification; PENDING for live sessions |

The account activity immediately before public enablement was a training
session. It verified the server trainer, persistence ledger, media inventory,
and dashboard refresh, but it did not exercise the certification Web Worker.

## Current latency interpretation

Use `answerToMediaReady` as the participant-facing next-question measurement.
It starts at answer dispatch and ends when `Bundle.segment` has fetched and
decoded the selected media and the viewer accepts it. `answerToItem` stops at
the earlier worker-to-UI item handoff and can understate a media-cache miss.
Engine and phase timings explain compute only.

A completed 2026-07-22 certification on the accepted `d7efd5a` baseline
provided a privacy-safe reference observation:

| Measure | Observation |
|---|---:|
| Questions / selected workers / serial fallbacks | 143 / 6 / 0 |
| `answerToMediaReady` p50 / p95 / max | 520.0 / 2711.4 / 3372.3 ms |
| `answerToItem` p50 / p95 | 446.5 / 2508.6 ms |
| Engine total p95 | 2459.8 ms |
| Selection p95 | 621.2 ms |
| Rejuvenation count / p95 | 21 / 1919.0 ms |
| MH-history likelihood p95 | 1013.9 ms |

This single device/session is a comparison point, not an SLO. The remaining
tail clusters around required rejuvenation. A one-second wait surfaces the
teal “Optimizing Question Trajectory” status so a valid long calculation does
not look stalled.

Schema-v2 retains a `runtimePool` adjustment array for backward-compatible
reading of earlier sessions. Current fixed-pool sessions should report zero
adjustments; the selected count comes from the startup profile. Post-start
heartbeat delay remains diagnostic only.

## Aggregate database observation

Use a read-only aggregate query approved for deployment operations. The query
must emit no participant code, email, session identifier, response, or question
identifier. It should report session stamps/lifecycle, completed-session
telemetry coverage, fallback totals, actual browser mode, coarse
hardware-concurrency strata, and per-session timing summaries. The approved
query is versioned at `deploy/scripts/observe_precision_compute.sql`; it is
coupled to the live schema and access controls, so re-validate it against the
deployed schema before each use. Stored summaries cannot reconstruct a pooled
per-question percentile; reported means must therefore be labelled as means of
session-level p50/p95 values.

Also inspect the service journal for the same UTC window:

```bash
sudo journalctl -u cortex.service --since '2026-07-20 05:05:00 UTC' \
  --no-pager -p warning
sudo journalctl -u cortex.service --since '2026-07-20 05:05:00 UTC' \
  --no-pager -o cat | grep 'POST /api/client-error'
```

## Rollback conditions and procedure

Immediately return new sessions to serial computation for an uncontained
worker exception, a worker-linked inability to continue or finalize, any
mathematical/result discrepancy, or a repeated fallback pattern across
sessions. A latency concern without a correctness or completion failure should
be stratified by hardware concurrency and concurrent device load before being
attributed to the worker.

Rollback does not alter existing session stamps:

1. Set `CORTEX_PRECISION_COMPUTE_ROLLOUT=off` in
   `/etc/cortex/cortex.env`.
2. Restart `cortex.service`.
3. Require active service state and `GET /api/health?deep=1` with `ok=true` and
   `db=ok`.
4. Verify the running process reads `off`; confirm the next new Precision
   session is stamped `serial`.

Do not change the PrecisionPolicy rollout, frozen policy, bank, particle
profile, or existing session rows in response to a compute-placement incident.

## Known adjacent limitation

The server-driven training path is separate from certification computation.
Its persisted trajectory point is currently captured from the pre-answer
snapshot, so the longitudinal training curve trails the server-authoritative
belief by one update. Training question selection uses the post-answer server
state and is unaffected. Track this as a history-telemetry issue; do not mix it
with certification-worker rollback decisions.
