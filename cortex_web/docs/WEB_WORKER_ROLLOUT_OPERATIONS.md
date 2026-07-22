# Precision browser-compute rollout operations

This is the authoritative operational record for the Web Worker architecture.
The numerical contract remains `WEB_WORKER_ARCHITECTURE.md`; qualification
evidence remains `WEB_WORKER_QUALIFICATION.md`.

## Production record

This table preserves the initial rollout event and the latest tagged source
known when this document was refreshed. The running release must always be
verified from the release stamp and deep-health endpoint; do not assume a
Markdown file proves current production state.

| Field | Value |
|---|---|
| Initial worker release | `dc2750853febe164716a8d1ba86ae982c755b64f` |
| Initial release activated | 2026-07-20 04:43 UTC |
| Public compute enabled | 2026-07-20 05:05 UTC |
| Latest repository source at 2026-07-21 refresh | `016c6d8` (fixed calibrated worker pool for each session) |
| Latest tagged deployed source recorded here | `c5d63567417b33a4d6a24e93de112d137a1862f8` (`cortex-web-prod-rejuvenation-loading-ux-2026-07-21`) |
| Policy exposure | `CORTEX_PRECISION_POLICY_ROLLOUT=all` |
| Compute exposure | `CORTEX_PRECISION_COMPUTE_ROLLOUT=all` |
| Previous compute exposure | `email_allowlist` |
| Environment backup | `/etc/cortex/cortex.env.pre-compute-all-20260720T0505Z` |

The compute flag affects only new sessions. A session's server-owned
`compute_mode` stamp is immutable across resume. AD6, unsupported browsers,
devices reporting fewer than four logical cores, and failures during worker
startup or branch calculation retain the exact serial path.

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

## Aggregate database observation

Use a read-only aggregate query approved for deployment operations. The query
must emit no participant code, email, session identifier, response, or question
identifier. It should report session stamps/lifecycle, completed-session
telemetry coverage, fallback totals, actual browser mode, coarse
hardware-concurrency strata, and per-session timing summaries. The operational
query is not part of the reviewer source archive because it is coupled to the
live schema and access controls. Stored summaries cannot reconstruct a pooled
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
