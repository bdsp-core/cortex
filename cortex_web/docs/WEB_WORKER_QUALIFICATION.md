# Web Worker qualification — 2026-07-19

Scope: canonical `cortex_web` release candidate, developed in the isolated
`cortex_web_optimized` tree and promoted from baseline commit
`8dd3b5829046815a5e4209ccb4610e9268f73fa0`. Qualification preceded production
deployment.

## Numerical and lifecycle gates

- In-process full-session serial/split result equality: exact.
- Real Chromium coordinator/helper serial/split result equality: exact (20
  questions, 1,200 particles).
- Actual 35,193-item manifest serial/split partial-session equality: exact in
  every benchmark below.
- RNG snapshots include the cached Gaussian; policy snapshots include fresh
  Precision statuses, persistence streaks, terminals, band counts, and last
  diagnostics.
- Injected helper failure: exact serial result, one `serial_fallback` event,
  then serial operation.
- Build emits separate coordinator and helper module-worker assets.

## Regression gates

All gates below passed from the isolated candidate directory:

| Gate | Result |
|---|---:|
| FastAPI/Python suite | 204 passed |
| Vitest TypeScript/React suite | 188 passed, 6 intentionally skipped |
| ESLint architecture/code-quality gate | passed with zero warnings |
| TypeScript typecheck | passed |
| Production Vite build + precompression | passed |
| CSP verification | passed |
| Full dependency audit (`npm audit`) | 0 vulnerabilities |
| Real-Chromium coordinator/helper smoke | passed, exact result equality |
| Full participant UI smoke | passed through signup, consent, tutorial, EEG-only spike questions, EEG+spectrogram IIIC questions, and responsive layouts |

Vitest reported 45 passing test files and one intentionally skipped file. The
skips are opt-in benchmark or uncommitted-real-bank cases; the production-bank
benchmark was run separately below. The API suite's only output beyond the
passes was the existing Starlette/httpx deprecation warning.

Canonicalization was performed in a clean worktree. No unrelated files from
the research workspace are part of the release diff.

## Production-bank browser timings

Machine: 48 reported logical cores, headless Chromium 140, 1,200 particles,
28 identical-answer-sequence questions. These are local comparative
benchmarks, not universal device promises.

| Scenario | Mode | p50 answer→item | p95 answer→item | max answer→item |
|---|---:|---:|---:|---:|
| Immediate, alternating response | serial | 421.6 ms | 2,024.9 ms | 2,086.7 ms |
| Immediate, alternating response | split branch | 419.2 ms | 1,461.9 ms | 1,505.4 ms |
| 750 ms think-time, alternating response (canonical release) | serial | 0.6 ms | 1,332.2 ms | 1,547.6 ms |
| 750 ms think-time, alternating response (canonical release) | split branch | 0.8 ms | 767.0 ms | 773.4 ms |

The final split-branch implementation preserves the median and reduces the
immediate-response p95/max by about 28%. In the final canonical release run,
modest think-time reduced p95 by 42.4% and max by 50.0%. The improvement is
concentrated where intended:
unpredicted responses and expensive late selections. A one-direction response
sequence on an idle host already had sub-millisecond answer latency after 750
ms think-time in both modes; the worker design does not claim speed where the
existing speculative result was already ready.

## Payload gate

| Artifact | Bytes |
|---|---:|
| Full UI manifest (EEG/spec metadata retained) | 34,865,653 |
| Minimal JSON calculation objects | 5,020,051 |
| Transferable typed calculation payload | 4,363,932 |

The packed payload is 87.5% smaller than the rendering manifest. The helper
receives one packed copy; the earlier two-helper prototype was rejected after
measurement because it added common-path overhead.

## Rollout gate

The compute flag remains `off`. Before changing it to `all`:

1. enable `email_allowlist` for a small device-diverse pilot;
2. confirm zero mathematical parity complaints, zero uncontained worker
   errors, and zero persistent `serial_fallback` concentration;
3. compare `_enginePerformance.answerToItem` p50/p95/max against serial
   sessions, stratified by reported hardware concurrency;
4. require a lower p95 with no material p50 regression on eligible devices;
5. switch back to `off` immediately if the gate is missed.

Absolute times depend on device load. A concurrent 40-core simulation can
still contend with browser computation; Web Workers keep UI rendering
responsive and reduce response-branch serialization, but cannot create CPU
capacity that the operating system has already allocated elsewhere.
