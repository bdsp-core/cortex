# Native n-way Web Worker qualification

Status: production architecture qualified and deployed in July 2026. This file
records the durable correctness boundary and measured performance evidence; it
does not turn host-specific benchmark timings into a universal latency promise.

## Frozen statistical contract

Every retained performance release preserves:

- the full exposure-eligible bank and native six-way response model;
- 1,200 particles, ESS threshold 0.5, and 30 MH steps;
- the exact total-variance selector, candidate/tie ordering, and response
  resolver;
- unchanged PrecisionPolicy/AD6, intervals, coverage, information, stopping,
  content, exposure, and reporting rules;
- centralized adoption of only the participant's actual raw response; and
- exact fail-closed serial recomputation.

No qualified change reduces the bank, particle count, rejuvenation, response
categories, or retained information.

## Exact replay contract

The privacy-safe 144-question production replay pins five independent outputs:

| Artifact | SHA-256 |
|---|---|
| Statistical result | `28b05b304a8eb1525e04f6499cbd8fd55a5a2231ad5bd6b39cf3fd9df0bde20a` |
| Result without trajectory | `e72a914242201319f892c6b6a3244281c6590b2e45a3742ffb02f7aef7f3a13f` |
| Complete trajectory | `27ed007ea01aae91d3d39c380b14c2eddac949c7f1f879efab35a8dca38f6687` |
| Served sequence | `0b0973a4e3aa374e876450d67d0f972cb6a8860a2830b4c2515846fe4419d532` |
| Trial diagnostics | `125d6de7aa60fb43f72583310caea65b040a00b52cacb0f848349e694d8becd0` |

The replay covers spike yes/no and every native IIIC raw category. Direct tests
also assert exact likelihoods, weights, ESS, particle arrays, rejuvenation
telemetry, selected segments, complete RNG snapshots, and the cached Gaussian.

Injected worker errors, wrong-kind and malformed messages, timeouts, and MH
shard failures are contained. MH failure recomputes exact history likelihood
on the coordinator; branch failure recomputes the full step from untouched
authoritative state. Pool calibration asserts exact loss equality before
selecting a worker count. The replay retains 31 required rejuvenations and zero
serial fallbacks under its frozen answer stream.

## Full live-replay performance progression

All rows use the same 144-trial stream, five golden hashes, and 31
rejuvenations. Browser times are comparative measurements on the qualification
host, not universal device guarantees.

| Stage | Answer p50 | Answer p95 | Selection p50 | Rejuvenation p95 |
|---|---:|---:|---:|---:|
| Untouched `01ca712` | 8,456.8 ms | 25,028.6 ms | 4,681.4 ms | 12,843.5 ms |
| Exact hot-loop mechanics | 8,423.9 ms | 18,084.0 ms | 4,689.3 ms | 7,442.6 ms |
| Analytical Fisher + exact refinement mechanics | 4,950.7 ms | 14,136.7 ms | 2,558.9 ms | 7,539.4 ms |
| Deterministic selector pool | 591.5 ms | 10,162.6 ms | 570.2 ms | 8,420.1 ms |
| Calibrated selector pool | 587.4 ms | 10,317.5 ms | 557.5 ms | 7,772.9 ms |
| Selector + particle-sharded MH | **569.3 ms** | **3,840.2 ms** | **558.4 ms** | **2,446.5 ms** |

Final Chromium distributions:

- answer-to-next-item p50/p90/p95/p99/max:
  569.3/3,039.7/3,840.2/4,749.8/5,402.8 ms;
- selection: 558.4/665.9/707.8/758.7/860.5 ms;
- rejuvenation: 0/1,982.6/2,446.5/3,327.8/3,758.9 ms;
- event-loop heartbeat mean 0.056 ms, max 25.9 ms; and
- serial fallbacks: zero.

Compared with the untouched n-way production base, that replay reduced answer
p50 by 93.3%, answer p95 by 84.7%, selection p50 by 88.1%, and rejuvenation
p95 by 81.0%.

Later exact hot-path reuse, persistent history caching, and bounded ranked
speculation further reduced ordinary high-core live/replay medians to roughly
0.43–0.48 seconds on the measured streams. A post-load-guard live certification
supplied 147 unique questions, 122 native categorical transitions, zero
fallback, and a 432 ms overall median; ordinary categorical work had a 460 ms
median. These are paired observations from specified devices and streams, not
an SLO for all hardware.

## Reported-core startup selection

Short full-bank Chromium measurements, with exact serial/adaptive result
equality:

| Reported cores | Answer p50 | Answer p95 | Profile |
|---:|---:|---:|---|
| 2 | 1,956.0 ms | 2,224.2 ms | exact serial; no pool |
| 4 | 1,159.6 ms | 1,601.4 ms | ceiling 2, calibrated |
| 8 | 308.5 ms | 722.1 ms | ceiling 5, calibrated |
| 12 | 201.2 ms | 449.9 ms | ceiling 6, calibrated |

The reported-two-core result was measured on a high-resource host using a
qualification override; it is not a physical low-power-device claim. Startup
calibration is real work and data-driven. Its selected worker count is fixed
for the session; post-start heartbeat data remains observational and cannot
revalidate or resize the pool.

## Browser and workload boundary

- Chromium full replay: exact, answer p50/p95 569.3/3,840.2 ms, heartbeat max
  25.9 ms.
- Firefox full replay: exact, answer p50/p95 620/4,645 ms, selection 578/720
  ms, rejuvenation p95 3,170 ms, no fallback. A rare 60 ms heartbeat exceeded
  the 50 ms startup target and remains an observation, not an in-session
  topology change.
- WebKit did not start on the shared qualification host because optional
  GStreamer, Flite, and libavif runtime libraries were absent. Shared OS
  packages were deliberately not changed; Safari/WebKit remains unqualified.
- With think time, the rank-one path measured 0.7 ms p50 and 152.6 ms p95. The
  remaining tail is concentrated in uncached observed ranks and required MH
  work; the release does not hide it by weakening inference.

Four-core/two-worker qualification remains materially slower, and required MH
rejuvenation continues to dominate the latency tail. Physical low-resource
devices, Safari/WebKit, and a broader powered statistical matrix remain
separate qualification obligations.

## Regression gate

The production candidate must pass:

| Gate | Requirement |
|---|---|
| ESLint and TypeScript | zero warnings/errors |
| Vitest and FastAPI/Python pytest | all non-intentionally-skipped tests pass |
| Production Vite build + precompression | pass |
| CSP verification and dependency audit | pass; zero high-severity findings |
| Exact replay and five golden hashes | exact match |
| Real Chromium coordinator/pool parity | exact serial/worker match |
| Failure/timeout/malformed-response injection | exact containment or fallback |
| Diff hygiene | clean release candidate with generated artifacts excluded |

The normal gate does not launch browser smoke. The production integration gate
sets `CORTEX_BROWSER_GATES=1` and additionally requires real-Chromium
serial/worker equality plus participant UI smoke. Full-bank benchmark and
exact-replay inputs are governed external artifacts; a missing artifact is not
evidence of a failed algorithm.

From `cortex_web/`, with locked Python dependencies and Chromium installed:

```bash
CORTEX_QUALITY_PYTHON=.venv/bin/python \
CORTEX_BROWSER_GATES=1 \
CORTEX_SMOKE_SYNTHETIC=1 \
  bash scripts/quality_gate.sh
```

## Historical binary reference

The prior two-thread binary worker release used a 35,193-item bank and 1,200
particles. On the same class of high-resource host, immediate alternating
answers measured 419.2 ms p50 and 1,461.9 ms p95; with 750 ms think time they
measured 0.8 ms p50 and 767.0 ms p95. Those results are retained only as a
historical reference: binary had two outcomes and a substantially cheaper
selector, so it is not an apples-to-apples n-way baseline.
