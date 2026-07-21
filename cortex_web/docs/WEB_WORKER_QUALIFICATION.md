# Native n-way Web Worker qualification - 2026-07-20

Scope: native six-way browser performance changes developed and measured in a
disposable linked worktree from production commit `01ca712`. The production
integration replays the qualified commits without changing their `cortex_web`
tree.

## Frozen statistical contract

- native six-way IIIC categorical likelihood;
- response artifact `iiic-f1-crossfit-ensemble9-rd-20260720`, SHA-256
  `0654fc210e67ece152dcaef6b40641fc9c94cb259d100ed3829d90224bc5ef8b`;
- exact categorical Fisher total-variance selector and frozen tie ordering;
- 1,200 particles, ESS threshold 0.5, and 30 MH steps;
- unchanged Precision stopping, interval, content, exposure, information, and
  result rules;
- central adoption of only the participant's actual raw outcome;
- exact fail-closed serial calculation.

No release change reduces bank size, particles, rejuvenation, categories,
coverage, or retained information.

## Exact numerical and lifecycle gates

The untouched serial baseline, every retained optimization phase, adaptive
pool sizes, the full Chromium replay, and the full Firefox replay share these
SHA-256 results:

| Evidence | SHA-256 |
|---|---|
| Statistical result | `28b05b304a8eb1525e04f6499cbd8fd55a5a2231ad5bd6b39cf3fd9df0bde20a` |
| Result without trajectory | `e72a914242201319f892c6b6a3244281c6590b2e45a3742ffb02f7aef7f3a13f` |
| Complete trajectory | `27ed007ea01aae91d3d39c380b14c2eddac949c7f1f879efab35a8dca38f6687` |
| Served sequence | `0b0973a4e3aa374e876450d67d0f972cb6a8860a2830b4c2515846fe4419d532` |
| Trial diagnostics | `125d6de7aa60fb43f72583310caea65b040a00b52cacb0f848349e694d8becd0` |

The 144-trial privacy-safe replay covers spike yes/no and every native IIIC raw
category. Direct tests additionally assert exact likelihoods, weights, ESS,
particle arrays, rejuvenation telemetry, selected segments, and complete RNG
snapshots, including the cached Gaussian.

Injected worker errors, wrong-kind and malformed messages, timeouts, and MH
shard failures are contained. MH failure recomputes exact history likelihood
on the coordinator; existing branch failure recomputes the full step from the
untouched authoritative state. Pool calibration asserts exact loss equality
before selecting a worker count.

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
- event-loop heartbeat mean 0.056 ms, max 25.9 ms;
- serial fallbacks: zero.

Compared with the untouched n-way production base, the final replay reduced
answer p50 by 93.3%, answer p95 by 84.7%, selection p50 by 88.1%, and
rejuvenation p95 by 81.0%.

## Reported-core adaptation

Short full-bank Chromium measurements, with exact serial/adaptive result
equality:

| Reported cores | Answer p50 | Answer p95 | Profile |
|---:|---:|---:|---|
| 2 | 1,956.0 ms | 2,224.2 ms | exact serial; no pool |
| 4 | 1,159.6 ms | 1,601.4 ms | ceiling 2, calibrated |
| 8 | 308.5 ms | 722.1 ms | ceiling 5, calibrated |
| 12 | 201.2 ms | 449.9 ms | ceiling 6, calibrated |

The reported-two-core result was measured on a high-resource host using a
qualification override; it is not a physical low-power-device claim. The
startup calibration is real work and data-driven, but post-start thermal and
heartbeat revalidation is future work.

## Browser and workload evidence

- Chromium full replay: exact, answer p50/p95 569.3/3,840.2 ms, heartbeat max
  25.9 ms.
- Firefox full replay: exact, answer p50/p95 620/4,645 ms, selection 578/720
  ms, rejuvenation p95 3,170 ms, no fallback. A rare 60 ms heartbeat exceeded
  the 50 ms startup target and remains a monitoring/downgrade follow-up.
- WebKit did not start on the shared qualification host because its optional
  GStreamer, Flite, and libavif runtime libraries were absent. Shared OS
  packages were deliberately not changed; Safari/WebKit remains unqualified.
- With think time, the rank-one path is already 0.7 ms p50 and 152.6 ms p95.
  The remaining tail is concentrated in uncached observed ranks and late MH
  work; the release does not hide this by running six competing full branches.

## Regression gate

The qualified candidate passed:

| Gate | Result |
|---|---:|
| Vitest TypeScript/React suite | 50 files passed, 3 skipped; 209 tests passed, 8 intentionally skipped |
| TypeScript typecheck | passed |
| Production Vite build + precompression | passed |
| Exact full replay and golden hashes | passed |
| Real Chromium coordinator/pool parity | passed |
| Injected failure/timeout/malformed-response gates | passed |
| Diff hygiene | passed |

The canonical production gate additionally runs ESLint, FastAPI/Python tests,
CSP verification, dependency audit, real-Chromium worker smoke, participant UI
smoke, clean-commit validation, a pre-deploy backup, staged remote build,
database-backed health, and live phone-browser smoke.

## Adjustment points and remaining work

Worker ceilings and the 5%/50 ms calibration guard live in
`engine/execution_profile.ts`. Calibration domains and shards are derived from
runtime arrays and dimensions, so additional approved domains can be added and
requalified without rewriting pool topology.

The next performance increment is probability-ranked expansion with
cancellation/deprioritization and immediate observed-answer priority. Runtime
heartbeat revalidation, versioned calibration caching, physical
2/4/8/12-core testing, Safari/WebKit qualification, fine-grained
cancellation/resume coverage, and the broader powered statistical matrix
remain follow-up gates. Complete-trajectory equality proves the frozen replay
did not change its statistical result; it does not claim performance on every
device or statistical generalization beyond the qualified inputs.

## Historical binary reference

The prior two-thread binary worker release used a 35,193-item bank and 1,200
particles. On the same class of high-resource host, immediate alternating
answers measured 419.2 ms p50 and 1,461.9 ms p95; with 750 ms think time they
measured 0.8 ms p50 and 767.0 ms p95. Those results are retained only as a
historical reference: binary had two outcomes and a substantially cheaper
selector, so it is not an apples-to-apples n-way baseline.
