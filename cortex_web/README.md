# CORTEX Web

Canonical full-stack implementation of the CORTEX adaptive EEG certification app.
The React UI, TypeScript testing engine, FastAPI service, deployment assets,
and regression harnesses live together so one versioned release can be tested,
deployed, and rolled back as a unit.

The browser-compute rollout is server-owned and disabled by application
default. Eligible native n-way sessions can use a bounded, calibrated worker
pool when the production rollout stamp enables it; every failure path remains
exact and fail-closed.

## Architecture

- `apps/web/src/` — participant UI, API client, bundle/media ownership.
- `apps/web/engine/` — deterministic TypeScript SMC engine, AD6 rollback,
  frozen PrecisionPolicy, worker protocols, snapshots, and branch scheduling.
- `apps/web/trainer/` — browser trainer code, separate from certification.
- `services/api/` — authenticated FastAPI API and SQLite/PostgreSQL storage.
- `deploy/` — Caddy, systemd, provisioning, backup, and release scripts.
- `research/` — export/research utilities that do not participate in serving.
- `docs/WEB_WORKER_ARCHITECTURE.md` — optimized worker and rollback contract.
- `docs/WEB_WORKER_QUALIFICATION.md` — reproducible correctness/performance
  evidence and the rollout gate.
- `docs/REPOSITORY_STRUCTURE.md` — module ownership, dependency rules, and
  release gates.
- `docs/MODULARIZATION_REPORT.md` — completed hygiene/modularization work and
  its regression evidence.
- `docs/README.md` — authoritative documentation index.

The full 35k manifest remains in the main-thread `Bundle` for exact EEG and
spectrogram rendering. Only a packed numerical index crosses into the engine
coordinator. On eligible native n-way devices, a bounded startup calibration
selects a conservative persistent pool for deterministic selector and MH
history shards. That selected worker count remains fixed for the full session;
heartbeat telemetry observes responsiveness but never resizes the pool. The
coordinator owns state and RNG, speculates the most likely response, and adopts
only the participant's actual response. AD6 and low-core devices stay serial. See
[WEB_WORKER_ARCHITECTURE.md](docs/WEB_WORKER_ARCHITECTURE.md).

## Local setup

```bash
cd cortex_web_optimized
npm ci

# API dependencies (use a dedicated virtual environment in a fresh checkout)
python3 -m venv .venv
.venv/bin/pip install -r services/api/requirements.lock
```

The production bank is intentionally not committed. Configure or copy the
bundle under `apps/web/public/bundle/` as described in `LOCAL_DEV.md`.

For hot reload, run the API and Vite processes described in `LOCAL_DEV.md`.
For the production-shaped local app:

```bash
npm run build
cd services
../.venv/bin/python -m api.run
```

## Qualification commands

```bash
# TypeScript engine and frontend
npm run typecheck
npm run lint
npm test
npm run build
npm run -w cortex-web worker-smoke

# Full-bank comparative benchmark; override the path when the bank is elsewhere
CORTEX_FULL_BANK_MANIFEST=/path/to/manifest.json \
  npm run -w cortex-web worker-benchmark

# API and research suites
.venv/bin/python -m pytest -q
```

The complete local/CI gate is available as one command. Set the Python path
when the environment is not named `.venv`:

```bash
CORTEX_QUALITY_PYTHON=.venv/bin/python npm run quality
CORTEX_QUALITY_PYTHON=.venv/bin/python npm run quality:browser
```

`worker-smoke` runs the coordinator and nested workers in real Chromium and
requires exact serial/parallel result equality. The native replay and
full-bank benchmarks also assert exact equality before reporting timings.

## Runtime controls

Stopping policy and compute execution are independent server-owned stamps:

```text
CORTEX_PRECISION_POLICY_ROLLOUT=off|email_allowlist|all
CORTEX_PRECISION_POLICY_EMAILS=...

CORTEX_PRECISION_COMPUTE_ROLLOUT=off|email_allowlist|all
CORTEX_PRECISION_COMPUTE_EMAILS=...
```

The compute default is `off`. Unknown values, AD6 sessions, unsupported
browsers, failed calibration, and devices reporting fewer than four logical
cores use serial execution. Eligible ceilings are two workers at four reported
cores, three at five through seven, five at eight through eleven, and six at
twelve or more; calibration may choose fewer. Existing sessions retain their
persisted stamps across resume.

## Accuracy boundary

Worker scheduling changes computation placement only. It does not alter the
frozen policy, particle count, MCSE guards, question-selection objective,
tie-breaking, RNG stream, stopping statuses, interval reporting, or downstream
cut classification. Helper failure is bounded and fails closed to exact serial
recomputation from untouched authoritative state.

The Python reference remains the scientific contract. Cross-engine exact
parity is asserted only in the fixed-cloud/no-resampling harness because the
Python and browser RNG implementations intentionally differ; each engine also
has its own deterministic full-session golden.
