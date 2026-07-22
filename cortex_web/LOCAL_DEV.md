# Local development — CORTEX web

The supported development stack is the `cortex_web` monorepo: a React/Vite SPA
and deterministic TypeScript certification engine, plus a FastAPI service with
SQLite for local state. Production uses the same source with PostgreSQL and an
externally managed EEG bundle.

## Prerequisites

- Python 3.11
- Node 20
- Git
- Chromium only when running browser smoke tests

From the repository root:

```bash
cd cortex_web
npm ci
python3.11 -m venv .venv
.venv/bin/pip install -r services/api/requirements.lock
```

Use `npm ci`, not `npm install`, when reproducing CI or a release candidate.

## EEG bundle boundary

Real EEG banks and production bundle blobs are governed external artifacts and
are intentionally absent from Git. Do not copy a production bank into a source
archive.

Authorized developers can build a versioned bundle from an approved HDF5 bank:

```bash
.venv/bin/python apps/web/scripts/prepare_web_bundle.py \
  --bank /approved/path/eeg_bank.h5 \
  --version v1.6-k7-35k \
  --include-spike
```

Place or mount the resulting version under `apps/web/public/bundle/` only for
local development. Validate the manifest hash against the governed production
record. For UI/browser smoke without restricted data, use the supported
synthetic smoke bundle instead.

## Development servers

Hot reload:

```bash
apps/web/scripts/dev.sh
```

This starts Vite on port 5173 and the local API on port 8000, with `/api`
proxied by Vite. For a production-shaped local build:

```bash
apps/web/scripts/run_local.sh
```

The scripts create/use local SQLite state and never require AWS.

## Required checks

The normal self-contained gate is:

```bash
CORTEX_QUALITY_PYTHON=.venv/bin/python npm run quality
```

It runs ESLint, TypeScript type checking, Vitest, the FastAPI/Python suite, the
production build, CSP verification, and the high-severity dependency audit.

The browser release gate additionally needs Chromium:

```bash
CORTEX_QUALITY_PYTHON=.venv/bin/python \
CORTEX_BROWSER_GATES=1 \
CORTEX_SMOKE_SYNTHETIC=1 \
  bash scripts/quality_gate.sh
```

Focused commands:

```bash
npm run lint
npm run typecheck
npm test
.venv/bin/python -m pytest -q
npm run build
npm run -w cortex-web worker-smoke
npm run -w cortex-web ui-smoke
```

Full-bank benchmarks and sanitized production-session replays are separate
governed gates. Supply their external fixture paths explicitly; never commit
those fixtures.

Visual regression tests require a production build and local Chromium:

```bash
npm run -w cortex-web visual-smoke
```

Update visual baselines only for an intentional reviewed UI change.

## Repository ownership

```text
cortex_web/
  apps/web/                    SPA and deterministic certification engine
  services/api/                FastAPI, auth, persistence, and trainer host
  learning-engine-cleaned/     runtime-authoritative server trainer
  research/                    non-serving export and analysis utilities
  deploy/                      release, backup, and rollback automation
  docs/                        current architecture and operations records
```

The browser trainer is a thin server-session adapter; it has no local learning
model fallback. The root `trainer-policy/` package is a reviewer representation,
not a serving dependency.

## Local state reset

The default local database is `services/api/cortex.db`. Stop local processes
before removing that file and its `-wal`/`-shm` companions. This destroys local
development accounts and sessions only; never apply the procedure to a
production database.

## Deployments

`main` is the release branch. Deploy only from a clean checkout whose `HEAD`
exactly matches `origin/main`:

```bash
bash cortex_web/deploy/scripts/deploy_app.sh
```

The deployment script stages an immutable release, backs up persistent state,
builds on the target, and activates only after health checks. See
`deploy/README.md`; do not deploy from the dirty research workspace.
