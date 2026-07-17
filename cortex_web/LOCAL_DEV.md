# Local development — CORTEX web

Two-process dev (Vite + FastAPI) on your laptop, SQLite as the DB, no AWS
involved. Suitable for everything except "shake-down test the real prod build."

Monorepo layout (Phase B): `apps/web/` is the SPA (engine, trainer, src, ui),
`services/api/` is the FastAPI backend, `research/` is the offline analytics +
de-identified export tooling.

## One-time setup (~5 min)

You need: macOS or Linux, Python 3.11+ (prod runs 3.12), Node 20+, git, ~2 GB
free disk.

```bash
git clone https://github.com/bdsp-core/ilae-skill-certification-test-multi.git
cd ilae-skill-certification-test-multi/cortex_web
npm install
pip3 install -r services/api/requirements.txt
```

That installs the SPA deps and the backend deps (FastAPI, uvicorn, psycopg).
psycopg's binary wheel is only used in prod-style runs; locally the SQLite
branch is the default and works without it.

### Drop in the EEG bundle (one-time)

The test needs the K=7 EEG bank in `apps/web/public/bundle/v1.5-k7/`. It's
gitignored (~325 MB, PHI-adjacent so not in the repo). Easiest path:

```bash
# A. If you have eeg_bank.h5 already on disk (e.g. scp'd from prod):
python3 apps/web/scripts/prepare_web_bundle.py --bank /path/to/eeg_bank.h5 \
    --version v1.5-k7 --include-spike

# B. If you don't, copy from the prod box (needs SSH to cortex-prod):
rsync -avz cortex-prod:/opt/cortex/cortex_web/apps/web/public/bundle/v1.5-k7/ \
    apps/web/public/bundle/v1.5-k7/
```

`prepare_web_bundle.py` produces the manifest + per-segment binaries in ~30 s
if the h5 is local.

## Day-to-day

### Hot-reload dev (the usual)

```bash
apps/web/scripts/dev.sh
```

- Vite on **:5173** with HMR — typing in `apps/web/src/` reloads instantly.
- FastAPI on **:8000** — `--reload` so server edits also bounce automatically.
- `/api/*` calls from the SPA are proxied to :8000 by Vite (no CORS gymnastics).

Open <http://localhost:5173>, click BEGIN, create a test account (any email,
any password ≥ 8 chars), do the test.

### Production-mode local smoke

When you want to test what users will actually see — the built SPA served by
the same uvicorn that handles the API + bundle:

```bash
apps/web/scripts/run_local.sh
```

Hits **<http://localhost:8000>**. Same as `dev.sh` from the user's point of
view, but no HMR — used as the final pre-deploy sanity check.

### Tests

```bash
# Frontend (vitest): engine drift-guards + trainer + src. The engine session
# sims do real adaptive runs, so the full suite takes ~5 min.
cd apps/web && npx vitest run

# Backend API (run from services/, which must be on sys.path):
cd services && python3 -m pytest api/test_server.py -q      # 117 tests

# Research (gold ETL + de-identified export):
python3 -m pytest research -q                                # 4 tests
```

### Visual regression smoke (screenshot diffs)

The "silent rewrite" net: screenshots of the key stable screens (sign-in
desktop/mobile, create-account, forgot-password, /report), pixel-diffed
against baselines committed in `apps/web/scripts/visual_baselines/`:

```bash
cd apps/web && npm run visual-smoke           # compare against baselines
cd apps/web && npm run visual-smoke:update    # after an INTENTIONAL restyle
```

Boots a throwaway server (no EEG bundle needed) and needs a production build
in `dist/` + headless Chrome. Baselines are machine-rendered: regenerate and
review the diff whenever a visual change is intentional.

### Type-check + production build

```bash
cd apps/web && npm run build                 # tsc --noEmit + vite build
```

Hard-fails on any TypeScript error before doing the bundle. Run this before
deploying. The build emits code-split chunks: one main chunk, one async chunk
per non-English locale, and lazy chunks for Cohorts + the standalone pages.

## Branches + PRs

Branch off `main`, push, open a PR. Standard squash-merge to `main`.

`main` is the deploy branch — what's on `main` is what's on prod.

## Deploying to prod (https://app.cortexeeg.org)

If you have SSH to `cortex-prod`, deploy from your laptop in one command:

```bash
bash cortex_web/deploy/scripts/deploy_app.sh
```

That rsyncs your working tree to the box, rebuilds the SPA on-box, restarts the
uvicorn service, and **gates on `GET /api/health?deep=1`** — a bad boot (e.g.
DB unreachable) aborts the deploy with a non-zero exit instead of a false green
check. The bundle on the box is preserved (rsync's `--delete` is scoped to skip
`apps/web/public/bundle/`).

If you DON'T have SSH access yet — open a PR, ping the maintainer, they deploy.

## Where things live

```
cortex_web/
  apps/web/
    engine/       The adaptive engine — particle filter, Φ/logΦ numerics,
                  AD6 stopping, item selection. Drift-guarded against the
                  Python reference (engine/core_mcmc.py up the repo).
    trainer/      The client-side adaptive trainer engine.
    src/          React SPA: App.tsx flow state machine, components/*,
                  api.ts (backend client), transport.ts (retry/outbox).
    ui/theme.ts   Single source of truth for colors, fonts, geometry.
    scripts/      prepare_web_bundle.py (h5 → bundle), dev.sh, run_local.sh.
    public/bundle/  Gitignored. EEG payload; drop the v1.5-k7 tree in here.
  services/api/   FastAPI: auth (PBKDF2 + HS256 JWT), routers/*, db.py
                  (SQLite dev / pooled Postgres prod), admin CLI.
  research/       Offline gold ETL + de-identified export (ilae-export).
  deploy/         systemd units, Caddyfile.template, provision.sh,
                  deploy_app.sh, backup_to_box.sh, deploy README.
```

## When the local DB needs resetting

The local SQLite lives at `services/api/cortex.db`. Wipe and restart fresh:

```bash
rm -f services/api/cortex.db services/api/cortex.db-wal services/api/cortex.db-shm
apps/web/scripts/dev.sh
```

Tables get recreated on next service start (`services/api/db.py`'s schema is
idempotent on both SQLite and Postgres).
