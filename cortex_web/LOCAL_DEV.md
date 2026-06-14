# Local development — CORTEX web

Two-process dev (Vite + FastAPI) on your laptop, SQLite as the DB, no AWS
involved. Suitable for everything except "shake-down test the real prod build."

## One-time setup (~5 min)

You need: macOS or Linux, Python 3.11+, Node 20+, git, ~2 GB free disk.

```bash
git clone https://github.com/bdsp-core/ilae-skill-certification-test-multi.git
cd ilae-skill-certification-test-multi/cortex_web
npm install
pip3 install -r server/requirements.txt
```

That installs the SPA deps and the backend deps (FastAPI, uvicorn, psycopg).
psycopg's binary wheel is only used in prod-style runs; locally the
SQLite branch is the default and works without it being usable.

### Drop in the EEG bundle (one-time)

The test needs the K=7 EEG bank in `public/bundle/v1.5-k7/`. It's
gitignored (~325 MB, PHI-adjacent so not in the repo). Easiest path:

```bash
# A. If you have the eeg_bank.h5 already on disk (e.g. scp'd from prod):
python3 scripts/prepare_web_bundle.py --bank /path/to/eeg_bank.h5 \
    --version v1.5-k7 --include-spike

# B. If you don't, copy from the prod box (needs SSH to cortex-prod):
rsync -avz cortex-prod:/opt/cortex/cortex_web/public/bundle/v1.5-k7/ \
    public/bundle/v1.5-k7/
```

`prepare_web_bundle.py` produces the manifest + per-segment binaries in
~30 s if the h5 is local.

## Day-to-day

### Hot-reload dev (the usual)

```bash
./scripts/dev.sh
```

- Vite on **:5173** with HMR — typing in `src/` reloads the page instantly.
- FastAPI on **:8000** — `--reload` so server edits also bounce automatically.
- `/api/*` calls from the SPA are proxied to :8000 by Vite (no CORS gymnastics).

Open <http://localhost:5173>, click BEGIN, create a test account
(any email, any password ≥ 8 chars), do the test.

### Production-mode local smoke

When you want to test what users will actually see — the built SPA served by
the same uvicorn that handles the API + bundle:

```bash
./scripts/run_local.sh
```

Hits **<http://localhost:8000>**. Same as `dev.sh` from the user's point of
view, but no HMR — used as the final pre-deploy sanity check.

### Tests

```bash
npm test                                    # vitest: 54 engine + frontend tests
python3 -m pytest server/test_server.py -q  # backend API: 17 tests
```

Both run in < 1 minute end-to-end. The session tests do real adaptive runs
against the local bundle, so they take ~45 s on their own.

### Type-check + production build

```bash
npm run build                               # tsc --noEmit + vite build
```

Hard-fails on any TypeScript error before doing the bundle. Run this before
deploying.

## Branches + PRs

Branch off `main`, push, open a PR. Standard squash-merge to `main`.

`main` is the deploy branch — what's on `main` is what's on prod.

## Deploying to prod (cortex-44-233-29-150.nip.io)

If you have SSH to `cortex-prod` (see the Brandon ↔ Eli handshake), deploy
from your laptop in one command:

```bash
bash cortex_web/deploy/scripts/deploy_app.sh
```

That rsyncs your working tree to the box, rebuilds the SPA on-box, restarts
the uvicorn service. ~20 seconds. The bundle on the box is preserved
(rsync's `--delete` is scoped to skip `public/bundle/`).

If you DON'T have SSH access yet — open a PR, ping Brandon, he merges + deploys.

## Where things live

```
cortex_web/
  engine/         The adaptive engine — particle filter, Φ/logΦ numerics,
                  AD6 stopping, item selection. Validated 45/45 against the
                  Python reference (engine/core_mcmc.py up the repo).
  src/            React SPA: App.tsx flow state machine, components/*.
  ui/theme.ts     Single source of truth for colors, fonts, geometry.
  server/         FastAPI: auth (PBKDF2 + HS256 JWT, stdlib-only),
                  endpoints, SQLite/Postgres DB layer, admin CLI.
  scripts/        prepare_web_bundle.py (h5 → bundle), dev.sh, run_local.sh.
  deploy/         systemd units, Caddyfile, provision.sh, deploy_app.sh,
                  backup_to_box.sh, deploy README.
  public/bundle/  Gitignored. EEG payload; drop the v1.5-k7 tree in here.
```

## When the local DB needs resetting

The local SQLite lives at `server/cortex.db`. Wipe and restart fresh:

```bash
rm -f server/cortex.db server/cortex.db-wal server/cortex.db-shm
./scripts/dev.sh
```

Tables get recreated on next service start (server/db.py's schema is
idempotent on both SQLite and Postgres).
