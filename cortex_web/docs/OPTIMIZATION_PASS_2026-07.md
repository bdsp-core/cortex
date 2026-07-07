# CORTEX web — efficiency & robustness pass (2026-07-07)

A gated, regression-tested pass over the production `cortex_web` app
(app.cortexeeg.org) for efficiency, robustness, and intuitive organization of
the backend and frontend. Driven by a three-part audit (backend `services/api`,
frontend `apps/web`, cross-cutting infra/docs).

Each gate was landed independently and gated on the existing test suites:
backend `python -m pytest api/test_server.py` (from `services/`), frontend
`npx vitest run` + `npx tsc --noEmit` + `npm run build`, research
`pytest research`. No production behavior changed except where noted; all
constants and the validated engine/trainer code were left untouched (the
`engine/` and `trainer/` drift-guards still pass byte-identically).

## Baseline

- Backend: 105 tests → **117** (12 added this pass).
- Frontend: single `index.js` = **536 KB / 170 KB gzip** → **349 KB / 109 KB
  gzip** main chunk (−36% first paint) plus per-locale + Cohorts + standalone-
  page lazy chunks.
- Research: 4 tests (unchanged, still green).

## G1 — Backend robustness

- **Account-existence oracle closed.** `/forgot` and `/verify/resend` let an
  SMTP/SES send error propagate → a 500 for real accounts vs an instant 200 for
  unknown emails, leaking which emails exist. Both now catch, log, and 200
  regardless (matching the already-hardened `/register`).
  `routers/auth.py`.
- **`/api/videos` meta validation.** Malformed JSON / missing or non-3-int
  `shape` now returns 400 instead of an unhandled 500. `routers/videos.py`.
- **Point-batch validation.** `POST /api/trajectories` and
  `/api/training-progress` validated `int(p["taskK"])` mid-transaction — a
  missing key 500'd and discarded the whole batch. A `_validate_points`
  precheck now 422s a bad batch up front and caps size at 1000 (413).
  `routers/dashboard.py`.
- **Orphaned render temp-dirs reclaimed.** Video-job dirs (`cortex_viz_*`) are
  in-memory only, so a restart leaked them forever. A once-per-process orphan
  sweep now reclaims dirs older than the job TTL. `routers/videos.py`.
- **API request-body caps.** A middleware rejects declared `Content-Length`
  over 4 MB (64 MB for `/api/videos`) before buffering, mirroring the Caddy
  edge limits so the API self-protects when reached directly. `app.py`.

## G2 — Backend efficiency

- **Dashboard no longer parses every historical result blob.** `/dashboard`
  and `/regimen` called `list_results_for_code` (JSON-parses *every* completed
  attempt's ~370 KB blob) then used only `[0]`. Switched to a `LIMIT 1`
  `latest_result_for_code`. `routers/dashboard.py`, `db.py`.
- **Training-monitor N+1 removed.** `_graduation_safety` looped
  `list_results_for_code` once per learner. Replaced with a single
  `WHERE code IN (...)` query. `db.py`.
- **Cohort-list N+1 removed.** `list_cohorts` fetched full member rows per
  cohort to count actives; replaced with one grouped `active_member_counts`.
  `routers/cohorts.py`, `db.py`.
- **Pool ceiling raised** from `max_size=4` to a `CORTEX_PG_POOL_MAX`-driven
  default of 10 (sync routes run on Starlette's ~40-thread pool). `db.py`.
- **Video blob parse off the event loop.** The only `async` route expanded two
  ~64 MB float64 arrays inline, stalling the loop on the single-worker box; now
  `run_in_threadpool`. `routers/videos.py`.

## G3 — Transactional integrity for `/results`

`POST /api/results` did three separate write units (store-result, finalize-
session, write-trajectory). A crash between them left a `results` row on a
session still `in_progress` (finished_utc NULL) — and on Postgres NULLs sort
*first* under `ORDER BY … DESC`, so that half-state shadowed the participant's
true latest attempt. Now one `store_result_finalized` transaction, plus
`NULLS LAST` in the latest/list ordering as defense-in-depth. `db.py`,
`routers/testing.py`. Covered by `test_results_writes_are_atomic`.

## G4 — Frontend robustness

- **Engine worker never leaked.** `EngineClient.dispose()` existed but was never
  called; each re-take spun up a new `Worker` and old ones survived sign-out. A
  `disposeClient()` now terminates on done/error/re-take/idle-logout/sign-out.
  `App.tsx`, `engineClient.ts`.
- **`worker.onerror`/`onmessageerror`** wired — an uncaught worker throw that
  posts no error message now surfaces instead of hanging the test.
- **"Computing…" strand fixed.** `savePending` (localStorage) could throw on
  quota/private-mode and reject `submitResults`, stranding the user after the
  result was computed. `savePending` is guarded; `submitResults` now delivers
  the result directly (not via a storage round-trip) and `onDone` is wrapped to
  route any throw to the error screen. `api.ts`, `App.tsx`. Covered by
  `api_resilience.test.ts`.
- **Segment-fetch failures surface.** `Viewer`/`SpikeViewer` had a `.then` with
  no `.catch` → unhandled rejection + permanent "loading EEG…". Now caught with
  a retryable error message.
- **Cache-poisoning fixed.** `idbcache`/`bundle` fetched without checking
  `res.ok`, so a transient 404/500 error body was decoded as EEG *and written to
  IndexedDB*. Both now require a 2xx before decode/cache.
- **Double-submit guards** on Settings email/password; the practice
  start→finish bridge moved off a `window` global to a `useRef`.

## G5 — Frontend efficiency

- **Code-splitting.** The whole UI was one 536 KB chunk. Now: the 4 standalone
  pages (Privacy/Terms/Citation/Report) and Cohorts are `React.lazy`; the 7
  non-English locales (~155 KB) load on demand via a per-locale dynamic import
  (English stays eager as the fallback). Main chunk 536 → **349 KB** (170 →
  **109 KB gzip**). `main.tsx`, `Shell.tsx`, `i18n/LanguageProvider.tsx`.
- **Duplicate dashboard fetch coalesced.** Shell + DashboardSurface both fetch
  `/dashboard` on first paint; an in-flight promise cache collapses them into
  one round-trip (clears on settle — dedup, not a stale cache). `api.ts`.
- **Chart memoization.** `Ring`/`Sparkline`/`Heatmap` wrapped in `React.memo` so
  selecting a mastery tile no longer re-runs all 7 tiles' SVG geometry.
  `charts/index.tsx`.
- **`EegCanvas` redraw** keyed on explicit deps instead of the whole `props`
  object (every parent render was a full canvas redraw). `EegCanvas.tsx`.
- **Training-start waterfall** — independent `createRegimen` + `startSession`
  now run in parallel. `App.tsx`.

## G6 — Frontend organization

- **Dead code removed**: `Login.tsx`, `Tutorial.tsx`, `trainerClient.ts` (no
  importers), and the unused `api.ts` exports `getManifest`, `health`,
  `appendTrajectories`, `listTrainingSessions` (+ orphaned types).
- **Duplicate `Item` interface** (declared in Viewer *and* SpikeViewer)
  consolidated into `bundle.ts`.
- **`Shell.tsx`** shrank ~1900 → ~1528 lines: the ~370-line `SHELL_CSS` string
  moved to `components/shell/shellCss.ts`; Cohorts lazy-loaded.

## G7 — Backend organization + env docs

- Stale run-command docstrings fixed (`server/...` → `api/...`) in
  `test_server.py` and `admin.py`; admin CLI now warns its `gen`/`add` accounts
  can't sign in through the email-gated `/api/auth`.
- `services/api/.env.example` completed: added the previously-undocumented
  `CORTEX_SPACING_*`, `CORTEX_TRAINING_*`, `GOOGLE_CLIENT_ID`,
  `CORTEX_REPORT_TO`, `CORTEX_PG_POOL_MAX`; removed the contradictory "prod runs
  700" note.
- The duplicated 7-task ontology (`backfill_eval_trajectories.py`) now imports
  the single source from `dashboard_logic.CANONICAL_TASKS`.

## G8 — Infra hardening

- **Deploy readiness gate.** `deploy_app.sh` now polls `GET /api/health?deep=1`
  after restart in both modes and fails the deploy (non-zero exit) on an
  unhealthy boot — the 2026-06-24 "process up, DB down" class that the old
  ungated `curl … || true` reported as success.
- **Caddyfile drift warning.** `deploy_app.sh` warns (does not auto-overwrite —
  the live file is hand-maintained) when `Caddyfile.template` differs from the
  live `/etc/caddy/Caddyfile`.
- **systemd unit** gained `MemoryHigh`/`MemoryMax`/`TasksMax`/`LimitNOFILE`
  (so a `/api/videos` render burst is OOM-killed in isolation, not the whole
  box), a `StartLimitBurst` restart-loop backstop, and uvicorn
  `--timeout-keep-alive`/`--limit-concurrency`.
- **`provision.sh`** now writes commented email/Google/report placeholders (a
  fresh box shipped broken signup + Google login silently) and a next-step to
  fill them in.
- **`backup_to_box.sh`** gained a `flock` single-instance lock and a dump
  size-sanity check *before* the retention prune (a truncated-but-exit-0 dump
  could evict good snapshots).
- **Dockerfile** bumped to `python:3.12-slim` (matches prod) and installs from
  `requirements.lock` when present.

## G9 — Docs

- `LOCAL_DEV.md` rewritten with the monorepo paths/commands/test counts.
- `STATUS.md` given a current-state banner (deployed; monorepo; real counts).
- This close-out.

## Deferred follow-ups (out of scope for a no-big-bang, test-gated pass)

- **Full surface-by-surface `Shell.tsx` split** (DashboardSurface,
  SettingsSurface, etc. into `components/shell/*`). High value but risky with
  zero component-render tests — do it *after* adding jsdom/@testing-library
  render coverage, per the incremental-edits convention.
- **Verdict→style single source** (`chipFor` / `VERDICT_STYLE` / `VERDICT_VAR`
  produce three related-but-different outputs). Consolidating risks subtle
  visual regressions without visual tests.
- **`api.ts` domain split** (auth/account/cohorts/dashboard/results/videos).
- **Client read-cache** for tab-switch refetches of `getRegimen`/`getHistory`
  (needs invalidation-on-write; deliberately not a blind TTL cache).
- **`config.py` env centralization** — several vars are still read ad-hoc in
  `app.py`/`deps.py`/routers.
