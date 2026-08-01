# Local cortex_web stack — draw-latent (atoms17) through the production UI

Owner-facing runbook for `scripts/local_cortex_stack.sh`: the fully local
cortex_web stack for sitting real sessions against the newly integrated
draw-latent n-way engine through the EXACT production SPA, isolated from the
prod server. Verified 2026-07-31 on branch `nway-f1-promotion-20260731`
(dark-integration commits `5c926a6..7de84fd`).

## Launch / stop

```bash
n-way-protocol/scripts/local_cortex_stack.sh          # (re)start
n-way-protocol/scripts/local_cortex_stack.sh stop     # stop
```

Idempotent: a run first stops its OWN previous instance (pidfile
`local_server/data/.cortex_stack_8788.pid`, killed only after the recorded
pid's cmdline matches `uvicorn api.app:app … --port 8788` — nothing else is
ever killed), then refuses to start if anything else is still serving the
port. It rebuilds `cortex_web/apps/web/dist` when sources are newer than the
built `index.html`, and refuses to serve a dist that does not contain the
draw-latent profile id (the stale-dist guard).

## Ports

| Port | What |
|---|---|
| 8788 | this stack: one uvicorn serving SPA + `/bundle` + `/api` (`CORTEX_SERVE_STATIC=1`, the `apps/web/scripts/run_local.sh` idiom). Override with `CORTEX_STACK_PORT`. |
| 8734 | OCCUPIED by the separate n-way-protocol `local_server` rig — never reused here. |

## Data home (isolation)

- Database: `n-way-protocol/local_server/data/cortex_ui_test.db`
  (SQLite/WAL; selected via `CORTEX_DB`, see `services/api/db.py`). The
  `local_server/data/` dir is gitignored. The dev default
  `cortex_web/services/api/cortex.db` and prod Postgres are never touched.
- `n-way-protocol/server/migration.sql` was NOT applied: the API never reads
  its columns (`engine_profile_id`, `response_model`, …) — grep of
  `services/api` finds no consumer. The served stack stamps the whole
  profile as JSON into the `sessions.nway_profile` column, which
  `persistence/schema.py` + `persistence/migrations.py` already create at
  boot. (migration.sql remains the isolated `local_server/schema.sql`
  specification only.)

## How a session gets the draw-latent engine

Two independent gates, both handled by the launcher:

1. **Per-account: precision_v1 policy rollout** (`policy_rollout.py`,
   consumed in `routers/testing.py::new_session`). The stack runs
   `CORTEX_PRECISION_POLICY_ROLLOUT=email_allowlist` with
   `CORTEX_PRECISION_POLICY_EMAILS=elikeldsen+nway-local@gmail.com`, so ONLY
   that account starts precision_v1 (n-way) sessions; any other local
   account fails closed to AD6 and never receives an nway stamp.
   (`compute_rollout.py` is execution placement only — left `off`, so
   `compute_mode=serial`; it never changes the response model.)
2. **Process-wide: the response-model rollout** (`services/api/
   nway_profile.py`, pinned by `test_nway_profile.py`). The stack runs
   `CORTEX_NWAY_RESPONSE_ROLLOUT=all`, so every NEW precision_v1 session
   is stamped `precision_nway_f1_nesting34_draw_latent_v1`
   (`responseAggregation: "draw_latent"`, artifact
   `iiic-f1-engine-frame-nesting34-20260730`). Off/unknown values fail
   closed to the qualified floor015 mixture profile — this is the same
   fail-closed gate production uses for the staged activation, not a
   local-only escape. (The pre-qualification
   `CORTEX_NWAY_RESEARCH_UNQUALIFIED` escape from commit `18b4423` was
   removed once the qualified path landed.)

The SPA passes the session's `nwayProfile` into the browser engine, and
`engine/nway_likelihood.ts` builds the draw-latent runtime exactly when
`responseAggregation === "draw_latent"`.

## Test account

- `elikeldsen+nway-local@gmail.com` / `nway-local-test-1`
  (display name "NWay Local Tester"; created through the real
  `POST /api/register` flow with `CORTEX_EMAIL_BACKEND=dev` +
  `CORTEX_EMAIL_EXPOSE_CODE=1`, so the 6-digit verify code comes back in
  the API response and no real email is sent).
- The launcher re-creates it only when absent from the dedicated DB.
  Register is rate-limited to 5/hour/IP by the IN-MEMORY limiter
  (`deps.py RATE_LIMITS`) — if you ever trip it, restart the stack (the
  limiter resets with the process); do not hammer the endpoint.

## Verification run (2026-07-31)

Headless Playwright/Chrome drive of the production SPA at
`http://127.0.0.1:8788`: sign-in → dashboard CTA → consent → tutorial →
answered 6 questions of a real sitting. Evidence from the dedicated DB:

- `sessions` row `81b1c376d70c457a9cbdd5512920a542`:
  `status=in_progress_nway`, `termination_policy=precision_v1`,
  `compute_mode=serial`, and `nway_profile` =
  `{"engineProfileId":"precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1",
  "responseAggregation":"draw_latent",
  "responseArtifactId":"iiic-f1-engine-frame-atoms17-rd-20260730",
  "responseArtifactSha256":"b25bd1c5e680b77df394eeccab0961bd1f67c1593ff3d4a144d0b92eb8f90fb8", …}`.
- 6 `trials` rows (trial_index 0–5) with `reaction_ms`, client timestamps,
  and `diag` populated; diag keys include `pi, lMean, tMean, sSd, ess,
  rejuv, nPerTask, domainStatuses, skillIntervals, biasIntervals, verdicts,
  determinations, precisionStreakCounts, guardedPrecisionStatistic,
  responseKind, terminationPolicy`.
- The dist served was built fresh the same day and greps positive for the
  atoms17 profile id (`dist/assets/worker-*.js`,
  `dist/assets/nway_selector_worker-*.js`).

Inspect captured sittings any time:

```bash
sqlite3 n-way-protocol/local_server/data/cortex_ui_test.db \
  "SELECT session_id, status, termination_policy, nway_profile FROM sessions;"
```

## Known traps (all guarded by the launcher)

- **Leaked/stale servers**: the launcher kills only its own pidfile'd
  process and refuses a port that anything else is serving. `python -m
  uvicorn` is started with `exec` so the pidfile records the real server
  pid (a killed wrapper otherwise orphans uvicorn, which keeps the port and
  serves a stale revision).
- **Stale dist**: rebuild-if-newer plus a hard grep for the atoms17 profile
  id before serving.
- **Register rate limit**: account creation is check-first; a stack restart
  resets the in-memory limiter.
- The API suite (`cortex_web/services/api`, 264 tests) passes unchanged
  with this stack's files added; nothing tracked in `cortex_web/` was
  modified.
