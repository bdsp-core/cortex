# CORTEX Web — status & handoff

> **CURRENT STATE (2026-07-07).** The sections below are the original
> 2026-05-29 build-session brief and are kept for history — several details are
> now stale. What is true today:
> - **DEPLOYED and live** at <https://app.cortexeeg.org> (EC2 + Caddy serving
>   the SPA + EEG bundle; uvicorn as a pure API; Postgres). Phase 6 "not yet
>   deployed" below is obsolete.
> - **Monorepo layout**: the SPA is `apps/web/` (engine, trainer, src, ui) and
>   the backend is `services/api/` (was `server/`). Any `server/` path below is
>   stale — see [`LOCAL_DEV.md`](LOCAL_DEV.md) for current paths + commands.
> - **Tests**: backend `services/api/test_server.py` = 117; research = 4;
>   frontend vitest = ~101 cases (engine drift-guards + trainer + src).
> - Since the original brief: Postgres backend + psycopg pool, routers/ split,
>   deep health probe, adaptive trainer (test→train→retest), cohorts, i18n (8
>   langs), Google sign-in, SES email, and the 2026-07-07 efficiency/robustness
>   pass (see `docs/OPTIMIZATION_PASS_2026-07.md`).

Last updated: 2026-05-29 (overnight build session). Cold-start brief so a
future session can pick up the browser port without re-deriving anything.
Pairs with [`PLAN.md`](PLAN.md) (architecture/design) — this file is "where we
actually are."

## TL;DR

A browser version of the CORTEX adaptive EEG certification test (desktop:
`../cortex_app`). The SMC particle-filter engine runs **client-side in a Web
Worker**; the UI is React; the EEG bundle is static (S3/CloudFront in prod,
Vite `public/` locally); a tiny FastAPI+SQLite backend does auth + results.

**The whole flow runs locally today** — landing → login → consent →
registration → tutorial → adaptive test → results — backed by a real auth +
results database. Two test suites green: engine 45, backend 11, plus 9
frontend unit tests (54 JS total).

Status by phase (PLAN §10):

| Phase | State |
|---|---|
| 1. Engine port (TS) | ✅ done |
| 1b. Engine validation vs Python | ✅ 45/45 tests, typecheck clean |
| 2. Data bundle (h5 → browser) | ✅ script + 300-seg local bundle |
| 3. Viewer (EEG + spectrogram + answer panel) | ✅ done (1–6 pick-and-advance, spec marker, counter, confidence) |
| 4. Flow screens (Landing/Login/Consent/Registration/Tutorial/Computing/Results) | ✅ done |
| 5. Auth + results API | ✅ FastAPI+SQLite; JWT; admin CLI; 11 tests; crash-safe retry |
| 6. AWS deploy | ⏳ scaffolded (Dockerfile + README steps); not yet deployed |
| 7. Fidelity polish + pilot | ⏳ in progress (see gaps) |

## How to run locally (full flow)

```bash
cd cortex_web
pip install -r server/requirements.txt   # fastapi + uvicorn (auth/db are stdlib)
npm install
./scripts/run_local.sh                    # builds SPA, mints a demo login, serves :8000
# → open http://localhost:8000, log in with the demo credentials it prints
```

Hot-reload dev: `./scripts/dev.sh` (Vite :5173 + FastAPI :8000, /api proxied).

Other commands:
- `npm test` — full vitest suite (engine 45 + frontend 9). No Python needed.
- `python -m pytest server/test_server.py` — backend (11).
- `npm run build` — production build + typecheck gate → `dist/`.
- `python -m server.admin gen --count N --out codes.csv` — mint credentials.
- `python -m server.admin export-results --out results/` — pull results JSON.
- `npx tsx engine/demo_session.ts` — headless strong/borderline/at-chance demo.

## What's built

### Engine (`engine/`) — faithful TS port, validated (unchanged this session)
Φ/logΦ/logsumexp numerics, seeded xoshiro256\*\*, lapse-probit likelihood
(λ=0.025), hierarchical MVN prior, SMC update + MH rejuvenation, A-optimal
`choose_item`, AD6 policy, the adaptive loop, the Web Worker. See PLAN §9 for
the validation story. N_PARTICLES=600, MAX_QUESTIONS=300.

### Data (`scripts/prepare_web_bundle.py`)
h5 `iiic/` bank + `iiic_segment_signals.csv` + `Sigma_l_fitted.npy` +
`cert_config.yaml` → `public/bundle/<version>/{manifest.json, seg/*.eeg,
seg/*.spec}`. New `--bank PATH` flag targets any bank. **Local bundle is now
the 300-seg `build-data-v2` bank** (50/class) — confident verdicts, not the
REFER-skew the old 100-seg bank gave. Bundle + banks are gitignored.

### Frontend (`src/`, `ui/theme.ts`)
- `App.tsx` — flow state machine: landing → login → consent → registration →
  tutorial → loading → running(Viewer) → computing → done(Results) → error.
- `components/` — Landing (wordmark + static-EEG backdrop), Login (code+pw),
  Consent ("Before You Begin"), Registration (3-page wizard; fields mirror the
  desktop participant record), Tutorial, Computing (spinner), Results (per-task
  verdict table + technical π/R/n grid), Viewer (Canvas EEG/Spec), `ui.tsx`
  shared primitives.
- `api.ts` — backend client: login(JWT), getManifest, createSession,
  postProgress (per-trial, fire-and-forget), submitResults (persist→POST→retry).
- `sampleSession.ts` — fresh difficulty-stratified ~500-sample per sitting.
- `progress.ts` — `resolutionConfidence = min_k max(π_k, 1−π_k)` (posterior).
- `bundle.ts` + `idbcache.ts` — lazy EEG/spec loader, IndexedDB-cached by
  bundle version (survives reload; degrades to network).
- `engineClient.ts`, `dsp.ts`, `montage.ts`, Canvas components.

### Backend (`server/`)
- `security.py` — PBKDF2-SHA256 password hashing + HS256 JWT, **stdlib only**
  (no PyJWT/bcrypt). Signing secret from env or a persisted dev file.
- `db.py` — SQLite (WAL): participants / sessions / trials / results.
- `app.py` — `POST /api/auth`→JWT; gated `GET /api/manifest`, `POST /api/session`,
  `POST /api/progress`, `POST /api/results`; admin endpoints behind
  `X-Admin-Token`; CORS for Vite dev; serves `/bundle` + the SPA `dist` so one
  uvicorn runs everything. `admin.py` CLI, `run.py` entry, `test_server.py` (11).
- Runtime artifacts (`*.db`, `.jwt_secret`, `codes.csv`, `results/`) gitignored.

### Deploy (`Dockerfile`, `scripts/run_local.sh`, `scripts/dev.sh`)
Single image: node build → python+uvicorn. README has the S3+CloudFront (SPA +
bundle) + container (API) steps and env vars.

## The 500-question pool — current state & the one remaining data step

The per-session sampler draws a fresh ~500-question subset each sitting, so the
shipped pool must EXCEED 500 for the "rarely-repeats" property. **The local
bundle is 300 segments** (the largest prebuilt bank), so today the sampler uses
all 300 each sitting (confident verdicts, but two sittings see the same 300 in
a different order — not yet the rare-repeat property).

To get a true >500 pool, the EEG+spectrogram payload must come from the
~100 GB `eeg_bank_spec.h5`. It is **not on disk** (canonical home: the external
SSD `/Volumes/Extreme SSD/`, not mounted this session) and also lives at
`s3://bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5` (99.5 GiB). I did
NOT pull 100 GiB unsupervised (egress cost). Everything else is ready — when the
payload is available, one command builds the pool and one rebuilds the bundle:

```bash
# Option A — mount the SSD, then:
python scripts/build_cortex_test_bank_v2.py \
    --per-class 150 --spec "/Volumes/Extreme SSD/eeg_bank_spec.h5" \
    --out data/eeg_bank_pool.h5            # 150×6 = 900-segment pool
# Option B — pull the 99.5 GiB spec bank from S3 first (one-time egress):
aws --profile opendata s3 cp \
    s3://bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5 /tmp/eeg_bank_spec.h5
python scripts/build_cortex_test_bank_v2.py --per-class 150 \
    --spec /tmp/eeg_bank_spec.h5 --out data/eeg_bank_pool.h5

# then regenerate the bundle from the bigger pool:
python cortex_web/scripts/prepare_web_bundle.py --version v1.1-local \
    --bank data/eeg_bank_pool.h5
```

`build_cortex_test_bank_v2.py` is now parameterized (`--per-class`, `--spec`);
`prepare_web_bundle.py` takes `--bank`. No code change needed — just the data.

## Known gaps / fidelity TODOs

1. **>500 pool** — see above (data step, not code).
2. **Aggregate pass/fail probability** — the displayed metric is the interim
   `min_k max(π_k,1−π_k)` (per the user's "for now"). The full intent is a
   forward Monte-Carlo rollout of P(eventually pass-or-fail, whichever larger);
   deferred because a correct rollout re-runs the adaptive loop per question
   (heavy in-browser) and needs validation vs the desktop. Documented in
   `src/progress.ts`.
3. **Visual fidelity not browser-verified** — the flow screens were built to
   the `ui/theme.ts` spec but NOT yet eyeballed in a browser side-by-side with
   the desktop. First morning task: open the app and check palette/spacing/fonts.
4. **DSP not scipy-exact**; **laplacian montage = average fallback** (neighbour
   sets not ported); **Palatino face** not bundled (CSS stack only).
5. **Strong examinee takes ~259 q on the 300 pool** — expected (adaptive
   exploration with 50/class); a bigger pool shortens this.
6. *(resolved)* A headless-Chrome click-through now exists: `npm run ui-smoke`
   drives landing→login→consent→registration→tutorial→viewer→keyboard-answering
   in real Chrome (Playwright). Still no visual/pixel diff vs the desktop.

## Recently done (this session)
- Per-session stratified 500-sampler (`sampleSession.ts`) + posterior
  resolution-confidence readout (replaced the budget heuristic).
- Parameterized `build_cortex_test_bank_v2.py` (`--per-class`, `--spec`).
- FastAPI+SQLite backend (auth, session, progress, results, admin CLI, 11 tests).
- Full onboarding flow + API wiring; crash-safe result retry; IndexedDB cache.
- Deployment scaffolding (run scripts, Dockerfile, README).
- Pulled the 300-seg bank, rebuilt the bundle → confident verdicts.

## Mobile (phone companion surface)

Phones get a SEPARATE module tree, `apps/web/src/mobile/` (2026-07-10):
auth (shared `AuthFlow` — the email verify/reset deep links are usually
opened on a phone) + dashboard status + past results. The certification
test and training are **desktop-gated by design**: the EEG/spectrogram task
was calibrated (ℓ*/σ*) at desktop scale, so phone-scale sittings would
change task difficulty under the certification's claims. Selection happens
once at boot (`src/device.ts`: coarse pointer AND screen short side <768px;
`?desktop=1` / `?mobile=1` overrides), the mobile chunk is lazy-loaded, and
the desktop↔mobile import boundary is ENFORCED by
`src/mobile/boundary.test.ts` (mobile may import only its own tree + an
explicit shared allowlist; nothing outside `main.tsx` may import mobile).
Growing the phone surface = adding files under `src/mobile/` only.

## Open decisions
- Result-delivery channel (backend stores in SQLite + admin export; PLAN
  mentioned S3 — mirror to S3 in prod if desired).
- ~~Session resume across a browser refresh mid-test~~ — SHIPPED 2026-07-10
  (Resume/Start-over prompt; deterministic engine replay of the per-trial
  server checkpoints; `engine/resume_replay.test.ts`).
- Whether `cortex_web/` splits into its own repo once it grows.

## Where this sits in the broader project
- **`cortex_app/`** — shipped desktop app (the reference implementation).
- **`engine/` + `scripts/cortex_*.py` + `calibration/`** — the certified
  Python engine this port mirrors.
- **`data/eeg_bank.h5`** (gitignored) — test bank; 300-seg on the
  `build-data-v2` GitHub release + `s3://bdsp-opendata-credentialed/eeg-test/`.
- **`bdsp-core/ilae-skill-certification-learning`** — separate learning workstream.
