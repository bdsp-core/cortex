# CORTEX Web

Browser port of the CORTEX adaptive EEG certification test. The SMC particle
filter + adaptive item selection run **client-side in a Web Worker** (no
per-question server round-trip), the EEG bundle is pulled once into IndexedDB,
and the UI is pixel-identical to the desktop app (`../cortex_app`).

**Read [`PLAN.md`](PLAN.md) first** — full architecture, the engine-port plan,
data format, concurrency/auth model, deployment, validation, and build order.

## Status

End-to-end working app: engine + full participant flow + backend. What's here:

```
PLAN.md                       the design doc (start here)
engine/                       TypeScript SMC engine (ported from engine/core_mcmc.py)
  mathfns.ts                  Φ, logΦ (log_ndtr), logsumexp  — the delicate numerics
  rng.ts                      seedable xoshiro256** + gaussian + multinomial resample
  linalg.ts                   fixed-size chol / inv / slogdet / cov / eigh-sqrt
  likelihood.ts               lapse-probit response model (λ=0.025)
  prior.ts                    hierarchical zero-mean MVN (Corr_l, both blocks)
  particles.ts                state / update / ess / resample + MH rejuvenation
  choose_item.ts              A-optimal expected-loss item selection
  policy.ts                   AD6 termination + verdicts
  session.ts                  the adaptive loop (async answer flow)
  worker.ts                   Web Worker entry + message protocol
  types.ts                    shared interfaces
src/                          React SPA
  App.tsx                     flow state machine (landing→…→results)
  api.ts                      backend client (auth / session / progress / results)
  bundle.ts + idbcache.ts     lazy EEG/spec loader, IndexedDB-cached by version
  sampleSession.ts            fresh difficulty-stratified 500-sample per sitting
  progress.ts                 posterior resolution-confidence readout
  components/                 Landing, Login, Consent, Registration, Tutorial,
                              Computing, Results, Viewer (+ Canvas EEG/Spec), ui
server/                       FastAPI + SQLite backend
  security.py                 PBKDF2 password hashing + HS256 JWT (stdlib-only)
  db.py                       participants / sessions / trials / results (WAL)
  app.py                      auth + manifest + session + progress + results + admin
  admin.py                    CLI: mint credentials, export sessions/results
  run.py                      `python -m server.run` dev entry
  test_server.py              11 tests (security + full API round-trip)
ui/theme.ts                   exact colors / fonts / geometry / montages / shortcuts
scripts/prepare_web_bundle.py h5 bank → browser bundle (manifest + int16 EEG + uint8 spec)
scripts/run_local.sh dev.sh   one-process serve / hot-reload dev
Dockerfile                    single-image build (SPA + API)
```

## Run it locally (before AWS)

One process serves the SPA, the EEG bundle, and the API on `:8000`:

```bash
cd cortex_web
pip install -r server/requirements.txt   # fastapi, uvicorn (auth/db are stdlib)
npm install
./scripts/run_local.sh                    # builds the SPA, mints a demo login, serves :8000
# open http://localhost:8000 — log in with the printed demo credentials
```

Hot-reload dev (Vite on :5173 proxying /api → FastAPI on :8000):

```bash
./scripts/dev.sh                          # open http://localhost:5173
```

Mint real participant credentials and export results:

```bash
python -m server.admin gen --count 100 --prefix cortex --out codes.csv  # one CSV row per participant
python -m server.admin export-sessions --out sessions.csv
python -m server.admin export-results  --out results/                   # one JSON per session
```

## Tests

```bash
npm test                                  # vitest: engine (45) + frontend unit (9)
python -m pytest server/test_server.py    # backend (13) — security + full API round-trip
npm run ui-smoke                          # headless-Chrome click-through of the whole flow
```

`ui-smoke` boots the server, mints a participant, and drives real Chrome
through landing → login → consent → registration → tutorial → viewer →
keyboard answering (via Playwright + the already-installed system Chrome).
Playwright is a dev dependency; it does not download a browser
(`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install` if a fresh install tries to).

## Data bundle

```bash
# from the repo root, with the desktop test bank present at data/eeg_bank.h5:
python cortex_web/scripts/prepare_web_bundle.py --version v1.1-local
# → cortex_web/public/bundle/v1.1-local/{manifest.json, seg/*.eeg, seg/*.spec}
```

The per-session sampler draws a fresh ~500-question subset each sitting, so the
shipped POOL must exceed 500. The local bank is 100 segments; build a larger
difficulty-spanning pool from the spec payload (on the external SSD), then
re-run prepare_web_bundle.py:

```bash
python scripts/build_cortex_test_bank_v2.py \
    --per-class 200 --spec /Volumes/Extreme\ SSD/eeg_bank_spec.h5 \
    --out data/eeg_bank.h5          # 200×6 = 1200-segment pool
python cortex_web/scripts/prepare_web_bundle.py --version v1.1-local
```

## Deploy (AWS)

Two pieces (PLAN §3): static SPA + bundle on **S3 + CloudFront**, API on a small
instance/Lambda.

```bash
# 1. SPA + bundle → S3 (edge-cached; identical bytes for everyone)
npm run build && aws s3 sync dist/   s3://<spa-bucket>/   --profile <p>
aws s3 sync public/bundle/ s3://<bundle-bucket>/bundle/ --profile <p>
# point CORTEX_BUNDLE_URL at the CloudFront origin so the SPA fetches it from the CDN

# 2. API (one process; auth + results only — no per-question load)
docker build -t cortex-web . && docker run -p 8000:8000 \
    -e CORTEX_JWT_SECRET=... -e CORTEX_ADMIN_TOKEN=... \
    -e CORTEX_BUNDLE_URL=https://<cloudfront>/bundle/v1.1-local \
    -e CORTEX_CORS_ORIGINS=https://<spa-domain> cortex-web
```

Env: `CORTEX_JWT_SECRET` (sign tokens), `CORTEX_ADMIN_TOKEN` (gate admin API),
`CORTEX_BUNDLE_URL`, `CORTEX_CORS_ORIGINS`, `CORTEX_DB`, `CORTEX_TOKEN_TTL`.

## Why a Web Worker

The engine is light (N=600 particles, K=6 tasks) but stateful and
latency-sensitive. Running it in a dedicated worker keeps the UI at 60 fps and
lets us compute the *next* question while the participant reads feedback, so
selection latency is invisible. 100 simultaneous users worldwide is a
non-issue because each runs its own engine locally — the only shared
infrastructure is a CDN for the (identical, edge-cached) data bundle plus a
tiny auth/results API. See PLAN.md §1, §7.

## Relationship to the desktop app

The engine is a direct port of `../engine/core_mcmc.py`,
`../scripts/cortex_policy.py`, `../scripts/cortex_engine_inputs.py`, and
`../scripts/session_controller.py`. It is validated to be statistically
equivalent (not bit-identical — different RNG) to the Python engine via the
suite in PLAN.md §9. The desktop remains the reference implementation.
