# CORTEX Web

Browser port of the CORTEX adaptive EEG certification test. The SMC particle
filter + adaptive item selection run **client-side in a Web Worker** (no
per-question server round-trip), the EEG bundle is pulled once into IndexedDB,
and the UI is pixel-identical to the desktop app (`../cortex_app`).

**Read [`PLAN.md`](PLAN.md) first** — full architecture, the engine-port plan,
data format, concurrency/auth model, deployment, validation, and build order.

## Status

Scaffold + de-risked engine core. What's here:

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
ui/theme.ts                   exact colors / fonts / geometry / montages / shortcuts
scripts/prepare_web_bundle.py h5 bank → browser bundle (manifest + int16 EEG + uint8 spec)
package.json / tsconfig / vite.config.ts
```

**Not yet built** (see PLAN.md §10 phases 3–7): the React screens (Landing /
Consent / Registration / Tutorial / Viewer / Results), IndexedDB bundle
loader, the auth + results API, deployment. The engine + theme + data-prep
are the foundation those build on.

## Engine quickstart (Node/Vitest)

```bash
npm install
npm run typecheck
npm test          # once engine/*.test.ts validation specs are added (PLAN.md §9)
```

## Data bundle

```bash
# from the repo root, with the desktop test bank present at data/eeg_bank.h5:
python cortex_web/scripts/prepare_web_bundle.py --version v1.1
# → cortex_web/public/bundle/v1.1/{manifest.json, seg/*.eeg, seg/*.spec}
# upload that dir to S3+CloudFront; the SPA fetches manifest.json then the segs.
```

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
