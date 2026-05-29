# CORTEX Web — status & handoff

Last updated: 2026-05-27. Written as a cold-start brief so a future session
(or person) can pick up the browser port without re-deriving anything. Pairs
with [`PLAN.md`](PLAN.md) (the architecture/design) — this file is "where we
actually are."

## TL;DR

A browser version of the CORTEX adaptive EEG certification test (desktop:
`../cortex_app`). The SMC particle-filter engine runs **client-side in a Web
Worker**; the UI is React; the EEG bundle is static (S3/CloudFront in prod,
Vite `public/` locally). **It runs locally today** — engine validated, a
working Viewer slice renders real EEG + spectrogram and drives a full
adaptive session to a verdict screen. No AWS or Python needed at runtime.

Status by phase (PLAN §10):

| Phase | State |
|---|---|
| 1. Engine port (TS) | ✅ done |
| 1b. Engine validation vs Python | ✅ 45/45 tests, typecheck clean |
| 2. Data bundle (h5 → browser) | ✅ script + local bundle generated |
| 3. Viewer (EEG + spectrogram + answer panel) | ✅ runnable first cut |
| 4. Flow screens (Landing/Consent/Registration/Tutorial/Computing) | ⏳ not started |
| 5. Auth + results API | ⏳ not started |
| 6. AWS deploy | ⏳ not started |
| 7. Fidelity polish + pilot | ⏳ in progress (see gaps) |

## How to run locally

```bash
cd cortex_web
npm install                                              # once
# generate the data bundle from the desktop test bank (needs ../data/eeg_bank.h5):
python scripts/prepare_web_bundle.py --version v1.1-local
npm run dev                                              # → http://localhost:5173
```

Other commands:
- `npm test` — engine validation suite (45 tests; self-contained, no Python).
- `npm run build` — production build (also the typecheck gate; emits to `dist/`).
- `npx tsx engine/demo_session.ts` — headless: prints n_questions + per-task
  verdicts for strong/borderline/at-chance simulated raters on the real bank.

## What's built

### Engine (`engine/`) — faithful TS port of the Python engine, validated
Ported from `../engine/core_mcmc.py`, `../scripts/cortex_policy.py`,
`../scripts/cortex_engine_inputs.py`, `../scripts/session_controller.py`.
- `mathfns.ts` — Φ, logΦ (`log_ndtr`, stable left tail), logsumexp. The
  numerically delicate part; erfc is the ~1e-7 NR form (Cody upgrade noted in
  PLAN §11 if 1e-12 ever needed).
- `rng.ts` — seedable xoshiro256\*\* + gaussian + multinomial resample. Seed =
  SHA-256(sessionId) like the desktop. NOT bit-identical to NumPy (by design).
- `linalg.ts` — fixed-size chol/inv/slogdet/cov + Jacobi eigh-sqrt fallback.
- `likelihood.ts` — lapse-probit `P(y=1|z)=λ+(1−2λ)Φ(z)`, λ=0.025.
- `prior.ts` — hierarchical zero-mean MVN, `Corr_l` for both t- and l-blocks.
- `particles.ts` — state / update / ess / multinomial-resample + 15-step MH
  rejuvenation (proposal_scale 2.38/√(2K)).
- `choose_item.ts` — A-optimal expected-total-posterior-variance selection +
  Q1 top-N variation. (Full-grid scan; the desktop's coarse-to-fine is a
  speed-only optimization not needed at N=600 / bank≤500.)
- `policy.ts` — AD6: π_k pass-mass, mcse, info-gate R_k; N_min=15, R*=0.30,
  α=0.10, Z=2.0; PASS/FAIL/REFER_BORDERLINE/REFER_UNINFORMATIVE.
- `session.ts` — the adaptive loop reshaped for the browser's async answer
  flow (binary-Y reduction Y=1 iff pick==k). N_PARTICLES=600, MAX_QUESTIONS=300.
- `worker.ts` — Web Worker entry + message protocol (init/answer/abort →
  item/trial/done/error).
- `__testdata__/gen_reference.py` + `reference.json` — ground truth dumped
  from the REAL Python engine.
- `mathfns.test.ts`, `pipeline.test.ts`, `session.test.ts` — the validation
  suite (the session test runs full adaptive sessions on the real bank or a
  synthetic fallback).
- `demo_session.ts` — headless demo runner.

### Data (`scripts/prepare_web_bundle.py`)
h5 bank (`../data/eeg_bank.h5` `iiic/` group) + `iiic_segment_signals.csv` +
`Sigma_l_fitted.npy` + `cert_config.yaml` → `public/bundle/<version>/`:
- `manifest.json` — engine inputs (taskCodes/Labels, ellStar[6], corrL,
  per-seg sMean/sSd/patternClass/channelNames/fsHz) + segment metadata.
- `seg/<id>.eeg` — int16 LE (nCh×nSamp), µV×4.
- `seg/<id>.spec` — uint8, sdata quantized to [-10,25] dB.
Bundle is **gitignored** (`public/bundle/`) — regenerate locally; on prod it
lives on S3.

### UI (`src/`, `ui/theme.ts`)
- `ui/theme.ts` — exact palette/fonts/geometry/montage pairs/jet LUT/gain
  ladder/keyboard shortcuts extracted from the desktop viewer.
- `src/bundle.ts` — manifest + lazy int16-EEG/uint8-spec loader (in-memory
  cache; IndexedDB persistence is a TODO).
- `src/engineClient.ts` — main-thread wrapper around the worker.
- `src/dsp.ts` — zero-phase Butterworth bandpass + RBJ notch (filtfilt).
- `src/montage.ts` — bipolar (exact pairs + separators + EKG) / average;
  **laplacian falls back to average** (neighbour sets not yet ported).
- `src/components/EegCanvas.tsx` — Canvas EEG (per-row offset, ±clip·gain,
  gain µV = 1 row-unit, time grid + pan, scale bar).
- `src/components/SpecCanvas.tsx` — Canvas spectrogram (4 region panels, jet
  LUT, uint8→dB, no recompute).
- `src/components/Viewer.tsx` — the screen: spectrogram-left/EEG-right, 6 IIIC
  answer buttons + Confirm, montage/gain/bandpass/notch/window controls,
  keyboard 1–6/Enter/arrows/Ctrl. Controlled (props: bundle, item, onAnswer).
- `src/App.tsx` + `src/main.tsx` — shell: loads the bundle, owns the engine
  client + session state, renders Viewer then a verdict panel on done. Boots
  straight into the Viewer (no flow screens yet).

## Known gaps / fidelity TODOs (the "not yet pixel-perfect" list)

1. **Flow screens not built** — Landing/Consent/Registration(3 pages)/Tutorial/
   Computing-results. Specs are in `ui/theme.ts` + the desktop
   `eeg_bank_viewer.py`. App currently boots into the Viewer.
2. **DSP not scipy-exact** — `dsp.ts` is a zero-phase Butterworth+notch that
   looks right; exact `scipy.butter(N=2)` coefficient parity is a refinement.
3. **Laplacian montage** = average fallback; port the per-channel neighbour
   sets from `eeg_bank_viewer.py` (~L91-103).
4. **Fonts/spacing** not pixel-matched (Palatino branding face, exact paddings).
5. **EEG vertical fit** — 20 montage rows in a fixed canvas height; row
   spacing may need tuning so labels don't crowd.
6. **DC offset** — some bank channels have large DC means (~2000 µV); the
   default 0.5 Hz highpass removes it, but with bandpass "off" the traces will
   be off-screen. Acceptable (matches raw), but worth a note in the UI.
7. **IndexedDB persistence** — currently in-memory cache only; add IndexedDB
   so a reload doesn't re-download, plus prefetch-ahead.
8. **Results screen** is minimal (verdict list); the desktop has a richer
   skill/bias narrative + "technical details" grid + optional MP4s.
9. **Local bank is 100 segs** → verdicts skew REFER_BORDERLINE (≈16 q/task,
   right at N_min=15). The 300-seg production bank reaches confident PASS/FAIL.
   Generate from the 300-seg bank for realistic local testing.

## Recently fixed
- **Flat EEG bug (2026-05-27):** `channel_names`/`fs_hz` are attrs on the
  `eeg30s` *dataset*, not the group — data-prep read them from the group →
  empty channel names → montage produced all-zeros → flat traces. Fixed to
  read from the dataset attrs; regenerate the bundle and refresh.
- **NaN EEG samples** → `nan_to_num` before the int16 cast.
- **Stray emitted .js** → tsconfig `noEmit`, build uses `tsc --noEmit`.

## Roadmap (next, in order)
1. Visual check in-browser; tune EEG row spacing/fonts to match the desktop.
2. Build the flow screens (phase 4).
3. IndexedDB loader + prefetch (phase 4).
4. Auth + results API — local FastAPI stub first (phase 5), then S3 results.
5. AWS deploy — S3+CloudFront for SPA+bundle, EC2/Lambda for the API (phase 6).
6. Generate the 300-seg production bundle; pilot dry-run.

## Open decisions
- Result-delivery channel (desktop uses Dropbox; web plan is POST /results→S3).
- Session resume across a browser refresh mid-test (default: restart).
- Whether `cortex_web/` splits into its own repo (`ilae-skill-certification-web`)
  once it grows — same move as the learning workstream.

## Where this sits in the broader project
- **`cortex_app/`** — the shipped desktop app (PyInstaller; releases
  `cortex-v1.0`…`v1.1.5`; CI in `.github/workflows/cortex-release.yml`). The
  reference implementation.
- **`engine/` + `scripts/cortex_*.py` + `calibration/`** — the certified
  Python engine this port mirrors.
- **`data/eeg_bank.h5`** (gitignored) — the test bank; on the
  `build-data-v2` GitHub release + `s3://bdsp-opendata-credentialed/eeg-test/`.
- **`bdsp-core/ilae-skill-certification-learning`** — the separate
  training/learning-with-feedback workstream.
