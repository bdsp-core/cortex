# CORTEX Web — design & build plan

Porting the CORTEX adaptive EEG certification test from the PyQt6 desktop
app (`scripts/eeg_bank_viewer.py` + `engine/` + `scripts/cortex_*.py`) to a
browser app that:

- runs **almost entirely client-side** (the SMC particle filter + adaptive
  item selection happen in the browser, not on a server) for imperceptible
  per-question latency,
- downloads **up to ~1 GB** of EEG (≤ 500 questions' worth) into the browser
  once per session, cached locally,
- uses **Web Workers** so the engine compute never blocks the UI,
- looks **pixel-identical** to the desktop app,
- supports **up to 100 simultaneous participants worldwide**,
- is **password-protected**.

This document is the architecture + build plan. The repo already contains a
scaffold (`engine/`, `scripts/prepare_web_bundle.py`, project config) — see
§10 for what's built vs. pending.

---

## 1. Why client-side

The desktop app's whole value is that each answer instantly updates a
600-particle SMC posterior over the examinee's 6-task skill/bias, and the
next EEG is chosen by an A-optimal design over that posterior. If we kept the
engine on a server, every question would pay a network round-trip for
selection + scoring → visible latency, and the server would have to hold
per-user particle clouds for 100 concurrent sessions.

Instead: **ship the engine to the browser.** The server then does almost
nothing per-question — it only authenticates, serves the (static, cacheable)
EEG bundle, and ingests the final results. That is what makes "100
simultaneous users worldwide, low latency" trivial: the heavy, stateful,
latency-sensitive work is distributed onto each participant's own machine,
and the only shared infrastructure is a CDN (for the data) plus a tiny API.

The engine is light enough for this: **N = 600 particles, K = 6 tasks.** A
question costs one `choose_item` scan over the shrinking bank (~300→0
candidates × 6 tasks, coarse-to-fine) and, when ESS drops below 0.5·N, a
15-step Metropolis rejuvenation that replays the answer history. At human
answering pace (one question per ~10–30 s) this is milliseconds of compute —
and we hide even that by computing the *next* item while the participant
reads feedback.

---

## 2. Architecture at a glance

```
┌─────────────────────────── Browser (per participant) ──────────────────────────┐
│                                                                                  │
│  React SPA (main thread)                    Engine Web Worker                    │
│  ─────────────────────────                  ──────────────────                   │
│  • Landing / Consent / Registration         • SMC particle cloud (t,l ∈ R^600×6) │
│  • Tutorial overlay                          • choose_item  (A-optimal design)    │
│  • Viewer: EEG <canvas> + spectrogram        • update + ESS + MH rejuvenation     │
│    + answer panel + controls                 • AD6 termination policy             │
│  • Results screen                            • emits: nextItem, trialDiag, done   │
│         │  postMessage(answer)  ▲                     │                           │
│         └───────────────────────┴─────────────────────┘                          │
│                                                                                  │
│  IndexedDB  ◀── one-time download of the EEG bundle (≤1 GB), cached              │
└──────────────────────────────────────────────────────────────────────────────┘
            │ HTTPS                          │ HTTPS (once at start / once at end)
            ▼                                ▼
   ┌──────────────────┐            ┌──────────────────────────────┐
   │ CloudFront + S3  │            │ API (EC2 t3.small or Lambda) │
   │ static EEG bundle│            │ • POST /auth  → JWT          │
   │ + SPA assets     │            │ • GET  /manifest (gated)     │
   │ (edge-cached)    │            │ • POST /results (gated)      │
   └──────────────────┘            └──────────────────────────────┘
                                              │
                                              ▼  results JSON → S3 (or RDS)
```

**Three deployables:**
1. **SPA + engine** — static files (HTML/JS/WASM-free), hosted on S3+CloudFront.
2. **EEG data bundle** — pre-converted binary, hosted on S3+CloudFront, pulled once per session into IndexedDB.
3. **API** — a tiny service (auth + manifest + results ingest) on EC2 or Lambda.

The EC2 instance the user asked about hosts the API and (optionally) origin-serves the SPA; CloudFront sits in front for global low latency + concurrency.

---

## 3. Tech stack (decisions, not options)

| Concern | Choice | Why |
|---|---|---|
| Frontend framework | **React + TypeScript + Vite** | Fast build, first-class Web Worker support, trivial static hosting. |
| EEG rendering | **Canvas 2D** (single full-width canvas, redraw on pan/gain/montage) | Matches pyqtgraph's look; 20 ch × ≤6000 samples redraws in <5 ms. WebGL (regl/PixiJS) only if profiling demands — it won't at these sizes. |
| Spectrogram | **Canvas 2D** from the precomputed `sdata` array, jet LUT, fixed [-10, 25] dB | Pixel-match the desktop; no recompute needed (bundle ships `sdata`). |
| Engine | **Pure TypeScript** SMC port, in a **Web Worker** | 600 particles is trivial; pure TS avoids WASM build complexity. WASM is a documented escape hatch (§7) if a future N≫600 or rejuvenation cost demands it. |
| DSP (bandpass/notch) | **TS biquad (SOS) filtfilt** port of scipy `sosfiltfilt` | Display filtering must match the desktop's Butterworth N=2 + iirnotch. |
| Linear algebra | **Hand-rolled fixed-size** (6×6, 12×12) Cholesky / inverse / slogdet / eigh | Only tiny matrices; avoids a heavy LA dependency in the worker. |
| RNG | **seedable PCG/xoshiro256\*\*** (TS) | Reproducible per-session; need NOT bit-match NumPy (the test must be statistically correct, not byte-identical to Python). |
| Local cache | **IndexedDB** (via `idb`) | Stores the ≤1 GB bundle so a reload doesn't re-download; survives refresh. |
| Auth | **JWT** issued from a participant-credentials table | Simple, stateless, gates manifest + results. |
| Static hosting / CDN | **S3 + CloudFront** | Edge-cached → 100 concurrent worldwide downloads of the bundle is a non-event. |
| API | **FastAPI (Python) on EC2** behind nginx+TLS, *or* API Gateway+Lambda | Low-QPS; either is fine. FastAPI keeps us in the same language as the data-prep + engine reference. |
| Results store | **S3** (one JSON per session) | Mirrors the desktop's per-session-file model; no DB contention. |

---

## 4. The engine port (the crux)

The desktop engine is `engine/core_mcmc.py` + `scripts/cortex_policy.py` +
`scripts/cortex_engine_inputs.py` + `scripts/session_controller.py`. The
browser port lives in `cortex_web/engine/` (TypeScript). Component-by-component:

### 4.1 State
Particle cloud: `t` and `l` as `Float64Array(N*K)` (row-major, N=600, K=6),
plus `w`, `logPrior`, `logLik` as `Float64Array(N)`, plus the answer
`history` (array of `{k, s, y, sSd}`) and cached prior pieces (`SigmaInv`,
`logDet`, Cholesky `L` for both the t- and l-blocks).

### 4.2 Prior — hierarchical MVN
Zero-mean MVN with the **same fitted `Corr_l` (6×6) for both blocks** (the
desktop precedent: `make_state_hier(..., Sigma_l=Corr_l, Sigma_t=Corr_l)`).
`Corr_l` ships in the bundle as a 36-float array. Need: Cholesky (sampling),
inverse + slogdet (log-density). Port of `_precompute_prior_pieces`,
`sample_prior_hier_K`, `log_prior_hier`.

### 4.3 Likelihood — lapse probit
`P(y=1 | z) = λ + (1−2λ)·Φ(z)`, `λ = 0.025`, `z = exp(l_k)·(s + t_k)`,
optionally attenuated by `1/√(1+(e^l·s_sd)²)`. Port of `_log_p_response`
(needs `logΦ` = `log_ndtr` and `logsumexp`) and `_p_response_yes` (needs `Φ`
= `norm.cdf`). **These two special functions are the only numerically
delicate primitives** — §4.8.

### 4.4 Update
`update(state,k,s,y,sSd)` — reweight by likelihood, normalize, append to
history. Direct port; trivial.

### 4.5 ESS + resample + rejuvenate
When `ESS < 0.5·N`: multinomial resample (needs the seeded RNG), then
`mh_rejuvenate` — 15 MH steps with a random-walk proposal whose covariance is
`proposal_scale² · cov(cloud)` (12×12 `np.cov` + Cholesky, with an `eigh`
fallback for non-PD). `proposal_scale = 2.38/√(2K) ≈ 0.687`. Each step
replays the full history likelihood (vectorized over particles). This is the
heaviest op; see §7 for the worker/latency strategy.

### 4.6 Item selection — A-optimal `choose_item`
`_expected_loss_vec` computes expected post-answer **total posterior
variance** over all 12 coordinates at each candidate signal; `choose_item`
picks the `(k, seg_id, s)` minimizing it, over the *remaining* (unserved)
bank. Port `_expected_loss_vec`, `_coarse_to_fine_argmin` (the top-M
coarse-to-fine optimizer), and `choose_item`. The controller supplies
per-task `(s_mean, s_sd, seg_id)` arrays from the manifest (port of
`IIICEngineInputs.as_engine_arrays`). **Q1 special case:** uniform pick among
the top-`FIRST_ITEM_TOPN=10` lowest-loss items (per-examinee variation).

### 4.7 Termination — AD6 policy
Port `AD6Policy`: per task k, pass-mass `π_k = Σ w_i·1[ℓ_k^(i) > ℓ*_k]`,
`mcse_k = √(π_k(1−π_k)/ESS)`, info-gate `R_k = 1 − Var_post(ℓ_k)/Var_prior(ℓ_k)`.
RESOLVED iff `n_k ≥ N_min(15) ∧ R_k ≥ R*(0.30)` and `π_k − Z·mcse_k ≥ 1−α`
(PASS) / `π_k + Z·mcse_k ≤ α` (FAIL), `α=0.10`, `Z=2.0`. Stop when all 6
RESOLVED, or MAX_QUESTIONS (300) hit, or bank exhausted. `finalize_verdicts`
maps leftover PENDING → REFER_BORDERLINE / REFER_UNINFORMATIVE. `ℓ*_k` ships
in the bundle (from `calibration/cert_config.yaml` `ell_star_unified_v13`);
`Var_prior` = `diag(Corr_l)`.

### 4.8 The two special functions
- **`Φ(x)` (normal CDF):** `0.5·erfc(−x/√2)`. Use a high-accuracy `erf`/`erfc`
  (e.g. the W. J. Cody rational-Chebyshev approximation, ~1e-15) — *not* the
  cheap Abramowitz-Stegun 7.1.26 (only ~1e-7, biases confident particles).
- **`logΦ(x)` (`log_ndtr`):** for `x > −5`, `log(Φ(x))`; for `x ≤ −5`, the
  asymptotic `−x²/2 − log(−x) − 0.5·log(2π) + log1p(−1/x² + …)` to avoid
  `log(0)` underflow on confident raters (this is *why* the desktop uses
  `log_ndtr` — clipped `norm.cdf` collapses to 1.0 at |z|≳6 and biases
  posteriors; see the repo's `CLAUDE.md` "what NOT to do").
- **`logsumexp`** — standard max-shift form.
A `engine/mathfns.test.ts` cross-checks these against values dumped from
scipy (§9 validation).

### 4.9 Session controller
Port of `CortexSession.run`: the loop owns `remaining` (de-dup), calls
`choose_item`, posts the chosen item to the UI, awaits the answer (the
binary-Y reduction `Y = 1 iff pick == k` happens here), calls `update`, ESS
check, policy check, repeat. In the browser this is an async loop in the
worker driven by `postMessage` answers. Seed derived from the session id
(SHA-256 → int), exactly as the desktop does.

---

## 5. Data pipeline & format

`scripts/prepare_web_bundle.py` (in this repo) converts the test bank into a
browser-optimized bundle. Inputs (same as the desktop engine):
`data/eeg_bank.h5` (the `iiic/` group), `data/labels/iiic_segment_signals.csv`,
`Sigma_l_fitted.npy`, `calibration/cert_config.yaml`.

**Output bundle layout (uploaded to S3):**
```
bundle/<version>/
  manifest.json         # {version, taskCodes, taskLabels, ellStar[6],
                        #  corrL[36], segments:[{segId, patternClass,
                        #  sMean[6], sSd[6], fsHz, nCh, nSamp, channelNames,
                        #  eeg:"seg/<id>.eeg", spec:"seg/<id>.spec"}]}
  seg/<id>.eeg          # int16 LE, (nCh × nSamp), µV×scale; ~240 KB/seg @20×6000
  seg/<id>.spec         # uint8, (nFreq × nTime × 4 regions) dB-quantized to
                        #   [-10,25]; ~tens of KB/seg  (or omit & recompute)
```
- **EEG as int16** (µV × a fixed scale, e.g. ×4 → 0.25 µV resolution): halves
  the size vs float32, lossless enough for display. 500 segs × 240 KB ≈ 120 MB.
- **Spectrogram as uint8** quantized to the fixed [-10, 25] dB display range:
  the desktop only ever shows it at those levels, so 8 bits/pixel is visually
  lossless. Ships `sdata` so the browser never recomputes (and stays
  pixel-identical). ~tens of KB/seg.
- **Total** for 500 segments: well under the 1 GB ceiling (~150–300 MB), so
  "≤1 GB" has comfortable headroom; we can ship raw float32 EEG instead of
  int16 and still fit if exactness ever matters.
- **Manifest carries the engine inputs** (`ellStar`, `corrL`, per-seg
  `sMean`/`sSd`) so the worker needs nothing else to run.
- **PHI:** the bundle is de-identified signals only (no names/MRNs), same as
  the desktop bank. Served from the credentialed S3 bucket behind auth.

**Download UX:** on session start, the SPA fetches `manifest.json`, then
streams the per-seg blobs into IndexedDB with a progress bar (the desktop has
no analog; this is the one new screen). Subsequent reloads read from
IndexedDB. Optionally prefetch only the first ~50 segs to start fast, then
background-fill the rest (the adaptive test rarely needs all 500).

---

## 6. UI replication (pixel-identical)

The Explore pass extracted the exact spec; `engine/`-adjacent `ui/theme.ts`
encodes it. Highlights the React build must hit verbatim:

- **Palette:** bg `#0b0d12`, cards `#14161c`/`#15171c`, borders
  `#3a3d45`/`#454a55`, focus/hover `#8a8f9b`, text `#eef1f5`/`#dde0e6`/`#aab0ba`/
  `#767b87`, accent `#f5a623`, pass `#7ed391`, fail `#d8806a`, refer-borderline
  `#d4b169`, refer-uninformative `#9aa0ab`.
- **Type:** Palatino for branding (Landing/Consent/Registration headings &
  CORTEX wordmark at 84 pt); system sans for the Viewer + Results. Web font:
  ship a Palatino-equivalent (e.g. *URW Palladio* / `"Palatino Linotype",
  "Book Antiqua", Palatino, serif` stack) so non-Mac browsers match.
- **Screens & geometry:** Landing 980×660 (CORTEX wordmark + static 16-ch EEG
  backdrop in `#2e425e`, "BEGIN ASSESSMENT" 252×50), Consent 980×660
  ("Before You Begin" + Decline/I-Accept), 3-page Registration wizard 980×720
  (exact field lists & dropdown options in the spec), Viewer 1500×950, Results
  fit-to-content. In the browser these become responsive layouts that
  reproduce the proportions (the desktop fixed sizes become max-widths +
  centering).
- **Viewer:** spectrogram LEFT (250–300 px fixed), EEG plot RIGHT (flex); top
  answer bar (6 IIIC buttons `1 · Seizure` … `6 · Other`, orange `#f5a623`
  3 px selection outline, `Confirm ⏎` disabled until selected); bottom control
  row (Montage / Gain / Bandpass / Notch / Window dropdowns + Pan ◀▶ +
  "Show spectrogram"). Info label format verbatim.
- **EEG:** bipolar (default) / average / laplacian montages with the exact
  channel pairs & separator rows from the spec; gain ladder
  `[1,2,7,10,15,20,30,50,70,100,200,300]` µV/div (default 100); ±3×gain clip;
  black traces, red EKG; 1 px; 1-sec + gain scale bar.
- **Spectrogram:** jet 9-stop LUT (exact RGB tuples in the spec), [-10,25] dB,
  0.5–25 Hz, 4 stacked region panels LL/RL/LP/RP.
- **Keyboard:** `1–6` select, `Enter` confirm, `←/→` pan, `↑/↓` gain ladder,
  `Ctrl` cycle montage. App-level key handling so dropdowns don't eat hotkeys
  (the desktop uses an eventFilter; the web uses a top-level keydown handler).
- **Tutorial overlay**, **Computing-results** spinner page, and **Results**
  verdict table (PASS/FAIL/REFER colors, skill/bias narrative, "Show technical
  details" grid) per the spec.

Acceptance: side-by-side screenshots of each desktop screen vs the web screen,
diffed by eye against the spec's hex/px values.

---

## 7. Concurrency, latency, Web Workers

- **Engine in a dedicated Web Worker.** The main thread never computes the
  particle filter, so the UI (EEG pan, montage cycle, answer selection) stays
  at 60 fps no matter what the engine is doing.
- **Hide rejuvenation latency:** the worker computes the *next* `choose_item`
  immediately after `update` while the participant is still reading
  correctness feedback / the next question is rendering. By the time they're
  ready, the next item is already chosen.
- **`SharedArrayBuffer`?** Not needed at N=600 — the per-question compute is
  single-Worker-fast. (If we ever push N≫600 or parallelize particle updates,
  we'd shard particles across multiple workers via `SharedArrayBuffer`, which
  requires COOP/COEP headers from CloudFront. Documented escape hatch, not in
  v1.)
- **100 concurrent users worldwide:** each runs its own engine locally, so
  there is *no shared compute*. The only shared load is (a) the one-time
  bundle download — absorbed by CloudFront edge caching, the same bytes for
  everyone — and (b) two tiny API calls per session (auth + results). A single
  `t3.small` handles thousands of such calls; 100 concurrent is negligible.

---

## 8. Auth, passwords, results

- **Participant credentials:** pre-generate N codes (e.g. `cortex-<8char>` +
  password, or a single-use access token each). Store hashed (argon2/bcrypt)
  in a small table (DynamoDB or a JSON in S3 for the pilot).
- **Flow:** `POST /auth {code, password}` → verify → issue a short-lived
  **JWT** (e.g. 6 h). The JWT gates `GET /manifest` (returns the bundle
  manifest URL — optionally a CloudFront signed URL) and `POST /results`.
- **Results:** at session end the worker serializes the full session
  (participant info, per-trial telemetry incl. reaction times & interaction
  trace, final verdicts, particle-trajectory summary) to JSON; the SPA
  `POST`s it to `/results`, which writes `s3://…/results/<session_id>.json`.
  Mirror the desktop's crash-safety: also persist to IndexedDB after each
  trial and re-POST on reconnect if the final upload failed.
- **No per-question server calls** — auth once, results once. That's the
  whole latency story.

---

## 9. Validation — does the web engine match the desktop?

The web engine must produce the *same adaptive behaviour* as the Python
engine (not bit-identical — different RNG — but statistically equivalent):

1. **Unit:** `mathfns.test.ts` checks `Φ`, `logΦ`, `logsumexp` against a JSON
   of scipy reference values (dump with a tiny Python script) to ~1e-12.
2. **Component:** feed a fixed `(history, prior)` to both engines and compare
   post-update `w`, `π_k`, `R_k`, `choose_item` selection on a fixed cloud
   (seed the TS RNG to a recorded particle set → deterministic compare).
3. **Behavioural:** run the desktop `CortexSession` and the web engine on the
   **same simulated rater** (`make_simulated_y_source`) over many seeds;
   compare distributions of `n_questions`, per-task verdicts, served-segment
   difficulty profiles. Tolerance: verdict-agreement ≥ 99%, median `n_q`
   within ±1.
4. **Golden session:** record one full desktop session's (item, answer,
   verdict) trace; replay the answers through the web engine; assert identical
   verdicts.

---

## 10. Build order (and scaffold status)

**Already scaffolded in this repo (`cortex_web/`):**
- `PLAN.md` (this file)
- `engine/` — TypeScript port: `mathfns.ts` (Φ/logΦ/logsumexp), `linalg.ts`
  (fixed-size chol/inv/slogdet/eigh), `rng.ts` (seeded), `prior.ts`,
  `likelihood.ts`, `particles.ts` (state/update/ess/resample/rejuvenate),
  `choose_item.ts`, `policy.ts` (AD6), `session.ts` (the loop),
  `worker.ts` (the Web Worker entry).  [first cut — see file headers for TODO]
- `scripts/prepare_web_bundle.py` — h5 bank → browser bundle.
- `ui/theme.ts` — the exact palette/typography/geometry constants.
- `package.json`, `tsconfig.json`, `vite.config.ts`, `README.md`.

**Phased plan from here:**
1. **Engine correctness** — finish the TS port; pass the §9 validation suite.
   *This is the highest-risk item; do it first.*
2. **Data bundle** — run `prepare_web_bundle.py` on the v1.1 bank; upload to S3;
   wire IndexedDB download + cache.
3. **Viewer** — Canvas EEG + spectrogram + answer panel + controls, pixel-matched.
4. **Flow screens** — Landing / Consent / Registration / Tutorial / Computing /
   Results, pixel-matched.
5. **Auth + results API** — FastAPI; participant credential table; JWT; S3 results.
6. **Deploy** — S3+CloudFront for SPA + bundle; EC2 (or Lambda) for API; TLS;
   COOP/COEP only if SharedArrayBuffer is ever needed.
7. **Validation + pilot** — golden-session test; 1-participant dry run; then open.

**Repo placement:** scaffolded as `cortex_web/` inside
`ilae-skill-certification-test-multi` for now (sibling to `cortex_app/`). When
it grows, split into its own repo `ilae-skill-certification-web` — same move
we made for the learning workstream.

---

## 11. Risks & open questions

- **`logΦ` tail accuracy** — the single most important numerical detail; a
  lazy `Φ` clip silently biases confident-rater posteriors (the desktop hit
  exactly this; see repo `CLAUDE.md`). Mitigation: Cody erf + asymptotic
  `log_ndtr`, unit-tested to 1e-12.
- **RNG divergence** — the web engine won't bit-match Python. Acceptable: the
  test is statistically defined, not byte-defined. Validated by §9.3/9.4.
- **Rejuvenation cost growth** — MH replays the full history each step, so the
  last questions of a 300-question session replay 300 observations × 600
  particles × 15 steps. Still ~ms in TS, but if it ever stutters, (a) precompute
  the next item during feedback (already planned), or (b) cap history replay
  via a sufficient-statistic cache.
- **Font fidelity** — true Palatino isn't web-safe; ship a metrically-close
  free face (URW Palladio) and verify the wordmark visually.
- **Bundle size vs. start latency** — 150–300 MB is fine on broadband but slow
  on poor links. Mitigation: prefetch first ~50 segs, background-fill the rest.
- **Result delivery** — desktop uses Dropbox; web uses `POST /results` → S3.
  Confirm that's the desired collection channel (vs. keeping Dropbox).
- **Open:** session resumption across a browser refresh mid-test (IndexedDB has
  the cloud trajectory; do we restore or restart?). Default: restart per the
  desktop's per-session model; revisit if dropouts are common.
