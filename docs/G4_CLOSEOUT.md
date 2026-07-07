# G4 CLOSE-OUT — web app wiring (staging)

Status: **staging exit criterion MET** — backend loop + client bridge + the
immersive UI runner all wired and code-verified; behavioral (visual/interaction)
verification of the runner needs a live-SPA Playwright/manual pass (§4).
Companion to `docs/TRAINER_INTEGRATION_PLAN.md` §G4.

## 1. What was already there (the scaffolding G4 fills)

The training DB tables (`regimens`, `training_sessions`, `training_trials`,
`param_trajectories` with `phase` + `is_real`), the session-lifecycle endpoints
(`GET /regimen`, `GET/POST /trajectories`, `GET/POST /training-sessions`,
`/finalize`), and the `is_real=1`-only trajectory read were all present and wired.
The two real gaps: **nothing wrote `training_trials`** (so retests couldn't exclude
trained segments) and there was **no way to create a regimen or persist real
trainer output**. The client trainer engine (`apps/web/trainer/`, G3) existed but
was not imported by the SPA.

## 2. Backend wiring (done + e2e-verified)

- **`db.record_training_progress(code, training_id, points)`** — the missing
  writer: per training trial it writes an idempotent `training_trials` exposure row
  (its `seg_id` then feeds the existing `get_exposure_exclusion`, so retests never
  re-show a trained segment — D-INT-4/7) **and** a REAL `param_trajectories` row.
  **`is_real`, `phase`, and `code` are set server-side** from the authenticated
  session — never trusted from the client (anti-tamper). `db.training_session_owner`
  is the ownership gate.
- **`POST /api/regimen`** — builds + activates a regimen from the latest cert
  result: one track per NON-PASSED task, carrying its ℓ*.
- **`POST /api/training-progress`** — persists trainer output for an OWNED training
  session (404 otherwise).
- The exam draw (`POST /api/session`) already excludes recently-seen segs via
  `get_exposure_exclusion` (test + train union), so **the retest exclusion now works
  end-to-end** once training_trials are written.

**Staging exit test — `test_g4_train_retest_loop_staging`** (in
`services/api/test_server.py`, SQLite + `TestClient`): one simulated participant
runs cert → regimen → train → **fresh retest, ×3**. It asserts: the regimen weak
set = the non-PASSED tasks; real (`is_real=1`, `phase='train'`) exposure +
trajectories are persisted; **every trained segment is excluded from every
subsequent retest draw**; and a raw synthetic trajectory post (no `isReal`) is
quarantined (never surfaces). Passing. Full backend suite green (no regression).

## 3. Client bridge (done + typecheck/test-verified)

- **`src/trainerClient.ts`** — the main-thread client for the trainer Web Worker
  (`trainer/worker.ts`), the trainer analogue of `engineClient.ts` (init / next /
  submit → item / done / state / error).
- **`src/trainerBank.ts`** — the data bridge: a web bundle manifest → the trainer's
  per-task `ArrayBank` candidates (SDT-frame labels y* = 1[s>0], matching the G2
  Python `SimBankAdapter`), cut-scores (σ* = exp(−ℓ*), σ_∞ above the cut), and
  seeded initial clouds. Unit-tested.
- **`src/api.ts`** — `createRegimen()` + `postTrainingProgress()` (with the
  `isReal`-implied server contract).

`tsc --noEmit` = 0 errors across the whole app; the trainer + bridge vitest tests
pass; the existing SPA tests are unregressed.

## 4. UI runner (done + typecheck/test-verified; behavioral pass pending)

The immersive learning-protocol runner, built to the CORTEX aesthetic (four UX
decisions confirmed with the user: full-screen immersive container, split
result-reveal step, minimal progress, fixed 120-item daily bound):

- **`src/components/TrainingRunner.tsx`** — a full-screen session that reuses the
  exam Viewer's look (`cx-test`, EEG + spectrogram canvases via `EegCanvas`/
  `SpecCanvas`, montage/gain/window/pan controls, sharp edges, teal actions) but is
  FEEDBACK-DRIVEN: each trial is a binary one-vs-rest **"Is this <pattern>?"** (Yes/No,
  keyboard Y/N/1/2), which maps directly to the trainer's belief update. Answering
  advances to a distinct **RESULT step** (✓/✗ in pass/fail color + the true label,
  Enter/Space to continue — the D-INT-1 feedback reveal the feedback-free exam
  deliberately lacks). Minimal on-screen progress (count / 120); belief + telemetry
  stay under the hood.
- **`src/trainingController.ts`** — a pure, DOM-free `question→result→done` state
  machine over a synchronous `TrainerSession` (the per-trial compute is light enough
  to run on the main thread; the worker is not needed for the runner). Collects
  trajectory points (ℓ/θ/sd/rt/seq per trial); unit-tested (`trainingController.test.ts`).
  Trajectory is batched to `POST /api/training-progress` (every 15 trials + on exit)
  and the session `finalize`d on completion; on `allMastered` the done screen nudges
  toward the retest.
- **`App.tsx`** — a new `training` phase. Dashboard "Start training" (`onStartTraining`,
  formerly a no-op) now runs `createRegimen` → `startTrainingSession` → a candidate
  draw (`startSession`, spacing-aware + media) → builds the `TrainerSession` from the
  manifest (`buildTrainerBank`/`cutScores`/`seedClouds`) → renders `TrainingRunner`.
- The regimen / trajectory / "Daily training" surfaces already render and light up
  automatically as real data lands (they read `is_real=1` trajectories).

`tsc --noEmit` = 0 errors; the trainer + controller + bank vitest suites pass. What
is NOT yet done: a **behavioral pass on the live SPA** (visual layout, canvas render,
keyboard/click interaction, real-network persistence) — this needs Playwright or a
manual browser session and cannot be exercised headlessly in this environment.

## 5. Deviations from the plan (noted)

- **Exposure ledger.** The plan called for refactoring `get_exposure_exclusion`
  into a Postgres `NoRepeatPolicy` (count-window) ledger with `domain_ordinal`. I
  instead **kept the existing days/sessions-window exclusion** (which already unions
  test + train and is proven) and closed the gap by *writing* `training_trials`. The
  hard retest exclusion (D-INT-4) is achieved and verified; the count-window policy
  (D-INT-7) is a config swap available as a follow-up. Rationale: changing the
  exam's live spacing semantics is riskier than reusing the proven exclusion.
- **Production bank into the exam + thin-domain budget audit** (carried from G2)
  remain follow-ups.

## 6. Exit

Staging loop wired + the e2e staging test passes (DB rows correct, `is_real`
quarantine holds, spacing verified across 3 retests) ✅; client bridge + data layer
+ API done and typecheck/test-clean ✅; the immersive UI runner built + wired into
`App.tsx`, `tsc` clean, trainer/controller/bank suites green ✅. Remaining before
G5: a behavioral (Playwright/manual) pass on the live SPA to verify the runner's
visuals + interaction. Then **G5** — the K=1–3 prod pilot (PI sign-off + real
users; not autonomously completable). **Nothing was deployed** — this is
local/staging wiring.
