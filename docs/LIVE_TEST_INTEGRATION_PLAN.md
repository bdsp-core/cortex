# Live-Test Integration Plan — CORTEX adaptive certification

**Status:** planning draft, implementation landed as commit `6f79119`
(2026-05-22). Author: investigation by four expert agents + synthesis,
2026-05-21. **Rev 2** — engine choice changed to the SMC particle-cloud
engine. **Rev 3 (2026-05-22)** — AD6 resolved as per-task three-way
band; see the AD6 section + `docs/AD6_RESOLUTION.md`.
**Audience:** future implementing agents and the PI.
**Scope:** integrate the adaptive certification engine into the live PyQt6
viewer (`scripts/eeg_bank_viewer.py` — the CORTEX app), capture and store live
test-taker data, produce per-examinee visualizations, and scale from the
200-segment test bank to the real 10,000+-segment bank.

---

## 0. Executive summary

The CORTEX viewer today is a polished but **self-contained demo**: it shows a
static set of 10 pre-picked EEG segments (`pick_examples()`), the answer buttons
flash and advance, and nothing is recorded, scored, or stored. Turning it into a
real adaptive certification instrument has four workstreams:

1. **Engine integration** — let the adaptive engine choose each question.
2. **Live data capture** — record every response, reaction time, and posterior
   state.
3. **Storage** — a crash-safe, PHI-aware per-session store.
4. **Visualization** — a per-examinee particle-collapse animation + overview
   report.

**Recommended architecture (see §3):** the **SMC + MCMC particle-cloud engine**
(`engine/core_mcmc.py`) drives the test — it is the Paper-1 certification
methodology, and its particle cloud makes the collapse animation native rather
than synthesized. It runs in a **background `QThread`**; its `y_source` callback
**blocks on a thread-safe queue** that the GUI fills when the human answers;
live data goes to a **per-session directory** with an append-only JSONL audit
log.

**AD6 (stopping / certificate rule) is now RESOLVED — per-task
three-way band with sample-size + information + probability-band
gates; see `docs/AD6_RESOLUTION.md` for the spec, the four-expert
derivation panel record, and the production-restore checklist for the
internal-test calibration overrides currently in
`scripts/cortex_policy.py`.**

**⚠ Headline blocker (see §4):** every bank segment the engine can serve as a
question needs a **calibrated per-segment signal `s`**. `s` is a latent
case-difficulty parameter fit from *many raters'* labels — it cannot be computed
from the EEG alone. The current `eeg_bank.h5` and the engine's signal bank use
**incompatible `seg_id` namespaces**: 0 of 100 spike segments and 5 of 100 IIIC
segments in the bank have any calibrated signal. **No live test can score a
spike segment today.** Resolving this — for the test bank and then for the 10k
real bank — is Phase 0 and gates everything else.

---

## 1. Goal & scope

Turn `scripts/eeg_bank_viewer.py` from a fixed-content demo into a live,
human-in-the-loop adaptive certification test that:

- selects each next question adaptively from the engine,
- records the test-taker's response, reaction time, and the running posterior,
- stores everything durably and PHI-safely,
- ends with per-task certificates and a results screen,
- produces a per-examinee particle-collapse animation and an overview report,
- scales to the real 10,000+-segment bank.

Out of scope for this plan: re-deriving the calibration math, the Paper-1
methodology, or modifying the engine's numerics.

---

## 2. Current state (findings)

### 2.1 The viewer — `scripts/eeg_bank_viewer.py` (~1700 lines, single file)

- **Page flow** (`main()`): `LandingPage → ConsentPage → RegistrationPage →
  BankViewer`, four separate top-level windows; a `goto()` closure shows the
  next at the previous window's geometry and closes the previous.
- **Segment sequence is frozen at construction.** `pick_examples()` returns a
  static list of 10 seg_ids; `BankViewer` only ever indexes into `self.seg_ids`
  via `self.current_idx`. The hidden `example_box` `QComboBox` is the *de facto*
  cursor — `_step_example()` → `example_box.setCurrentIndex()` →
  `_load_segment()` → `_render()`. Advance wraps **modulo** — after the last
  segment it silently returns to segment 0. **There is no terminal state and no
  results screen.**
- **The answer path is a no-op stub.** `_answer_chosen()` → `_flash_and_advance()`
  flashes the chosen button for 85 ms and advances. Nothing is recorded; there
  is no reaction-time measurement anywhere in the file (`QTimer` is used only
  for the flash).
- **Registration** (`RegistrationPage.commit()` / `_save_row()`) appends an
  8-field identity row to `results/registrations.csv`. There is **no session
  ID**, and the collected row is discarded — never handed to `BankViewer`.
- **Missing for a real test:** engine integration, session management,
  reaction-time capture, audit logging, a stopping rule wired to the UI, a
  results screen, registration→session linkage.

### 2.2 The two engines (D1 architecture — they never import each other)

- **`engine/` — SMC + MCMC particle-cloud engine** (`core_mcmc.py`), Paper-1
  "Mode-A". The posterior over θ = (t_k, ℓ_k) per task is a weighted **particle
  cloud**; `choose_item` selects the next item; `run_session_mcmc_auroc` runs an
  adaptive session and accepts a `y_source` callback; native stopping is AUROC
  **half-width precision** (δ). MCMC rejuvenation re-evaluates the answered
  history when it fires, so per-trial latency rises through a session (handled
  in §3 AD1/§8). **This is the chosen engine** — see AD1.
- **`deployment/` — Laplace / EKF runtime** (`simulate_test.py`), a
  deployment-speed Gaussian approximation of the same model: `select_next_case`,
  `simulate_candidate`, three-way per-task stopping, flat sub-ms latency. Not
  used to drive the live test (it has no particle cloud, so the collapse
  animation would have to be synthesized). Retained as a cross-check / fast
  simulator.
- **The `y_source` contract** — both engines accept
  `y_source(k: int, seg_id: int, s_val: float) -> int` returning the binary
  response. In `core_mcmc.py`, `y = int(y_source(k, seg_id, s))`; a real
  `seg_id` is passed only when `bank_segids` is also supplied. **The viewer must
  become this callable.** The engine loop is a synchronous blocking loop that
  calls `y_source` once per trial.

### 2.3 The data

- `data/eeg_bank.h5` — the **test** bank: exactly 100 IIIC + 100 spike
  segments, monolithic, 175 MB. Groups `iiic/<seg_id>` and `spike/<seg_id>`;
  per-segment `eeg30s` dataset; IIIC segments also carry a precomputed
  `sdata/sfreqs/stimes` 10-min spectrogram.
- Engine inputs: a per-`(task, seg_id)` signal bank with `s_mean`/`s_sd`
  (`data/deployment_prior/case_bank.csv`, and the per-segment IIIC source
  `data/labels/iiic_segment_signals.csv`), the prior `Sigma`/`Sigma_l`, and the
  per-task ℓ* thresholds (`calibration/cert_config.yaml` v13).
- `data/iiic_precompute_manifest.csv` — 69,806 IIIC rows; the natural seed
  manifest for the real bank.

### 2.4 The existing particle-collapse visualization

- `scripts/viz_smc_collapse.py` re-runs the SMC engine's inner loop **from its
  public primitives** (`make_state_hier`, `choose_item`, `update`, `ess`,
  `resample_and_rejuvenate`) on *simulated* gold-standard raters, snapshotting
  the particle cloud after every question → `results/viz/snapshots_<slug>.npz`.
- `scripts/viz_render_collapse.py` renders `mbw_collapse.mp4` (solo, 2×3 panel
  grid) and `four_examinee_collapse.mp4`.
- **`mbw_collapse.mp4` is the output of this exact engine.** The particle-cloud
  engine and the capture/render code already exist and work. What does *not*
  exist is feeding a **live human's** answers into the engine and capturing
  *that* session — the simulated raters have a known fitted truth (the cloud
  collapses toward a gold ★); a live examinee does not.

---

## 3. Architecture decisions

AD1–AD5 are recommendations; **AD6 is now RESOLVED (2026-05-22) —
per-task three-way band; see `docs/AD6_RESOLUTION.md`.** PI sign-off
status: AD1 and AD5 may still need confirmation; AD6 is implemented
under internal-test calibration overrides that must be reverted to
panel-derived production values before public release (revert
checklist in `docs/AD6_RESOLUTION.md`).

### AD1 — Engine: the SMC + MCMC particle-cloud engine (`engine/core_mcmc.py`)

The live test runs on the SMC particle-cloud engine. Rationale:

- It **is** the Paper-1 certification methodology ("Toward Objective
  Certification of Epileptiform-Discharge Identification Skill"); the Laplace
  deployment runtime is a speed approximation of it.
- The posterior **is** a particle cloud, so the per-examinee collapse animation
  is **native** — captured directly from the engine, not synthesized.
- It already exposes adaptive selection (`choose_item`) and the `y_source`
  hook (`run_session_mcmc_auroc`) needed for a live human-in-the-loop session.

Trade-off — **latency**: MCMC rejuvenation re-evaluates the answered history
when it fires, so question-selection late in a session can pause ~a second or
two. Mitigation: the engine runs in a background thread (AD2), and the GUI shows
a brief "selecting next case…" state between questions (the human is reading the
next EEG for far longer anyway). The full 173-question `mbw` session runs in
tens of seconds total — real but not disqualifying at human pace.

(The earlier draft of this plan recommended the Laplace deployment engine for
its flat latency and built-in three-way certificate. That was the wrong call
for this project's priorities — it would have forced the collapse animation to
be synthesized from a Gaussian. Superseded.)

### AD2 — Threading: engine in a background `QThread`, `y_source` blocks on a queue

The engine numerics must run **unchanged** (the engine is invariant-pinned by
drift-guard tests). Therefore:

- The `EngineWorker` runs the adaptive loop inside a worker `QThread`. The loop
  is built from the engine's **public primitives** — `make_state_hier`,
  `choose_item`, `update`, `ess`, `resample_and_rejuvenate` — exactly the
  pattern `viz_smc_collapse.py` already uses (a blessed, engine-modification-free
  pattern), with `simulate_response` replaced by the live `y_source` and a
  snapshot recorded after every question.
- `y_source` is backed by a `queue.Queue(maxsize=1)`; the call
  `y = answer_queue.get()` **blocks the engine thread only** — the GUI thread
  stays live.
- **Engine → GUI** via Qt signals (`itemReady`, `snapshotReady`,
  `testComplete`) — thread-safe, delivered as queued connections onto the GUI
  thread.
- **GUI → engine** via the `queue.Queue` (not a signal — the engine thread is
  blocked inside Python code, not in an event loop).
- Set `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` before the engine thread
  starts (the engine's bit-exact reproducibility contract). The bank h5 handle
  stays **GUI-thread-only** (h5py handles are not thread-safe); the engine
  thread touches only the pre-loaded signal-bank arrays.

### AD3 — Storage: per-session directory, append-only JSONL audit log

Each session writes its own directory `results/sessions/<session_id>/`:
`participant.json` (PHI), `trials.jsonl` (append-only, `fsync` per trial —
crash-safe source of truth), `certificate.json`, `trajectory.npz` (the particle
snapshots). No shared mutable file across stations. An **offline collector**
later merges sessions into a central SQLite/parquet store; SQLite is never
written live (multi-station + network-share + SQLite locking is unreliable).
See §7 for schemas.

### AD4 — The collapse visualization: native particle cloud

Because the engine is the SMC particle-cloud engine (AD1), the per-examinee
collapse animation is captured **directly** — no Gaussian synthesis. The
`EngineWorker` records the particle cloud after every question (the
`snapshots_<slug>.npz` schema in §2.4 is the template). For a **live examinee**:

- there is **no ground truth** — drop `true_t`/`true_l`; the cloud collapses
  toward its own converged estimate;
- replace the gold ★ with the per-task **ℓ\* decision line** and the resulting
  PASS/FAIL/REFER verdict;
- reuse `viz_render_collapse.py` (plasma spread→color, weight-fade alpha) — the
  solo `render_solo` layout is the template.

### AD5 — Binary-Y reduction (PI decision required)

The engine consumes one **bit** per trial; the viewer collects a 6-way IIIC
class or a 2-way spike answer.

- **Spike task** (`k=0`): `Y = 1` if the rater chose "Spike", else `0` — direct.
- **IIIC tasks** (`k=1..6`): each task is a *one-vs-rest* binary judgment. The
  engine selects a specific task `k`; `Y = 1` iff the rater's 6-way choice
  equals task `k`'s class, else `0`. The answer panel stays 6-way (it mirrors
  clinical practice); the controller derives `Y` for the asked task.

This discards information (a 6-way clinical answer is richer than one bit). **It
is a methodological decision for the PI** — confirm before building Phase 2.

### AD6 — Stopping / certificate rule ✅ RESOLVED (2026-05-22)

**The "per-task three-way band" option below was selected — see
`docs/AD6_RESOLUTION.md` for the resolved spec, the gate semantics
(N_min + R* + α/Z bands), the four-expert derivation panel record,
and the production-restore checklist for the internal-test
calibration overrides currently in `scripts/cortex_policy.py`.**
Shipped as commit `6f79119` on `bdsp-core/ilae-skill-certification-test-multi`
`main`. The legacy Mode-A δ-stopping rule below remains available via
the `cortex_policy.default_policy_for` precedence ladder
(`DeltaStopPolicy`), and the never-stop variant (`NoStopPolicy`) is
used by the audit / OC harnesses.

The original AD6 discussion below is preserved as historical context:

The question has two coupled parts: (a) **when the test ends** (per task and
overall), and (b) **how PASS / FAIL / REFER is declared** per task. Candidate
approaches identified so far — *not yet chosen*:

- **Mode-A AUROC-precision (δ-stopping)** — the SMC engine's native rule: keep
  asking until each domain's AUROC posterior half-width < δ (e.g. 0.05), then
  compare the precise estimate to the threshold. Estimates every domain to a
  fixed precision regardless of how clear the pass/fail is.
- **Per-task three-way band** — the deployment D5 rule: stop a task as soon as
  `P(ℓ_k > ℓ*_k)` crosses a pass-band or fail-band; unresolved tasks at a trial
  cap → REFER. Computable directly from the particle cloud (the weighted
  fraction of particles with ℓ_k > ℓ*_k). Stops earlier on clear cases.
- **Mode-B certificate mode** — an SMC cert/gray-zone path that exists in the
  engine but was retired in favour of Mode-A; would need re-validation.

Whichever is chosen also sets the **session length** distribution and feeds the
consent copy ("the assessment is adaptive"). Until it is decided, the
`SessionController` must treat the stopping check as a **single injectable
policy function** `stop_policy(state) -> {continue | pass | fail | refer}` so
the rest of the integration does not depend on the choice.

---

## 4. ⚠ THE CRITICAL BLOCKER — bank / signal reconciliation

This section gates Phases 1–7. **Resolve it first, or in parallel from day one.**
It is engine-independent — the SMC engine needs the same per-segment signals
(consumed as `bank_signals` / `bank_segids` by `choose_item`).

### 4.1 What `s` is and why it is hard

The likelihood at trial `(k, seg)` uses `η = e^ℓ·(s + t)`, where `s` is the
segment's calibrated **signal** (`s_mean`, with uncertainty `s_sd`). `s` is a
latent IRT case-difficulty parameter, **jointly fit from many raters' labels of
that segment**. It is *not* computable from the EEG waveform — a segment with no
multi-rater label history has no `s` and **cannot be used as a question**.

### 4.2 The namespace mismatch (verified)

- The bank's **IIIC** seg_ids (20533–22159): 95/100 are in `case_bank.csv`;
  **all 100** are in `data/labels/iiic_segment_signals.csv`.
- The bank's **spike** seg_ids (≥95327, `source_dataset='morgoth1:spikes'`):
  **0/100** are in any calibrated signal file. The bank's spike segments are a
  **separate seg_id namespace** with no calibrated signals anywhere in the repo.

**Consequence: today the engine cannot serve any spike segment from
`eeg_bank.h5`, and 5 IIIC segments are also unservable.**

### 4.3 Two more bank-data defects (verified)

- **`fs_hz` / `channel_names` are missing** from `eeg_bank.h5` segments (they
  *are* present on the `data/test_h5_mini/` per-segment files). The viewer falls
  back to `fs=200 Hz` — but **spike segments are 128 Hz**, so spike EEG
  currently renders at the wrong time scale, with a wrong duration label and a
  wrong filter design.
- **All 100 spike segments in the test bank are `spike_category=1`** (positive).
  A spike test needs negatives too.

### 4.4 What Phase 0 must deliver

1. A **`seg_id ↔ (domain, h5 path, case_key)` crosswalk** unifying the bank and
   the signal tables into one namespace.
2. **Calibrated `(s_mean, s_sd)` per task for every test-bank segment.** IIIC:
   source from `iiic_segment_signals.csv` (full coverage). Spike: the
   `morgoth1:spikes` segments must be run through the same Phase-3 calibration
   (Rasch → SDT → ×1/1.7, `LAPSE_RATE=0.025`) — this requires multi-rater spike
   labels for those segments.
3. **Spike negatives** added to the test bank.
4. **`fs_hz` + `channel_names`** written onto every segment of the production
   bank (or surfaced via a bank manifest the viewer reads).
5. A **bank manifest** (`seg_id, test_class, pattern_class, per-task
   s_mean/s_sd, fs_hz, h5_path, ...`) so the viewer and engine never iterate the
   h5 to discover content.

### 4.5 The 10,000-segment real bank

The real bank multiplies the problem: **every one of the 10k+ segments needs a
calibrated `s`.** That requires a multi-rater labeling campaign and a
calibration re-run. `data/iiic_precompute_manifest.csv` (69,806 IIIC rows) is the
natural seed manifest for the IIIC side. **This is a data-pipeline project in
its own right and is the true critical path** — engine/viewer wiring can proceed
against the (fixed) 200-segment test bank in parallel, but a real certification
cannot launch until the real bank is calibrated.

---

## 5. Phased implementation plan

Each phase lists concrete tasks. Phases 1–6 can use the **fixed 200-segment test
bank** once Phase 0 has reconciled it; Phase 7 is the real-bank scale-up.

### Phase 0 — Data foundation (PREREQUISITE, see §4)

- [ ] 0.1 Build the `seg_id` crosswalk unifying `eeg_bank.h5`,
      `iiic_segment_signals.csv`, `case_bank.csv`, `segments.csv`.
- [ ] 0.2 Produce per-task `(s_mean, s_sd)` for all 200 test-bank segments;
      run the spike `morgoth1:spikes` segments through the calibration.
- [ ] 0.3 Add spike-negative segments to the test bank.
- [ ] 0.4 Write `fs_hz` + `channel_names` onto every `eeg_bank.h5` segment.
- [ ] 0.5 Emit a bank manifest CSV; build the engine's per-task
      `bank_signals` / `bank_segids` arrays from it.

### Phase 1 — Engine ↔ viewer scaffold

- [ ] 1.1 New `SessionController(QObject)` (new module
      `scripts/session_controller.py`): owns `session_id`, the loaded engine
      inputs (`Sigma`/`Sigma_l`, the per-task `bank_signals`/`bank_segids`, ℓ*),
      the `EngineWorker` + its `QThread`, the GUI→engine `queue.Queue`, the
      audit-log writer, and the **injectable `stop_policy`** (AD6).
- [ ] 1.2 New `EngineWorker(QObject)` moved onto a `QThread`: runs a
      capture-instrumented adaptive loop over the SMC engine's public primitives
      (the `viz_smc_collapse.py` pattern) with a queue-backed `y_source`; emits
      `itemReady(k, seg_id, s_val, trial)`, `snapshotReady(...)`, and
      `testComplete(state)`. Engine numerics unchanged.
- [ ] 1.3 Set the BLAS single-thread env before the engine thread starts; keep
      the bank h5 handle GUI-thread-only.
- [ ] 1.4 `BankViewer` connects `itemReady → show_item`,
      `testComplete → show_results`. Construct the controller in `open_viewer()`
      (load engine inputs off-thread / behind a loading indicator).

### Phase 2 — Answer capture, timing, engine-driven selection

- [ ] 2.1 Reaction-time instrumentation: `time.perf_counter()`; start the clock
      *after paint* via `QTimer.singleShot(0, ...)` following `_redraw()`; stop
      at the first line of `_answer_chosen()`.
- [ ] 2.2 Add an `_awaiting_answer` guard so one item yields exactly one `Y`
      (today `_answer_chosen` is re-entrant during the 85 ms flash).
- [ ] 2.3 Implement the binary-Y reduction (AD5) in the controller.
- [ ] 2.4 `_answer_chosen()` → `controller.submit_answer(raw_choice, rt)` →
      derive `Y`, write the audit row, `answer_queue.put(...)`.
- [ ] 2.5 New `BankViewer.show_item(seg_id, domain, k, trial)` (replaces the
      `example_box`/`_load_segment` list-index path).
- [ ] 2.6 Remove free navigation incompatible with an adaptive posterior:
      delete `_step_example` from the answer path, remove/disable the prev/next
      buttons and the `n`/`p` hotkeys. (Pan ◀▶, gain, montage, filters stay —
      they change the display only.) Retire `example_box` and `pick_examples`
      from the live path (keep `pick_examples` only for the tutorial example).

### Phase 3 — Registration → session linkage

- [ ] 3.1 Generate a `session_id` (UUID4) in `RegistrationPage.commit()`; write
      it as the first column of `registrations.csv`; thread it (and
      `expertise`) through `open_viewer()` into the `SessionController`.
- [ ] 3.2 Write `participant.json` per session (identity quarantined here only).

### Phase 4 — Storage (schemas in §7)

- [ ] 4.1 Create `results/sessions/<session_id>/` at session start.
- [ ] 4.2 `trials.jsonl` — append-only, one JSON object per trial, `flush()` +
      `os.fsync()` after each write.
- [ ] 4.3 `certificate.json` — written once at test end (its decision fields
      depend on the AD6 rule).
- [ ] 4.4 `trajectory.npz` — the particle-cloud snapshots for the visualization.
- [ ] 4.5 `BankViewer.closeEvent` must flush the audit log and signal the engine
      thread to exit (push a sentinel onto `answer_queue`) — handle the
      participant closing the window mid-test.
- [ ] 4.6 (Later) an offline collector merging sessions → `index.sqlite`.

### Phase 5 — Results / end screen

- [ ] 5.1 New terminal `QWidget` shown on `testComplete`: per-task certificate
      (the verdict labels depend on the AD6 rule), in the established CORTEX
      aesthetic.
- [ ] 5.2 Replace the modulo-wrap dead end.

### Phase 6 — Per-examinee visualization (see §6)

- [ ] 6.1 Capture: the `EngineWorker` records each question's particle cloud
      (`t`, `l`, `w`), per-domain AUROC half-widths, and the rejuvenation flag
      → `trajectory.npz` (the `snapshots_<slug>.npz` schema, minus `true_*`,
      plus `ell_star`).
- [ ] 6.2 Collapse animation adapted per AD4 — fork `viz_render_collapse.py`:
      no ground-truth ★, ℓ* decision lines, per-task verdict, K=7 task set.
- [ ] 6.3 Static overview report (one composite figure): per-task certificate
      panel, posterior-over-ℓ_k vs ℓ*, AUROC/half-width convergence,
      reaction-time profile, accuracy-by-task. Adapt `deployment/plot_deploy.py`
      `figure_2` and `scripts/plot_phase1_figure.py` `panel_c`.
- [ ] 6.4 Render post-test as a **background job**; ffmpeg is a hard MP4
      dependency — degrade gracefully (static filmstrip / GIF) if absent, and
      keep rendering non-gating (do not fail the session on a render error).

### Phase 7 — Scale to the 10,000+-segment real bank

- [ ] 7.1 Deploy the bank as **per-segment h5 files + a manifest** (the
      `test_h5_mini/` format) for test stations; keep a monolithic h5 only as
      the archival master. (A 10k IIIC-heavy bank is ~10–15 GB monolithic.)
- [ ] 7.2 The viewer reads the manifest, never iterates the h5; lazy-load only
      the segments actually served.
- [ ] 7.3 Complete the real-bank calibration campaign (§4.5) — the gating
      data-pipeline deliverable.

---

## 6. Per-examinee visualization detail

**Capture (during the session).** The `EngineWorker` runs the capture-
instrumented adaptive loop (Phase 1.2) and, after every question, records the
particle cloud — `t` and `l` particles per task `(N, K)`, the weights `(N,)`,
the per-domain AUROC half-widths `(K,)`, and the rejuvenation flag. This is the
same data `viz_smc_collapse.py` captures for the simulated raters. It is emitted
to the GUI via `snapshotReady` (for any live HUD) and written to
`trajectory.npz` at session end. Reaction time is recorded per trial as a new
field (RT exists nowhere in the codebase today).

**Collapse animation (post-test).** Native, per AD4 — fork
`viz_render_collapse.py`. Per task, the (t_k, ℓ_k) particle cloud collapses as
questions are answered; cloud color encodes spread, particle alpha encodes
weight. For a live examinee: **no ground-truth ★**; overlay the per-task `ℓ*_k`
decision line; annotate the per-task verdict. The K=6 methodology task set in
the current viz code must move to the K=7 deployment task set (spike + seizure +
lpd + gpd + lrda + grda + other). One MP4 per examinee.

**Static overview report (post-test, one composite figure).**

- **Certificate panel** — K=7 grid of per-task verdict badges (verdict, task
  trial count, the relevant posterior summary); colors from
  `plot_deploy.DECISION_COLORS`.
- **Posterior-over-ℓ_k vs ℓ\*** — per task, the final weighted posterior of ℓ_k
  (from the last snapshot's `l`/`w`) with the `ℓ*_k` line and PASS/FAIL shading.
  Adapt `plot_deploy.figure_2`.
- **AUROC half-width convergence** — `max_k` AUROC half-width vs question with
  the δ reference line(s). Adapt `plot_phase1_figure.panel_c`; data = the
  captured `hw` trajectory.
- **Reaction-time profile** — RT per trial / per task (needs 6.1 capture).
- **Accuracy-by-task** — empirical hit rate per task from the recorded
  `(k, Y)` history.

`deployment/plot_deploy.py` is K-agnostic and is the strongest reusable basis;
the population-level Paper-1 figures are not per-examinee and are useful only
for their shared Okabe-Ito plotting style.

---

## 7. Storage schemas

`results/sessions/<session_id>/` contains:

**`participant.json`** (PHI — gitignored, excluded from any release):
`session_id, participant_id` (salted hash of email, links repeat takers),
`schema_version, app_version, engine_version, bank_build_utc, started_utc,
finished_utc, station_id, consent_accepted, consent_text_hash`, an `identity`
block (name/age/gender/institution/email/expertise/credentials), the
`stop_policy` id + parameters used (AD6), the `ell_star` snapshot used.

**`trials.jsonl`** (append-only, one object per line): `session_id,
trial_index, ts_shown_utc, ts_answered_utc, reaction_time_ms, seg_id,
test_class, task_k, task_name, pattern_class_true, s_mean, s_sd, response_raw,
response_label, response_y, is_correct, montage, gain_uv, bandpass, notch,
window_s`, plus a per-trial posterior summary: `ell_post_mean[K]`,
`ell_post_ci[K][2]`, `auroc_post_mean[K]`, `auroc_halfwidth[K]`,
`p_pass[K]` (weighted fraction of particles with ℓ_k > ℓ*_k), `rejuv` (bool),
`ess`, `decision_after[K]`, `n_per_task_after[K]`. (The full particle clouds are
too large per line — they go in `trajectory.npz`.)

**`certificate.json`** (written once at end): `session_id, app_version,
engine_version, finished_utc, total_trials, stop_policy, stopped_reason`, and a
`per_task` list of `{task_name, task_k, decision, n_trials, ell_hat, ell_ci,
auroc_hat, p_pass, ell_star, bank_exhausted}`, plus `all_pass`, `any_fail`. The
exact `decision` semantics depend on the AD6 rule and **must not be hard-coded
until AD6 is settled**. `ell_true` is **omitted** (no ground truth for a human).

**`trajectory.npz`** — the particle-collapse snapshots (the
`snapshots_<slug>.npz` schema, adapted for a live examinee): `t_traj
(T,N,K)`, `l_traj (T,N,K)`, `w_traj (T,N)`, `hw_traj (T,K)`, `rejuv (T,)`,
`q_idx (T,)`, `ell_star (K,)`, `domains (K,)`, `n_q`, `stopped`, `N_particles`;
**no `true_t`/`true_l`** (no ground truth). Per-trial `(k, seg_id, s, Y,
reaction_time_ms)` arrays are included for the overview graphs. Regenerable in
part from `trials.jsonl` — the JSONL remains the crash-safe source of truth.

`registrations.csv` is kept (add a `session_id` column) but treated as PHI; the
per-session-directory design removes the multi-station append-contention risk
for everything except that one file.

---

## 8. Risks & open questions

| # | Risk / question | Severity | Owner |
|---|---|---|---|
| R1 | `s`-signal / `seg_id` namespace mismatch — no spike segment is engine-servable today (§4) | **Blocker** | Data pipeline + PI |
| R2 | The real 10k bank needs a full multi-rater labeling + calibration campaign (§4.5) | **Blocker** | PI / data pipeline |
| R3 | `fs_hz`/`channel_names` missing from `eeg_bank.h5` → spike EEG renders at the wrong sample rate | High | Phase 0.4 |
| R4 | Binary-Y reduction (AD5) discards information vs a 6-way clinical answer — methodological call | High | **PI decision** |
| R5 | Test bank has only positive spike cases — need negatives | High | Phase 0.3 |
| R6 | **Stopping / certificate rule is undecided (AD6)** — when the test ends and how PASS/FAIL/REFER is declared. Build it as an injectable policy; do not hard-code | **High — open** | **PI decision** |
| R7 | SMC rejuvenation latency rises through a session (~1–2 s pauses late on) — mitigated by the background thread + a "selecting…" indicator (AD1) | Medium | Implementer |
| R8 | The engine is invariant-pinned — the adaptive loop must use the public primitives unchanged (AD2 keeps it so); keep BLAS single-threaded | Medium | Implementer |
| R9 | Session length depends on the AD6 rule — could be long for a human; the consent says "adaptive length" but a human-appropriate cap is needed | Medium | **PI decision** (tied to R6) |
| R10 | Abort/resume — is a crashed/resumed session a valid certification? The JSONL supports replay-resume, but policy must decide | Medium | **PI decision** |
| R11 | h5py handles are not thread-safe — the engine thread must never touch the bank h5 | Medium | Implementer |
| R12 | RT "stimulus onset" with pyqtgraph is the post-paint moment — measure after the event loop paints | Low | Implementer |
| R13 | `expertise` is collected at registration but the engine has no per-candidate prior hook beyond the prior `Sigma` — record as metadata unless the PI wants it to inform the prior | Low | **PI decision** |
| R14 | ffmpeg is a hard dependency for the MP4 — degrade gracefully, keep rendering non-gating | Low | Phase 6.4 |
| R15 | The viz code is K=6 methodology domains; the live test is K=7 — the capture/render must move to the K=7 task set | Low | Phase 6 |

**Decisions needed from the PI before implementation starts:** AD1 (engine
choice — confirm), AD5/R4 (binary-Y reduction), **AD6/R6 (stopping & certificate
rule — the headline open decision)**, R9 (max session length, tied to R6), R10
(abort/resume validity), R13 (expertise as prior).

---

## 9. Testing strategy

- **Engine invariance** — the adaptive loop uses the SMC engine's public
  primitives unchanged; the existing engine drift-guard tests must remain green.
  AD2 (thread + queue) touches no engine numerics. Keep BLAS single-threaded so
  bit-exact reproducibility holds.
- **A new `y_source`-equivalence test** — a queue-backed `y_source` fed a fixed
  response sequence must produce the same particle trajectory as a direct
  in-process `y_source` fed the same sequence.
- **`stop_policy` isolation test** — the stopping/certificate logic (AD6) is a
  single injectable function; test each candidate rule independently of the
  GUI/threading.
- **Offscreen GUI smoke tests** — extend the existing `QT_QPA_PLATFORM=offscreen`
  pattern: drive a full scripted session (controller + worker + a fake
  responder) and assert the JSONL, certificate, and trajectory outputs.
- **Phase 0 data validation** — assert every test-bank `seg_id` resolves to a
  `(domain, h5 path)` AND a per-task `(s_mean, s_sd)`; assert `fs_hz` matches
  the segment's true rate.
- **Thread-lifecycle tests** — closing the window mid-test joins the engine
  thread cleanly and flushes the log.

---

## 10. File-by-file change inventory (quick reference)

| File | Change |
|---|---|
| `scripts/eeg_bank_viewer.py` | Drop `example_box`/`pick_examples` from the live path; add `show_item`, RT capture, `_awaiting_answer` guard; `_answer_chosen` → controller; remove free-nav; add results screen; `closeEvent` thread shutdown; `session_id` in `RegistrationPage`. |
| `scripts/session_controller.py` *(new)* | `SessionController` + `EngineWorker`; engine thread, queue-backed `y_source`, capture-instrumented adaptive loop, audit-log writer, injectable `stop_policy`. |
| `scripts/viz_render_live.py` *(new)* | Per-examinee particle-collapse animation — fork of `viz_render_collapse.py` (no ★, ℓ* lines, K=7). |
| `scripts/viz_report_live.py` *(new)* | Per-examinee static overview composite figure. |
| `data/eeg_bank.h5` | Add `fs_hz`/`channel_names` attrs; add spike negatives (Phase 0). |
| `data/` *(new manifest)* | Bank manifest CSV: `seg_id → (domain, h5_path, per-task s_mean/s_sd, fs_hz, ...)`. |
| `engine/core_mcmc.py` | **No change** — the adaptive loop calls its public primitives unchanged from the worker thread. |
| `results/sessions/` *(new tree)* | Per-session storage (see §7). |

---

## 11. Recommended sequencing

1. **Settle the PI decisions in §8 first** — especially **AD6 (the stopping /
   certificate rule)**, since it shapes the `stop_policy`, the certificate
   schema, the results screen, and the session-length expectation. The plan is
   built so the rest of the integration does not block on it (the stopping
   check is injectable), but it must be decided before Phase 5.
2. **Phase 0 in parallel from day one** — it is the critical path and is a data
   project, not a code project. Kick off the bank reconciliation and the
   spike-signal calibration immediately.
3. While Phase 0 runs, build **Phases 1–2** against the (soon-to-be-fixed) test
   bank — the engine/viewer wiring depends only on a correctly-signalled
   200-segment test bank, not the real bank.
4. **Phases 3–5** (registration linkage, storage, results screen) follow
   directly; Phase 5 needs AD6 settled.
5. **Phase 6** (visualization) after a full session can run end-to-end.
6. **Phase 7** (real-bank scale-up) gated on the Phase 0 calibration campaign
   completing for all 10k+ segments.
