# CHANGELOG — ilae-skill-certification (unified)

Consolidated 2026-05-15 (F3.5).  Supersedes the per-day `STATE_LOG_*.md`
and per-wave `CHANGES_W*.md` files (archived under
`docs/historical/`).  Forward-chronological; the box below is the
fast-path orientation for a new contributor.

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.3.2` — collapse.mp4 panels centered on (0,0) (symmetric axes)

Visualization-only release. No engine, policy, calibration, or data change.

`scripts/cortex_render_videos.py` `render_collapse`: each per-task panel's
axes are now **symmetric about zero** (centered on (0,0)) instead of fit to
the raw asymmetric data extremes. New helper `_symmetric_data_limits` returns
`[-M, M]` with `M = max |value| + margin` (per panel, from the full
trajectory), so the bias (x = t) and skill (y = ℓ) axes each span an
equal-magnitude range and the 0 reference lines cross the middle of every
box. Still static (no per-frame jitter) and still captures every value.

Scope: collapse.mp4 only. The engine-explainer's (t, ℓ) manifold + cloud
panels keep their v1.3.0 static (asymmetric, data-fit) limits.

### Tests (+1)

* `test_v1_3_2_collapse_axes_symmetric_about_zero`: `render_collapse` uses
  `_symmetric_data_limits`; the helper returns limits symmetric about 0 that
  contain the largest |value|, and a collapsed cloud still gets a
  non-degenerate symmetric window. Validated by frame-grabbing the collapse
  video on a real 176-trial K=7 session (per-panel limits e.g. spike x ±4.1 /
  y ±5.1, all lo = −hi).

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.3.1 → 1.3.2`.
* `.github/workflows/cortex-release.yml` release-body "What is new" rewritten
  for v1.3.2.

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.3.1` — drop the (inaccurate) ETA + "2 to 3 minutes" copy on the loading page

UI-copy only. No engine, policy, calibration, data, or video-render change.

The post-test `ComputingResultsPage` (`eeg_bank_viewer.py`) no longer shows a
time estimate, because the estimate was not accurate enough to be useful:

* Removed the `eta_lbl` widget and the `_fmt_eta` helper; `set_progress`
  still accepts the 3-arg `(stage, frac, eta)` progress signal for
  compatibility but no longer displays `eta`. The determinate progress bar
  and the stage label ("Building engine-explainer video", etc.) stay, so the
  page still communicates real progress and what the engine is doing.
* Removed the "This takes 2 to 3 minutes." line from the opt-in caption (now
  just "Computing your results and generating personalized visualizations.").
  The opt-out caption keeps "This will take just a few seconds." (the
  no-video path really is near-instant). `finalize` still computes/emits the
  ETA value; it is simply not shown.

### Tests

* Updated `test_computing_page_set_progress_is_determinate` (drops the ETA
  assertions) and `test_computing_results_page_opt_in_text` (asserts no "2 to
  3 minutes"); removed `test_fmt_eta_formatting`; added
  `test_v1_3_1_no_eta_no_minutes_estimate` (no `eta_lbl`/`_fmt_eta`/"2 to 3
  minutes"; `set_progress` still safe with a 3-arg call). Validated by
  offscreen-rendering the page.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.3.0 → 1.3.1`.
* `.github/workflows/cortex-release.yml` release-body "What is new" rewritten
  for v1.3.1.

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.3.0` — static MP4 axes + signal-graph trend line + professional titles

Visualization-only release. No engine, policy, calibration, or data change.
Three changes to the per-session MP4s, in `scripts/cortex_render_videos.py` +
`scripts/render_engine_explainer.py`.

### 1. Static axes from full-session extremes (replaces v1.2.8 breathing)

v1.2.8 autoscaled each panel per frame, which fit the data but made the axes
jitter as the cloud collapsed. v1.3.0 instead computes each panel's extremes
ONCE from the whole session and uses fixed axes that capture every value:

* `render_collapse` precomputes `task_xlim`/`task_ylim` per task from
  `t_traj[:, :, k].ravel()` / `l_traj` (every particle, every frame) and sets
  them once; the per-frame `set_xlim`/`set_ylim` in `update()` is removed.
* `render_engine_explainer` precomputes per-task `task_tlim`/`task_llim` and a
  global `q_xlim`/`q_ylim` for the question scatter, passed into `_draw_frame`
  as static `t_range`/`l_range`/`q_xlim`/`q_ylim`. The 3D manifold, 2D cloud,
  and bottom scatter no longer autoscale per frame.

Net effect: the cloud never runs off a panel (the extremes contain it) and
the axes hold still during playback. The question scatter's x-axis now spans
all questions from the start, so dots fill in left to right within a fixed
frame.

passfail's π-axis stays fixed at [0, 1] as before (π is a probability).

### 2. Neutral grey trend line through the signal-question dots

The engine-explainer's "Question difficulty by domain" panel now draws a thin
neutral grey line (`_MUTED`, the muted slate already in the palette)
connecting the questions in order, under the domain-colored dots, so the
difficulty trajectory reads as a sequence.

### 3. Professional titles and descriptions

Cleaned up the rendered copy across all three videos (sentence case, no em
dashes, clearer phrasing): collapse "Posterior cloud collapse" / "Question N
of M"; explainer "Task K of N: <domain> · trial j of T", panel titles
"Posterior density", "Posterior cloud and mean path", "Question difficulty by
domain", and a clean one-line caption; passfail "Question N of M".

### Regression tests (updated +1, retargeted +1)

* `test_v1_3_0_collapse_static_axes_from_extremes` (collapse precomputes
  `task_xlim` and `update()` no longer autoscales);
  `test_v1_3_0_explainer_static_axes_and_grey_connector` (static precompute,
  `_draw_frame` uses the static ranges + draws the grey connector, no
  per-frame `_data_limits(t_k)`). The v1.2.8 bottom-title assertion retargeted
  to "Question difficulty by domain". Validated by frame-grabbing the collapse
  + explainer on a real 175-trial K=7 session.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.9 → 1.3.0`.
* `.github/workflows/cortex-release.yml` release-body "What is new" rewritten
  for v1.3.0 (carries forward the two mac downloads).

### Test gate at ship

cortex viewer + render + storage + results + end-to-end + phase9:
**138/138 PASS**.

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.9` — results-page recordings breakdown + real progress bar with ETA

Two test-taker-facing UX improvements. No engine, policy, calibration, or
data change; verdicts are identical to v1.2.8.

### 1. "Recordings reviewed" line shows the spike/pattern split

`ResultsScreen` (`eeg_bank_viewer.py:2512`) printed only the combined total
(`result.n_questions`, which already sums all 7 tasks). It now also shows the
breakdown derived from per-trial `task_code`, e.g. "175 recordings reviewed
(20 spike, 155 pattern)." Falls back to the plain total when a result carries
no trials (older/stub results).

### 2. Post-test loading page: determinate progress bar + stage + real ETA

`ComputingResultsPage` previously showed an indeterminate busy-chase spinner
with a static "2 to 3 minutes" hint for the whole 2-3 min finalize. It now
shows a determinate bar, the current stage, and a live countdown:

* `cortex_storage.finalize(result, progress=cb)` gained an optional
  `progress(stage, frac01, eta_seconds)` callback. It splits the fraction
  budget: results+trajectory write (small), the three MP4 renders (the bulk,
  `[0.05, 0.92]`), then save + share. ETA is `elapsed * (1-frac)/frac`, which
  self-corrects as real frames flow. A throttle suppresses near-duplicate
  frames.
* The three renderers (`cortex_render_videos.render_collapse` /
  `render_passfail` / `render_all`, `render_engine_explainer.render_engine_explainer`)
  gained a `progress_callback` forwarded to matplotlib's per-frame
  `anim.save(progress_callback=...)`, so the bar tracks genuine render
  progress (not a guess). `render_all` reports a labeled
  `(stage, frame, total)` so finalize can slice collapse vs pass/fail.
* `_FinalizeWorker` gained a `progress` signal (emitted from the worker
  thread, queued to the GUI); `ComputingResultsPage.set_progress` switches
  the bar to determinate on first update and renders the stage label + a
  formatted ETA ("about 1m 13s remaining"). Stages a tester sees: Calculating
  results → Building collapse / pass/fail / engine-explainer video → Saving
  results → Sharing results → Done.

### Regression tests (+6)

* `test_finalize_emits_progress_stages` (monotonic 0→1, named stages, ETA
  non-negative); `test_render_all_forwards_labeled_progress`;
  `test_render_engine_explainer_forwards_progress_callback`;
  `test_results_screen_shows_spike_pattern_breakdown`;
  `test_computing_page_set_progress_is_determinate`; `test_fmt_eta_formatting`.
  Existing finalize/render stubs updated to the new `progress`/
  `progress_callback` signatures. Validated by rendering the progress page +
  results line offscreen.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.8 → 1.2.9`.
* `.github/workflows/cortex-release.yml` release-body "What is new" rewritten
  for v1.2.9 (carries forward the two mac downloads).

### Test gate at ship

cortex viewer + render + storage + results + end-to-end + session-controller
+ phase9: **148/148 PASS** (+6 new for v1.2.9).

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.8` — end-of-session MP4 layout cleanup

Visualization-only release. No engine, policy, calibration, or data change;
verdicts and session behavior are identical to v1.2.7. Four changes to the
three per-session MP4s (`collapse.mp4`, `passfail.mp4`, `engine_explainer.mp4`),
all in `scripts/cortex_render_videos.py` + `scripts/render_engine_explainer.py`.

### 1. Participant name removed from all three videos

The HUD titles no longer embed the test-taker's name. Removed from
`render_collapse` (`cortex_render_videos.py:253,274`), `render_passfail`
(`:387,427`), and the explainer title (`render_engine_explainer.py`). Titles
now read e.g. "question 91 / 175" and "task 3/7: LPD — trial 88/175".

### 2. AD6-verdict caption dropped from passfail.mp4

The sub-caption "trajectory color shows running AD6 verdict" is removed
(`cortex_render_videos.py:390`); the caption now reads only "green band =
PASS · red band = FAIL". Per Eli's direction this is **caption only** — the
π-line is still colored by the running PASS/FAIL/REFER verdict and the final
verdict badge still locks at session end. (The caption existed only in
passfail; collapse and the explainer never had it.)

### 3. engine_explainer.mp4 bottom panel replaced

The wide bottom panel was the expected-posterior-variance score curve
(`_expected_loss_vec` over a signal grid). It is replaced by a session-level
scatter: x = question number, y = that question's signal strength (`s_mean`,
probit-scale case difficulty), one dot per question colored by domain
(spike / sz / lpd / gpd / lrda / grda / iic, Okabe-Ito palette). Dots
**reveal progressively** up to the trial in focus, with the most-recent
question ringed white. The explainer now loads `trials.jsonl` (for `s_mean`
+ `task_code`) and no longer imports the engine. Also fixed a latent K=7
cosmetic bug: the explainer's `DOMAIN_TITLES` was missing the `"spike"` key.

### 4. Axes adapt to the data (no values run off-axis)

Per-frame "breathing" autoscale on every panel where values can exceed a
fixed box: the collapse (t, ℓ) clouds, the explainer's 3D manifold + 2D
cloud, and the new bottom scatter now fit their live data each frame
(`_data_limits` helper, `cortex_render_videos.py`). passfail's π-axis stays
fixed at [0, 1] on purpose: π is a probability and the PASS/FAIL bands are
defined on that scale, so values there cannot run off and breathing it would
break the band reference.

### Regression tests (`tests/test_cortex_render_videos.py`, `tests/test_render_engine_explainer.py`; +7)

* names dropped from titles; AD6 caption gone; collapse breathes via
  `_data_limits`; explainer dropped `_expected_loss_vec` + the
  "expected posterior variance" ylabel; `DOMAIN_COLORS` covers all 7 tasks;
  `_load_session` parses `questions` from `trials.jsonl`. The synthetic
  explainer fixture now carries `s_mean` + `task_code` so the e2e render
  exercises the new panel. Validated by frame-grabbing all three MP4s on a
  real 175-trial K=7 session.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.7 → 1.2.8`.
* `.github/workflows/cortex-release.yml` release-body "What is new" rewritten
  for v1.2.8 (carries forward the two mac downloads from v1.2.7).

### Test gate at ship

`tests/test_cortex_viewer.py + tests/test_cortex_render_videos.py + tests/test_render_engine_explainer.py + tests/test_cortex_storage.py + tests/test_cortex_results_screen.py + tests/test_phase9_*.py`: **130/130 PASS** (+7 new for v1.2.8).

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.7` — separate Intel-Mac DMG (CORTEX-mac-intel.dmg)

Packaging-only release. No engine, policy, UI, or data change; the app
behaves identically to v1.2.6. This build fixes a macOS architecture gap.

### The gap

The mac build job runs on `runs-on: macos-latest`, which GitHub Actions
now provisions as an Apple Silicon (arm64) runner. A PyInstaller binary is
single-architecture (`cortex.spec` has `target_arch=None`, so it targets
the build host), and the bundled ffmpeg is `ffmpeg-macos-aarch64`
(`cortex.spec:96`). The resulting `CORTEX-mac.dmg` is therefore arm64-only
and does **not** launch on an Intel Mac. Rosetta 2 does not help: it runs
Intel binaries on Apple Silicon, never the reverse. The release notes
nonetheless advertised "macOS 12 Monterey or newer (Intel or Apple
Silicon)", so an Intel-Mac tester would download the DMG and hit a launch
failure. Surfaced by Eli.

### Change

`.github/workflows/cortex-release.yml`:

* New `build-mac-intel` job that cross-builds the x86_64 app on the SAME
  Apple Silicon runner as `build-mac`, under Rosetta 2. GitHub is retiring
  the standalone Intel macOS runners (a first attempt on `macos-13` sat
  un-serviced in queue for 25+ minutes), so instead of depending on one the
  job installs a universal2 python.org CPython and drives venv + pip +
  PyInstaller through `arch -x86_64`. A `lipo -archs` guard fails the job if
  the output is not x86_64. It emits `CORTEX-mac-intel.dmg` with the same
  Applications drag-target staging. **`build-mac` is untouched**, so the
  arm64 DMG is built exactly as before.
* `release` job: `needs` gains `build-mac-intel`; a download step pulls the
  new artifact; `CORTEX-mac-intel.dmg` is added to the release `files`
  list. Releases now attach four artifacts (mac arm64, mac Intel, Windows,
  Linux).
* Release-body rewritten for v1.2.7: two mac downloads with an "About This
  Mac" chooser, and the system-requirements mac line corrected (no longer
  claims a single DMG covers both architectures).

`cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.6 → 1.2.7`.

### Not changed

No Python source changed, so the source-tree test gate
(`tests/test_cortex_viewer.py + tests/test_phase9_*.py`, 82/82) is
unaffected. The Intel build is exercised by CI on the next tag push; there
is no local mac runner to validate it here.

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.6` — tutorial note: spectrogram not shown for spike questions

A documentation-only clarity fix on top of v1.2.5. No engine, policy, or
data change; behavior is byte-identical to v1.2.5.

### Context

v1.2.5 made the spike block hide the spectrogram panel and expand the EEG
to fill the freed space (`_redraw` hides `spec_container` when
`family == "spike"`). But the tutorial renders an **IIIC** example
(`start_tutorial(..., tutorial_domain="iiic")`), so during the coach-marks
walkthrough the spectrogram panel IS on screen while the "The spectrogram"
step describes it. A first-time tester therefore had no warning that the
panel disappears once the live test enters the spike block — the v1.2.5
hide could read as a bug rather than intended behavior.

### Change

`scripts/eeg_bank_viewer.py:start_tutorial` — the "The spectrogram"
coach-mark step gains a closing sentence:

> "The spectrogram appears only for these pattern-classification
> recordings. The spike-present questions show the EEG on its own, with
> no spectrogram panel."

This pairs with the existing "Choosing an answer" step, which already
notes that "Some recordings instead ask only whether an epileptiform
spike is present." Wording matches the terse tutorial voice and the
`spike-present` phrasing used elsewhere in the walkthrough.

### Regression test added (`tests/test_cortex_viewer.py`; +1 test)

* `test_tutorial_spectrogram_step_notes_spike_has_no_spectrogram`:
  AST-parses `start_tutorial` and asserts the "The spectrogram" step
  contains the `no spectrogram panel` note, so a future tutorial-copy
  edit cannot silently drop it.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.5 → 1.2.6`.
* `.github/workflows/cortex-release.yml` release-body "What is new in
  v1.2.6" describes the tutorial note and carries forward the v1.2.5
  spike-display + MP4 fixes (fresh-slate page, no stacked history); the
  "earlier v1.2.x" re-download callout bumped to v1.2.6.

### Test gate at ship

`tests/test_cortex_viewer.py + tests/test_phase9_*.py`: **82/82 PASS**
(was 81; +1 new for v1.2.6).

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.5` — spike-screen cleanup + K=7 collapse/passfail MP4 fix + release-notes rewrite

Three things in this build. The first two are bugs Eli surfaced on his
v1.2.4 self-test; the third is a documentation cleanup.

### Issue 1: spectrogram from the tutorial stays visible on spike screens

v1.2.4's `_redraw` family-aware gate skipped DRAWING the spectrogram for
spike segments, but did not HIDE the `spec_container` widget. So after
the tutorial rendered an IIIC spectrogram into the panel, the panel
stayed visible (with the stale tutorial spectrogram still on it) when
the live test transitioned into the spike phase. Eli reported it as
"the spectrograms from the tutorial stayed rendered for the spike
questions".

Fix in `scripts/eeg_bank_viewer.py:_redraw`:

* When `family == "spike"`, call `self.spec_container.setVisible(False)`
  outright. The `eeg_plot` carries `stretch=1` in `plots_row` while
  `spec_container` has `stretch=0`, so Qt re-distributes the freed
  space to the EEG plot. The EEG view expands to fill the entire plot
  area for the 10-s by 128-Hz spike clip, which is what Eli requested
  with "have the eeg for the spikes questions extend to replace the
  location of the spectrograms".
* Also disable the spec_cb checkbox (`setEnabled(False)`) for spike so
  the user cannot toggle on a panel that has no data anyway.
* Re-enable both when the family transitions back to IIIC.

`_on_spec_toggle` also gets a defensive short-circuit so that if the
checkbox is somehow toggled while a spike segment is showing, the
panel stays hidden.

### Issue 2: `collapse.mp4` and `passfail.mp4` did not generate at K=7

Eli reported that on his v1.2.4 finalize the `engine_explainer.mp4`
rendered but `collapse.mp4` and `passfail.mp4` did not. Root cause in
`scripts/cortex_render_videos.py`:

* `render_collapse` (line 218 pre-fix) and `render_passfail` (line 339
  pre-fix) used `GridSpecFromSubplotSpec(2, 3, ...)` which is 6 cells.
  For K=7 the loop tried to add the 7th subplot at `inner[2, 0]` which
  is out of bounds for a 2-row grid, throwing IndexError. The renderers
  are wrapped in try/except inside `cortex_storage.SessionRecorder.finalize`
  with an independent failure budget per renderer, so the engine
  explainer (a separate code path) still ran while collapse + passfail
  silently failed.
* `engine_explainer.mp4` uses a 2x2 layout that cycles tasks one at a
  time inside the same frame, so it was unaffected by K=7 vs K=6 grid
  sizing.

Fix: compute `n_cols` and `n_rows` from `K` (K<=6 keeps the original
2x3 grid; K=7 widens to 2x4). The figsize widens proportionally so each
per-task panel stays the same size as the K=6 version. The hardcoded
`if r == 1:` last-row check becomes `if r == n_rows - 1:`. Also added
`"spike": "Spike"` to `DOMAIN_TITLES` so the spike panel gets a friendly
title instead of falling back to the raw task code.

Verified by smoke-rendering both MP4s for both K=6 (no regression) and
K=7 synthetic sessions: all four files produced cleanly.

### Issue 3: release notes rewrite (human voice, fresh-slate scope)

The release-page body in `.github/workflows/cortex-release.yml` had
stacked v1.2.0 through v1.2.4 hotfix sections that read as a wall of
third-person technical changelog text full of em dashes. Per Eli's
direction, the body is now a fresh, first-person, em-dash-free document
that covers: a one-paragraph "what the test does" intro (adaptive,
MCMC engine, K=7 task list, typical session length); a short "what's
new in v1.2.5" section; an "if you took an earlier v1.2.x" callout;
system requirements; detailed install instructions per OS (mac
Gatekeeper walkthrough preserved); and a reporting-bugs paragraph.
Full historical changelog stays in this file (`CHANGELOG.md`).

### Regression tests added (`tests/test_cortex_viewer.py`; +3 tests)

* `test_redraw_hides_spec_container_for_spike_and_disables_checkbox`:
  AST-asserts `_redraw` calls `spec_container.setVisible(False)` AND
  `spec_cb.setEnabled(False)` for spike (not just the silent-skip bug
  from v1.2.4).
* `test_render_collapse_passfail_k7_grid_layout`: AST-asserts both
  renderers compute `n_cols` / `n_rows` from K and no longer have the
  hardcoded `GridSpecFromSubplotSpec(2, 3` or `r == 1` patterns.
* `test_render_collapse_passfail_actually_render_k7_session`: drives
  both renderers end-to-end on synthetic K=6 + K=7 session payloads
  and asserts both MP4 files materialise.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.4 → 1.2.5`.
* `.github/workflows/cortex-release.yml` release-body fully rewritten
  (fresh-slate v1.2.5-only page, no stacked history).

### Test gate at ship

`tests/test_cortex_viewer.py + tests/test_phase9_*.py`: 81/81 PASS
(was 78; +3 new for v1.2.5).

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.4` — four-issue fix from Eli's v1.2.3 self-test: tutorial loader, spike spectrogram, session sectioning, empty-bank crash

Four user-visible issues Eli surfaced on his v1.2.3 self-test (Linux),
all fixed in this release. The first two are small UI fixes; the last
two are a single architectural change in `scripts/session_controller.py`
that introduces phase-aware item selection.

### Issue 1: tutorial fails to load EEG + spectrogram

`scripts/eeg_bank_viewer.py:open_viewer()` picked `inputs.all_seg_ids[0]`
as the tutorial example. Under K=7 the manifest puts the 50 spike segs
at indices 0..49, so the tutorial got a spike seg_id — but the tutorial
UI is hard-coded for IIIC (6-button answer panel + `eeg30s` renderer).
The `_render` call tried `bank['/iiic/<spike_seg_id>']` which doesn't
exist → silent failure (the bank's exception handler set `self.data = None`
and returned). The tutorial appeared frozen.

Fix: explicit `next(s for s in inputs.all_seg_ids if inputs.family(s) == 'iiic')`
when the inputs object has a `family` attribute (K=7 path). K=6 fallback
preserved (`IIICEngineInputs` has no `family` attr; first seg is always
IIIC there).

### Issue 2: spike questions still show spectrogram panel

`scripts/eeg_bank_viewer.py:_redraw()` called `self._draw_spectrogram()`
whenever the spec checkbox was checked. Spike segments don't carry
`sdata`/`sfreqs`/`stimes` — and the spike-paper methodology (Jing et al)
uses the 10s × 128 Hz EEG window only, no spectrogram panel. The
on-the-fly compute fallback would have produced a meaningless 10-s
spectrogram strip if the precomputed data were absent.

Fix: family-aware gate in `_redraw`:
`if self.spec_cb.isChecked() and family != "spike": self._draw_spectrogram()`.

### Issues 3 + 4: session sectioning + empty-bank crash (single fix)

Eli's v1.2.3 self-test froze at ~question 270 on a spike question. Root
cause traced to `engine/core_mcmc.py:choose_item:607`:
`best_k, best_s = domains[0], float(np.asarray(candidates[domains[0]])[0])` —
when the spike pool exhausts (all 50 spike segs served), `bank_signals[0]`
becomes `np.array([])` of shape `(0,)`, and the engine crashes with
`IndexError: index 0 is out of bounds for axis 0 with size 0`. The
engine has an `active_domains` parameter (`engine/core_mcmc.py:605`)
exactly for this case — but the K=7 session controller was not passing
it.

Eli's separate request — section the test (spike phase first, then IIIC,
then aggregate results at the end) — uses the same `active_domains`
mechanism. One fix addresses both:

`scripts/session_controller.py` adds `_compute_active_domains` on
`CortexSession`:

  * **Phase A (spike sectioning)**: if K=7 AND spike (k=0) verdict is
    still PENDING AND the spike bank is non-empty, return `[0]` — the
    engine asks spike questions only until AD6 locks the verdict or the
    spike pool exhausts.
  * **Phase B (IIIC)**: once the spike phase has ended (verdict locked
    OR pool empty), return all IIIC task indices (1..6) with non-empty
    banks AND PENDING verdicts.
  * **K=6 (legacy) / non-AD6 policies**: return all k with non-empty
    banks (no phase awareness needed).
  * **All resolved**: empty list → `_select` returns None → run loop
    stops cleanly with new `stop_reason="all_active_resolved"`.

`_select` now consults `_compute_active_domains` each trial + passes the
result to `choose_item` and `_pick_top_n`. The hierarchical Σ_l coupling
is preserved across phases (spike evidence informs IIIC priors via the
K=7 model) — only the question ordering changes. Final ResultsScreen
aggregates spike + IIIC verdicts as before.

### Regression tests (`tests/test_cortex_viewer.py`)

Four new tests, 30/30 viewer suite pass:

  * `test_main_open_viewer_picks_iiic_tutorial_seg_under_k7` — AST-asserts
    that `open_viewer` uses `inputs.family(...)` to filter `all_seg_ids`
    for the tutorial seg.
  * `test_redraw_skips_spectrogram_for_spike_family` — AST-asserts
    `_redraw` reads `_cur_family` and the body mentions `spike` (the
    family-aware gate).
  * `test_compute_active_domains_k7_spike_first_sectioning` — drives all
    4 phase-transition cases (Phase A, spike-resolved Phase B, empty-
    spike-pool Phase B, all-resolved → empty) on a real K=7 inputs +
    AD6Policy.
  * `test_session_skips_empty_spike_bank_without_indexerror` — runs a
    real random-rater K=7 session for up to 100 trials and asserts it
    completes without IndexError.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.3 → 1.2.4`.
* `.github/workflows/cortex-release.yml` release-body prepends a "Fixes
  in v1.2.4" section above the v1.2.3 calibration entry.

### Test gate at ship

`tests/test_cortex_viewer.py + tests/test_phase9_*.py`: **78/78 PASS** (was 74; +4 new for v1.2.4).

---

## ▶ CORTEX bundle (2026-05-30) — `cortex-v1.2.3` — AD6 calibration tweak: N_MIN 12 → 20 (random-rater edge-case hardening)

Calibration tweak surfaced by Eli's v1.2.0/v1.2.2 self-test as a deliberate
random guesser. Session `f3da305d-b4a8-4a01-afa5-1f813dd9c519` ended with
2 of 6 IIIC tasks (gpd, iic) returning PASS verdicts despite 19.4%
accuracy (near random for 6-way questions). The AD6Policy verdict-lock
fired at the N_MIN=12 floor (trial 67 for gpd with n_per_task=12,
π=0.823; trial 54 for iic with n_per_task=12, π=0.808) when the posterior
was momentarily elevated. The K=7 hierarchical posterior continued
updating after the lock via Σ_l cross-task pooling — by session end
gpd ell_mean drifted to +0.197 (below ℓ\*=0.330) and iic to -0.313
(below ℓ\*=0.442) — but monotonic verdicts could not revoke.

### PhD-agent analysis

Two PhD-level agents (Bayesian decision theory + CAT psychometrics)
independently analysed the failure mode in this session's transcript.
Quantitative summary:

| Setting | Per-task spurious-PASS | Familywise (K=7) |
|---|---:|---:|
| v1.2.0 (N_MIN=12, ALPHA=0.25) | ~5.5% | ~50% (matches observed 2/6) |
| **v1.2.3 (N_MIN=20, ALPHA=0.25)** | **~3%** | **~25%** |
| v2.0 production (N_MIN=60, ALPHA=0.05) | ~0.5% | ~3.5% |

Both agents independently confirmed Eli's hypothesis: the failure mode
is statistically improbable at v2.0 production strictness. v1.2.3
bridges internal-test speed with reduced false-PASS at the calibration
midpoint. Architectural alternative (revocable verdicts per AERA
Standard 4.10) was considered and explicitly deferred — monotonic-lock
is kept as a deliberate adaptive design choice consistent with the SPRT
OC framing (Wald 1947; Robbins 1956). See Agent 1's posterior-precision
math + Agent 2's CAT-literature review in the session transcript.

### Change

  * `scripts/cortex_policy.py` `DEFAULT_N_MIN` `12 → 20`. `DEFAULT_ALPHA`
    unchanged at `0.25` by design (keeps the internal test fast while
    raising the bar against momentary-posterior spurious locks).
  * `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.2 → 1.2.3`.
  * `.github/workflows/cortex-release.yml` release-body prepends a
    "Calibration tweak in v1.2.3" section above the v1.2.2 hotfix entry.

### Sim coverage

N_MIN=20 is **outside** the `sim_v1_2_0/` sweep grid (which covered
NMIN ∈ {8, 10, 12, 15}). Extrapolated from the nearest measured cell
(NMIN=15, ALPHA=0.25: median=221, p95=300, all_resolved=64%) to NMIN=20
→ expected median ~245 trials, p95 ~300, all_resolved ~55%. Empirical
re-validation deferred. Acceptable because v1.2.3 is an internal-test
calibration setting; v2.0 production will run its own OC sweep at the
panel-target N_MIN=60 + ALPHA=0.05.

### Test gate at ship

`tests/test_cortex_viewer.py + tests/test_phase9_*.py`: **74/74 PASS**.

---

## ▶ CORTEX bundle (2026-05-29) — `cortex-v1.2.2` — HOTFIX: bundle the K=7 data files the v1.2.1 builder reads

Critical packaging hotfix for the cortex-v1.2.1 ship. v1.2.1 correctly
re-wired `scripts/eeg_bank_viewer.py:open_viewer()` from the K=6 inputs
builder to `cortex_engine_inputs_k7.build_k7_engine_inputs()`, but did
NOT update the PyInstaller `datas` list in `cortex_app/cortex.spec` to
ship the two data files the K=7 builder reads. The v1.2.1 bundle still
carried only the K=6 files (`data/labels/iiic_segment_signals.csv` +
`Sigma_l_fitted.npy`), so launching the live test crashed at startup:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '.../CORTEX/_internal/data/labels/segment_signals.csv'
```

Surfaced on Eli's Linux v1.2.1 self-run (`./CORTEX/CORTEX`).

### Root cause

The K=6 builder (`cortex_engine_inputs.py`) reads
`data/labels/iiic_segment_signals.csv` + `Sigma_l_fitted.npy` — exactly
the two files `cortex.spec` bundled, so v1.2.0 (K=6 path) was
self-consistent. The K=7 builder (`cortex_engine_inputs_k7.py`) reads
`data/labels/segment_signals.csv` + `Sigma_l_fitted_k7.npy` instead.
v1.2.1 changed the code's file dependencies without changing the
bundle. The 432/432 + 26/26 source-tree suites did not catch it: they
run where both K=7 files exist on disk; nothing exercised the packaged
bundle's data completeness.

Two files were missing, not one — `segment_signals.csv` is the first
`FileNotFoundError`; `Sigma_l_fitted_k7.npy` (read by
`load_fitted_Sigma(SIGMA_PATH)` at `cortex_engine_inputs_k7.py:282`) is
the second, surfacing only once the CSV is present.

### Fix

`cortex_app/cortex.spec` `datas` now also ships:

* `data/labels/segment_signals.csv` → `data/labels`
* `Sigma_l_fitted_k7.npy` → bundle root

The K=6 `iiic_segment_signals.csv` is retained (still read by
`session_controller` for segment metadata). The v14 ℓ\* policy path was
verified to read only the already-bundled `calibration/cert_config.yaml`
— no third missing file.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.1` → `1.2.2`.
* `.github/workflows/cortex-release.yml` release-body "Hotfix in v1.2.2"
  prepended above the v1.2.1 section.

### What testers see

v1.2.2 launches and runs all 7 tasks. No code/behavior change vs v1.2.1
beyond the bundle now containing the K=7 data files — v1.2.1 never
reached a usable session on a clean install, so there are no v1.2.1
results to compare against.

---

## ▶ CORTEX bundle (2026-05-29) — `cortex-v1.2.1` — HOTFIX: K=7 engine-inputs import + ℓ\* block consistency

Critical hotfix for the cortex-v1.2.0 ship. The v1.2.0 bundle compiled
all the K=7 backbone code + bank correctly but the live-test entry
point in `scripts/eeg_bank_viewer.py` still imported the K=6 engine
inputs builder, so internal-test sessions ran with K=6 inputs (300 IIIC
seg_ids only) and the 50 spike segments in the bundled K=7 bank were
never asked. Sessions completed with **6 task verdicts** (sz, lpd, gpd,
lrda, grda, iic) and **NO SPIKE verdict** — defeating the whole point
of Phase 9.

Surfaced by Eli's v1.2.0 self-test (session
`f3da305d-b4a8-4a01-afa5-1f813dd9c519` at
`/home/exx/.local/share/CORTEX/sessions/`). 299/300 trials, stop_reason
= bank_exhausted, certificate.json had 6 verdicts, participant.json
recorded `n_iiic_segments: 299`.

### Two bugs fixed

**1. `scripts/eeg_bank_viewer.py:2844`** — `open_viewer()` was importing
`build_iiic_engine_inputs` from `cortex_engine_inputs` (the K=6 module),
not `build_k7_engine_inputs` from `cortex_engine_inputs_k7`. The
back-compat alias `build_iiic_engine_inputs = build_k7_engine_inputs`
at `session_controller.py:67` only applied to consumers importing FROM
`session_controller`; this entry point imported directly from the
cortex_engine_inputs module, bypassing the alias. Fixed by changing the
import to the K=7 module explicitly. The misleading alias is preserved
for back-compat with `audit_selection.py` + `cortex_smoke.py` (dev
tools; their K=7 update is non-blocking and deferred).

**2. `scripts/cortex_policy_k7.py:103`** — `default_policy_for_k7`'s
default `block_name` parameter was `"ell_star_unified_v13"` while
`load_ell_star_k7`'s default was `"ell_star_unified_v14"`. After
fixing bug 1, the live engine would have computed verdicts against
v13 ℓ\* while `ResultsScreen._safe_load_ell_star` rendered narratives
against v14 ℓ\* — same screen, disagreeing numbers. Fixed by bumping
the policy default to v14 to match the loader.

### Regression tests (`tests/test_cortex_viewer.py`; 3 new, 26 total pass)

* `test_main_open_viewer_uses_k7_engine_inputs` — AST-parses
  `eeg_bank_viewer.py` and asserts the `open_viewer` nested function
  imports `build_k7_engine_inputs` (not `build_iiic_engine_inputs`).
  Any future refactor reverting the import fails this test.
* `test_k7_engine_inputs_actually_returns_7_tasks_with_spike` —
  asserts `build_k7_engine_inputs()` returns the 7-task list starting
  with spike + exposes the `family()` method needed for the UI dispatch.
* `test_default_policy_for_k7_uses_v14_block_by_default` — asserts
  the v13/v14 block_name default consistency between policy + loader.

### Packaging

* `cortex_app/cortex.spec` `CFBundleShortVersionString` `1.2.0` → `1.2.1`.
* `.github/workflows/cortex-release.yml` release-body "What's new in
  v1.2.1" prepended above the v1.2.0 section explaining the hotfix.

### What testers see

v1.2.1 bundle runs all 7 tasks. Internal testers who already took v1.2.0
should retake v1.2.1 to get a spike verdict — their v1.2.0 results in
`Dropbox/Apps/CORTEX/results/v1.2.0/` are missing the spike task and are
not directly comparable to v1.2.1 results in the same folder.

---

## ▶ CORTEX bundle (2026-05-29) — `cortex-v1.2.0` — Phase-9 K=7 unified joint hierarchical ground-up release (internal-test cohort)

Closes Phase 9 — the K=6 → K=7 ground-up rebuild that adds spike-vs-no-spike
as the 7th unified task alongside the 6 IIIC pattern_classes. Phase-9
methodology shift: **modular Bayesian inference** (Plummer 2015) — joint
posterior over per-segment latent signals `s_j` only, with per-rater
parameters `c_j` / `d_j` cut via byte-verbatim two-stage SDT fits. SVI
becomes the production inference path; NUTS retained for validation only.

Full close-out: `docs/PHASE9_CLOSEOUT.md`.

### Six layers shipped (48/48 tests pass)

* **Layer 1** — Joint K=7 SVI for 7 tasks under V_B variant (Gate A applied;
  Gate B reverted). 7× `*_k7_B_posterior.npz`; ~25 min wall-time on 2×
  A4500 with `chain_method='vectorized'`. NUTS dropped from primary path
  after Phase-3.5 Kong-crowd-dominance multi-modal posterior diagnosed.
* **Layer 2** — `assemble_outputs_k7.py` → `data/labels/segment_signals.csv`
  (89,138 segs × 7-task signals). Replaces `iiic_segment_signals.csv` as
  the canonical per-segment signal source.
* **Layer 3** — `build_engine_inputs_k7.py` → `sdt_fits_k7.csv` (D2-pinned
  per-rater parameters) + `MANIFEST.json` + `cross_domain_matrix.npy`.
* **Layer 4** — `run_youden_calibration_k7.py` + `emit_cert_config_v14.py`:
  uniform CV-top-14 Youden across all 7 tasks. Headline: min J = 0.6493,
  mean J = 0.7818. 5 of 6 IIIC ℓ\* byte-stable vs v13; sz + spike change
  by documented amount (cross-task expert panel + 70/30→CV-top-14 spike
  methodology). Appends `ell_star_unified_v14` to `cert_config.yaml`.
* **Layer 5** — `build_sigma_l_k7.py` → `Σ_l_fitted_k7.npy` from
  D2-preserved two-stage per-rater fits (1,949 raters; matches K=6 v13
  precedent). Fixed pre-flight from joint-posterior to two-stage source
  (consistency with V_B's modular cut).
* **Layer 6a** — CORTEX K=7 code + UI + 350-seg internal bank:
  - `scripts/cortex_engine_inputs_k7.py` + `cortex_policy_k7.py` (default
    ℓ\* block → `ell_star_unified_v14`)
  - `scripts/eeg_bank_viewer.py` family-aware UI: spike → 2-button Yes/No
    + `eeg10s` renderer (128 Hz, 10 s); IIIC → 6-button + `eeg30s`
    unchanged. Engine `_y_source` contract preserved byte-equivalently.
  - `data/eeg_bank.h5` rebuilt at 350 segs (50 calibrated SN1 spike + 300
    IIIC). K=6 bank preserved as `data/eeg_bank_v1.1.0_legacy.h5`.
  - `ResultsScreen` extended to K=7 (`_TASK_LABELS` + `_safe_load_ell_star`).
* **Layer 6b** — production-bank infrastructure (deployment deferred):
  - `scripts/build_cortex_bank_k7_production.py` — strict filter
    (morgoth1_recompute + (20,6000) shape); 36,771-seg production-pool
    target; ~43 GB at full build; 91.6% IIIC coverage after strict filter
    documented in MANIFEST.
  - `scripts/cortex_session_bank_fetch.py` — deterministic per-session
    sampler (sha256(session_id) seed); stratified by V_B `s_mean_<task>`;
    `BankBackend` interface (LocalBackend works; HTTPBackend placeholder).
  - `cortex_app/cortex_offline_fallback.h5` (1.14 GB; 970 segs stratified)
    + manifest — built for testing.

### v1.2.0 release packaging (this commit)

* **AD6 stop-policy defaults RECALIBRATED** for the K=7 350-seg internal
  bank — sim-driven, not panel-derived:
  * `DEFAULT_N_MIN`  `15` → `12`
  * `DEFAULT_ALPHA` `0.10` → `0.25`
  * `DEFAULT_R_STAR` unchanged at `0.30`
  Selected by `sim_v1_2_0/run_ad6_sweep.py` from a 500-session sweep
  over {NMIN×ALPHA} = 4×5 grid. Achieves median **188 trials** (target
  200; bank is 350 → 53.7% utilization, 46% selector headroom),
  **76% all_resolved** rate. Engine quits early (~110 trials) for
  confident-skill raters, runs long for borderline (ℓ=+0.5 hits the
  300-cap in ~60% of sessions) — correct adaptive behavior. Full
  analysis: `results/sim_v1_2_0/report.md`. Sim infrastructure lives in
  `sim_v1_2_0/` and is removable with one `rm -rf` (zero runtime code
  modified).
* **Bank delivery:** the 350-seg K=7 internal bank ships bundled in the
  PyInstaller installer (same fetch path as v1.1.5; new GitHub-release
  tag `build-data-v3-k7` when published).
* **Per-session fetch default lowered** `200` → `60` (`per_task`):
  empirical AD6 settings (now NMIN=12, ALPHA=0.25) produce median ~27
  asks/task; 60 gives 2× choice margin while keeping session download
  at ~500 MB (bandwidth-friendly for internal Dropbox).
* **Dropbox folder** `/results` → `/results/v1.2.0`: v1.2.0 cohort
  recordings logically isolated from v1.1.5. Same Dropbox app key +
  refresh_token + secret as v1.1.5 (no new OAuth flow needed).
* **`cortex.spec` `CFBundleShortVersionString`** `1.1.5` → `1.2.0`.

### K=6 vs K=7 head-to-head (50 matched-seed synthetic raters)

* K=7 wall time: **1.41× K=6** at recommended ship params (NMIN=12,
  ALPHA=0.25) — sub-linear scaling for the added 7th task.
* K=7 median session length: **151 trials** vs K=6 median **131
  trials** (+15%).
* K=7 AUROC half-width: median **0.099** vs K=6 median **0.094** —
  essentially the same per-task precision; K=7 just tracks one more
  task.
* K=7 PASS rate for borderline rater (ℓ=+0.5): **64%** vs K=6 **38%** —
  K=7's joint hierarchical posterior pools cross-task evidence under
  the modular Bayesian inference framing, reaching confident verdicts
  on borderline cases that K=6 left as REFER_BORDERLINE. Expected
  behavior; not a regression.

### Utilization analysis (per `docs/PHASE9_CLOSEOUT.md` §3)

```
Production pool:      36,771 segs       (full filtered production pool;
                                          v1.2.0 ships ~350 internally)
Per-session fetch:       420 segs (1.14%)  (60/task × 7; ~500 MB)
Per-session asked:       210 segs (50.0%)  (median; AD6 production settings)
Net pool consumption:  ~0.57% per session
Pair-wise overlap:     ~0.5% per session pair
```

### What's deferred (operational, not methodology)

* Full 38K production-bank build (~3-4 h compute on the workstation).
* Cloud backend choice + HTTPBankBackend implementation (~30 LOC).
* Production manifest URL → `cortex_cloud_config.yaml`.
* v1.3.0 will wire cloud-fetch live once these operational items land.

### Tests

`tests/test_phase9_layer6b.py` — 10 new tests for build + fetch
infrastructure (schema, determinism, balance, e2e, manifest). Phase-9
suite total: **48/48 PASS**.

---

## ▶ UNIFIED MERGE (2026-05-18) — Phases 0–8 COMPLETE — v1.0.0-rc1 SHIPPED

Methodology repo + PI deployment repo merged into one shippable repo
(plan: `../UNIFIED_REPO_MERGE_PLAN.md`, decisions D1–D9).

- **Phase 0/1** — skeleton/packaging/provenance; PI corpus adopted
  (proven strict superset, 0 obs lost); Centaur-IIIC 4-expert gold panel
  ingested into the canonical lake; R1–R6 (see `data/DATA_PROVENANCE.md`).
- **Phase 2 — engine consolidation.** The hardened methodology engine
  (`core*.py`, `engine_mode_b.py`, `auroc.py`, `diagnostics.py`,
  `bridge/`, `tests/`, `scripts/`) adopted **byte-identical** into
  `engine/` + `bridge/`; flat bare-name imports preserved (conftest/path
  shims) so the validated suite runs verbatim. Path-only integration
  edits: `engine_paths._REPO`, bridge `ENGINE_REPO` (+`engine/` on
  sys.path); `Sigma_l_fitted.npy` is one frozen repo-root file with
  `archive/` + `engine/` symlinks (zero drift). Single-thread BLAS caps
  set in conftest (the engine's documented bit-exact-reproducibility
  contract).
- **PI variants ported onto the hardened likelihood.** Code audit showed
  the variants already delegate to `core.response_prob`/`core_mcmc`
  (auto-hardened once importing the canonical core); the ONLY genuinely
  unhardened code was `core_K.py` `expected_loss_{hier,brute}_vec_K`
  (`np.clip(norm.cdf(z),1e-9,1-1e-9)`, collapses to 1.0 at |z|≳6).
  Replaced with the spike-paper Eq. 2 lapse mixture
  `LAPSE_RATE + (1−2λ)·Φ(z)` (single-sourced `LAPSE_RATE` from `core`;
  identical to `core_mcmc._p_response_yes`). No other variant rewrite was
  warranted (the plan's blanket "rewrite 7 files" was over-scoped vs the
  actual code).
- **FIX-T1.9 IUT dispute RESOLVED (decision 2026-05-18).** Mode-B binary
  certification now defaults to the **joint-posterior** stopping rule
  (`iut_rule="joint"`) that the Bayesian-expert audit holds correct —
  PASS the conjunction iff `P(all active l_k>l*_k | data)` from the JOINT
  particle cloud, MCSE-buffered, ≥ stop_thresh. The FIX-T1.9 Berger
  (1982) min-of-marginals rule is **preserved verbatim** and selectable
  via `iut_rule="berger_marginals"`. Threaded with brute parity
  (`core_mcmc_brute_k`: brute "joint" = product of independent marginals).
  **Paper-1 is unaffected** (Mode-A has no IUT; Mode-B is deprecated for
  Paper-1, Paper-2 scope). Joint is strictly ≥-conservative than Berger.
  Tests: `tests/mode_b/test_iut_rule.py`.
- **Phase 3 — reference-faithful calibration.** Byte-verbatim Rasch +
  per-rater probit-lapse + CV-top-14 two-stage Youden ell* carried into
  `pipeline/reference_calibration/`; spike un-fold (clean sn1 binary,
  Centaur-IED excluded from cert task, J 0.366→0.632); erratum fix
  (uniform `{bipd,birds}→other`, iic J 0.643→0.814). `cert_config` v13.
- **Phase 3.5 — joint s_j unification + engine s_sd propagation —
  CLOSED.** Joint hierarchical cross-source SVI fit (split-anchor
  gauge); engine marginalises item uncertainty
  (`z/=√(1+(e^ℓ·s_sd)²)`, `s_sd=0` ⇒ **bit-identical** default).
  Release-gate battery COMPLETE, results reported honestly in
  `docs/DATA_UNIFICATION_ANALYSIS.md` §8 (+ `cert_config` v13
  `provenance.phase35.release_gate`): (1) plug-in-vs-uncertainty AUROC
  — real, correctly-signed, **small** (CI ratio 1.019; bounds the
  plug-in cost in the favourable regime); (2) δ_Centaur — **robust**
  vs an SVI noise floor (cheap probe; ratio<1, r≥0.984); (3) **engine
  SBC — strongest result**: PLUGIN genuinely miscalibrated under real
  item noise (cov 0.88, KS≫control), UNCERT restores calibration to
  the CONTROL baseline (cov 0.93) ⇒ s_sd propagation is *not
  cosmetic*. Honest caveats recorded (CONTROL-KS = baseline SMC
  approximation over-detected at large n; δ_Centaur smoke-scale;
  SVI-vs-NUTS s_mean r≈0.72–0.78). Tests: `tests/test_phase35_*.py`.
  Full suite 197 passed / 1 xfailed.
- **Phase 4 — deployment integration (4.1 → 4.7 COMPLETE).** PI
  Laplace/EKF clinical-deployment engine merged onto
  the unified corpus + reference-faithful v13 calibration, strict
  incremental + two-step-gated (prove port fidelity, then layer each
  intended change with a signed-off delta). Per-sub-step attribution +
  the definitive verdict in **`docs/DEPLOYMENT_INTEGRATION.md`**.
  Highlights: 4.1 bit-faithful path-only port; 4.2 config→YAML
  contract; 4.3 λ-lapse = one likelihood def shared with Paper-1
  (λ=0≡PI bare-probit ~2e-16); 4.4-A K-agnostic runtime (bit-identical
  @K=6); 4.4-B fit/freeze port, delete degenerate `iic=OR`, real
  erratum-correct `other` (Y=204,163=v13 `sparcnet_iic`; plan's 79,383
  was stale pre-merge); 4.5 single **v13 ℓ* lineage** (D2; isolated
  delta large+expected+signed-off); 4.6-A re-fit/re-freeze **K=7** on
  the unified corpus (Σ 14×14, clean-sn1 spike, PI baseline archived,
  **D3** `data/labels` sha256 provenance); 4.6-B **RNG decoupled**
  (the 4.3 population-coupling confound fixed — proven). **DEFINITIVE
  4.6-C verdict:** the λ-lapse pass-share-of-decisive leniency shift
  is **stable + reproducible** across PI-K6 → v13-K6 → shipped-K7
  (+0.0145 → +0.0157 → **+0.0144**, ~5·SE) — a *known, quantified
  characteristic of the reference-correct lapse likelihood* (also
  +6 pp more REFER = more conservative), **NOT a bug**; closed, no
  further re-assessment owed. **4.7**: `ilae-deploy` =
  `deployment/cli.py` orchestrator (`freeze`/`simulate`/`plot`/`all`);
  `plot_deploy.py` ported + made **K-agnostic** (renders all 5 figures
  at K=7; `plot` best-effort in the pipeline); entry point retargeted.
  Close-out: `docs/DEPLOYMENT_INTEGRATION.md`. Tests:
  `tests/test_phase4_*`. Full suite 232 passed / 1 xfailed.
- **Phase 5 — engine_inputs provenance/layout reconciliation (D3),
  COMPLETE.** Audit found the plan stale: `build_engine_inputs.py`
  read the legacy out-of-repo prepared fits (219-row → re-running
  regresses + violates D9); the live `sdt_fits.csv` (14,214 rows) is
  *already* the gated Phase-3 recompute on `data/labels`
  (`run_unified_calibration` STEP d). Decision (no content regen):
  builder **repurposed** into a no-sibling MANIFEST
  verifier/regenerator that self-checks the gated contract + **proves
  D3** (`engine_inputs` `data_labels` sha256 == `deployment_prior`
  == live corpus). Stale `MANIFEST.json`/`README.md` rewritten
  truthfully; stray materially-divergent `sdt_fits.phase3.csv`
  removed; nonexistent `consolidated/` dropped from the gate.
  Findings in `data/DATA_PROVENANCE.md` §6 + `data/engine_inputs/`
  `README.md`/`MANIFEST.json`. Tests:
  `tests/test_phase5_engine_inputs.py`. Full suite 237 passed /
  1 xfailed.
- **Phase 6 — invariant audit (plan §"Phase 6") COMPLETE
  (2026-05-19).** Each of the 9 invariants audited against actual
  code; findings + signed-off dispositions in
  `docs/INVARIANT_AUDIT.md`. **Invariants 1–5 PASS** with in-code
  evidence pointers: `LOGIT_TO_PROBIT=1.0/1.7` at
  `fit_sdt_per_domain.py:38` + runtime-asserted in
  `run_unified_calibration.py:201`; `λ=0.025` single-sourced at
  `engine/core.py:21` and imported by Mode-B/deployment/MCMC paths;
  the two `fit_sdt_per_domain.py` copies remain
  **md5 `6b90d59dcd0aaa878f9a52802254044b`** (gated also by Phase-3
  + new Phase-6 drift-guard test); σ*/ℓ* on TRAIN-only via
  `EXPERT_TRAIN_FRAC=0.70` at `youden_sigma_star_ref.py:19` (70/30
  expert + 50/50 non-expert split in `run_unified_calibration.py`
  STEP g); CLAUDE.md placeholder records the 70/30 correction for
  Phase-8 assembly. **Four signed-off deviations**: §6
  `GRAY_ZONE_DELTA` and §8 `T_TOL` are Mode-B/Paper-2 constants that
  the merge plan didn't carry (N/A for the unified Paper-1 scope —
  the paper-2-validation scripts were not ported); §7 the plan's
  `EXPERTS` set / `gold_standard_raters.yaml` both name non-existent
  sources — *actual* authoritative expert membership is the
  `expertise_level` column in `data/labels/raters.csv`
  (`run_unified_calibration.py:450`, `freeze_deployment_prior.py`),
  a wording clarification not a content gap; §9 `r_ℓ=0.326` is the
  retired K=6 PI-era value (preserved at
  `_pi_baseline_frozen/sim_summary.json:3`) — the current K=7
  shipped value is **0.36656350316581954** (derived, not pinned,
  from the frozen Σ ℓ-block at `freeze_deployment_prior.py:126`,
  recomputed at simulate-time `run_deployment_sim.py:104`),
  superseded by Phase-4.6-A re-freeze. **Phase-6 gate satisfied.**
  Tests: `tests/test_phase6_invariants.py` (the §3 byte-equivalence
  drift-guard explicitly requested by the plan; audit-doc
  consistency; spot-check the 5 PASS invariants point at code that
  actually defines them). Full suite **240 passed / 1 xfailed**.
- **Phase 7 sub-step 1 — Phase-2 validation suite synthetic at K=7,
  α-scope (per user 2026-05-19).** Pre-execution audit
  `docs/PHASE7_AUDIT.md`: traced all 5 plan sub-steps to actual
  code; **critical finding**: `deployment_replay.csv` is a
  per-candidate Bernoulli-sim *output* (80 cands × 6 tasks, K=6 PI
  era), NOT held-out real-rater response sequences as the plan's
  wording implied — so D6 is a true *build* (the real-rater source
  is `data/labels/labels.csv`). `validate.py` does not exist in
  this repo (Tier-2 OC `run_phase4_simstudy.py` lives in sibling
  methodology). Phase-2 validation scripts ARE carried byte-
  identical (SBC/coverage/lapse/sigma/test-retest/gold-chain) +
  hardcode K=6 in their entry points (K-agnostic internally).
  Sub-1 scope α: synthetic K=7 pins via NEW tests in
  `tests/test_phase7_k7_validation.py` — `test_sbc_skeleton_l_
  domain0_k7`, `test_posterior_coverage_k7`, `test_lapse_algebra_k7`
  (all slow-marked, ~8 s); the carried K=3 methodology skeleton
  tests are NOT edited (md5 byte-identity preserved). Engine
  K-agnostic at K=7: SBC mean rank ≈ 0.5, 90/95 % CI coverage
  > 0.7, lapse algebra floor/ceiling holds per-task. Doc:
  `docs/PHASE7_VALIDATION_K7.md` enumerates what does NOT need a
  K=7 re-run + why (lapse-sensitivity K-independent; sparcnet
  test-retest's K=6 IIIC ⊆ K=7's 6 IIIC, `combined_spike` has no
  SPARCNET analog; Σ-sensitivity K-agnostic in its own grid; gold-
  chain sampler correctness is K-independent; Phase-3.5 SBC engine
  artifact at K=6 IIIC covers the 6 IIIC of K=7). Full suite
  **243 passed / 1 xfailed**.
- **Phase 7 sub-step 2 — Tier-2 OC simstudy port + K=7 added (per
  user 2026-05-19, Q2 = port + re-run at K=7).** Faithful port of
  the methodology reference `run_phase4_simstudy.py` →
  `scripts/run_tier2_oc_simstudy.py` (the "Paper-1 Results
  centerpiece" synthetic ℓ-grid OC surface: ℓ∈[-1.5..1.0],
  K∈{2,4,6,8}, Σ_l∈{empirical,independent,cs0.7}, methods∈{random,
  brute,hier}). **K=7 added** to K_GRID; MAX_Q_BY_METHOD_K K=7 caps
  interpolated between K=6/K=8 (hier 5000, random 24000). Path-
  only edits per Phase-4.4-B precedent: (a) `sys.path` adds
  `engine/` (unified layout vs methodology root-flat), (b) output
  dir `results/phase2_validation/` (vs methodology
  `results/phase4_simstudy/`, per plan §"Phase 7"). Renamed
  consistently with plan terminology ("Tier-2 OC"). Port gated by
  `tests/test_phase7_tier2_oc_port.py` (7 fast tests, 0.28 s):
  design constants preserved, methodology MAX_Q caps no-drift,
  K=7 in K_GRID, monotone caps cap(6)<cap(7)<cap(8) per method,
  Σ_l well-shaped K=7 matrices (symmetric PSD; independent=I_7;
  cs0.7[0,1]=0.7; empirical=matched-mean CS), 30-task graph at
  K=7 (6 ℓ × 5 methods × correctly-crossed Σ_l-conds), output
  dir. Functional probe: one K=7 hier-empirical ℓ=0.0 session
  end-to-end in 42.5 s (reached δ=0.05@q834, δ=0.10@q204; right-
  censored at δ=0.025 within probe cap max_q=2000 — expected).
  **Honest pilot-progress evidence:** the in-session --n-reps 1
  K=7 pilot reached **20/30 sessions in 591 s** (per the script's
  progress line) before being terminated to keep the sub-step
  boundary clean — projected ETA total ~16 min; no JSON artifact
  persisted (parent SIGTERM propagated mid-`parallel_map`, main()
  never reached `with open(...)`). Sub-7.2 deliverable is the
  port + tests + probe + doc; the full --n-reps 1 K=7 pilot and
  the --n-reps 25 full-grid campaign are scoped as **separate
  paper-grade executions** (re-launchable via the same CLI). Doc:
  `docs/PHASE7_TIER2_OC.md`. Suite **250 passed / 1 xfailed** in
  15:31 (slower wall time documented as CPU-contention from the
  concurrently-running pilot, since terminated).
- **Phase 7 sub-step 3-A — D6 real-rater replay bank-builder (per
  user 2026-05-19, Q1=both engines, Q2=floor=10, Q3=fitted-θ
  Bernoulli comparator, Q4=three sub-sub-steps).** The first of
  three sub-sub-steps inside D6. New
  `pipeline/replay/build_rater_replay_bank.py` joins
  `data/labels/labels.csv` (with the erratum-correct {bipd,birds,
  other}→other mapping for IIIC tasks; spike typed-coerced to
  {0,1}) with `data/deployment_prior/case_bank.csv` (the K=7
  frozen production item bank) → per-(rater, task) strict-A bank
  with engine inputs `s_mean`/`s_sd` attached. Build 27.8 s on
  single-thread. Outputs: `data/replay/rater_replay_bank.csv.gz`
  (102 MB; 5,502,146 long-form rows) + `rater_replay_summary.csv`
  (801 KB; per-(rater, task) aggregates with `expertise_level`
  joined from raters.csv). Both gitignored as regenerable build
  artifacts (matches the precedent for Phase-3
  `pipeline/_calib_work/`). Headline inventory:
  *14,823 (rater, task) cells at the deployment N_min_per_task=10
  floor* (per-task headline cohort, 13,724 at floor=20); the
  *per-candidate cohort = 21 raters with ≥10 segs/task on ALL 7
  tasks* (19 expert + 2 experienced — the headline full-7
  cohort). 421,084 of 5,923,230 observations (7.1 %) dropped on
  the case_bank join — segs the rater scored that aren't in the
  production K=7 frozen bank (no s_mean signal). Pre-build audit
  found 22 per-candidate / 14,955 per-task; the 1-rater /
  132-cell drop is honest data hygiene (one rater had a task
  whose scored segs all lacked an s_mean in the production
  bank). Gate: `tests/test_phase7_replay_bank.py` (9 tests,
  2.25 s): outputs exist; schema = engine inputs (rater_id,
  task, seg_id, y, s_mean, s_sd); all 7 tasks populated;
  erratum-correct mapping (no `bipd`/`birds` task survives);
  Centaur 4-expert gold cohort verifies (cal/matt/tianyu =
  exactly 5,000 segs/IIIC task each, no spike; mbw=rater 97 on
  all 7 tasks); summary schema; per-task cohort ≥10,000 cells at
  floor=10 (each task ≥1,000 raters); per-candidate cohort
  within 20–23 raters (mostly expert); Y distributions
  sensible (spike pos rate 0.3–0.9, each IIIC task 0.05–0.40).
  Docs: `docs/PHASE7_REPLAY_DESIGN.md` (the full sub-7.3 design
  audit + this sub-step's close-out). Remaining: 7.3-B
  deployment replay driver; 7.3-C Mode-A replay driver +
  Bernoulli comparator + close-out. Suite **259 passed / 1
  xfailed** in 8:38 (+9 from sub-7.2).
- **Phase 7 sub-step 3-B — D6 strict-A deployment replay driver
  (2/3, per user 2026-05-19, Q1=both engines, Q3 design=A,
  Q4=three sub-sub-steps).** Engine attach: two minimal-surface
  optional kwargs added to `deployment/simulate_test.
  simulate_candidate` (defaults BYTE-IDENTICAL to pre-edit
  Phase-4 contract): `y_source` (callable replacing the default
  Bernoulli draw — the strict-A Y-lookup attach) and
  `initial_decision` (overrides the default `["pending"] * K_`;
  lets per-task replay pre-mark the 6 non-target tasks as
  `"refer"`). Drift-guard:
  `tests/test_phase7_replay_engine_drift.py` (2 tests, 5 s) pins
  the default behaviour at a fixed seed + verifies the new kwarg
  branches are actually consumed when provided. New
  `deployment/replay/run_deployment_replay.py` (CLI: `--floor`,
  `--limit`, `--out-dir`, `--seed`) drives the K=7 deployment
  engine with the rater's strict-A bank from
  `data/replay/rater_replay_bank.csv.gz`; produces
  `results/replay/deployment_replay_per_task.csv` (per (rater,
  task) row), `deployment_replay_per_candidate.csv` (per rater,
  full 7-task decisions + roll-up), and
  `deployment_replay_run_summary.json` (config + headline
  aggregates). **Per-task interpretation-(i) fix discovered in
  smoke**: the pathology where every per-task replay halted at
  exactly n=10 with REFER was root-caused to `select_next_case`'s
  `n_min_per_task` constraint forcing the engine to try selecting
  from the 6 empty-bank tasks → `None` → outer loop breaks. Fix:
  `initial_decision = ["refer"] * K_; initial_decision[k_target]
  = "pending"` removes the 6 from `pending` so the engine focuses
  on the target task naturally (up to `N_max_per_task=120`).
  Smoke result (rater 97 = mbw, per-task replay): gpd PASS@n=14
  (pos_rate=0.64), grda PASS@n=34, lpd PASS@n=41, lrda PASS@n=22,
  `other` REFER@n=120 — real verdicts driven by real data. Tests:
  `tests/test_phase7_deployment_replay.py` (8 tests, 24 s):
  `_RaterYLookup` returns labels.csv Y on every (k, seg);
  contract-violation raise; strict-A bank construction;
  engine-Y-matches-labels (per-task + per-candidate via call-
  logging y_source — state.history records signal s_val, not
  seg_id); end-to-end smoke; per-task interpretation-(i) skips
  the 6 non-target tasks. .gitignore: adds `results/replay/` as
  regenerable build artifact alongside the bank.
  Doc: `docs/PHASE7_REPLAY_DESIGN.md` §10 (engine attach-point +
  smoke + tests + roadmap to 7.3-C). Remaining inside sub-7.3:
  **7.3-C** = Mode-A replay driver + fitted-θ Bernoulli paired
  comparator + parallelization (`parallel_map` integration for
  the 14,823-cell headline run) + sub-7.3 close-out. Suite **269
  passed / 1 xfailed** in 15:55 (+10 from sub-7.3-A's 259 = 2
  drift-guard + 8 replay correctness; the slower wall is
  documented as poking-during-gate contention, not a regression).
- **Phase 7 sub-step 3-C — D6 real-rater replay HEADLINE (3/3,
  closes sub-7.3; per user 2026-05-19, Q1=both engines, Q3
  design=A, comparator=fitted-θ Bernoulli, Q4=both cohorts).**
  Mode-A engine attach: two minimal-surface optional kwargs added
  to `engine/core_mcmc.choose_item` (`bank_segids` — appends
  chosen seg_id to return tuple) and
  `engine/core_mcmc.run_session_mcmc_auroc` (`bank_segids` +
  `y_source` — pair-wired with the per-iteration choose_item
  call). Defaults BYTE-IDENTICAL to pre-edit; gated by
  `tests/test_phase7_mode_a_replay_drift.py` (6 tests, 0.43 s):
  default 2-tuple / return_sd 3-tuple / bank_segids 3-tuple /
  bank_segids + return_sd 4-tuple return signatures; the new
  y_source path is consumed; the K-agnostic session driver
  preserves structural invariants. NEW `pipeline/replay/_common.
  py` (shared helpers used by both engine drivers): pre-indexed-
  pickle bank loader (`rater_replay_bank.indexed.pkl`, ~60×
  faster vs gz CSV; built from the bank-builder); pre-cached
  per-rater y-table lookup; fitted-θ table joining `sdt_fits.
  csv` + `sdt_fits.spike.csv` with the canonical name index;
  ENGINE_TASK_TO_SDT_DOMAIN map (matches v13 _V13_TASK_KEY).
  NEW `pipeline/replay/run_replay.py` (unified parallel driver):
  `--engines`/`--arms`/`--cohorts` product × parallel_map at 14
  workers; engine-agnostic CLI; fitted-θ Bernoulli comparator
  reuses the engine's default Bernoulli path (true_params =
  fitted θ; y_source=None) — no extra code. Performance tuning:
  Mode-A replay-grade params (per_task K=1 N=200 max_q=120,
  per_cand K=7 N=200 max_q=350, n_subsample=100) — ~10× faster
  than paper-grade (verified via cProfile: 99.7% of time in
  `_expected_loss_vec`; bottleneck is K=7 EV item selection, not
  bank scan).
  **HEADLINE RUN COMPLETED** (14 workers, caffeinate -is for
  macOS App Nap defeat):
    Deployment full × both arms: **29,688 sessions in 7:36 wall**
        (65 sess/sec; zero errors)
    Mode-A per_cand × both arms: **42 sessions in 25:21 wall**
        (36 s/session; zero errors)
    Combined: 32:57 wall for the full v1.0-blocker D6 headline.
  **Major scientific finding**: real-rater replay produces
  SUBSTANTIALLY more PASS verdicts than the fitted-θ SDT
  Bernoulli predicts — spike +43%, seizure +409%, gpd +235%,
  grda +533%, lrda +1,154%, other +24%. Mode-A AUROC posteriors
  echo the directional finding: replay yields +0.017 to +0.027
  higher mean AUROC across 6 IIIC tasks (the lone exception is
  `other`, −0.010), with **25-46% tighter CI halfwidths** and
  **1.94× fewer questions to δ=0.05 stop** (mean n_q 145 vs
  281). Interpretation: the SDT-fit Bernoulli systematically
  UNDER-predicts real-rater skill — raters are more skilled and
  more coherent than their (σ, θ) fit suggests; the v1.0
  deployment will be MORE lenient than the Bernoulli OC tables
  predict. Favourable finding for the v1.0 claim, honest to
  reviewers. Outputs (gitignored): `results/replay/` (deployment
  per_task + per_cand), `results/replay_mode_a/` (Mode-A
  per_cand). Doc: `docs/PHASE7_REPLAY_HEADLINE.md` (full
  sub-7.3-C close-out + headline tables + scientific reading).
  Mode-A per_task headline (14,823 × 2 = 29,646 sessions)
  documented as paper-grade follow-on (re-launchable via the
  same CLI). Suite **275 passed / 1 xfailed** in 9:10 (+6 from
  sub-7.3-B's 269 = the 6 new Mode-A drift-guard tests).
- **Phase 7 sub-step 4 — Paper-1 figures regeneration at K=7
  (CLOSED, 3-of-3).** Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7"
  sub-4 (user scope 2026-05-20). Three sub-sub-steps; total wall
  ~63 min across two figure pipelines + one bank-vendor +
  scaffolding commit; zero engine drift (sub-7.1-style synthetic
  K=7 pins from `tests/test_phase7_k7_validation.py` re-confirmed
  by the production OC numbers below).

  * ✅ **7.4-A — Phase-1 main figure** (commit `6aeb036`).
    SPARCNET item-bank vendored in-repo (7 JSONs, 168 KB, md5
    verified vs `docs/_manifests/reference_consulted.md5`);
    `bridge/_common._autodetect_banks_dir()` precedence flipped
    (in-repo first → D9 self-contained; sibling fallback retained
    for back-compat). `scripts/run_phase1_experiments_v2.py`
    `_KS_B`: `{2,4,6,8}` → `{2,4,6,7,8}`, K=7 caps interpolated
    between K=6 and K=8 per method (matches sub-7.2
    `MAX_Q_BY_METHOD_K` convention). `scripts/run_phase4_figures.
    py` retargeted to the Tier-2 OC rows path
    (`results/phase2_validation/tier2_oc_simstudy_rows.json`)
    with `_subplot_grid(n)` K-adaptive (1→1×1, 4→2×2, 7→3×3) +
    `--rows`/`--out-dir` CLI overrides. Headline run (14-core
    MacBook, `caffeinate -is`): **ExpA 405 sessions in 56:18 +
    ExpB 375 sessions in 2:16 = 780 sessions in 58:34 wall,
    zero errors**. ExpA K=6 SPARCNET median n_q @ δ=0.05:
    random 967 / brute 458 / hier **412** → **hier-vs-random
    2.35×**; brute-vs-hier (pooling alone) 1.11× (consistent
    with the weak-correlation regime r≈0.378; ablation gain
    dominated by adaptive item selection at this Σ_l). ExpB
    synthetic K-scaling now covers production K=7: hier vs
    random speedup band 1.6–1.9× across K∈{2,4,6,**7**,8},
    with K=7 = **1.63×** (hier 477 vs random 776) — cleanly
    interpolating between K=6 and K=8 in absolute n_q AND in
    speedup. No engine pathology at production K. Tests:
    `tests/test_phase7_sub74_figures.py` (7 tests, 1.01s):
    bank vendor + md5 + bridge precedence; ExpB K=7 + monotone
    caps; `run_phase4_figures.py` ROWS_PATH + subplot grid
    1..7. Doc: `docs/PHASE7_PAPER1_FIGURES.md`. Outputs
    (gitignored): `results/phase1_figures/`. Suite **282 passed /
    1 xfailed** in 9:17 (+7 from sub-7.3-C's 275 = the 7 new
    sub-7.4 drift-guard tests).
  * ✅ **7.4-B — Tier-2 OC K=7 pilot + figures** (commit
    `2627ccd`). `python scripts/run_tier2_oc_simstudy.py
    --k-grid 7 --n-reps 1 --max-workers 14` → 30 sessions
    in **4:33 wall** (273.2s, 9.11s/session amortized; 3.5×
    faster than sub-7.2's ~16-min projection thanks to 14
    workers vs the methodology default 10). `python scripts/
    run_phase4_figures.py` rendered three 1×1 single-K
    figures via the sub-7.4-A `_subplot_grid` helper.
    **Censoring=False at BOTH δ=0.05 AND δ=0.025** — the
    sub-7.2 K=7 max_q cap interpolation (hier/brute=5000,
    random=24000) is budget-sufficient. K=7 OC at δ=0.05
    hier-vs-random speedup ranges 1.01×–2.34× across the
    6-AUROC sweep; the directional finding matches the
    sub-7.4-A ExpB K=7 1.63× synthetic. Honest pilot
    caveats documented in the close-out doc: n_reps=1 makes
    bootstrap CIs degenerate (CI=median); paper-grade
    `--n-reps 25 --k-grid 2 4 6 7 8` re-launchable. At
    AUROC 0.687/0.841 the brute/hier pooling-only ratio
    inverts (<1.0×) — single-rep noise + the K=7
    `_sigma_l("empirical")` being a matched-mean CS
    approximation of the K=6 empirical fit. Outputs
    (gitignored): `results/phase2_validation/
    tier2_oc_simstudy_rows.json`, `oc_summary.json`,
    `fig_oc_{surface,delta_censored,hier_gain}.{pdf,png}`.
    No code delta vs sub-7.4-A (scaffolding was front-
    loaded); commit is doc-only.
  * ✅ **7.4-C — close-out** (this commit). CHANGELOG entry +
    final full-suite gate + `docs/PHASE7_PAPER1_FIGURES.md`
    close-out section. **Phase 7 sub-step 4 SHIPPED**: the
    Paper-1 main ablation figure regenerated on the unified
    engine+corpus (ExpA K=6, ExpB K∈{2,4,6,**7**,8}); the
    Tier-2 OC pilot at production K=7 (uncensored even at
    δ=0.025); the deployment figures already shipped at K=7
    by Phase 4.7. The Phase-2 inference-validation 4-panel
    composite is **explicitly deferred** (user scope 2026-
    05-20): the K=6 SBC 12/12 + coverage |Δ|≤0.007 + sparcnet
    retest 6/6 ICC≥0.70 + gold-chain 35/36 finding is already
    the Paper-1 calibration claim, and sub-7.1 added the
    synthetic K=7 engine-soundness pins via tests; regen is a
    Phase-8 figures-only follow-on if a reviewer asks.

  **Phase 7 sub-7.4 gate satisfied** — all three deliverable
  figure sets exist at the unified-engine + K=7 production
  configuration: `results/phase1_figures/` (Paper-1 main + supp),
  `results/phase2_validation/` (Tier-2 OC), and
  `data/deployment_prior/figures/` (deployment, from Phase 4.7).
  Suite **282 passed / 1 xfailed**. **Remaining**: 7.5 close-out
  + Phase-7 gate.
- **Phase 7 sub-step 5 — Close-out + Phase-7 GATE (CLOSED).** Per
  `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7" sub-5. Doc-only commit
  consolidating evidence across sub-7.1–7.4 in `docs/PHASE7_CLOSEOUT.
  md`: Phase-7 commit chain (9 commits from `d9cda94` sub-7.1 →
  this commit); per-sub-step evidence map → gate criteria; 7-row
  delta table from prior validated runs (every delta signed off
  in the originating close-out doc); retrospective sign-off for
  the "pre-registered bounds" on the real-rater replay OC (four
  bounds in order of strictness — hard / hard / soft / directional
  — all satisfied). **Phase-7 gate SATISFIED**: coverage within
  target band ✓ (sub-7.1 K=7 + carried K=6 Phase-2); real-rater
  replay OC computed and within bounds ✓ (sub-7.3); no unexplained
  regressions; all deltas attributed to intended corpus /
  likelihood / calibration / K=7 changes and signed off ✓. Paper-
  grade follow-ons documented as re-launchable + non-blocking
  (Mode-A per_task headline at full cohort; paper-grade Mode-A
  precision N=2500; Tier-2 OC at n_reps=25 K=7-only or full K=
  {2,4,6,7,8}; Phase-2 inference-validation figure regen if a
  Phase-8 reviewer asks; formal hypothesis-test layer for replay-
  vs-Bernoulli paired tests). Suite **282 passed / 1 xfailed**
  in 9:13, unchanged from sub-7.4-A; 36 Phase-7-specific tests
  passing (6 slow deselected by default). **Phase 7 SHIPPED.**
  Phase 7 → Phase 8 handoff is clean; remaining work is
  documentation + packaging + open-decisions carrying. No further
  engine / calibration / scientific build required.
- **Phase 8 — Shippability (5-of-5 sub-steps CLOSED; v1.0.0-rc1
  SHIPPED).** Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 8". User
  scope locked 2026-05-20: contributor-facing developer README +
  reviewer-grade CLAUDE.md; per-task certificates as v1.0
  per-candidate roll-up policy; `ilae-calibrate` rewired through a
  thin `--help`-safe wrapper; in-place gate (this venv).

  * ✅ **8.1** (`6d836ca`) — README.md (193 lines, was 19) +
    CLAUDE.md (162 lines, was 10) merged; Phase-6 invariant
    corrections baked in (70/30 expert split, raters.csv:
    expertise_level, K=7 r_ℓ=0.36656…, λ=0.025, LOGIT_TO_PROBIT=
    1/1.7); two-engine architecture documented; "what NOT to do"
    rules + contributor conventions in CLAUDE.md.
  * ✅ **8.2** (`07bc5d8`) — `docs/OPEN_DECISIONS.md` created
    with 5 open shipping decisions (per-task certs adopted as
    v1.0 working policy, supported by sub-7.3-C empirical 0/21
    all_pass finding); `data/SENSITIVE.md` refreshed to cover
    Phase-5 + Phase-7 files; D9 retrieval-path doc added
    (`SN1_combined_v2.h5` stays external). `tests/test_phase1_
    data.py:test_carry_forwards_present` byte-pin replaced with
    IRB-coverage content invariant (the Phase-8 refresh
    intentionally diverges from the methodology-repo copy per
    merge plan §"Phase 8" sub-3).
  * ✅ **8.3** (`b968f9e`) — packaging: `pyproject.toml` version
    `0.0.0.dev0` → **`1.0.0rc1`** (PEP 440 form of the v1.0.0-rc1
    git tag). `ilae-calibrate` entry-point rewired
    `calibration.run_youden_calibration:main` (Phase-0 stub) →
    `calibration.cli:main` (NEW thin wrapper, 88 lines).
    Foot-gun fix: the Phase-3 orchestrator runs ~hours on ANY
    invocation including `--help`; the wrapper argparse-
    intercepts BEFORE any heavy import. `--help` in 0.23s,
    `--dry-run` in 0.02s. Phase-0 stub
    `calibration/run_youden_calibration.py` DELETED (orphaned).
    6 packaging drift-guards in
    `tests/test_phase8_packaging.py` (foot-gun guard included).
  * ✅ **8.4** (`1ce4341`) — in-place gate + **real
    reproducibility bug surfaced + fixed in flight**:
      Step 1 — full slow suite: **288 passed / 1 xfailed** in
        9:28 (+6 from sub-7.5's 282 = sub-8.3 packaging guards).
      Step 2 — `ilae-deploy all` end-to-end:
        FIRST run revealed ~1-ULP drift in `hat_*`/`sd_*` columns
        of `sim/candidates.csv` vs the committed Phase-4.6-B
        baseline. Diagnosis: `OPENBLAS_NUM_THREADS=` unset →
        multi-thread BLAS active → engine's bit-exact-
        reproducibility contract violated (conftest.py enforces
        single-thread for pytest; the CLI did NOT).
        Fix: BLAS env-var setter (5 env vars) at the top of
        `deployment/cli.py` AND `bridge/run_multi_auroc_bridge.
        py` BEFORE any numpy-importing code. Deferred imports
        keep numpy out of module load. Re-run: working-tree diff
        byte-clean except for `sim/summary.json:wall_time_s`
        (intentional timing noise). +2 BLAS source-inspection
        drift-guards in `tests/test_phase8_packaging.py` (now
        8 packaging tests).
      Step 3 — `ilae-paper --max-raters 3 --method both --n-reps
        1 --no-audit`: **6 / 6 sessions in 28:17 wall, zero
        errors**. hier ~440s/session (δ=0.05 stops at q=141 / 239
        / 447); brute ~125s/session (q=943 / 701 / 449); hier-
        vs-brute n_q ratio @ δ=0.05 = 6.7× / 2.9× / 1.0× across
        3 raters — directionally consistent with the Phase-1 v2
        paper-grade finding.
      Phase-8 hygiene caught + fixed: `data/eeg_bank.h5` (175 MB
        PHI-bearing EEG, mtime 2026-05-20 11:16 from local
        `scripts/eeg_bank_viewer.py` exploration) was untracked
        but NOT in `.gitignore`. **D9 defensive ignore added**.
        `results/mode_a_auroc/` also added (oversight from
        sub-8.3).
  * ✅ **8.5** (this commit) — close-out + `v1.0.0-rc1` tag.
    `docs/PHASE8_CLOSEOUT.md` consolidates per-sub-step evidence
    + ships-list + carries-list; CHANGELOG entry; final full-
    suite gate re-confirmed (288 / 1 xfailed); annotated git
    tag `v1.0.0-rc1`.

  **Phase 8 gate satisfied** (all 6 criteria per merge plan
  §"Phase 8"): README/CLAUDE merged with Phase-6 corrections;
  env.yml/requirements.txt union pinned (since Phase 0);
  SENSITIVE.md + D9 retrieval doc; OPEN_DECISIONS.md created;
  `v1.0.0-rc1` tagged at this commit; full gate triad green.
  **v1.0.0-rc1 SHIPS.** Post-tag carries documented in
  `docs/PHASE8_CLOSEOUT.md` §7: anonymizer runs at journal
  acceptance; open-decision resolution + external-cohort
  validation deferred to v1.0 review meeting + paper revision;
  paper-grade Tier-2 OC / Mode-A Phase-1 / per_task replay
  re-launchable via their respective CLIs. Tag `v1.0.0` after
  v1.0 review meeting signs off open decisions.

---

## ▶ CORTEX click-to-run app (2026-05-26) — `cortex-v1.0` → `cortex-v1.0.5` — SHIPPED

PyInstaller-bundled standalone test-taker app distributed via GitHub
Releases as `CORTEX-mac.dmg` (~308 MB) and `CORTEX-windows.zip`
(~319 MB). Zero Python install required for end users; clinicians
download, double-click, Gatekeeper "Open Anyway" once, run. Build
infra: `.github/workflows/cortex-release.yml` triggers on `cortex-v*`
tags, fetches the test bank from the `build-data-v1` release (no AWS
deps in CI), stages `cortex_config.yaml` from the
`CORTEX_CONFIG_YAML` repository secret, runs PyInstaller against
`cortex_app/cortex.spec` on parallel macOS + Windows runners, and
attaches both artifacts to a new GitHub Release. Complementary to
`scripts/build_internal_test_zip.py` (bash-launcher zip for users
willing to install Python 3.11 themselves).

Six iterations on 2026-05-26 — each a CI build + ship + test cycle.
Bug surface mostly: PyInstaller's static AST tracer cannot see
dynamic / lazy / pickle-driven imports, and macOS `console=False`
bundles redirect stdout to /dev/null.

- **v1.0.0** (initial) — PyInstaller scaffold under `cortex_app/`
  (`cortex.spec`, `build_mac.sh`, `build_windows.bat`,
  `fetch_test_bank.sh`, `BUILD.md`). Dedup pass at commit `a102ec5`
  removed duplicated engine source under `cortex_app/scripts/`;
  spec points at repo-root `scripts/` / `engine/` / `Sigma_l_fitted.npy`
  as the single source of truth.

- **v1.0.1** — fix: registration write into the read-only `.app`
  bundle under macOS App Translocation (downloaded unsigned bundles
  run from a temporary `/private/var/folders/.../AppTranslocation/...`
  path that is read-only). Added `cortex_storage.user_data_root()`
  to route per-session output to the platform-appropriate user-data
  dir when `sys.frozen`: `~/Library/Application Support/CORTEX/`
  (macOS), `%LOCALAPPDATA%\CORTEX\` (Windows),
  `$XDG_DATA_HOME/CORTEX/` (Linux). Dev runs still write to
  `<repo>/results/`.

- **v1.0.2** — fix: `ModuleNotFoundError: No module named 'core_mcmc'`
  at session start. `_REPO = Path(__file__).parent.parent` doesn't
  resolve to the data unpack root in PyInstaller — PYZ-module
  `__file__` is a synthetic path inside the archive, and the entry
  script's `__file__` is `<MEIPASS>/<entry>.py` whose `parent.parent`
  is one level ABOVE MEIPASS. Anchored `_REPO` on `sys._MEIPASS`
  when `sys.frozen` in `cortex_engine_inputs.py`,
  `session_controller.py`, `cortex_storage.py`.

- **v1.0.3** — fix: `ModuleNotFoundError: 'numpy._core'` on
  `load_fitted_Sigma()`. `Sigma_l_fitted.npy`'s pickle references
  `numpy._core.multiarray._reconstruct` (a numpy 2.x compat path);
  numpy 1.26.4 ships a 7-module `numpy/_core/` shim package that
  forwards to `numpy.core.*` lazily — invisible to the AST tracer.
  Parallel PhD-level audit (two independent agents in isolated
  worktrees) synthesized into:
    * `collect_submodules('numpy._core')` + explicit pin list
      (Agent A + Agent B converged)
    * `collect_data_files('certifi')` — Dropbox HTTPS via `requests`
      loads the CA bundle via `importlib.resources` (Agent A only;
      Agent B's probe didn't make HTTPS calls)
    * `pyqtgraph.{graphicsItems.ImageItem, graphicsItems.TextItem,
      widgets.PlotWidget, colormap}` — concrete classes used by the
      viewer (Agent A only; Agent B didn't drive GUI init)
    * `scipy.special.cython_special` + `scipy.special._ufuncs` +
      `scipy.stats._continuous_distns` — defensive coverage for
      engine particle-update calls (Agent A only)
    * `yaml` / `_yaml` / `dropbox` + `collect_submodules('dropbox')`
      — explicit, defensive (both agents)
    * `cert_config.yaml` bundle-root fallback for
      `cortex_policy.load_ell_star_iiic`'s `alt = _REPO / "cert_config.yaml"`
      branch (Agent B)
    * Latent v1.0.1-pattern bug fix in `cortex_policy.py:_REPO`
      (Agent B; would have crashed AD6Policy construction once
      `numpy._core` was unblocked). Agent B empirically built the
      bundle locally with PyInstaller 6.20 + numpy 1.26.4 and ran a
      console probe that loaded Sigma + AD6Policy + dropbox + yaml +
      matplotlib + eeg_bank.h5 inside the frozen bundle: ALL CHECKS
      PASSED. v1.0.3 mac build first failed at the
      `actions/upload-artifact@v4` step (transient GitHub Actions
      storage hiccup); re-running failed jobs via API succeeded.

- **v1.0.4** — fix: `FileNotFoundError` at `BankViewer.__init__`
  opening `data/eeg_bank.h5`. Entry-script `__file__` in PyInstaller
  one-folder mode resolves to `<MEIPASS>/eeg_bank_viewer.py`, so
  `Path(__file__).parent.parent` goes ONE LEVEL ABOVE the data
  unpack root (resolved to `/Applications/CORTEX.app/Contents/`
  rather than `Contents/Frameworks/`). Same `sys._MEIPASS` gate
  applied to `BANK_PATH` / `SPEC_PATH` in `eeg_bank_viewer.py`;
  defensive same fix in `cortex_render_videos.py` `_THIS_DIR` /
  `_REPO`. Verified end-to-end: session creates the user-data dir,
  writes `participant.json` + `trials.jsonl` + `events.jsonl` +
  `certificate.json` + `trajectory.npz` + two CSVs.

- **v1.0.5** — feature: file-based logging. macOS `.app` bundles
  built with `console=False` send stdout to /dev/null at the
  bootloader, even when launched from a shell. Sessions in v1.0.4
  ran clean but the operator had zero post-session visibility into
  Dropbox upload status, MP4 render outcome, or warnings (only
  uncaught tracebacks survived via `sys.excepthook`'s low-level
  syscalls). Added `cortex_storage.setup_logging()` — idempotent,
  `FileHandler` on `user_data_root() / "cortex.log"` always,
  `StreamHandler` on stderr in dev (`sys.frozen=False`). Called from
  `eeg_bank_viewer.main()` first thing before `QApplication`
  construction. Migrated 13 `print()` sites across
  `cortex_storage.py`, `cortex_render_videos.py`,
  `cortex_engine_inputs.py`, `eeg_bank_viewer.py` to
  `logger.{info, warning}`. `logging.basicConfig(force=True)` also
  captures stdlib loggers (`matplotlib.animation`, `dropbox`,
  `matplotlib.font_manager`) for free.

**Empirical verification (v1.0.5, 99-trial session through bank
exhaustion):**

```
17:10:15 INFO cortex_storage: logging initialised; frozen=True
17:13:33 WARNING matplotlib.font_manager: building the font cache (first run)
17:13:39 INFO matplotlib.animation: MovieWriter._run: ffmpeg ... collapse.mp4
17:13:48 INFO cortex_render_videos: collapse.mp4 rendered in 8.5s (4.8s video, 0.7 MB)
17:13:56 INFO cortex_render_videos: passfail.mp4 rendered in 8.6s (4.8s video, 0.2 MB)
17:13:56 INFO cortex_storage: Dropbox client constructed (refresh-token mode)
17:13:57 INFO dropbox: Refreshing access token.
17:13:57 INFO dropbox: Request to files/upload
17:13:58 INFO cortex_storage: uploaded 2 result CSV(s) to Dropbox /results/
```

Result: 99 trials, `stop_reason=bank_exhausted`, 20/99 correct
(20.2%), one PASS verdict (`lrda`: AUROC 0.907 ± 0.037), MP4 render
✓ (system ffmpeg available), Dropbox upload ✓ (refresh-token mode +
access-token refresh + 2× `files/upload`).

**Known issues, deferred (NOT blocking v1.0.5):**

- **AD6 calibration overrides still active.** `scripts/cortex_policy.py`
  ships `DEFAULT_N_MIN = 6` + `DEFAULT_ALPHA = 0.30` (internal-test
  settings, panel-derived production values are 15 + 0.05). Documented
  in `docs/AD6_RESOLUTION.md` "Internal-test override (2026-05-22)".
  Must revert before any push to the public 10k bank. No CI gate
  enforces this today.
- **ffmpeg not bundled.** MP4 renderer uses
  `matplotlib.animation.FFMpegWriter` which shells out to system
  `ffmpeg`. Dev machines (Homebrew) succeed; clinician machines
  without ffmpeg log `WARNING cortex_storage: per-test-taker video
  render failed: ...` and skip MP4 production (session itself
  completes cleanly). To fix: bundle a static `ffmpeg` binary in
  `binaries=` and point `matplotlib.rcParams['animation.ffmpeg_path']`
  at it (~80 MB to the .dmg).
- **Eli's `scripts/build_internal_test_zip.py` bash-launcher path
  remains** as the complementary distribution for tech-comfortable
  users with Python 3.11 already installed.

Tags (annotated; each triggered `cortex-release.yml` CI): `cortex-v1.0`,
`cortex-v1.0.1`, `cortex-v1.0.2`, `cortex-v1.0.3`, `cortex-v1.0.4`,
`cortex-v1.0.5`. Test bank pinned at `build-data-v1`.

---

## ▶ CORTEX bundle (2026-05-26 → 2026-05-27) — `cortex-v1.0.6` → `cortex-v1.1.0`

Six intermediate shipments between the v1.0.5 close-out above and the
v1.1.1 entry below. Catalogued here so the CHANGELOG covers the full
release timeline; details live in the per-tag commit messages.

- **v1.0.6** (`73694a8`) — drop 38 MB of `calibration/` dev-only outputs
  from the PyInstaller bundle. Pure size reduction; no behaviour change.
- **v1.0.7** (`685b573`) — add Linux x86_64 to the release pipeline.
  `.github/workflows/cortex-release.yml` now produces three artifacts
  per tag (macOS DMG + Windows ZIP + Linux bash launcher).
- **v1.0.8** (`e98db34`) — DMG layout fix: include the
  `/Applications` symlink as a drag-target so users land in the right
  install location on first download; expand the release-notes
  language for the macOS Sequoia first-launch / Gatekeeper dance.
- **v1.0.9** (`fc05bf0`) — diagnostic-log upload on uncaught exception.
  `cortex_diagnostics.py` registers a `sys.excepthook` that ships the
  exception payload (type/message/traceback + OS/Python/app version +
  redacted tail of `cortex.log`) to Dropbox `/debugging-incidents/`
  using the same refresh-token client as `cortex_storage`. CSV-filename
  log lines containing the safe-name-sanitised participant name are
  redacted before upload (privacy contract: **no PHI ships**).
- **v1.1.0** (`9bdc4a2`) — the headline release of this block:
  doubled the IIIC bank to **300 segments (50 per class)** so the
  certification algorithm has a real 300-question runway, and
  advanced the AD6 strictness knobs from internal-test (`N_MIN=6`,
  `ALPHA=0.30`) toward panel-derived production (`N_MIN=15`,
  `ALPHA=0.10`). Test bank pinned at `build-data-v2`. AD6 evolution
  documented in `docs/AD6_RESOLUTION.md` "v1.1.0 update". Bank
  construction in `scripts/build_cortex_test_bank_v2.py`
  (deterministic seed=42; quality-biased — sort by `n_raters DESC`;
  v1 segments preserved for backward comparability; +200 new picks
  with median `n_raters=99` vs v1 median 81).

Tags (annotated; each triggered `cortex-release.yml` CI): `cortex-v1.0.6`,
`cortex-v1.0.7`, `cortex-v1.0.8`, `cortex-v1.0.9`, `cortex-v1.1.0`.

---

## ▶ CORTEX bundle (2026-05-27) — `cortex-v1.1.1` — participant-info wizard + test-suite cleanup

**Headline:** participant registration form rewritten as a structured
3-page wizard with mineable dropdown demographics, in preparation for
a Nature-Medicine-grade public-release dataset. Engine, calibration,
deployment, and bank are untouched (no impact on `cert_config.yaml`
v13 or the Phase-6 invariants); this is a CORTEX-app-only release.

**Participant form (the v1.1.0 free-text form → v1.1.1 wizard).**
Free-text fields that resisted analysis (gender, credentials) replaced
with controlled dropdowns; new fields added to support equity,
generalizability, and calibration analyses. `RegistrationPage` in
`scripts/eeg_bank_viewer.py` is now a `QStackedWidget` with three
steps, preserving the class-level API (`continue_btn`, `commit()`,
`f_name`/`f_email`/`f_expertise` widget handles for back-compat with
`cortex_smoke.py` and tests):

  * **Page 1 — Identity & eligibility:** name (required), email
    (required), age (optional free text), eligibility checkbox
    ("I am a healthcare professional or student in a clinical/research
    role") that gates advance.
  * **Page 2 — Clinical background:** institution (optional),
    `expertise` (now required), `practice_setting` (Academic medical
    center / Community hospital / Tele-EEG service / Private practice
    / Training only / Other), `years_reading_eeg` (5-yr bins, 0–4 …
    30+; **replaces** v1.1.0 free-text `credentials`),
    `eeg_volume_per_month` (fewer than 5 / 5–20 / 21–50 / 51–100 /
    more than 100), `self_rated_confidence` (1=Very low … 7=Very
    high, optional), `color_vision` (optional), `prior_test_taken`
    (optional, test-retest control).
  * **Page 3 — Demographics (all optional, every dropdown defaults
    to a "Prefer not to say" answer):** `sex` (Male / Female /
    Prefer not to say; **renamed from** v1.1.0 `gender`),
    `gender_identity` (SAGER-style separate field), `country` of
    practice (curated 24-country list), `race_ethnicity`
    (NIH categories + MENA per Federal Register 2024 OMB SPD 15).

**Schema migration.** `registrations.csv` columns went from 9 to 20.
Schema-v2 fieldnames pinned in `eeg_bank_viewer.REGISTRATION_FIELDS_V2`.
`RegistrationPage._save_row` detects a v1-header file on disk, rotates
it to `registrations.v1.csv.bak`, and writes the new schema — preventing
silent column drift on upgrade (DictWriter does not validate against
existing headers). Required-field validation now hard-gates Continue
on pages 1 & 2; demographic page is fully optional.

**Hidden bookkeeping fields.** Each row stamps `consent_version`
("v1.1.1-placeholder" until the public-release IRB amendment lands) and
`irb_protocol_id` (empty placeholder; the in-repo IRB language at
`ConsentPage._TERMS` still flags itself as placeholder and is bracketed
for IRB replacement before public deployment).

**Upload surface.** `cortex_storage._summary_row` now flows all 12 new
demographic + clinical-background fields into the per-session summary
CSV that uploads to Dropbox — back-compatible with pre-v1.1.1
participant dicts (missing keys default to ""). The local
`registrations.csv` continues to hold the same fields plus name +
email + age + institution; analysis joins on `session_id`.

**Test-suite cleanup (the unblocker on this ship).** Pre-v1.1.1 the
suite ran 350 passed / 6 failed / 9 skipped on a fresh clone. After
cleanup: **357 passed / 0 failed / 11 skipped** (all skips honest).

  * **`requirements.txt` install** of the missing pip packages
    (`openpyxl`, scikit-learn, statsmodels, arviz, pingouin,
    jax/jaxlib, numpyro, et al — 30 packages) into the project venv.
  * **Symlink** `/data/eli-work/repos/ilae-skill-certification-test-main`
    → `ilae-skill-certification-test` so the cross-repo byte-id tests
    can resolve the role-C reference repo (per
    `docs/MERGE_SOURCE_MANIFEST.md`).
  * **`test_render_failure_does_not_break_finalize`** — replaced
    `capsys` capture with `caplog.at_level(WARNING, "cortex_storage")`;
    the production code at `cortex_storage.py:375` migrated from
    `print` to `logger.warning` in v1.0.5 (commit `27a27f4`) but the
    test wasn't updated.
  * **Phase-3 cross-repo byte-equivalence tests** (3 in
    `test_phase3_calibration.py`) — added `pytest.skip()` guards
    matching the existing pattern at line 86 (`test_run_youden_byte_identical`).
    `test_youden_sigma_star_extract_byte_identical` refactored to find
    `youden_sigma_star` by function name rather than hard-coded line
    numbers (the role-C `train_val_split_and_fit.py` was refactored
    post-merge and the function moved from line 211 to line 171).
    In-repo md5 pins on `pipeline/reference_calibration/*` remain
    active — the cross-repo arm is defence-in-depth and skips when
    its prerequisite is absent.
  * **`test_phase6_invariants.py::test_fit_sdt_two_copies_byte_equivalent`**
    — turned the file-existence `assert` for the derived
    `pipeline/_calib_work/src/fit_sdt_per_domain.py` into a
    `pytest.skip()`, matching the convention used by the 8 other
    `_calib_work/` skips. In-repo md5 pin
    (`6b90d59dcd0aaa878f9a52802254044b`) on the canonical copy
    remains actively enforced.

**Drift finding (carried as an open item).** `src/fit_sdt_per_domain.py`
in the role-C reference repo no longer exists (no git history; the
file was apparently in the working tree at merge time but never
committed). The merge manifest at
`docs/_manifests/reference_consulted.md5` still records it with the
expected md5. The multi-repo's own
`pipeline/reference_calibration/fit_sdt_per_domain.py` is unaffected
and still hashes correctly. To restore the cross-repo byte-equivalence
check, the file would need to be re-added to the single repo with the
byte-equivalent content. Not blocking v1.1.1.

**Tests added (`tests/test_cortex_viewer.py`, `tests/test_cortex_storage.py`).**
  * `test_registration_generates_session_id` updated to set the new
    required-field surface.
  * `test_registration_blocks_without_eligibility` — eligibility
    checkbox hard-gates commit and pulls the user back to page 1.
  * `test_registration_requires_sex_dropdowns` — parametrised over the
    4 page-2 required dropdowns (expertise, practice, years_eeg,
    eeg_volume); commit must fail when any one is unset.
  * `test_registration_csv_schema_v2` — persisted CSV header equals
    `REGISTRATION_FIELDS_V2` exactly; sex, race, consent_version,
    eligibility_confirmed all flow through.
  * `test_registration_schema_migration_rotates_old_csv` — a legacy
    v1-header file is rotated to `.v1.csv.bak` instead of getting
    mismatched-column rows appended.
  * `test_registration_wizard_advances_through_pages` — page 1 ←→ 2
    ←→ 3 transitions gate correctly on per-page validators.
  * `test_summary_csv_includes_v1_1_1_demographic_fields` — full
    demographics flow from participant dict → summary CSV.
  * `test_summary_csv_back_compat_missing_demographics` — pre-v1.1.1
    participant dicts produce empty strings, not KeyError.

**Bundle version.** `cortex_app/cortex.spec`
`CFBundleShortVersionString` bumped `'1.1.0'` → `'1.1.1'`. Test bank
unchanged at `build-data-v2` (300 IIIC + 100 spike, v1.1.0 build).
AD6 strictness unchanged from v1.1.0 (`N_MIN=15`, `ALPHA=0.10`;
production target `ALPHA=0.05` deferred to v1.2.0). No engine,
deployment, calibration, or `cert_config.yaml` changes — Phase-6
invariants intact.

**Known deferred items** (not blocking v1.1.1):
  * Plaintext name + email continue to ship to Dropbox in the summary
    CSV. PII at-upload hashing scoped for v1.1.2 or v1.2.0.
  * IRB amendment for race/ethnicity + country collection needs MBW
    coordination before any public deployment. `CONSENT_VERSION`
    stamp and `IRB_PROTOCOL_ID` field are in place to keep cohort
    splits reproducible across language revisions.
  * Age remains a free-text field (binning to dropdown is a clean
    follow-on if/when the public-release dataset volume justifies it).

Tag: `cortex-v1.1.1`. Test bank pinned at `build-data-v2`.

---

## ▶ CORTEX bundle (2026-05-27) — `cortex-v1.1.2` — verdict-first results screen + cross-platform MP4

**Headline:** the post-session results screen now shows a per-task
PASS / FAIL / REFER verdict with a one-line skill+bias narrative and
a collapsible technical-details panel, replacing the agreement-with-
reference accuracy line that the methodology had moved past. MP4
renders (`collapse.mp4` + `passfail.mp4`) now produce on any platform
without a system ffmpeg install thanks to `imageio-ffmpeg`. No
engine, deployment, calibration, or `cert_config.yaml` changes —
Phase-6 invariants intact; v13 calibration unchanged.

### Results screen rewrite (`scripts/eeg_bank_viewer.py:ResultsScreen`)

The methodology evolved past per-trial agreement as a certification
quantity (Phase 3/4); the AD6 termination policy decides PASS / FAIL
/ REFER per task from the posterior pass-mass π_k. The v1.0–v1.1.1
screen still rendered AUROC point estimates + an
agreement-with-reference summary, neither of which mapped to the
AD6 verdict. The v1.1.2 layout:

  * **Verdict table** (1 row per task): clinician-friendly label
    (PASS → "Pass"; FAIL → "Did not pass"; REFER_BORDERLINE →
    "Refer (borderline)"; REFER_UNINFORMATIVE → "Refer (need more
    data)") colour-coded (soft green / soft red / amber / grey) with
    a two-line narrative beside it:
       — *Above / Below threshold by 0.XX* (or *Near the passing
         threshold* when |ℓ̂ − ℓ\*| < 0.05) — the threshold-relative
         skill summary, driven by `cortex_policy.load_ell_star_iiic`.
       — *Slight / Strong* {liberal | conservative} bias — driven
         by the sign + magnitude of θ̂ (negative ⇒ liberal /
         over-calls; positive ⇒ conservative / under-calls; |θ̂|
         < 0.10 ⇒ "Bias near neutral").
  * **Collapsible technical-details panel** (hidden by default;
    toggled by a "Show / Hide technical details" button): compact
    grid showing the raw posterior numbers — TASK | VERDICT | ℓ̂ |
    ℓ\* | θ̂ | π — with a caption explaining each symbol.
  * **Agreement-with-reference line removed.** Per-trial accuracy is
    still tallied in `BankViewer._on_trial_done`, written to
    `cortex.log`, and persisted in the per-session summary CSV
    (`accuracy` + `n_questions` columns) — only the user-facing
    screen surface dropped it.
  * **AUROC + 95% CI columns removed** from the main panel. The
    AUROC point estimate / half-width remain in the engine result
    (`final_auroc_mean` / `final_auroc_hw`) and the summary CSV; the
    main panel is now strictly the AD6-verdict view, with the raw
    ℓ̂ / θ̂ / π behind the details toggle for transparency.
  * Class-level API preserved (`close_btn`; `ResultsScreen(result,
    n_correct, n_answered)` constructor signature) so the
    `BankViewer._on_session_complete` call site at line 953 works
    unchanged. `n_correct` / `n_answered` are accepted but no
    longer rendered.

Threshold loader degrades gracefully: if `calibration/cert_config.yaml`
is missing / malformed (e.g., a partial install), the screen logs a
WARNING and falls back to a raw "ℓ̂ = ±X.XX" narrative without the
threshold-relative context — the verdict + bias still render.

### Cross-platform MP4 generation (`scripts/cortex_render_videos.py`)

The CHANGELOG v1.0.5 close-out flagged "ffmpeg not bundled" as a
deferred issue: clinician machines without a system ffmpeg install
saw a WARNING in `cortex.log` and the session completed without
MP4s. v1.1.2 resolves this by adopting `imageio-ffmpeg` (v0.6.0
pinned), whose wheel ships per-platform ffmpeg binaries
(~25 MB Linux, ~75 MB macOS / Windows) inside
`imageio_ffmpeg/binaries/`. At module-import time
`cortex_render_videos` resolves the binary via
`imageio_ffmpeg.get_ffmpeg_exe()` and points
`matplotlib.rcParams["animation.ffmpeg_path"]` at it BEFORE any
`FFMpegWriter` is constructed. A `_ensure_ffmpeg_logged()` one-shot
emits an INFO line to `cortex.log` on the first render so the
binary path is auditable post-session.

Fallback path is preserved: if `imageio_ffmpeg` is unavailable at
import (dev clones without the venv populated, or an interpreter
running the script outside the bundle), the module falls back to
the system ffmpeg on PATH and logs a WARNING. So existing dev
workflows keep working.

**PyInstaller integration** (`cortex_app/cortex.spec`):
`collect_data_files('imageio_ffmpeg', include_py_files=False)`
appended to `datas`; `'imageio_ffmpeg'` added to `hiddenimports`.
The wheel ships ONE binary per platform, so each release artifact
gains the corresponding ~25–80 MB.

**License**: `imageio-ffmpeg`'s bundled ffmpeg is GPL-licensed (it
includes libx264). CORTEX is CC BY-NC 4.0 and shells out to ffmpeg
as a separate executable (mere aggregation, not linking), which is
compatible. The spec carries an inline attribution comment.

### Dependency pin

`imageio-ffmpeg==0.6.0` added to `requirements.txt`,
`requirements-cortex.txt`, and `pyproject.toml` (so
`test_phase0_skeleton.py::test_requirements_and_pyproject_pins_are_consistent`
stays green).

### Tests

  * `tests/test_cortex_render_videos.py::test_imageio_ffmpeg_resolves_binary`
    — `cv.FFMPEG_EXE` is non-None and points at an extant file;
    `matplotlib.rcParams["animation.ffmpeg_path"]` matches it.
  * `tests/test_cortex_render_videos.py::test_imageio_ffmpeg_binary_executable`
    — `subprocess.run([FFMPEG_EXE, "-version"])` exits 0 with
    "ffmpeg version" in stdout. Catches a corrupt wheel install.
  * `tests/test_cortex_viewer.py::test_results_screen_skill_narrative_function`
    + `test_results_screen_bias_narrative_function` — pure-function
    coverage of the threshold-relative skill text and the
    sign-aware bias text across the band breakpoints.
  * `tests/test_cortex_viewer.py::test_results_screen_renders_verdicts`
    — instantiates `ResultsScreen` with a synthetic result + stubbed
    `load_ell_star_iiic`; asserts all four verdict labels are
    present, the threshold-relative narrative is present, the
    agreement line is GONE, and the details toggle is bidirectional
    (a v1.1.1 toggle bug — `isVisible()` always returned False until
    the parent window was shown, which would have stuck the toggle
    in "show" mode — was caught by the bidirectional check and
    fixed via `isHidden()`).
  * `tests/test_cortex_viewer.py::test_results_screen_handles_missing_threshold`
    — graceful degradation when `cert_config.yaml` is absent.
  * `tests/test_cortex_results_screen.py::test_results_screen_builds`
    updated for the v1.1.2 contract (asserts the removed AUROC /
    agreement strings are absent; clinician-friendly verdict label
    is present; details panel collapsed by default).

### Bundle version

`cortex_app/cortex.spec` `CFBundleShortVersionString`
`'1.1.1'` → `'1.1.2'`. Test bank unchanged at `build-data-v2`
(300 IIIC + 100 spike). AD6 strictness unchanged from v1.1.0/v1.1.1
(`N_MIN=15`, `ALPHA=0.10`).

Tag: `cortex-v1.1.2`. Test bank pinned at `build-data-v2`.

---

## ▶ CORTEX bundle (2026-05-27) — `cortex-v1.1.3` — sex-only demographic + engine-explainer MP4

**Headline:** the participant form drops `gender_identity` (keeping
`sex` as the canonical demographic field). A third per-session MP4
(`engine_explainer.mp4`) renders alongside `collapse.mp4` and
`passfail.mp4`, showing the differential-geometry framing of how the
SMC engine picks the next best question — a rotating 3D posterior
manifold, a 95%-credible-ellipse cloud with a "geodesic" mean-trail,
and the engine's 1D info-gain landscape with the optimal-next-signal
argmin marked.

### Gender-identity field removed (`scripts/eeg_bank_viewer.py`)

v1.1.1/v1.1.2 added a SAGER-style separate `gender_identity` dropdown
on page 3 of the wizard. v1.1.3 drops it — `sex` is the analytic
field of interest for the credentialing dataset, and a second
gender-identity dropdown added intake friction without proportional
analytic value. The `_GENDER` constant, the `f_gender` widget, the
form row, and the `gender_identity` entry in the persisted CSV are
all removed.

**Schema migration is now version-aware.** The single
`REGISTRATION_FIELDS_V2` constant in v1.1.1/v1.1.2 is split into
three frozen historical snapshots (`_REGISTRATION_FIELDS_V1`,
`_REGISTRATION_FIELDS_V2`) plus the new canonical current alias
`REGISTRATION_FIELDS` (which points at `REGISTRATION_FIELDS_V3` —
the 19-column v1.1.3 schema). `RegistrationPage._save_row` detects
the on-disk header version and rotates v1 → `registrations.v1.csv.bak`,
v2 → `registrations.v2.csv.bak`, anything else →
`registrations.legacy.csv.bak`. The earlier collision-incrementing
suffix logic is preserved. New tests cover both v1→v3 and v2→v3
rotation paths.

`cortex_storage._summary_row` drops the `gender_identity` column from
the uploaded summary CSV. `CONSENT_VERSION` bumped to
`"v1.1.3-placeholder"` (still placeholder pending IRB amendment).

### Engine-explainer MP4 (`scripts/render_engine_explainer.py` — new)

A third per-session MP4 (`engine_explainer.mp4`) auto-renders into
the session directory during `SessionRecorder.finalize()` (alongside
`collapse.mp4` and `passfail.mp4`), and is also exposed as a
standalone library + CLI for reuse in methodology figures, talks,
or off-line replay of any session directory:

    .venv/bin/python scripts/render_engine_explainer.py <session_dir>

The video cycles through all K=6 IIIC tasks sequentially. For each
task block, three panels:

  * **Top-left — rotating 3D posterior manifold.** Weighted 2D
    histogram of the particle cloud over (t_k, ℓ_k) rendered as a
    plasma-colormap surface. Camera azimuth rotates per frame for
    the 3D feel. Manifold contracts visibly over trials as more
    data tightens the posterior.
  * **Top-right — 2D overhead cloud + 95% credible ellipse.**
    Posterior cloud projected to (t_k, ℓ_k); the ellipse's principal
    axis is the steepest-uncertainty direction (the "Riemannian
    gradient" the engine descends). A trail of past posterior means
    shows the **geodesic** the cloud has traced through parameter
    space so far.
  * **Bottom — 1D info-gain landscape.** The score curve
    `engine.core_mcmc._expected_loss_vec(state, k, signals)` over a
    dense 61-point signal grid. Argmin is the optimal next-item
    signal level (the engine's "gradient-descent step" in question
    space). A small marker rolls from the current frame's posterior
    mean toward the argmin to make the descent visually explicit.

**Pure replay — no engine instrumentation.** The renderer
reconstructs the engine `state` dict from each saved trial's
particle cloud and re-calls `_expected_loss_vec`. The engine's
runtime path is untouched — Phase-6 invariants (`LAPSE_RATE=0.025`,
`LOGIT_TO_PROBIT=1/1.7`, byte-equivalent `fit_sdt_per_domain` md5)
remain intact and no drift-guard tests need re-validation.

**Defaults**: 24 fps, 30 trials per task, 0.8s end-of-task hold →
~12 sec video, ~5 MB, ~60 s wall on a default machine for a
realistic 60-trial session. Tunable via `fps`, `trials_per_task`,
`hold_seconds` kwargs / CLI flags. ffmpeg comes from imageio-ffmpeg
(introduced in v1.1.2) so the binary is portable across mac / win /
linux without a system ffmpeg install.

**Independent failure budget.** `SessionRecorder.finalize()` wraps
each renderer in its own try/except so a crash in `render_all`
(collapse + passfail) does not suppress `render_engine_explainer`
and vice versa. Tested by
`test_finalize_render_failures_are_independent`.

**Aborted sessions skip the explainer** for the same reason
`render_all` does — the partial trajectory isn't worth the render
time. Tested.

### Tests

  * `tests/test_cortex_viewer.py`:
    - `test_registration_csv_schema_v3` — header is V3, gender_identity
      column is absent; `f_gender` widget is gone.
    - `test_registration_schema_migration_rotates_v1_csv` — pre-
      v1.1.1 CSV header rotates to `registrations.v1.csv.bak`.
    - `test_registration_schema_migration_rotates_v2_csv` — new
      for v1.1.3: v1.1.1/v1.1.2 CSV header rotates to
      `registrations.v2.csv.bak` (proves the version-aware migration
      names backups by what was actually on disk).
  * `tests/test_cortex_storage.py`:
    - Updated to drop `gender_identity` from the demographic-flow
      test (and to ASSERT it's absent from the summary).
  * `tests/test_render_engine_explainer.py` (new, 9 tests):
    - Pure-math: weighted mean/cov recovery, ellipse closure +
      centering, zero-weight-cloud defensive path, frame-plan cycle.
    - End-to-end: tiny synthetic session renders a valid MP4
      (verified via ffprobe through the bundled imageio-ffmpeg
      binary).
    - Zero-trials guard: empty session is a clean no-op, not a
      crash.
    - Cortex finalize wiring: both renderers called, independent
      failure budgets, aborted-session skip.

### PyInstaller spec

`'render_engine_explainer'` added to `hiddenimports` so the frozen
bundle picks up the new module. No additional `datas` /
`collect_data_files` calls needed — the renderer depends only on
matplotlib (already vendored) + the engine package (already vendored)
+ imageio-ffmpeg (already vendored via v1.1.2).

### Bundle version

`cortex_app/cortex.spec` `CFBundleShortVersionString`
`'1.1.2'` → `'1.1.3'`. Test bank unchanged at `build-data-v2`. AD6
strictness unchanged from v1.1.0–v1.1.2 (`N_MIN=15`, `ALPHA=0.10`;
production target `ALPHA=0.05` deferred to v1.2.0). No engine,
deployment, calibration, or `cert_config.yaml` changes —
Phase-6 invariants intact.

Tag: `cortex-v1.1.3`. Test bank pinned at `build-data-v2`.

---

## ▶ CORTEX bundle (2026-05-27) — `cortex-v1.1.4` — visualizations opt-in + background-thread finalize

**Headline:** the post-session ~3-minute UI freeze caused by the v1.1.3
MP4 renders is gone. Visualizations are now an explicit opt-in
question on the registration form, and `SessionRecorder.finalize()`
runs on a background thread so the participant sees an immediate
"Thank you, computing your results" page after the last question
regardless of which path they chose.

### The v1.1.3 UX problem

v1.1.3 shipped three per-session MP4s (`collapse.mp4`,
`passfail.mp4`, `engine_explainer.mp4`) that auto-rendered on
`SessionRecorder.finalize()`. The renders ran synchronously on the
GUI thread, freezing the UI for ~3 minutes between the last question
and the `ResultsScreen` swap. Participants would close the window
thinking the app had crashed.

### v1.1.4 fix — three changes

**1. Opt-in question on the registration form.** Page 3 of the
wizard gains a new dropdown:

>  Generate personalized visualizations?
>     • "No — faster results"  (default)
>     • "Yes — generate visualizations (2-3 min)"

The default is **No**: silent participants take the fast path. Only
explicit opt-in triggers the renders. Value is persisted to
`registrations.csv` + the per-session summary CSV as
`wants_visualizations` for downstream analysis (who opts in / out
becomes a known covariate).

A new module-level helper `is_opt_in_for_visualizations(reg_row)`
centralises the truthiness check so the dropdown label format can
change without scattering string comparisons across the codebase.

**2. `SessionRecorder.render_videos` wired to the opt-in
preference.** `eeg_bank_viewer.py:2484` now passes
`render_videos=is_opt_in_for_visualizations(reg.registration)` to
the recorder constructor. Opt-out → recorder's existing
`render_videos=False` skip path (proven by
`test_render_videos_false_skips_render`) → no MP4 calls at all,
finalize completes in ~1 sec. Opt-in → all three MP4s render as
before, ~3 min total. The plumbing-level skip already existed; v1.1.4
adds the user-facing preference that controls it.

**3. Background-thread finalize + `ComputingResultsPage`
transition.** `BankViewer._on_session_complete` no longer calls
`recorder.finalize` synchronously. Instead:

  * Immediately swaps the central widget to `ComputingResultsPage`
    (new class) — "Thank you for participating" + a busy-chase
    `QProgressBar` + dynamic subtitle copy chosen by opt-in path
    (opt-in: "Computing your results and generating personalized
    visualizations. This takes 2 to 3 minutes."; opt-out: "Computing
    your results. This will take just a few seconds.").
  * Spawns `_FinalizeWorker` (`QObject` + `QThread`) that calls
    `recorder.finalize(result)` off the GUI thread. The recorder is
    a plain Python object (no Qt internals); finalize() does file
    I/O + matplotlib Agg renders — both thread-safe.
  * When the worker emits `finished(result)`, the main thread swaps
    `ComputingResultsPage` → `ResultsScreen` via
    `_on_finalize_done`.
  * On `failed(err_msg)`, the main thread logs + still shows
    `ResultsScreen` so the participant gets a verdict even when
    persistence partially failed.

`ResultsScreen` is unchanged.

If the participant closes the window before the worker finishes
(opt-in path, ~3 min), the daemon thread keeps writing until done
— partial / no MP4 acceptable, won't crash. The `ComputingResultsPage`
footer reads "Please do not close this window."

### Schema migration V3 → V4

The single `wants_visualizations` column addition required a schema
bump. `REGISTRATION_FIELDS_V3` is now frozen alongside V2 and V1;
`REGISTRATION_FIELDS_V4` is the new canonical (20 columns);
`REGISTRATION_FIELDS` alias points at V4. The version-aware
migration in `RegistrationPage._save_row` rotates a V3-header file
to `registrations.v3.csv.bak` (new path; complements the v1→v3
and v2→v3 rotations from v1.1.3).

`CONSENT_VERSION` bumped `"v1.1.3-placeholder"` →
`"v1.1.4-placeholder"`. `cortex_storage._summary_row` continues to
flow the demographic columns (now including `wants_visualizations`
implicitly via the participant dict) into the uploaded summary CSV.

### Tests (full sweep: 379 pass / 0 fail / 11 skipped)

  * `tests/test_cortex_viewer.py`:
    - `test_registration_csv_schema_v4` — renamed from `_v3`; asserts
      the new `wants_visualizations` column is present and that an
      opt-in row produces an `is_opt_in_for_visualizations()` True.
    - `test_registration_default_opts_out_of_visualizations` — silent
      participant ⇒ default "No" persisted; opt-in helper returns
      False.
    - `test_registration_schema_migration_rotates_v3_csv` — new
      v1.1.4 migration path: V3-header file rotates to
      `registrations.v3.csv.bak`.
    - `test_computing_results_page_opt_out_text` /
      `test_computing_results_page_opt_in_text` — direct render of
      `ComputingResultsPage`; opt-out has "few seconds" copy, opt-in
      has "2 to 3 minutes" copy; spinner is indeterminate; opt_in
      attribute matches the constructor arg.
  * `tests/test_cortex_results_screen.py`:
    - `test_session_complete_swaps_to_results` updated to drain the
      `QTimer.singleShot(0)` path used when there's no recorder.
    - `test_session_complete_shows_computing_page_first` — new:
      asserts `ComputingResultsPage` is shown BEFORE finalize
      completes (via a stub recorder that sleeps 250ms in
      finalize()), then waits up to 5s for the background worker to
      emit and verify the swap to `ResultsScreen`.
    - `test_session_complete_opt_out_path_shows_fast_copy` — new:
      opt-out recorder's `ComputingResultsPage` carries the
      "few seconds" copy, not the "2 to 3 minutes" copy.
  * `tests/test_cortex_storage.py`:
    - `CONSENT_VERSION` expectations bumped `"v1.1.3-placeholder"`
      → `"v1.1.4-placeholder"` in both summary-CSV tests.

### Bundle version

`cortex_app/cortex.spec` `CFBundleShortVersionString`
`'1.1.3'` → `'1.1.4'`. Test bank unchanged at `build-data-v2`. AD6
strictness unchanged from v1.1.0–v1.1.3 (`N_MIN=15`, `ALPHA=0.10`).
No engine, deployment, calibration, or `cert_config.yaml` changes
— Phase-6 invariants intact.

### What v1.1.3 cohort sees on upgrade

Participants who took v1.1.3 sessions had `wants_visualizations`
implicit (always on). v1.1.4 marks them as a separate cohort via
the `consent_version="v1.1.4-placeholder"` stamp + the new
`wants_visualizations` column. Analysis pipelines can join on
session_id + filter by `consent_version` to compare cohorts.

### Deferred (carried from prior releases)

  * IRB amendment for `CONSENT_VERSION` + race/ethnicity + country
    + the new `wants_visualizations` field before public deployment.
  * LICENSE / README ffmpeg-GPL attribution (v1.1.2).
  * Plaintext PII (`participant_name` + email) in summary CSV.
  * 11 Dependabot vulnerabilities on the default branch.

Tag: `cortex-v1.1.4`. Test bank pinned at `build-data-v2`.

---

## ▶ CORTEX bundle (2026-05-28) — `cortex-v1.1.5` — show-results-folder button + engine-explainer slowdown

**Headline:** two UX-only changes. (1) `ResultsScreen` now has a
"SHOW RESULTS FOLDER" button next to CLOSE that opens the
per-session output directory in the OS file manager — solves the
"where are my MP4s?" discoverability problem reported by a v1.1.3
test taker. (2) `engine_explainer.mp4` defaults retuned so the
video runs ~60 seconds across the six IIIC tasks (was ~12 s);
each frame is now visible long enough to absorb the manifold +
ellipse + score-curve story.

### Diagnostic context

A v1.1.3 test taker reported their MP4s "had no content" when
opened. Investigation confirmed the files were valid H.264
(extracted frames showed correct content); the actual issue was
**Ubuntu Totem's default install lacking H.264 codecs** — the
file plays fine in VLC / mpv / a Totem with `gstreamer1.0-libav`
installed. Files live at the OS-blessed per-user data dir
(`~/.local/share/CORTEX/sessions/<uuid>/` on Linux,
`~/Library/Application Support/CORTEX/sessions/<uuid>/` on macOS,
`%LOCALAPPDATA%\CORTEX\sessions\<uuid>\` on Windows) — chosen for
macOS App-Translocation read-only constraints (the v1.0.1 fix
explained at `cortex_storage.user_data_root:49-54`). Moving
storage into the app-install location would re-break that, so
v1.1.5 instead adds a discoverability button.

### 1. SHOW RESULTS FOLDER button (`scripts/eeg_bank_viewer.py`)

`ResultsScreen.__init__` gains an optional `session_dir=None`
kwarg. When set:

  - A second button ("SHOW RESULTS FOLDER", 240×48 px) appears
    LEFT of CLOSE in the bottom button row.
  - Click → `subprocess.Popen([opener, str(session_dir)])` where
    `opener` is `open` (macOS) / `explorer` (Windows) /
    `xdg-open` (Linux/*BSD). Per-platform dispatch by
    `sys.platform`.
  - Opens the **session subfolder only** (not the
    `sessions/` root), so a shared-machine participant sees only
    their own files — no risk of seeing past participants' data.
  - Defensive: if the session dir no longer exists (manual
    deletion), the click logs + no-ops rather than crashing.

`BankViewer._show_results` plumbs the session_dir from
`self.recorder.dir` via `getattr(self.recorder, "dir", None)` so
no-recorder fast paths and test stubs without a `.dir` attribute
still work. ResultsScreen with `session_dir=None` hides the
button — keeps the pre-v1.1.5 constructor contract for the 5
existing test fixtures that don't pass it.

Bundle / install location explicitly NOT changed. See `cortex_storage.
user_data_root:43-67` for the rationale (macOS App Translocation
read-only constraint, Windows Program Files admin requirement,
Linux portability).

### 2. engine_explainer.mp4 slowdown (`scripts/render_engine_explainer.py`)

New module-level defaults:

  FPS = 10                    (was 24)
  TRIALS_PER_TASK = 60        (was 30)
  HOLD_SECONDS_PER_TASK = 4.0 (was 0.8)

Per task block: 60 trial frames + 40 hold frames = 100 frames =
10 sec. Total across 6 tasks: 600 frames = exactly 60.00 sec at
10 fps. Verified end-to-end with a synthetic T=60 session:
`Duration: 00:01:00.00, 10 fps, 1210×770 H.264`. For sessions
with T<60 trials the video shortens proportionally (T=20 → ~36 s)
rather than padding.

Side effects:
  - Camera rotation per real-time second halves (1.5°/frame ×
    10 fps = 15°/s vs 36°/s previously) — more contemplative,
    matches the slower per-frame display rate.
  - Render wall-time roughly doubles (~63s → ~122s on a default
    machine for a T=60 session). On the opt-in path (already
    accepting 2-3 min per [[v1.1.4]]) this is well within
    budget — total render is now ~152s = ~2.5 min including
    collapse + passfail.
  - File size grows (5 MB → 21 MB) due to ~2× the frame count
    at the same bitrate. Stays local — never uploaded to
    Dropbox.

Pure fps reduction alone (e.g., dropping to fps=4.9 to hit 60s
with the v1.1.4 frame count) would have looked too jerky.
Combining fps reduction with more frames + longer hold
preserves smoothness while hitting the 60s target.

### Tests (full sweep: 385 pass / 0 fail / 11 skipped)

  * `tests/test_cortex_viewer.py` (3 new):
    - `test_results_screen_show_folder_button_hidden_when_no_session_dir`
      — pre-v1.1.5 callers (no `session_dir`) get the button
      hidden; CLOSE still works.
    - `test_results_screen_show_folder_button_dispatches_to_os` —
      with session_dir set, the button is visible; click invokes
      `subprocess.Popen` with the platform-correct opener
      (`open` / `explorer` / `xdg-open`) and the session
      subfolder path. Mocked subprocess.
    - `test_results_screen_show_folder_button_no_op_when_dir_missing`
      — defensive: a deleted-since session_dir → click is a
      logged no-op, no Popen call.
  * `tests/test_render_engine_explainer.py` (2 new):
    - `test_v1_1_5_default_timing_hits_60_seconds` — module
      constants pinned to FPS=10 / TRIALS_PER_TASK=60 / HOLD=4.0;
      `_frame_plan(T, 6, ...)` produces exactly 600 frames =
      60.00 sec for T ∈ {60, 100, 300}.
    - `test_v1_1_5_short_sessions_proportionally_shorter` —
      T=20 session yields ~36 s (60 frames + 40 hold per task,
      6 tasks @ 10 fps), proving short sessions shrink rather
      than padding.

### Bundle version

`cortex_app/cortex.spec` `CFBundleShortVersionString`
`'1.1.4'` → `'1.1.5'`. Test bank unchanged at `build-data-v2`.
AD6 strictness unchanged from v1.1.0-v1.1.4 (`N_MIN=15`,
`ALPHA=0.10`). No engine, deployment, calibration, or
`cert_config.yaml` changes — Phase-6 invariants intact.

### Deferred (carried from prior releases)

  - IRB amendment for `CONSENT_VERSION` + race/ethnicity +
    country + `wants_visualizations` before public deployment.
  - LICENSE / README ffmpeg-GPL attribution (v1.1.2).
  - Plaintext PII in summary CSV → Dropbox.
  - 11 Dependabot vulnerabilities on the default branch.

Tag: `cortex-v1.1.5`. Test bank pinned at `build-data-v2`.

---

## ▶ CURRENT STATE (read this first)

- **Paper 1 = the Multi-AUROC Precision Protocol (Mode-A).**  A per-examinee
  Bayesian adaptive test across K=6 SPARCNET domains.  Skill is reported as
  per-domain AUROC with a credible interval; the session stops when
  `max_k halfwidth_0.95(AUROC_k) < δ`.  Item selection = Global-EV
  (A-optimal posterior-variance reduction).  Engine:
  `core_mcmc.run_session_mcmc_auroc` + `choose_item` + `post_hoc_delta_sweep`.
- **Mode-B (binary PASS/FAIL certification)** — boundary prior, Berger-IUT
  joint stopping, Fisher/n_min guards, Šidák — is **DEPRECATED for Paper 1**.
  Relocated to `engine_mode_b.py` + `tests/mode_b/`.  Preserved verbatim as
  groundwork for Paper 2 (binary credentialing with a prospectively
  validated panel).
- **Inference validated three independent ways** (Phase 2):
  SBC 12/12 params; AUROC-CI coverage |Δ|≤0.007 at the production config
  (N=1000, ess=0.9); gold-chain 35/36 posterior-moment agreement.
- **Production config:** `cert_config.yaml` v11 (`mode: mode_a`,
  `auroc_delta=0.025`, `delta_sweep=[0.025,0.05,0.10]`, `n_particles=1000`,
  `ess_threshold_frac=0.9`, `max_q=3000`, unstructured `Corr_l` prior).
- **Compute:** process-pool parallelization (`scripts/_parallel.py`,
  spawn + single-thread BLAS, bitwise-deterministic vs serial, ~10×).
- **Tests:** 59 pass, 4 deselected (slow), 1 xfail (documented Mode-B
  termination issue, addressed by the reframe).
- **Repo is private until journal acceptance.**  PHI inventory:
  `data/SENSITIVE.md`.  IRB 2016P000058 (BIDMC), 2013P001024 (MGH).

---

## Load-bearing fix index (FIX-T*) — referenced throughout the code

These IDs appear in inline comments across `core_mcmc.py`,
`core_mcmc_brute_k.py`, `engine_mode_b.py`, `cert_config.yaml`.  Look them
up here.

| ID | What it did | Where it lives now |
|---|---|---|
| FIX-T0.5 | Removed broken KL item selection (`choose_item_kl`): conditioned on a θ̂ point estimate, selected near-ceiling items with ≈0 Fisher info for ℓ when θ_true≠0.  Replaced by EV/A-optimal `choose_item`. | shared (`core_mcmc.choose_item`) |
| FIX-T0.7 | Numerical hardening: `scipy.special.log_ndtr` + `logsumexp` everywhere instead of `log(clip(norm.cdf))`.  Fixes tail underflow at \|z\|≳6. | shared |
| FIX-T1.1 | Removed virtual certification (sz→grda Fréchet pass-through).  The joint particle posterior subsumes it; `virtual_pairs` now warns + is ignored. | Mode-B |
| FIX-T1.3 | MCSE-buffered stopping: PASS needs `p − Z·MCSE ≥ τ`, FAIL `p + Z·MCSE ≤ 1−τ`, `Z_BUFFER=2`. | Mode-B (shared `Z_BUFFER`) |
| FIX-T1.4 | Introduced lapse rate λ=0.025 in the response model (later corrected — see **F0.1**). | shared |
| FIX-T1.6 | Unstructured prior covariance `Σ_l` loaded from `Sigma_l_fitted.npy` (replaces compound-symmetry `r`). | shared (`make_state_hier`) |
| FIX-T1.7 | (a) `n_min_guard`: block FAIL until ≥20 questions in a domain.  (b) Use `Corr_l` (unit-diagonal) not raw `Σ_l` (marginal var ~0.01-0.06 → premature FAIL). Empirically justified by F2.4. | Mode-B / shared |
| FIX-T1.8 | Classification-boundary prior `l_k ~ N(l*_k, σ²·Corr_l)`, `P(PASS\|prior)=0.5`.  Mode-A-inert `l_prior_mean` kwarg in shared `make_state_hier`/`sample_prior_hier_K`/`log_prior_hier`. | Mode-B (kwarg shared) |
| FIX-T1.9 | Berger-1982 IUT joint stopping (min-of-marginals).  **Disputed** — the Bayesian-expert audit holds the prior joint-posterior rule was the correct one; min-of-marginals trades correctness for sensitivity.  Moot for Paper 1 (Mode-A has no IUT). | Mode-B |
| FIX-T1.10 | Boundary-targeted item selection `choose_item_cert`: maximises Fisher info for ℓ at l*_k (z² factor → peak at \|z\|=1). | Mode-B |
| FIX-T1.11 | Fisher-info guard: block FAIL until `domain_fi[k] ≥ fisher_imin` (I_min=96 at ε=0.10, σ=0.5). | Mode-B |

---

## Pre-history — Wave 1–4 + Phase 6 (2026-05-11 → 2026-05-14)

Source: `STATE_LOG_2026_05_11/13/14.md`, `CHANGES_W{1,2,3}{A,B,C}.md`
(archived).  This era operated under the **Mode-B** assumption (binary
PASS/FAIL was the product).  Key durable outcomes:

- **2026-05-11 audit (`STATE_LOG_2026_05_11` §§1–10):** the "hier ≈ brute"
  claim was an artifact of two bugs — a handicapped single-2K-D `brute_joint`
  and an asymmetric post-c* calibration metric.  Fix: methodology-compliant
  `core_mcmc_brute_k.py` (K independent 2-D SMCs) + `raw HW<δ` comparison.
  Corrected v3 result: hier up to 2.82× faster at K=8, r=0.9.
- **SMC + MCMC rejuvenation** replaced a kernel-jitter SMC that collapsed to
  0.40 coverage at K=8.  MCMC-rejuvenation reaches 0.94.
- **Wave 1–4 (`CHANGES_W*`):** virtual cert removed (T1.1), MCSE buffer
  (T1.3), lapse rate (T1.4), unstructured Σ_l (T1.6), EV item selection
  (T0.5), numerical hardening (T0.7), audit trail (W2-C).
- **Phase 6 (`STATE_LOG_2026_05_14`):** boundary prior (T1.8), Berger-IUT
  (T1.9), boundary-Fisher selection (T1.10), Fisher guard (T1.11), brute
  parity, CV top-14 Youden calibration, matrix N=15→29, NaN audit-log fix.
- **Four-expert audit (2026-05-15, pre-reframe):** flagged blocking issues —
  stale 2.8× headline, IUT YAML/code mismatch, brute/hier prior+selector
  confound, CV-panel sz/grda inversion, posterior coverage 0.75–0.91,
  min J=0.372.  Triggered the Multi-AUROC reframe (Phase 0+).

The Mode-B engine and its 17 tests remain green under `engine_mode_b.py`
+ `tests/mode_b/`; the unresolved Mode-B items (IUT semantics, panel
circularity, undecided-as-modal) are Paper-2 scope, not Paper-1 blockers.

---

## Phase 0 — Root-cause fixes (2026-05-15)

- **F0.1 — Lapse parametrization unification (CRITICAL).**  The engine used
  `(1−λ)Φ(z)+0.5λ` (ceiling 1−λ/2, a 4AFC-style guess rate) which is
  *algebraically inconsistent* with the spike paper's Eq. 2
  `λ+(1−2λ)Φ(z)` (ceiling 1−λ).  Every σ̂/ℓ̂/AUROC was computed under the
  wrong response model.  Fixed in `core_mcmc.py`, `core_mcmc_brute_k.py`,
  `core.py`; Fisher-info `(1−λ)²`→`(1−2λ)²`.  New regression test asserts
  numerical equivalence with `train_val_split_and_fit.py:probit_lapse_nll`
  to 1e-12.  **This was the primary driver of the pre-fix coverage
  catastrophe (0.75–0.91).**  `test_smoke_brute` budget 150→300 (corrected
  likelihood is less concentrated); `test_oc_borderline_pass_rate` xfail'd.
- **F0.2 — Dependency pinning.**  `requirements.txt` + `environment.yml` +
  `.python-version=3.11.9` in both repos (test-main was on broken 3.14).
- **F0.3 — CLAUDE.md §16.7** (wrong "Option B" factor model) moved to
  `docs/architecture-history.md`.
- **F0.4 — 12 broken `exp*.py`** (importing archived `core_K` etc.) moved
  to `docs/historical/experiments/`.
- **F0.5 — Stale 392 MB** of `…-051326/`, `…-old/` sibling snapshots + 2
  stale slide PDFs moved to `~/Documents/Research/_archive/`.

## Phase 1 — Mode-A (Multi-AUROC) reactivation (2026-05-15)

- **F1.1** — `run_session_mcmc_auroc` given `Sigma_l`/`Sigma_t` kwargs (the
  borrowing-of-strength prior); returns `delta_auroc`, `method`.
- **F1.2** — `cert_config.yaml` v11: `mode: mode_a`, `auroc_delta=0.025`,
  `delta_sweep`, Mode-B keys nested under `mode_b_legacy`.
- **F1.3** — new `bridge/run_multi_auroc_bridge.py`; `AuditTrail`
  `record_session_mode_a`.
- **F1.4** — `post_hoc_delta_sweep(lo_traj, hi_traj, deltas)`: one run at
  the tightest δ yields all δ stop-points.
- **F1.5** — `tests/test_smoke_mode_a.py` (5 tests).
- Real K=6 SPARCNET signal: with `Corr_l`, hier reaches δ=0.05 in ~165 q
  vs brute ~287 q (≈1.7×).

## Phase 1 figures (2026-05-15)

- v1 (4 raters) → **v2 (all 27 SPARCNET raters)**.  Headline: hier vs brute
  K=6 — δ=0.05 1.5×, δ=0.10 1.8×; K-scaling hier ahead K=4/6/8; per-rater
  speedup heatmap (Hiba Haider 5.78×, Olga Taraschenko 3.97×, MBW 2.54×;
  honest outliers Marcus Ng 0.57×).  Bug found+patched mid-run: rater
  column is `confirmed_canonical_name` (the v2 script + post-hoc patch fix
  it).  `results/phase1_figures/README.md`.

## Phase 2 — Inference validation (2026-05-15)

Process-pool parallelization built first (`scripts/_parallel.py`, 14
workers, bitwise serial==parallel pinned by
`tests/test_parallel_determinism.py`).  Measured cost model: Global-EV
`choose_item` ≈ 104 ms/q solo, ~250 ms/q under 14× contention; ess=0.9
adds a large rejuvenation tax on long sessions.  **Phase-4 prerequisite:
the 252k-examinee OC campaign is infeasible without a `choose_item`
signal-subsampling optimisation (~6× speedup, near-identical item).**

| Fix | Result |
|---|---|
| F2.5 MCMC diagnostics | `diagnostics.py` + 15 tests (ESS, lag-1 autocorr, Sokal τ_int, split-R̂) |
| F2.1 SBC | **12/12** params calibrated, 0 bins out of band |
| F2.2 + F2.2b coverage | production config (N=1000, ess=0.9) **\|Δ\| ≤ 0.007** all CI levels; the F2.2 "mild under-coverage" was an artifact of a speed-test `--N-particles 500` override |
| F2.4 Σ_l sensitivity | speedup ordering corr_l(1.44×)>cs>diagonal(1.15×) → **attributable to correlation**; `fitted_raw`/`ledoit_wolf` = documented false-precision negative control (justifies FIX-T1.7) |
| F2.6 SPARCNET test-retest | **6/6 domains ICC(3,1)≥0.70** (sz 0.90, grda 0.90, iic 0.88, lpd 0.86, gpd 0.80, lrda 0.71) — resolves audit F8 |
| F2.7 lapse sensitivity | Wichmann-Hill: robust λ∈[0.01,0.05] (excess bias ≤0.005), bounded at λ=0.10 (0.036) — quantifies audit F15 |
| F2.3 gold-chain | **35/36** posterior-moment agreement vs exact long-run MH.  Two methodology corrections: SMC-cov preconditioning of the reference chain (naive isotropic mixed at 8% accept); moment criterion not N-sensitive KS/TV |

Net: the central audit question — "is the engine calibrated or
miscalibrated like the pre-fix version?" — is answered **calibrated**, the
F0.1 lapse bug being the primary cause of the pre-fix failure.
`results/phase2_validation/README.md`.

## Phase 3 — Module separation, repro hygiene, figures (2026-05-15)

- **F3.1** — Mode-B engine extracted `core_mcmc.py` (988→690 lines, Mode-A
  only) → `engine_mode_b.py`.  Shared primitives stay in `core_mcmc`;
  `l_prior_mean` remains an optional Mode-A-inert kwarg.  Importers
  (Mode-B bridge + 5 tests) redirected.  0 regression.
- **F3.2** — mode-agnostic bridge helpers → `bridge/_common.py`; 3 bridge
  modules + 6 scripts redirected (scripts no longer couple to the
  Mode-B-named module).
- **F3.3** — 6 Mode-B test files → `tests/mode_b/` (+ `__init__.py`,
  `README.md`); `test_sidak` repo-root path fixed.  tests/ root is now
  cleanly Mode-A/shared.
- **F3.4** — `data/SENSITIVE.md` PHI inventory (inventory only; no
  gitignore/anonymizer changes — repo private until acceptance).  Corrected
  the architecture audit: `youden_ell_star.json` + Phase-1 result JSONs DO
  carry clinician names.
- **F3.5** — this CHANGELOG; 11 STATE_LOG/CHANGES files archived to
  `docs/historical/`.
- **F3.6 / F3.7** — Nature-quality inference-validation composite + Σ_l /
  lapse robustness figures (see `results/phase2_validation/`).

### Deferred to Phase 4 (flagged, not dropped)

- Bridge `--banks-uri` decoupling + vendored `data/curated_banks_v10/`
  (deployment-readiness, tied to the repro recipe).
- `REPRODUCING_PAPER_1.md` + Makefile (paper-grade figures don't exist
  until Phase 4).
- `choose_item` signal-subsampling optimisation (OC-campaign enabler).
- `scripts/anonymize_rater_data.py` (runs at journal acceptance).

---

## Provenance

Full day-level detail (process narrative, dead-ends, intermediate status)
is preserved verbatim in `docs/historical/state_logs/` and
`docs/historical/changes_waves/`.  This CHANGELOG keeps the durable
technical decisions and their rationale only.
