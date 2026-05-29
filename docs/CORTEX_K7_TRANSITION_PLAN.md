# CORTEX K=6 → K=7 transition plan

**Date:** 2026-05-28
**Goal:** transition the CORTEX live-test app from K=6 (IIIC only) to K=7
(spike + 6 IIIC), using the existing training corpus to its fullest extent so
that the certification methodology is mathematically rigorous for Nature
Medicine submission.
**Scope:** code path under `/data/eli-work/repos/ilae-skill-certification-test-multi`;
inputs/outputs that flow into CORTEX from the unified data + calibration layer.
**Companion docs:** `docs/LIVE_TEST_INTEGRATION_PLAN.md` (the
implementation-landed-as-of-2026-05-22 baseline), `docs/AD6_RESOLUTION.md`
(stopping/verdict policy), `docs/NATURE_MEDICINE_AUDIT.md` (broader audit),
`data/DATA_PROVENANCE.md` (corpus + derivation chain), `CLAUDE.md` (D1–D9).

---

## 1. Headline finding

The engine is K-agnostic by construction (Phase-4.4 — `K`, `DIM`, `TASKS`
derived from Σ slot-name index; `tests/test_phase4_port_fidelity.py:80-100`).
The deployment side is already K=7 (`data/deployment_prior/Sigma.csv` is
14×14, `summary.json::tasks` has all 7 entries, `cert_config.yaml::
ell_star_unified_v13` has 7 ℓ\* values).

**The CORTEX-side blockers are entirely on the data + the input-assembly +
the UI layers.** Specifically:

1. The 100 spike segments in `data/eeg_bank.h5::spike/` come from the
   `morgoth1:spikes` source and have **0 entries in `labels.csv`** — they
   are uncalibrated and unlabeled in our corpus.
2. The 17,346 spike segments with calibrated `(s_mean, s_sd)` at
   `data/labels/fits/spike/cases.csv` are from the SN1 / Bonobo / SpikeEd
   subset of `sn1_combined_v2`; their EEG payload lives in the external,
   PHI-bearing `SN1_combined_v2.h5` and is NOT in `data/eeg_bank_spec.h5`.
3. `data/labels/iiic_segment_signals.csv` has K=6 IIIC signal columns
   only; no `s_mean_spike` / `s_sd_spike` column.
4. The repo-root `Sigma_l_fitted.npy` is K=6 only; the K=7 ℓ-block lives
   in `data/deployment_prior/Sigma.csv` but is not exposed in
   `engine_paths.SIGMA_L`.
5. `scripts/cortex_engine_inputs.py` is hard-coded K=6 IIIC; its
   `TASKS` tuple, its signals-CSV reader, and its Sigma loader all
   assume K=6.
6. The GUI today reads pattern_class once per segment and lets the
   user pick one of 6 IIIC buttons (`scripts/eeg_bank_viewer.py`);
   spike is a binary task with different EEG geometry (no spectrogram)
   and needs a Yes/No UI variant.

Once the data blockers are resolved, the code surgery is **moderate scope**
(~5 files, ~300–500 LOC changed, plus tests). The engine itself does not
need to change.

---

## 2. The data state — concrete numbers

### 2.1 Two seg_id namespaces, zero overlap

Three layers of namespace touch spike data; none of them line up cleanly:

| Layer | seg_id range / type | n | Source | Has EEG? | Has calibration? |
|---|---|---:|---|---|---|
| `data/eeg_bank.h5::spike/` | 95327–103764 (int) | 100 | `morgoth1:spikes` | ✅ `eeg30s` only (no spectrogram) | ❌ no labels.csv entries |
| `data/labels/fits/spike/cases.csv` | 0–20519 (int) | 17,346 | `sn1_combined_v2:{sn1,bonobo_only,fabio_spikeed}` (12,801 + 3,966 + 579) | ❌ EEG external (`SN1_combined_v2.h5`; D9 PHI) | ✅ `(s_mean, s_sd, n_raters, pos_rate)` |
| `data/curated_banks/combined_spike.json` | string keys (`sn1::SEG_…`, `bonobo_only::…`) | 225 | `sn1_combined_v2:{sn1,bonobo_only}` (spike-paper bank) | ❌ EEG external | ⚠️ `s_probit` only (2-bin stratified at ±1.59); no `s_sd` |

Verified by direct join: bank-spike ∩ fits-spike = ∅; bank-spike ∩ curated-spike = ∅; fits-spike ∩ eeg_bank_spec.h5 = ∅.

### 2.2 The IIIC equivalent (for comparison)

For IIIC the equivalent layers DO align — that's how the K=6 build works:

| Layer | n | Source | Note |
|---|---:|---|---|
| `data/eeg_bank.h5::iiic/` | 300 | `sparcnet50K` (299) + `centaur_2025_iiic` (1) | The v1.1.0 bank |
| `data/labels/iiic_segment_signals.csv` | 69,806 | `sparcnet50K`+`pd_rda_profiler`+`kong2025`+`centaur` | Per-IIIC-task signals |
| `data/eeg_bank_spec.h5::segments/` | 132,291 | Multi-source spec payload | 100 GB |

Bank IIIC ∩ signals = 300/300 (full overlap). This is why IIIC "just works."

### 2.3 Engine-side prior

| Artefact | Where | K | Source | Consumed by |
|---|---|---:|---|---|
| `Sigma_l_fitted.npy` | repo root (+ symlinks `engine/`, `archive/`) | **6** | 15-rater-era frozen; `data/INVARIANT_AUDIT.md §9` | Engine Mode-A (`engine/core_mcmc.load_fitted_Sigma`) |
| `data/deployment_prior/Sigma.csv` | derived | **14 (= 7 t + 7 ℓ)** | PI block fit on `data/labels/fits_hier_block` (`r_iiic=0.334`, `r_cross=0.447`, ρ=0) | Deployment Laplace+EKF |

The 7×7 ℓ-block of the deployment Σ is a **block-constant matrix**: diagonal = 1.0, off-diagonal = `r_ell = 0.36656`. This is what an engine-side K=7 `Sigma_l_fitted_k7.npy` would carry as `Corr_l`.

### 2.4 cert_config v13 ℓ\* (all 7 already present)

`calibration/cert_config.yaml::ell_star_unified_v13::tasks` carries:

| YAML key | ℓ\* | Maps to CORTEX task code |
|---|---:|---|
| `sparcnet_sz` | 0.4550 | sz |
| `sparcnet_lpd` | 0.5337 | lpd |
| `sparcnet_gpd` | 0.3297 | gpd |
| `sparcnet_lrda` | 0.4793 | lrda |
| `sparcnet_grda` | 0.4865 | grda |
| `sparcnet_iic` | 0.4418 | iic ("other") |
| **`combined_spike`** | **0.2543** | spike — note key prefix differs |

The naming mismatch (`sparcnet_*` for IIIC vs `combined_spike` for spike) is currently a one-line problem in `scripts/cortex_policy.py:122` which hard-codes `f"sparcnet_{code}"`. Trivial fix: a `_KEY_FOR_CODE` lookup table.

---

## 3. The three data blockers (in priority order)

### 3.D1 — Spike bank with calibrated signals

**This is the headline blocker** (per LIVE_TEST_INTEGRATION_PLAN.md §0, §4 — "every bank segment the engine can serve as a question needs a calibrated per-segment signal `s`. ... 0 of 100 spike segments and 5 of 100 IIIC segments in the bank have any calibrated signal").

Three options:

**Option A — Build a new spike bank from the SN1 calibration corpus (recommended).**
- Pick N = 50, 75, or 100 spike segments from the 17,346 calibrated rows in `data/labels/fits/spike/cases.csv`, stratified by `s_mean` (e.g. quantile-stratified to span the difficulty range).
- Their EEG payload lives in the external `SN1_combined_v2.h5` (PHI; D9 enforced). Build script needs read access during build only.
- Output: vendor the EEG slice (10s window per the spike paper, or 30s to match IIIC framing) + the `(s_mean, s_sd)` into `data/eeg_bank.h5::spike/` (replacing the current uncalibrated `morgoth1:spikes` 100).
- Pro: uses the fitted spike calibration directly; no new modeling needed.
- Con: requires temporary access to `SN1_combined_v2.h5`; the build can be run once and the derived signals vendored thereafter (D9 is preserved — raw PHI never goes into the repo, only derived per-segment signals and the publication-shippable spectrogram/spec do).

**Option B — Calibrate the existing `morgoth1:spikes` bank.**
- Requires labels.csv entries (=many-rater ratings) for the 100 morgoth1 spike segments. Verified: **0 ratings in our corpus.**
- Would need to either (i) collect new ratings (e.g. via CORTEX itself as a pilot data-collection arm), or (ii) recover the morgoth1 source pipeline's ratings (external).
- Not actionable in the near term.

**Option C — Use the curated `combined_spike.json` (225 segs).**
- Already has `s_probit` per segment, but stratified into ±1.59 (two bins only — extremes-only design from the spike paper).
- No `s_sd` — would need to derive (e.g. by joining back to fit posteriors, or by analytic SD from `frac_yes` + `n_annotations`).
- case_keys are strings (e.g. `sn1::SEG_11449::scr0_1091`) — need a deterministic mapping back to integer seg_ids in SN1_combined_v2.h5 to fetch EEG payload.
- Pro: zero new calibration work.
- Con: 2-bin stratification reduces psychometric information content; the engine prefers a difficulty *range*; harder to demonstrate adaptive testing power on extremes-only data.

**Recommendation:** Option A. Stratified sampling from `fits/spike/cases.csv` is the cleanest path and uses the existing calibration to its fullest extent. The build script mirrors `build_cortex_test_bank_v2.py`'s structure but for spike.

### 3.D2 — `segment_signals.csv` with spike columns (or sibling file)

`scripts/cortex_engine_inputs.py` reads `data/labels/iiic_segment_signals.csv`, joins on `seg_id`, and reads `s_mean_<code>` + `s_sd_<code>` for each task code. For K=7, two clean designs:

**Design A — Single unified `segment_signals.csv` (recommended).**
Rename `iiic_segment_signals.csv` → `segment_signals.csv` with new columns `s_mean_spike`, `s_sd_spike`. For IIIC-pool segments (~69,806), spike values are NaN (those segments aren't spike-eligible). For spike-pool segments (the ~17,346 from `fits/spike/cases.csv`), the IIIC columns are NaN. Cortex_engine_inputs.py reads each segment, asks "which tasks does this segment have valid signals for?", and emits a task-mask per segment.

Pros: one file; clean schema evolution; producer is `pipeline/joint_calibration/assemble_outputs.py` extended once.
Cons: many NaNs in the file; needs slight schema change in the loader.

**Design B — Sibling `spike_segment_signals.csv`.**
Keep `iiic_segment_signals.csv` IIIC-only; add a new sibling file with `seg_id, s_mean_spike, s_sd_spike, n_raters, pos_rate, n_<tier>, votes_spike_yes`. Cortex_engine_inputs.py reads both and merges.

Pros: zero schema change to IIIC consumers; minimal blast radius.
Cons: two files; engine_inputs has to handle both.

**Recommendation:** Design B for v1.1.x (lowest blast radius); Design A as a Phase-3.5 schema unification in a later major version. The file `fits/spike/cases.csv` already has the exact schema we'd want; symlinking or copy-renaming + adding source/tier columns matches IIIC.

### 3.D3 — K=7 `Sigma_l_fitted.npy`

Mirrors Pipeline-G5 finding in `docs/NATURE_MEDICINE_AUDIT.md §4.G5`. The 7×7 ℓ-block of the deployment Σ is a block-constant matrix with `r_ell = 0.36656` off-diagonal:

```
     l_spike  l_seizure  l_lpd  l_gpd  l_lrda  l_grda  l_other
l_*    1.000      0.367  0.367  0.367   0.367   0.367    0.367   (etc.)
```

Two options:

**Option A — Direct extraction (recommended for v1.1).**
Write `scripts/build_sigma_l_k7.py` that reads `data/deployment_prior/Sigma.csv`, extracts the 7-name ℓ-block + the 7-name t-block, saves as `Sigma_l_fitted_k7.npy` with the same internal layout `core_mcmc.load_fitted_Sigma` expects (i.e. an npz/npy with `domains`, `Corr_l`, `Corr_t`, `Sigma_t`, `Sigma_l`). Update `engine/engine_paths.py:14-17` to consume the K=7 variant when running K=7 sessions.

Pros: bit-exact reproducible from the deployment Σ; no new fit; ~30 minutes of work + tests.
Cons: the K=7 Corr_l carries the block-constant constraint (r=0.367 on all off-diagonals) — psychometrically debatable but **matches the deployment** (consistency with the v1.0 ship state is more important than psychometric optimality for v1.1.x).

**Option B — Refit from `data/labels/fits_hier_block`.**
Run a K=7 hierarchical fit fresh from the corpus; produce a free-Σ K=7 file with task-specific correlations.

Pros: psychometrically richer; would close NatMed §5.B5 / Σ-block-structure objection.
Cons: meaningful compute; would shift the engine prior away from the deployment Σ, creating a new engine/deployment drift.

**Recommendation:** Option A for v1.1.x ship. Option B as a Sprint-2 follow-on (consistent with the audit's T2.7) — but do it as a coordinated engine + deployment refit, not just engine.

---

## 4. The code surgery — file by file

### 4.C1 — `scripts/cortex_engine_inputs.py` (236 lines; central)

**Current state:** K=6 hard-coded.
- `TASKS = [("sz", "Seizure", "seizure"), …, ("iic", "Other", "other")]` — 6 entries (line 61).
- `SIGNALS_CSV = data/labels/iiic_segment_signals.csv` (line 53).
- `SIGMA_PATH = Sigma_l_fitted.npy` (K=6; line 54).
- `K = len(TASKS) = 6` (line 72).
- Validation gates: every bank IIIC seg must be in signals; every pattern_class must map to one of 6 tasks (lines 184-186).

**K=7 changes:**

```python
# Add a 7th task entry. Bank H5 attr `pattern_class` already supports 'spike'
# for spike segs; the lookup falls through to spike when not in IIIC set.
TASKS = [
    ("spike",  "Spike",  "spike"),
    ("sz",     "Seizure","seizure"),
    ("lpd",    "LPD",    "lpd"),
    ("gpd",    "GPD",    "gpd"),
    ("lrda",   "LRDA",   "lrda"),
    ("grda",   "GRDA",   "grda"),
    ("iic",    "Other",  "other"),
]

# Either single unified file or sibling pattern (Design B from §3.D2):
SIGNALS_CSV = os.path.join(_REPO, "data", "labels", "iiic_segment_signals.csv")
SPIKE_SIGNALS_CSV = os.path.join(_REPO, "data", "labels",
                                 "spike_segment_signals.csv")
SIGMA_PATH = os.path.join(_REPO, "Sigma_l_fitted_k7.npy")
```

The manifest assembly (`build_iiic_engine_inputs`, renamed to `build_engine_inputs`) needs to:
- Iterate both `iiic/` and `spike/` H5 groups.
- For IIIC segs: pull `s_mean_<sz,lpd,gpd,lrda,grda,iic>` from IIIC signals; set `s_mean_spike` and `s_sd_spike` to NaN. Task-mask = [F, T, T, T, T, T, T].
- For spike segs: pull `s_mean_spike, s_sd_spike` from spike signals; set IIIC columns to NaN. Task-mask = [T, F, F, F, F, F, F].
- Drop segments with all-NaN signals (defensive); allow per-task NaN within mask.
- In `as_engine_arrays(seg_ids)`, return a `task_mask` array shape (N, K) alongside `bank_signals` (also list-of-K arrays) — so the engine can skip task k for any segment where mask[i, k] = False.

This is the deepest change but it's clean.

### 4.C2 — `engine/core_mcmc.py` (the K-agnostic engine)

The engine reads K from Σ slot-name index (`load_fitted_Sigma` returns a `domains` field, currently `['sz','lpd','gpd','lrda','grda','iic']`). Once `Sigma_l_fitted_k7.npy` exists with `domains = ['spike','sz','lpd','gpd','lrda','grda','iic']`, the engine should "just work" — K, DIM, TASKS all derive from this.

**Verify K-agnosticism:** the test `tests/test_phase7_k7_validation.py:108-186` already exercises K=7 at the engine level (SBC mean-rank + 90/95 % CI empirical coverage). Confirms the engine has been validated at K=7 already.

**Open question:** does `choose_item` properly handle per-segment task masks? The current API (`bank_signals: list of K arrays per task`) assumes every candidate is valid for every task. With sparse masks, either (a) extend `choose_item` to take a `bank_mask: list of K bool-arrays per task` and only consider mask[i] = True candidates per task, or (b) pre-filter the bank per-task in `as_engine_arrays`. (b) is simpler and what `IIICEngineInputs.as_engine_arrays` should produce: for task k, the bank is the subset of segments where mask[, k] = True.

**Decision needed (asked below):** confirm (b) — task-specific banks — is acceptable to the AD6 stopping rule (which iterates per-task).

### 4.C3 — `scripts/cortex_policy.py` (the AD6 verdict policy)

**Current state:** loads ℓ\* via `f"sparcnet_{code}"` key — would miss spike's `combined_spike` key (line 122).

**K=7 changes:**

```python
# Map CORTEX task codes to cert_config v13 keys.
_KEY_FOR_CODE = {
    "spike": "combined_spike",
    "sz":    "sparcnet_sz",
    "lpd":   "sparcnet_lpd",
    "gpd":   "sparcnet_gpd",
    "lrda":  "sparcnet_lrda",
    "grda":  "sparcnet_grda",
    "iic":   "sparcnet_iic",
}

def load_ell_star(task_codes, config_path=None):
    ...
    for code in task_codes:
        key = _KEY_FOR_CODE.get(code)
        if key is None:
            raise KeyError(f"no cert_config key for task code {code!r}")
        ...
```

Rename `load_ell_star_iiic` → `load_ell_star` (it's no longer IIIC-specific). The docstring "K=6 IIIC tasks" updates to "K=7 spike + IIIC tasks." Same `AD6Policy` class works unchanged because it operates on the K-dim arrays — just give it 7-len ell_star + var_prior.

### 4.C4 — `scripts/session_controller.py` (the run loop)

**Current state:** docstrings say K=6 IIIC; the actual code derives `K = len(inputs.task_codes)` (line 155, 447). It's already K-agnostic.

**K=7 changes:**
- Update the module docstring (lines 1-19) — "K=6 IIIC" → "K=7 spike + IIIC."
- The `make_simulated_y_source` (line 341) takes `true_t, true_l` arrays of length K — works at K=7 unchanged.
- `_y_source(self, k, seg_id, s)` (line 380) → the live binary-answer callback — needs UI awareness of whether the segment is spike or IIIC, because the UI is different. This is the touchpoint with C5.

### 4.C5 — `scripts/eeg_bank_viewer.py` (the GUI) — the biggest UX decision

**Current state (per LIVE_TEST_INTEGRATION_PLAN.md):** 6 IIIC pattern buttons. Per-segment pattern_class attribute is read from H5; the user picks one of 6.

**K=7 UX options:**

**Option A — Two UI variants per segment type.**
- IIIC segment: existing 6-button pattern picker. Click "Seizure" → engine sees (k=seizure, y=1) AND (k=lpd, y=0), (k=gpd, y=0), (k=lrda, y=0), (k=grda, y=0), (k=iic, y=0). Spike is NOT asked for IIIC segments (task_mask).
- Spike segment: binary Yes/No buttons ("Is there a spike?"). Click "Yes" → engine sees (k=spike, y=1).

Pros: matches the underlying SDT model exactly; spike binary task = published spike-paper methodology; IIIC remains 6-way to keep IIIC information density per question.
Cons: two UIs to maintain; user mental-mode switching between segment types.

**Option B — Unified 7-button UI for every segment.**
- Every segment shows 7 buttons: Spike / Seizure / LPD / GPD / LRDA / GRDA / Other.
- Click any one → engine sees one (k=clicked, y=1) and 6 (k≠clicked, y=0).
- The pattern_class attr is the ground truth.

Pros: uniform UI; one button = one question for every segment; reduces task-switch cognitive load.
Cons: spike vs IIIC are not directly comparable semantically (spike is a 10s focal phenomenon; IIIC is a 30s pattern); asking "spike?" of a 30s IIIC clip is awkward; calibration for spike was fit on a different EEG window length; the spike-paper methodology is explicitly binary, so a 7-button choice forces ambiguous priors on "spike absent" for IIIC segments.

**Option C — Hybrid task batching.**
- Run a CORTEX session as two phases: Phase A (IIIC, 6-button) → Phase B (spike, Yes/No). Engine state crosses both phases (same particle cloud).

Pros: minimal UI disruption per phase; clear mental model.
Cons: requires session-level phase switching; harder to make adaptive across the K=7 latent space (the engine would prefer to pick the most-informative (segment, task) at each step regardless of phase).

**Recommendation:** Option A. Matches the methodology, matches the spike-paper UX, and lets the engine pick the optimal (segment, task) at each step. The two UIs share a common parent widget; the per-segment renderer dispatches on `test_class` attr.

### 4.C6 — `scripts/build_cortex_test_bank_v2.py` (the bank builder)

**Current state:** builds 300 IIIC (50 per class) + verbatim copy of 100 spike (uncalibrated).

**K=7 changes:** add a `--include-spike-calibrated` stage that picks N spike segments (stratified by `s_mean`) from `data/labels/fits/spike/cases.csv` and pulls their EEG from the external SN1 source (or a vendored derivative). The current verbatim 100 spike copy stays as legacy fallback until validated.

Stratification (recommended): 10 strata on `s_mean` × 5 segments per stratum = 50 calibrated spike segments. Quality-bias by `n_raters` within each stratum, same as the IIIC sampler.

### 4.C7 — `cortex_app/cortex.spec` (PyInstaller bundle)

PyInstaller spec needs to include the new `Sigma_l_fitted_k7.npy` and `spike_segment_signals.csv` in the bundled `datas` list. ~5-line change.

### 4.C8 — `cortex_app/fetch_test_bank.sh` (release-fetch)

Bumps the `build-data-v1` reference to `build-data-v2` (or `-v3` if v2 has been used). The bank itself needs to be re-uploaded to a new GitHub release after rebuild. Documented in `cortex_app/BUILD.md`.

### 4.C9 — `scripts/cortex_storage.py` (per-session JSONL audit log)

Storage already uses generic per-trial schema; should be K-agnostic. **Verify** by inspecting the schema columns — if they include `k_index` / `task_code` rather than hard-coded IIIC, no change needed.

### 4.C10 — `scripts/cortex_diagnostics.py`

Same as C9: per-task diagnostics; should be K-agnostic via `len(task_codes)`. **Verify**.

### 4.C11 — Tests

New / updated tests:

| Test file | Action |
|---|---|
| `tests/test_cortex_engine_inputs.py` | Add K=7 cases; verify spike segment loading + IIIC-only task masks |
| `tests/test_cortex_session_controller.py` | K=7 simulated session with mixed segments |
| `tests/test_cortex_end_to_end.py` | Full session K=7 smoke test |
| (new) `tests/test_sigma_l_k7_build.py` | Drift-guard: K=7 Sigma_l matches deployment Σ ℓ-block extraction |
| (new) `tests/test_spike_bank_calibrated.py` | Every spike segment in bank has valid `s_mean_spike` + `s_sd_spike` |
| `tests/test_phase3_calibration.py` | Add `test_v13_ell_star_pinned()` (per NatMed audit T1.1) — pin all 7 ℓ\* numerically |

---

## 5. Engine readiness check — what's already K=7

The following K=7 paths are already in place and tested:

| Path | Status | Test |
|---|---|---|
| Engine K-agnostic via Σ slot names | ✅ | `tests/test_phase4_port_fidelity.py:80-100` |
| K=7 SBC mean-rank 0.25–0.75 | ✅ | `tests/test_phase7_k7_validation.py:108-139` |
| K=7 empirical 90/95 % CI coverage > 0.7 | ✅ | `tests/test_phase7_k7_validation.py:142-186` |
| Deployment 14×14 Σ from PI block fit | ✅ | `data/deployment_prior/Sigma.csv`; `summary.json` |
| cert_config v13 has all 7 ℓ\* values | ✅ | `calibration/cert_config.yaml:58-112` |
| Phase-4.5 v13 ℓ\* deployment integration | ✅ | `deployment/phase4_5_v13_ell_delta.json` |
| Phase-7 deployment replay drift-guards K=7 | ✅ | `tests/test_phase7_*_drift.py` |

The K=7 work is already done at the engine + deployment + calibration level. What remains is CORTEX-app-specific: input assembly + UI + bank.

---

## 6. Sequencing — three concrete options

All three sequences end with v1.2.x: K=7 CORTEX shipping with rigorous calibration. They differ in scope of spike bank and engine prior.

### Sprint 1A — Minimal viable K=7 (recommended for fastest CORTEX K=7)

| # | Title | Effort | D-risk |
|---|---|---|---|
| S1A.1 | `scripts/build_sigma_l_k7.py`: extract K=7 Corr_l from deployment Σ; emit `Sigma_l_fitted_k7.npy`; pin test | 0.5 day | None |
| S1A.2 | `scripts/build_spike_signals.py`: emit `data/labels/spike_segment_signals.csv` from `fits/spike/cases.csv`; add source/tier vote breakdown | 0.5 day | None |
| S1A.3 | `scripts/build_spike_bank_v3.py`: pick stratified N=50 spike segs from calibrated set; fetch EEG from SN1_combined_v2.h5; vendor into new bank file | 1 day (needs SN1 access) | None (D9 preserved — derived signals only) |
| S1A.4 | `scripts/cortex_engine_inputs.py`: add spike to TASKS; read spike signals; emit per-segment task_mask; new K=7 Sigma loader | 1 day | None (additive) |
| S1A.5 | `scripts/cortex_policy.py`: `_KEY_FOR_CODE` mapping; rename `load_ell_star_iiic` → `load_ell_star` | 30 min | None |
| S1A.6 | `scripts/session_controller.py`: docstring + spike UI dispatch via `test_class` attr | 0.5 day | None |
| S1A.7 | `scripts/eeg_bank_viewer.py`: per-segment-type UI (6-button for IIIC, Yes/No for spike) | 1–2 days | None |
| S1A.8 | `scripts/build_cortex_test_bank_v2.py`: update to optionally include calibrated spike (gated on S1A.3 output) | 0.5 day | None |
| S1A.9 | Tests: per-section above (C11) | 1 day | None |
| S1A.10 | `cortex_app/cortex.spec` + `fetch_test_bank.sh`: bundle new files | 30 min | None |
| S1A.11 | New `build-data-v3` GitHub release with the K=7 bank | 30 min | None |
| S1A.12 | Per-task replay smoke test on a few raters using a calibrated K=7 session | 1 day | None |

**Total: ~7–9 days focused effort.** All preserve D1–D9. Output: CORTEX v1.2.0 candidate with K=7 + 50 calibrated spike + 300 IIIC bank.

### Sprint 1B — K=7 with the deeper Σ refit (audit T2.7 + matching v1.0 audit Sprint 2)

Adds:
| # | Title | Effort | D-risk |
|---|---|---|---|
| S1B.13 | `pipeline/joint_calibration/fit_joint_k7.py`: 7-task hierarchical fit on `fits_hier_block` substrate; emit free-Σ K=7 ℓ-block | 1–2 weeks (compute-heavy) | Engine + deployment Σ both update; coordinated re-freeze; Paper-1 figures regenerate |
| S1B.14 | Phase-2 byte-equivalence re-run at K=7 free-Σ | 0.5 day | Possibly minor numerical deltas; document in CHANGELOG |

**Recommendation:** do S1B only after S1A is shipped + the CORTEX K=7 v1.2.x is validated in a live pilot. The block-constant K=7 Σ is good enough for the engine and matches deployment; the free-Σ refit is a Sprint-2 polish that affects the methodology paper.

### Sprint 2 — Bank scale-up to the real publication scope

| # | Title | Effort | D-risk |
|---|---|---|---|
| S2.1 | Expand spike bank from N=50 to N≥100 (or to match the publication enrollment scope) | 0.5 day | None (re-stratify) |
| S2.2 | Expand IIIC bank from N=300 to publication scope | 0.5 day | None |
| S2.3 | Spike-segment validation: verify the SN1-vendored 10s slices are clinically valid (review by MBW) | 1–2 days | None |
| S2.4 | Per-task drift-guard tests (Phase-7 pattern) for K=7 CORTEX bank | 0.5 day | None |

---

## 7. The decision points blocking Sprint 1A

These four decisions need user input before kicking off S1A.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| Q1 | **Spike bank source** | A. Build from SN1 calibration (need SN1_combined_v2.h5 access) <br> B. Recalibrate the morgoth1:spikes bank (needs new ratings; not in our corpus) <br> C. Use curated_banks/combined_spike.json 225 segs (2-bin stratified; no s_sd) | **A** |
| Q2 | **Σ_l K=7 source** | A. Extract from deployment 14×14 Σ (block-constant r=0.367) <br> B. Refit free-Σ from `fits_hier_block` (Sprint 1B) | **A for v1.2.x; B in Sprint 1B** |
| Q3 | **UI for spike segments** | A. Two UIs: 6-button for IIIC, Yes/No for spike <br> B. Unified 7-button for every segment <br> C. Phased (IIIC phase then spike phase) | **A** |
| Q4 | **Cross-bank task masking** | A. Each segment serves exactly one segment-type's tasks (spike serves task=spike only; IIIC serves the 6 IIIC tasks) <br> B. Allow spike segments to also probe IIIC and vice versa (would need cross-task signals computed) | **A** |

The audit-converged 5 of these were already aligned with Eli's earlier scope choices (Nature Medicine; full-scope) — Q1–Q4 are the new ones specifically gating Sprint 1A.

---

## 8. Cross-links

- `docs/LIVE_TEST_INTEGRATION_PLAN.md` §4 — the headline blocker (calibrated spike signals)
- `docs/AD6_RESOLUTION.md` — per-task three-way verdict policy (already K-agnostic via task codes)
- `docs/NATURE_MEDICINE_AUDIT.md` — broader audit; §4.G5 covers the K=6→K=7 Σ_l prior mismatch
- `data/DATA_PROVENANCE.md` — corpus + derivation chain
- `calibration/cert_config.yaml::ell_star_unified_v13` — already has all 7 ℓ\* values
- `data/labels/fits/spike/cases.csv` — the spike calibration substrate (17,346 segs)
- `data/deployment_prior/Sigma.csv` — the K=7 14×14 deployment Σ (ℓ-block is r=0.367 constant)
- `scripts/cortex_engine_inputs.py` lines 11-14 — the comment that documents the K=6 blocker
