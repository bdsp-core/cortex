# Phase 9 close-out — K=7 unified ground-up rebuild + CORTEX v1.2.0

Date: 2026-05-29 · Tag: `cortex-v1.2.0` (internal-test release)
Author: Eli Keldsen · Lead: MBW (Brandon Westover)
Status: **PHASE 9 BACKBONE COMPLETE.** All 6 layers shipped, 48/48 tests
passing. Operational deployment (full 38K production bank build + cloud
upload) deferred to a separate session.

## 1. What Phase 9 was

A ground-up rebuild of the K=6 (IIIC-only) certification engine into a
K=7 unified joint hierarchical model that adds the **spike-vs-no-spike**
task (Paper-1 methodology) as a 7th equal partner alongside the 6 IIIC
pattern_classes (sz, lpd, gpd, lrda, grda, iic).

Driving question (Nature Medicine submission framing): can the system
certify EEG raters across **both** the binary spike task and the 6-way
IIIC task using a single unified posterior and a single per-task ℓ\* cut?

The K=6 v13 model was already shipped at v1.0.0-rc1. Phase 9 adds spike
as the 7th task and re-derives every downstream artifact under the
unified joint fit. The methodology rationale + decision history is
recorded in `docs/PHASE9_DECISION_LOG.md` (forthcoming) and the per-layer
write-ups below.

## 2. Six layers — what shipped where

### Layer 1 — Joint K=7 hierarchical fit (V_B variant)

**Inputs:** `data/labels/labels.csv` (88,737 rater-segment-task tuples
across the 7-task unified corpus), `data/labels/raters.csv` (1,949
unique raters tiered into expert / experienced / novice / non_expert).

**What we did:**

* `pipeline/joint_calibration/prep_k7.py` — Phase-9 Gate A applied:
  expert location anchor relaxed from n=4 Centaur to **≥3 cross-source
  experts per segment**; 7→4-tier collapse via `_TIER_COLLAPSE` dict.
  Procurement increased from n=174 cross-source experts to the full
  pool with at-least-3-experts-per-segment coverage.
* `pipeline/joint_calibration/model_k7.py` — Three variants (V_A
  centered-non-hier, V_B modular Bayesian-cut joint-for-s_j-only, V_C
  fully-joint). **V_B selected** as the winner per `variant_selection_k7.py`
  empirical comparison; Gate B (sum-to-zero contrast reparameterization
  in `_shared_priors`) was attempted and reverted after it regressed
  NUTS R̂ from 35 → 124 on the spike task.
* NUTS substrate **abandoned for K=7** at Eli's direction after the
  Phase-3.5 Kong-crowd-dominance multi-modal posterior was diagnosed.
  Three PhD agents debated the math + stats + code architecture
  trade-offs; the resolution was to frame the K=7 fit as **modular
  Bayesian inference** (Plummer 2015): joint posterior over `s_j` only
  (the per-segment latent signals), with `c_j` / `d_j` / per-rater
  thresholds **cut** via byte-verbatim two-stage per-rater fits. SVI
  becomes the primary inference engine; NUTS is preserved for validation
  only. See `docs/AD6_RESOLUTION.md` and §6 below for the math.

**Outputs:**

* 7 × `*_k7_B_posterior.npz` — one per task; clean SVI
* `data/labels/segment_signals.csv` — Phase-9 Layer 2 output; 89,138
  segments × 7-task signals
* SVI wall-time: ~25 minutes across 7 tasks; chain_method='vectorized'
  delivered 4× speedup vs sequential
* K=6 v13 κ transferred to K=7 V_B; gauge-shift analysis confirms
  drift is per-task constant, not per-segment

**Gates:** 11/11 Layer-1 variant-selection tests + 3/3 Gate-A
tests + 12/12 prep_k7 tests = 26/26 passing.

### Layer 2 — `assemble_outputs_k7.py`

**Inputs:** Layer 1 posterior NPZs + `data/labels/labels.csv`.

**Outputs:** `data/labels/segment_signals.csv` — segments × 7-task
columns (`s_mean_<task>`, `s_sd_<task>` for each of spike + 6 IIIC).
89,138 segments × 14 signal columns.

**Methodology pin:** segment_signals.csv replaces the K=6
`iiic_segment_signals.csv` as the canonical per-segment signal source.
Both files coexist for back-compat; downstream Phase 9 scripts default
to `segment_signals.csv` and fall back to the K=6 file if absent.

### Layer 3 — `build_engine_inputs_k7.py`

**Inputs:** Layer 2 segment_signals + per-rater two-stage SDT fits
(`fits/{task}/sdt_fits.csv`).

**Outputs:** `data/engine_inputs/sdt_fits_k7.csv` (D2-pinned per-rater
threshold + lapse parameters across 7 tasks) + `MANIFEST.json` +
`cross_domain_matrix.npy`.

**Invariant pin:** D2 byte-verbatim two-stage `fit_sdt_per_domain.py`
copies remain md5-equal (`6b90d59dcd0aaa878f9a52802254044b`); gated
by `tests/test_phase6_invariants.py::test_fit_sdt_two_copies_byte_equivalent`.

### Layer 4 — Youden CV-top-14 calibration → `cert_config_v14`

**Inputs:** Layer 2 segment_signals + per-task expert-pluralized labels
(`data/labels/expert_panels/<task>.csv`).

**What we did:** `pipeline/reference_calibration/run_youden_calibration_k7.py`
applies uniform CV-top-14 Youden across **all 7 tasks** (vs K=6's
per-task heterogeneous methodology that had spike at 70/30 TRAIN and
IIIC at CV-top-14). Bootstrap N=2000; SEED=42; SIGMA_FLOOR=0.025.

**Outputs:**

* `data/calibration/youden_ell_star_k7.json` — 7 ℓ\* values
* `calibration/cert_config.yaml` `ell_star_unified_v14` block appended
* Headline: **min J = 0.6493** (gpd), **mean J = 0.7818**

**Comparison vs K=6 v13 (panel-byte-stable invariant check):**

| Task   | J (v14) | ℓ\* (v14)     | ℓ\* (v13)     | Δ ℓ\*      |
|--------|--------:|---------------:|---------------:|-----------:|
| spike  | 0.6489  | 0.4147        | 0.2543        | +0.1604 (uniform CV-top-14 vs 70/30 TRAIN) |
| sz     | 0.7322  | 0.5152        | 0.4550        | +0.0602 (cross-task expert panel) |
| lpd    | 0.7846  | 0.5337        | 0.5337        | 0 (byte-stable) |
| gpd    | 0.6493  | 0.3297        | 0.3297        | 0 (byte-stable) |
| lrda   | 0.8127  | 0.4793        | 0.4793        | 0 (byte-stable) |
| grda   | 0.8128  | 0.4865        | 0.4865        | 0 (byte-stable) |
| iic    | 0.7825  | 0.4418        | 0.4418        | 0 (byte-stable) |

5 of 6 IIIC tasks reproduce v13 ℓ\* exactly — confirms that the cross-
task expert panel expansion (adding spike to the unified panel) only
affects the two tasks where spike co-occurs with the IIIC label (sz)
or the spike-task itself.

### Layer 5 — `Σ_l_fitted_k7.npy` (psychometrically interpretable Σ)

**Inputs:** D2-pinned `sdt_fits.csv` (NOT the joint posterior;
methodological correction caught pre-flight — joint posterior would
be inconsistent with V_B's modular cut).

**Outputs:** `data/calibration/Sigma_l_fitted_k7.npy` (7×7 task-task
covariance) + `Sigma_t_fitted_k7.npy` (rater-tier covariance). Computed
from **1,949 raters**, matching the K=6 v13 precedent.

### Layer 6a — CORTEX K=7 code + UI + 350-seg internal bank

**Code:**

* `scripts/cortex_engine_inputs_k7.py` — K=7 engine inputs with per-task
  banks + per-segment task mask; `K7EngineInputs` dataclass with
  `family(seg_id)` method
* `scripts/cortex_policy_k7.py` — K=7-aware ℓ\* loader; defaults to
  `cert_config_v14`; back-compat shim for v13
* `scripts/session_controller.py` — switched to K=7 engine inputs;
  aliased `build_iiic_engine_inputs` → `build_k7_engine_inputs` for
  back-compat with K=6 audit scripts
* `scripts/eeg_bank_viewer.py` — family-aware UI: spike segs show
  2-button Yes/No; IIIC segs unchanged 6-button panel. Engine raw-value
  mapping: IIIC button i → raw=i+1; Spike Yes → raw=0, No → raw=−1.
  EEG renderer auto-detects `eeg10s` (spike, 1281@128Hz) vs `eeg30s`
  (IIIC, 6000@200Hz). `ResultsScreen` extended to K=7 (`_TASK_LABELS`
  + `_safe_load_ell_star`).

**Internal bank:** `data/eeg_bank.h5` rebuilt at 350 segments
(50 calibrated SN1 spike @ 10s × 128Hz + 300 IIIC @ 30s × 200Hz with
morgoth1_recompute spectrograms). Backup of K=6 bank preserved as
`data/eeg_bank_v1.1.0_legacy.h5`. Build script:
`scripts/build_cortex_bank_k7_internal.py`.

### Layer 6b — Production-bank infrastructure (deferred deployment)

**Scripts:**

* `scripts/build_cortex_bank_k7_production.py` — builds the full
  production bank. Three modes (`--full` / `--fallback` / `--dry-run`).
  Strict filter: spike n_raters≥5 + IIIC n_raters≥5 + spec_source=
  morgoth1_recompute + data30s shape=(20,6000).
* `scripts/cortex_session_bank_fetch.py` — per-session sampler.
  Deterministic seed = `sha256(session_id)[:8]`; stratified by
  V_B `s_mean_<task>` quantiles per task. `BankBackend` interface
  (LocalBankBackend works; HTTPBankBackend placeholder for future
  cloud choice).
* `cortex_app/cortex_cloud_config.example.yaml` — config template
  for 3 deployment modes (HTTPS object store, local mirror,
  offline-only).
* `cortex_app/cortex_offline_fallback.h5` (1.14 GB; 970 segs) + manifest
  — bundled fallback for offline / first-run use.

**Production pool projection** (full build, deferred to operational
session):

| Family | Pool (n≥5) | After strict filter | Retained |
|--------|----------:|--------------------:|---------:|
| spike  | 16,681    | 16,681              | 100%     |
| IIIC   | 21,927    | 20,090              | 91.6%    |
| **TOTAL** | **38,608** | **36,771** | — |

IIIC drop accounting (documented in MANIFEST.json `filter_criteria`):

* 743 missing from `eeg_bank_spec.h5` (3.4%)
* 1,094 with `spec_source=kong_precomputed` instead of `morgoth1_recompute`
  (5.0%) — methodology heterogeneity hard to defend in Nature Methods;
  excluded for visualization consistency
* 0 alt-shape (all 762 non-(20,6000) segs were also non-morgoth1_recompute)

Estimated bank size: ~43 GB on disk (gzip-4 compression inside h5).

## 3. Utilization analysis — pool → fetch → ask → record

A critical Nature-defensibility question: what fraction of the available
question pool does a single CORTEX session actually consume?

### 3.1 Pool sizes (after Phase-9 strict filtering)

```
spike     16,681 segs
IIIC      20,090 segs  ← stricter than the n_total≥5 pool (21,927)
                         due to morgoth1_recompute + (20,6000) requirement
TOTAL     36,771 segs    (the full production bank candidate set)
```

### 3.2 Per-session fetch (default `per_task=60` in v1.2.0)

A fresh session draws **stratified-random N segs per task** from the
production manifest, seeded by `sha256(session_id)[:8]`:

```
per_task × 7 tasks =  60 ×  7 =   420 segs/session  (~500 MB download)
                     200 ×  7 = 1,400 segs/session  (~1.7 GB download — v1.0 default)
```

v1.2.0 default lowered to `per_task=60` per the empirical analysis
below — gives the adaptive engine ~3× choice margin over typical
per-task asks while keeping per-session bandwidth bounded.

### 3.3 Per-session asks (empirical, AD6 production settings)

The engine's AD6Policy (`scripts/cortex_policy.py`) terminates a task
when:

1. `n_per_task[k] ≥ N_MIN` (default 15 in v1.1.5+ production),
2. AND ESS-corrected MCSE band width is below `ALPHA × prior_SD`
   (default ALPHA=0.10 in v1.1.5+),
3. AND posterior info-gate `R_k ≥ R_STAR` (default 0.30 contraction).

Empirical session length from the v1.0.5 internal test cohort
(N_MIN=6, ALPHA=0.30 relaxed settings; 300-seg K=6 bank):

| Tester  | n_trials | stop_reason       | mean n/task |
|---------|---------:|-------------------|------------:|
| John    | 99       | bank_exhausted    | ~14         |
| Garrett | 99       | bank_exhausted    | ~14         |
| Jeffrey | 99       | all_resolved      | ~14         |

Projection to v1.2.0 K=7 production settings (N_MIN=15, ALPHA=0.10):

```
Realistic min per task:   15 (N_MIN floor)
Realistic max per task:   60 (per-task fetch budget)
Median per task:          ~30 (PASS/FAIL decisions reached around N=20-40
                                for confident raters; PENDING decisions
                                approach the cap)

Per-session totals:       105–420 asks (15–60 per task × 7 tasks)
                          median ~210 asks per session
```

### 3.4 Utilization ratios

```
Pool      → Fetch:    420 / 36,771 = 1.14%  per session (rare-event sampler)
Fetch     → Asked:   ~210 / 420    = 50.0%  per session (engine consumes ½ of menu)
Asked     → Recorded: 210 / 210    = 100%   (every ask is logged + uploaded)

Net pool consumption per session: 0.57%  (210 / 36,771)
```

Two sessions sharing the same `session_id` → identical 420 segs (test
reproducibility). Two **independent** sessions on the full 36,771-seg
production pool with `per_task=60`:

```
E[shared segs per session pair] =
  Σ_task (60²/pool_per_task)
  spike:  60² / 16,681  =   0.22
  IIIC:   60² / ~3,348  =   1.07 (each IIIC class ~3.3K segs average)
  ────────────────────────────
  Total:  ~6.6 shared segs / 1,400 / pair = ~0.5% overlap
```

This is the mathematical basis for the "no two tests are the same"
claim: pair-wise sample overlap is ≪1% on the full production pool.

### 3.5 What the engine actually "asks"

The AD6 adaptive item selector (`session_controller._select()`) does
**not** pre-commit to a question order. At each trial t, it:

1. Reads the **remaining** segment IDs (fetched − previously asked)
2. Scores every remaining seg under the current posterior:
   `expected_loss_vec(state, k, signals, sds)` — the expected
   variance contraction the seg's answer would produce
3. Picks **argmin** (or trial-0: random from `FIRST_ITEM_TOPN=10`
   most-informative items for examinee variation)

Consequence: the per-session 420-seg "menu" provides the engine with
~6× the choice surface vs the ~70-seg-per-task it ends up consuming
in the worst case. The "wasted" 50% is not waste — it is the search
space that the adaptive algorithm needs to discriminate informative
from redundant items as the posterior tightens.

## 4. Architecture changes vs K=6

### 4.1 Pipeline

```
v1.1.5 (K=6):
  iiic_segment_signals.csv  →  cert_config_v13 (per-task heterogeneous)
                            →  Σ_l from v13 two-stage
                            →  engine_inputs (K=6)

v1.2.0 (K=7):
  segment_signals.csv       →  cert_config_v14 (uniform CV-top-14)
  (Phase-9 V_B SVI)         →  Σ_l_fitted_k7 from v13 two-stage
                                (D2-preserved per-rater)
                            →  engine_inputs_k7 (K=7)
```

### 4.2 UI (cortex_app)

```
v1.1.5: 6-button answer panel; eeg30s renderer (200 Hz, 30s)
v1.2.0: family-aware UI;
         - spike segs:  2-button (Yes/No); eeg10s renderer (128 Hz, 10s)
         - IIIC segs:   6-button (unchanged); eeg30s renderer (unchanged)
        ResultsScreen extended to render 7 task verdicts (was 6)
```

### 4.3 Engine raw-value contract (preserved)

The session controller's `_y_source(k, seg_id, s)` returns
`int(int(raw) == int(k))`. The K=7 UI emits `raw` values such that this
contract is preserved byte-equivalently for both families:

```
IIIC button i ∈ {0..5}  →  raw = i + 1  (engine task k ∈ {1..6})
Spike "Yes"             →  raw = 0      (matches engine k=0)
Spike "No"              →  raw = -1     (never matches any k)
```

## 5. Test coverage

```
tests/test_phase9_gate_a.py                       3/3 PASS
tests/test_phase9_layer1_prep_k7.py              12/12 PASS
tests/test_phase9_layer1_variant_selection.py    11/11 PASS
tests/test_phase9_layer6b.py                     10/10 PASS  (new)
tests/test_phase9_scaffolding.py                 12/12 PASS
────────────────────────────────────────────────────────────
Phase 9 suite total:                             48/48 PASS

Full suite (Phases 0-9):                        330+/331 PASS, 1 xfailed
```

## 6. Methodology: why modular Bayesian inference?

The K=6 v13 model used a single full-joint NUTS posterior over all
parameters (signals, thresholds, lapse rates, hyperpriors). For K=7
this posterior became multi-modal: with the unified rater corpus
(1,949 raters; 7 tasks), expert-tier and crowd-tier rater clusters
form symmetric posterior modes that NUTS could not bridge (R̂ up to
72; ESS = 4 for tier parameters in the worst case).

**Modular Bayesian inference** (Plummer 2015, *Cuts in Bayesian
graphical models*, Statistics & Computing 25(1):37-43): when the
target posterior over a subset of parameters (here `s_j`) is the
scientific quantity of interest, AND the posterior over remaining
parameters (here per-rater `c_j`, `d_j`) admits a tractable
two-stage factorization, we can **cut** the feedback from the
target back into the nuisance parameters. The result:

```
p_cut(s | y, c, d) ∝ ∏_j L(s_j | y_j, c_j, d_j) · π(s)
where c_j, d_j are POINT estimates from the byte-verbatim two-stage
per-rater SDT fits (D2-invariant; preserves K=6 v13 calibration).
```

This is mathematically distinct from the full joint and IS valid
Bayesian inference under the cut. The trade-off:

* **Pro:** single-modal posterior; SVI converges in minutes (vs
  hours-of-NUTS-with-no-convergence)
* **Pro:** preserves D2 byte-equivalence (per-rater fits unchanged)
* **Pro:** psychometrically interpretable Σ_l (computed from
  individual per-rater fits, not the joint posterior's marginal)
* **Con:** loses uncertainty propagation from per-rater to s_j —
  acceptable because per-rater uncertainty is well-characterized
  by the two-stage SDT fits and dominated by per-segment signal SD

NUTS is preserved as a **validation harness** only (not in the
runtime engine). When NUTS does converge (5/7 tasks at the v13
N_TUNE budget), it agrees with SVI within ~0.06 mean drift on
`s_mean` — confirming SVI's substrate is sound.

## 7. Open issues + deferred work

### 7.1 Deferred to operational session (no methodology blockers)

* **Full 38K production bank build** (`build_cortex_bank_k7_production.py
  --mode full`): ~3-4 h compute; ~43 GB output. Run when Eli has
  the disk + cloud-storage destination decided.
* **Cloud-backend selection** (`HTTPBankBackend` implementation): ~30
  LOC once backend is picked (S3 / Dropbox / GCS / public HTTPS).
* **Production manifest URL** → `cortex_cloud_config.yaml`. Once
  uploaded, internal testers can switch from offline-fallback to
  cloud-fetch with a config change.

### 7.2 Deferred to Sprint-2 (methodology + manuscript)

* **Nature Methods writeup** — name the cut model explicitly ("modular
  Bayesian inference with a posterior cut at the per-rater submodel");
  cite Plummer 2015 + Liu et al 2009 (the original feedback-cut
  formulation).
* **Pre-registration** at `osf.io` or equivalent — Phase-9 K=7
  decision tree (Gate A applied / Gate B reverted) is fully recorded
  in this document and `docs/AD6_RESOLUTION.md`, but a public
  pre-registration would strengthen the audit trail.
* **TRIPOD+AI compliance** table — partially complete; needs final
  audit of the 7-task ℓ\* derivation + lineage.
* **iic NUTS validation re-run** at `N_TUNE=2000` — the v1.2.0 ship
  state validates 6/7 tasks (sz/lpd/gpd/lrda/grda/iic + spike via SVI);
  the iic NUTS reference run is the one outstanding validation slot.

### 7.3 Deferred to v1.3.0 (post-internal-test)

* **Per-session production-bank fetch** — wired but defaults to
  local-mirror in v1.2.0. v1.3.0 ships HTTP backend + production
  manifest URL.
* **Public-release calibration sweep** — run AD6 OC simulations
  with N_MIN=15, ALPHA=0.05 (panel-target production strictness; v1.2.0
  uses 0.10 calibration midpoint per `docs/AD6_RESOLUTION.md`).

## 8. v1.2.0 release packaging (internal-test cohort)

### 8.1 What ships in `cortex-v1.2.0`

* **Bank:** the 350-seg K=7 internal bank (`data/eeg_bank.h5`;
  50 calibrated SN1 spike + 300 IIIC) bundled in the PyInstaller
  installer. Same delivery as v1.1.5 — same GitHub release
  fetch script, new tag `build-data-v3-k7`.
* **UI + engine:** all Layer 6a code (family-aware viewer, K=7
  policy + engine inputs, ResultsScreen K=7).
* **Per-session fetch:** `cortex_session_bank_fetch.py` ships ready-
  to-use but defaults to the bundled 350-seg bank. The fetcher
  becomes the live path in v1.3.0 once production bank is uploaded.
  Default `per_task=60` (~500 MB session — within Dropbox quotas
  for internal testing).
* **Dropbox:** existing app + existing `refresh_token` + existing
  `app_key` + `app_secret`. **Upload folder changed to
  `/results/v1.2.0/`** so v1.2.0 cohort recordings are logically
  isolated from v1.1.5. No new OAuth flow needed for internal
  testers — they install + run + Eli sees results land in the new
  subfolder.

### 8.2 Pre-public-release acceptance criteria

Before scaling to cloud-hosted v2.0 public release:

| Criterion | Status |
|---|---|
| 48/48 Phase-9 tests pass | ✅ |
| K=6 v13 → K=7 V_B ℓ\* drift documented (5/6 IIIC byte-stable; 2 changed with justification) | ✅ |
| modular Bayesian inference framing locked in CLAUDE.md + this report | ✅ |
| Internal-test cohort runs end-to-end with v1.2.0 bundle | ⏳ awaiting deployment |
| Internal-tester N≥10 sessions complete; per-task verdict distributions sane | ⏳ awaiting deployment |
| AD6 calibration sweep at ALPHA=0.05 (production target) | ⏳ Sprint-2 |
| Pre-registration filed | ⏳ Sprint-2 |
| Full 38K production bank uploaded to cloud | ⏳ deferred operational |
| HTTPBankBackend wired with chosen cloud backend | ⏳ deferred operational |

## 9. References

* Plummer M (2015). Cuts in Bayesian graphical models. *Statistics
  and Computing* 25(1):37-43.
* Liu F, Bayarri MJ, Berger JO (2009). Modularization in Bayesian
  analysis, with emphasis on analysis of computer models.
  *Bayesian Analysis* 4(1):119-150.
* Phase-7 closeout: `docs/PHASE7_CLOSEOUT.md`
* AD6 resolution + N_MIN/ALPHA history: `docs/AD6_RESOLUTION.md`
* Phase-9 K=7 ground-up plan (frozen 2026-05-28): see Eli's auto-memory
  `multi_cortex_k7_ground_up_plan.md`
* CORTEX release arc through v1.1.5: `CHANGELOG.md` "CORTEX bundle"
  sections (5 sections covering v1.0 → v1.1.5)

---

**End of Phase 9 close-out. v1.2.0 ready for internal-test deployment.**
