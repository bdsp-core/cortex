# Nature Medicine Readiness Audit — Multi-rater CORTEX system

**Date:** 2026-05-28
**Scope:** end-to-end audit of `ilae-skill-certification-test-multi` at v1.0.0-rc1
candidate, targeting **Nature Medicine** submission for the CORTEX rater-certification
system. Full scope (data + methodology + new acquisition all considered).
**Method:** five PhD-level domain audits run in parallel — biostatistics,
psychometrics/IRT-SDT, clinical EEG epidemiology, clinical-AI translation/TRIPOD+AI,
data pipeline + research-software methodology — each given D1–D9 invariants,
`docs/OPEN_DECISIONS.md` carry state, and an explicit Nature Medicine bar.
Findings synthesised below with file:line references.

This document is a **reviewer-grade synthesis**, not a plan. The recommendations
are sequenced under §6 (Recommended sequencing); the per-finding deep dives in
§4–§5 are intended to be readable by a co-author, reviewer, or board member.

---

## 1. Executive summary

**The engineering substrate is unusually strong.** Across 282 passing tests, byte-md5
pinned reference fitters, a single likelihood definition consumed by 5 code paths
under runtime assertion, single-thread BLAS determinism, sha256 D3 lineage between
`engine_inputs/` and `deployment_prior/`, and a five-dataset 2,115,793-label corpus
that dwarfs all prior published EEG-rater-certification work, the system is built to
a level of reproducibility hygiene Nature Medicine reviewers will recognise as
exceptional.

**Three categories of gap need to close before submission:**

1. **Two integrity gaps the test suite does not catch.** (i) The seven `ell_star`
   values in `calibration/cert_config.yaml` v13 — the *clinical decision boundary*
   for every PASS/FAIL verdict — are not pinned numerically in any of 282 tests; a
   hand-edit of the YAML would pass CI silently. (ii) The Phase-7 D6 headline
   real-rater replay cohort (n=21) substantially **overlaps** the calibration
   panels used to set ℓ\*: 16 of 21 replay raters appear in the 29-rater Q2-locked
   candidate pool, and 8–9 of the 14 raters in each domain's CV-Youden expert
   panel are also in the 21-rater replay cohort. The +43 % to +1154 % "replay
   over Bernoulli" headline is therefore partially confounded by calibration/test
   non-independence; the magnitude of confounding has not been quantified.

2. **Six methodological gaps that are non-negotiable for the Nature Medicine bar.**
   The probabilistic deployment outputs have *no* frequentist sanity checks (Brier,
   calibration-in-the-large, reliability diagrams, expected calibration error); the
   D6 headline finding has *no* formal hypothesis test (mixed-effects, paired
   McNemar, signed-rank); the 7-task per-candidate verdict scheme has *no* explicit
   family-wise error treatment (1 − 0.95⁷ ≈ 30 % per-candidate FWER); the lapse-
   sensitivity sweep that the team built (`scripts/run_lapse_sensitivity.py`) has
   *never been run on the unified corpus*; subgroup/fairness analyses are absent;
   decision-curve analysis (Vickers et al. 2019) is absent. The first four cost
   ≤ 2 days each on existing data; the last two are 6–8 hours of analysis.

3. **Three data gaps that bound what claims the manuscript can make.** The corpus
   has *no race / ethnicity / comorbidity / clinical-setting (ICU vs EMU vs
   outpatient)* fields, *4.2 % sex coverage, 16.4 % age coverage*, and where age
   is recorded **51 % of segments are pediatric** while the deployment is framed
   for adult-ICU IIIC monitoring. All five datasets trace to the MGH /
   BIDMC / Harvard / Yale circle under two Harvard-affiliated IRBs (BIDMC
   2016P000058, MGH 2013P001024); no external US-academic, VA,
   community-hospital, or international site contributes signals. The "external
   panel" the team has is n=4 (Centaur gold). Nature Medicine reviewer 2 will
   refuse to accept n=4 as an external cohort.

**The recommended sequencing is at §6.** The 4 highest-leverage actions are
(1) numerically pin v13 ℓ\* in the test suite (30 minutes, no risk); (2) run
the existing lapse-sensitivity sweep (minutes, no risk); (3) compute
reliability diagrams + Brier + CITL from the existing replay output (4 hours,
no risk); (4) document the replay/calibration overlap with a leakage-aware
sensitivity rerun (1–2 days). These four close two integrity gaps + two
TRIPOD+AI gaps with no D-decision changes. The expert-panel expansion
(n=4 → n≥10 non-PI) and the demographic backfill from BDSP source records
are the heavier-but-essential medium-term actions; without them the paper's
external-validity claim and fairness reporting are bounded.

---

## 2. What is strong (reviewer-ready)

Convergent across all five PhD audits. Cited with file:line where it lives.

### 2.1 Engineering hygiene Nature Medicine reviewers will recognise

- **Single-source likelihood (D2 invariant).** `P(y=1 | c, σ, θ, λ) = λ + (1 − 2λ) · Φ((c/1.7 − θ)/σ)` is defined exactly once at `pipeline/reference_calibration/fit_sdt_per_domain.py:55` and consumed bit-equivalently by the Mode-A SMC engine (`engine/core.py:21, 93-112`), the Mode-B engine (`engine/engine_mode_b.py`), the Laplace+EKF deployment runtime (`deployment/simulate_test.py:22-40`), and the joint hierarchical fit (`pipeline/joint_calibration/fit_joint_iiic.py:89`). The runtime assertion at `pipeline/run_unified_calibration.py:201` ensures the single source survives any future edit. This is genuinely rare and reviewers will appreciate it.

- **Numerical-stability hygiene** (`engine/core.py:93-112`). `scipy.special.log_ndtr` + `scipy.special.logsumexp` everywhere a probit log-likelihood is consumed, instead of the seductive `np.clip(norm.cdf(z), 1e-9, 1-1e-9)`. The CLAUDE.md "what NOT to do" section explicitly bans the clipped form because it collapses to 1.0 at |z| ≳ 6 and biases the posteriors of confident raters. This is the right call and the kind of correctness gate Nature Medicine reviewers grade highly.

- **Byte-md5 reference-fitter pinning.** Three calibration scripts are carried verbatim from the reference paper with md5 drift-guard tests:
  - `fit_sdt_per_domain.py` ≡ `6b90d59dcd0aaa878f9a52802254044b` (`tests/test_phase3_calibration.py:79`, `tests/test_phase6_invariants.py:67`)
  - `fit_main_effects.py` ≡ `fcf149b4f9c173ab14e9980ee3b82a13` (`tests/test_phase3_calibration.py:65`)
  - `run_youden_calibration.py` ≡ `f8bcfbd61b7a89c70e4dfec3cf8227db` (`tests/test_phase3_calibration.py:88`)

  Means no silent methodology drift between methodology paper and production engine.

- **D3 sha256 lineage anchor.** `engine_inputs/MANIFEST.json::data_labels_provenance.sha256` is cross-checked against `deployment_prior/summary.json::data_labels_provenance.sha256` and the live `data/labels/{labels,raters}.csv` sha256 (`tests/test_phase5_engine_inputs.py:91-116`). Means `engine_inputs/` and `deployment_prior/` are provably derived from the same labels corpus. **Caveat at §4.G3 below — the chain anchors only the *input*, not the intermediate fit outputs.**

- **BLAS single-thread bit-exactness.** `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=OMP_NUM_THREADS=1` set in `tests/conftest.py:22-25` *before* numpy imports; runtime parity in `scripts/_parallel.configure_blas_single_thread()`; verified by `tests/test_parallel_determinism.py:89-103` (parallel == serial bitwise on a seeded 64×64 eigvalsh/solve workload). Rare in clinical-AI papers.

- **JAX/calibration isolation.** Phase-3.5's NumPyro/JAX path is calibration-stage-only — `engine/core_mcmc.py` runtime never imports JAX. Deliberate isolation because XLA uses internal threading that would break the BLAS contract if it crossed module boundaries. (CLAUDE.md.)

- **K-agnostic engine.** The engine derives `K, DIM, TASKS` from the Σ slot-name index rather than hard-coded constants (`tests/test_phase4_port_fidelity.py:80-100`), with a K=7 SBC mean-rank test (`tests/test_phase7_k7_validation.py:108-139`) and K=7 90/95 % CI empirical coverage > 0.7 (`tests/test_phase7_k7_validation.py:142-186`).

### 2.2 Methodology that is Nature Medicine-defensible

- **Non-circular CV-top-N expert selection.** `pipeline/reference_calibration/run_youden_calibration.py:139-150` picks each domain's expert panel by ranking candidates on their mean ℓ over the *other five* domains, not on the held-out domain's own ℓ. The minimum-J figure jumps from the circular legacy ALL_7 (0.286) to the CV-top-14 0.778 — a real, defensible psychometric improvement, not a hyperparameter win. **Caveat at §4.G2 — the candidate pool itself is non-randomly selected, and the bootstrap CIs do not resample the selection step.**

- **TRAIN-only Youden** (σ\*/ℓ\* fitted exclusively on the 70 % TRAIN expert split; `pipeline/reference_calibration/youden_sigma_star_ref.py:19`, `EXPERT_TRAIN_FRAC=0.70`). Avoids the obvious leakage trap. **Caveat at §4.G1 — the 70/30 partition is over the σ-pool, not over rater identity; replay raters can overlap calibration raters.**

- **Joint hierarchical signal unification with gauge anchoring** (`pipeline/joint_calibration/fit_joint_iiic.py:102-117`). Split-anchored deterministic affine — scale `b = std(s) over weld set`, location `a = mean(s/b) over the n=4 gold panel` — with the numerical assertion that the gauge transformation is likelihood-invariant (`max |Δp| < 1e-5`). The cleanest treatment of the classic IRT shift/scale identifiability gauge I have seen in clinical-AI code.

- **Phase-3.5 SBC plug-in vs UNCERT contrast is genuinely high-quality science.** `calibration/joint/sbc_engine.json`: CONTROL mean-rank 0.521 ≈ 0.5 (harness sound); PLUGIN coverage 0.884 (miscalibrated under realistic item noise); UNCERT coverage 0.932 (restored to nominal under uncertainty propagation). The team honestly disclosed that the CONTROL KS marginally exceeds the critical value (0.048 vs 0.045) at n=900 due to SMC finite-particle approximation — this is what reviewers reward.

- **Variance-calibration NUTS-vs-SVI κ pinning** (`pipeline/joint_calibration/variance_calibration.py`; `tests/test_phase35_variance_calibration.py:31-42`): κ per task ∈ [1.18, 1.42] conservatively inflates the SVI s_sd to match exact-Bayes (NUTS); κ ≥ 1.0 enforced (the inflation never shrinks variance); honest "upper bound on needed inflation" framing.

- **Honest D7 reporting** (`calibration/CALIBRATION_PROVENANCE.md:259-279`). The Centaur n=4 expert-panel reproduction failed point-equivalence on 4/6 IIIC tasks; instead of spinning, the docs re-scoped D7 to "consistency / direction check" with the systematic gold > CV ℓ\* shift documented and Spearman ρ reported. Nature Medicine reviewers reward this kind of disclosure.

- **D1 two-engine separation is hard-walled.** `engine/` and `deployment/` never cross-import; the deliberate sharing point is the likelihood definition and the data layer. CLAUDE.md "what NOT to do" enforces.

### 2.3 The D6 finding is itself publishable

Per `docs/PHASE7_REPLAY_HEADLINE.md:75-98`: real-rater replay yields **+43 % (spike) to +1154 % (lrda) more PASS verdicts** than fitted-θ Bernoulli predicts, with 16–46 % tighter Mode-A AUROC posteriors. This is genuinely novel methodologically — most published psychometric certification systems are validated only against synthetic OCs and have no way to detect the within-rater autocorrelation / over-skill that real raters exhibit. The framing as the **favourable** direction (deployment will be more lenient than the OC tables predict) is honest, scientifically interpretable in two equivalent ways (reliability vs skill), and tells the reviewer the engine errs on the safe side. **Caveats at §4.G1 (leakage), §5.B1 (no formal test).**

### 2.4 Corpus scale comparison

| Reference | Source | n_segments | n_raters | Notes |
|---|---|---:|---:|---|
| Esteva 2017 (Nature) | Dermatology images | 129,450 | 21 (test) | Headline n=21 ref. |
| Gulshan 2016 (JAMA) | Diabetic retinopathy | 128,175 | 54 | |
| Rajpurkar 2017 CheXNet | Chest radiographs | 112,120 | 4 (test) | |
| Jing 2023 SPARCNET | EEG IIIC | ~50,000 | ~20 experts | Substrate of our `sparcnet50K`. |
| Kong 2025 (Epilepsia) | EEG IIIC (DiagnosUs) | ~9,000 | 1,537 raters | Substrate of our `kong2025`. |
| **CORTEX (this corpus)** | EEG IIIC + spike | **95,327** | **5,306** | Headline replay subset: **21 raters × 7 tasks × ≥10 segs/task**; 14,823 per-task cells. |

The label volume is therefore not a Nature Medicine blocker on its own — **the
21-rater per-candidate cohort is**, because it is *the* set on which the all-7
verdict claim rests. The cohort is also 90 % expert, which makes the
"replay-vs-Bernoulli leniency" generalisation to a true credentialing population
(mostly non-expert) unsupported (§4.G1 / §5.B1).

---

## 3. Data structure tour

This section is the prose tour of the on-disk data structures the
recommendations refer to. Schemas captured at audit time (2026-05-28).

### 3.1 Corpus tables (`data/labels/`)

Single source of truth (D3); 5 datasets unioned in one canonical layout. Post-
Centaur-ingest sizes from `data/DATA_PROVENANCE.md:18-27`.

| File | Rows | Schema head | Role |
|---|---:|---|---|
| `labels.csv` | 2,115,793 | rater_id, seg_id, label_type, label_value, source_dataset, ... | Per-(rater, segment, label_type) read; canonical observation table |
| `raters.csv` | 5,306 | rater_id, canonical_name, expertise_level, neurologist, epileptologist, board_certified, years_eeg, affiliation, source_dataset | Authoritative rater roster + expertise tier (the *only* expert-membership oracle per `docs/INVARIANT_AUDIT.md:113-116`) |
| `segments.csv` | 95,327 | seg_id, source_dataset, pat_key, mrn, pat_age, pat_sex, … | Per-segment metadata; **demographics ~4% sex / ~16% age coverage** (see §4.E2) |
| `datasets.csv` | 5 | source_dataset, n_segments, n_labels, description, s3_path | Dataset registry; `SN1_combined_v2.h5` recorded by path only (D9) |
| `segment_labels.csv` | 95,327 | seg_id, source_dataset, pattern_class, ... (33 cols incl. `iiic_vote_other` for K=7) | One row per segment, post-R1 regen on post-gold canonical tables |

Source-dataset decomposition (from agent inventory; `wc -l` verified):

| source_dataset | n_labels | n_segments | n_raters | Era / context |
|---|---:|---:|---:|---|
| `sn1_combined_v2` (3 sub-sources) | 964,206 | 20,521 | 2,551 | SN1 + Bonobo + SpikeEd; substrate of the spike-skill paper; Nascimento 2024 RCT lives here |
| `sparcnet50K` | 290,268 | 50,478 | 124 | Hong/Jing 2023 SPARCNET 6-class IIIC; 60 % of all IIIC segments |
| `pd_rda_profiler` | 48,727 | 13,556 | 9 | Internal curated 8-feature PD/RDA profiler |
| `centaur_2025_iiic` | 128,872 | 4,646 | 736 | Centaur novice IIIC contest |
| `centaur_2025_ied` | 167,503 | 5,000 | 643 | Centaur novice IED contest (BETS / vertex / POSTS / wickets) |
| `iiic_crowdsourcing:kong2025:crowd` | 487,683 | 7,817 | 1,529 | Kong et al. 2025 (Epilepsia 66:4366) DiagnosUs |
| `iiic_crowdsourcing:kong2025:expert` | 8,534 | 1,126 | 8 | Kong 2025 expert arm |
| `centaur_iiic_expert` (gold) | 20,000 | 5,000 | 4 | Phase-1 ingest: mbw/cal/matt/tianyu; the n=4 D7 panel |

Pattern-class prevalence in IIIC label pool: seizure 188,189 / other 173,650 /
lpd 151,946 / gpd 144,208 / grda 143,905 / lrda 130,056 / birds 18,397 / bipd
12,116. Crowd/Kong reads dominate the absolute counts.

### 3.2 Calibration intermediate (`data/labels/fits*/`, `pipeline/_calib_work/`)

| Subdir | Producer | Consumed by | Role |
|---|---|---|---|
| `data/labels/fits/{task}/` | PI scripts (upstream) | `deployment/freeze_deployment_prior.py:65-74` | Per-task Rasch case difficulties (vendored output) |
| `data/labels/fits_hier/` | PI hierarchical fit | (snapshot, free Σ) | Free-parameter K=6 hierarchical fit |
| `data/labels/fits_hier_block/` | PI block-constrained fit | `freeze_deployment_prior.py` | 3-param (`r_iiic`, `r_cross`, `ρ=0`) block fit → 14×14 Σ |
| `data/labels/fits_hier_factor/` | PI factor fit | (snapshot) | Factor decomposition variant |
| `data/labels/fits_hier_free_ml{300,1000}/` | PI free fit | (snapshot, sensitivity) | min_labels_per_rater sensitivity |
| `pipeline/_calib_work/data/prepared/{task}_fit/` | `pipeline/run_unified_calibration.py:step_b` | downstream steps | In-repo Rasch outputs (separate from PI fits/) |

**Note (G3 below):** `data/labels/fits/` is consumed by deployment;
`_calib_work/data/prepared/` is the in-repo recompute. The two are
computed at different times against the same `labels.csv`. The D3 sha256
anchor proves they share the *input* labels.csv version but does **not**
chain the intermediate Rasch outputs — a contributor could regenerate
`fits/` without re-running `_calib_work/`, breaking the consistency
silently while D3 still passes.

### 3.3 Engine inputs (`data/engine_inputs/`)

Per `data/engine_inputs/README.md` + `MANIFEST.json` (audited 2026-05-19).

| File | Rows | Schema | Role |
|---|---:|---|---|
| `sdt_fits.csv` | 14,214 | domain, rater_id, rater_name, sigma, theta, se_sigma, se_theta, n_trials, converged | **Authoritative** Phase-3 SDT fits, 6 IIIC domains × 2,030–2,369 raters/domain. Produced by `pipeline/run_unified_calibration.py:step_d`. |
| `sdt_fits.spike.csv` | – | same schema, spike-domain only | Phase-3 spike fits. |
| `sdt_fits.legacy_oldcorpus.csv` | 219 | same schema | Historical sibling-prepared build (15/29-rater era); **recorded, never rebuilt**. Pin status: sha256 in MANIFEST but no test enforces (§4.G8). |
| `cross_domain_rater_matrix.csv` | 29 | confirmed_canonical_name, sigma_{task}, theta_{task}, l_{task} × 6 IIIC | 29 raters who pass cross-domain criteria (Q2-locked candidate pool) |
| `cross_domain_rater_matrix.q2locked.csv` | 29 | same | Audit anchor; same 29 with Q2-locked names |
| `MANIFEST.json` | – | producer + sha256 + D3 anchor + self-check | Provenance witness regenerated by `scripts/build_engine_inputs.py` |

`sdt_fits.csv` is consumed by `engine/engine_paths.SDT_FITS` (Paper-1 Mode-A
SMC + MCMC). It is `pipeline/run_unified_calibration.py:step_d`'s recompute on
the unified `data/labels/` corpus, NOT the legacy 219-row sibling artifact.

### 3.4 Deployment prior (`data/deployment_prior/`)

Per `data/deployment_prior/summary.json` (audited).

| File | Schema | Role |
|---|---|---|
| `Sigma.csv` | 14×14 indexed by `{t_*, l_*} × {spike, seizure, lpd, gpd, lrda, grda, other}` | The 14×14 hierarchical prior (deployment-side; built by `freeze_deployment_prior.py:65-74` from `fits_hier_block`); 3 free params: `r_iiic=0.334`, `r_cross=0.447`, `ρ=0` |
| `case_bank.csv` | 230,599 rows | task × seg_id × s_mean × s_sd × n_raters × pos_rate (the deployment-side per-segment signal bank) — task-segment count: spike 17,346 / seizure 40,984 / lpd 30,476 / gpd 30,476 / lrda 39,747 / grda 40,701 / other 30,869 |
| `ell_thresholds.csv` | 7 rows | task × ell_star × youden_j × auroc_expert_vs_nonexpert × n_experts × n_non_experts. **Recorded but NOT consumed** (`engine_paths` reads ℓ\* from `cert_config.yaml::ell_star_unified_v13`) |
| `deployment_replay.csv` | 480 rows | tier × task × decision × n_per_task × ell_hat × ell_true × total_trials × candidate_id (the 21-rater replay sessions) |
| `summary.json` | – | sigma + r + tasks + thresholds + data_labels_provenance.sha256 (the D3 anchor) |

**Critical to know (§4.G4):** the 7 `ell_star` values that actually drive
PASS/FAIL/REFER live in `calibration/cert_config.yaml::ell_star_unified_v13`
(hand-maintained YAML), **not** in `deployment_prior/ell_thresholds.csv`
(which is recorded for provenance but not consumed). No test pins the
authoritative ones numerically.

Authoritative ℓ\* values per task (at audit time):

| Task | ell_star | Youden J | AUROC | n_expert | n_non_expert |
|---|---:|---:|---:|---:|---:|
| spike | 0.6236 | 0.6133 | 0.847 | 28 | 1048 |
| seizure | 0.9762 | 0.466 | 0.809 | (per cert_config) | |
| lpd | 0.4164 | 0.894 | 0.960 | | |
| gpd | 1.5254 | 0.886 | 0.930 | | |
| lrda | 0.6157 | 0.787 | 0.934 | | |
| grda | 1.4482 | 0.828 | 0.949 | | |
| other | 0.0499 | 0.806 | 0.924 | | |

*Note: AUROC here is expert-vs-non-expert discrimination on ℓ̂, NOT classifier
AUROC on labels (§4.G_AUROC).*

### 3.5 Real-rater replay artefacts (`data/replay/`)

| File | Rows | Schema | Role |
|---|---:|---|---|
| `rater_replay_bank.csv.gz` | – | per (rater, seg, task) historical reads | The full replay bank (compressed) |
| `rater_replay_bank.indexed.pkl` | – | rater×task lookup index | Hot loader for `deployment/simulate_test.py` y_source replay |
| `rater_replay_summary.csv` | 16,705 | rater_id, task, n_segs, n_pos, canonical_name, expertise_level | Per (rater, task) replay summary |

The 21-rater per-candidate cohort is the subset with `n_segs ≥ 10` on **all
seven** tasks. The 14,823 per-task cells flag the looser at-least-10-per-task
inclusion criterion across more raters (≈2,090–2,282/task).

### 3.6 Curated banks (`data/curated_banks/`)

Public-shippable derived per-segment signal banks (s_probit, frac_yes, integer case_keys):

| File | Domain | Role |
|---|---|---|
| `combined_spike.json` | spike | The K=7 spike bank, derived |
| `sparcnet_sz.json` ... `sparcnet_iic.json` | 6 IIIC tasks | Per-IIIC-task derived signal banks |

These are the **public reproducibility surface**: aggregate per-segment
s_probit + integer case_keys + frac_yes, no raw EEG (D9 enforced). Total ~168 KB.
The anonymizer spec is at `data/SENSITIVE.md:170-194` — invoked at post-acceptance
public-fork time.

### 3.7 Pipeline provenance map

```
                                           ┌─ EXTERNAL (D9; never vendored) ─┐
                                           │ SN1_combined_v2.h5 (PHI EEG)    │
                                           │ pd-rda-profiler / sparcnet50K   │
                                           │ Centaur2025 / Kong2025          │
                                           └──────────┬──────────────────────┘
                                                      │  build_unified_labels.py
                                                      │  (NOT re-runnable in-repo)
                                                      ▼
                                  ┌─────────────────────────────────────────┐
                                  │  data/labels/   ★sha256 anchor (D3)     │
                                  │   labels.csv      2,115,793 rows        │
                                  │   raters.csv      5,306     rows        │
                                  │   segments.csv    95,327    rows        │
                                  │   datasets.csv    5         rows        │
                                  │   fits/           (PI per-task Rasch)   │
                                  │   fits_hier_block/(PI 14×14 block fit)  │
                                  └────────┬─────────────────────────┬──────┘
                                           │                         │
                            ┌──────────────┴──────────────┐    ┌─────┴───────────────┐
                            │ pipeline/run_unified_       │    │ deployment/freeze_  │
                            │   calibration.py (~hours)   │    │   deployment_prior  │
                            │                             │    │   .py               │
                            │ STEP a build_calibration_   │    │  reads PI fits_     │
                            │   inputs (7 tasks)          │    │   hier_block        │
                            │ STEP b fit_main_effects     │    │  computes r_ell     │
                            │   (Rasch c_j logit)         │    │  builds 14×14 Σ     │
                            │ STEP c fit_sdt_per_domain   │    │  Youden ℓ\* by     │
                            │   (×1/1.7; probit-lapse)    │    │   group-label       │
                            │ STEP d sdt_fits.csv         │    │   (expert vs        │
                            │ STEP e Q2-locked 29-pool    │    │    non-expert)      │
                            │ STEP f CV-top-14 Youden ℓ\* │    │                     │
                            │   IIIC (×6 domains)         │    │                     │
                            │ STEP g spike σ\* (70/30     │    │                     │
                            │   TRAIN, raters.csv expert) │    │                     │
                            │ STEP h D7 Centaur n=4 panel │    │                     │
                            │ STEP i emit cert_config v13 │    │                     │
                            └──────────┬──────────────────┘    └─────────┬───────────┘
                                       │                                 │
                                       ▼                                 ▼
                        ┌─────────────────────────┐         ┌────────────────────────┐
                        │ data/engine_inputs/     │         │ data/deployment_prior/ │
                        │  sdt_fits.csv  14,214 r │         │  Sigma.csv  14×14       │
                        │  sdt_fits.spike.csv     │         │  case_bank.csv 230,599 │
                        │  cross_domain_rater_    │         │  ell_thresholds.csv     │
                        │    matrix.csv  29       │         │   (LEGACY, not consumed)│
                        │  …q2locked.csv  audit   │         │  summary.json  ★sha256  │
                        │  legacy_oldcorpus.csv   │         │   data_labels_provenance│
                        │   (219 r; never rebuilt)│         │                        │
                        │  MANIFEST.json ★sha256  │         │                        │
                        └──────────┬──────────────┘         └───────────┬────────────┘
                                   │                                    │
                                   ▼                                    ▼
                        ┌─────────────────────────┐                                  
                        │ calibration/             │                                 
                        │  cert_config.yaml v13    │ ←── consumed by ──→  Sigma_l_fitted.npy
                        │  (per-task ell_star;     │  ENGINE side ONLY    (repo root, 6×6   
                        │   hand-maintained body)  │                       FROZEN 15-rater  
                        │  ell_star_unified_v13    │                       era; symlinks   
                        └──────────┬───────────────┘                       to archive/ +
                                   │                                       engine/)
            ┌──────────────────────┴────────┬───────────────────────────────────┐
            ▼                               ▼                                   ▼
    ┌──────────────────┐         ┌──────────────────────┐        ┌────────────────────┐
    │ engine/ (Mode-A  │         │ deployment/          │        │ deployment/replay/ │
    │   SMC + MCMC,    │         │   simulate_test.py   │        │   strict-A replay  │
    │   Paper-1)       │         │   Laplace + EKF      │        │   (D6 v1.0 result) │
    │ reads Σ via      │         │ reads Σ from         │        │   y_source =       │
    │ load_fitted_     │         │   data/deployment_   │        │   labels.csv       │
    │ Sigma (6×6) +    │         │   prior/Sigma.csv    │        │   reports 21-rater │
    │ matched-mean CS  │         │   (14×14)            │        │   AUROC headline   │
    │ approx for K=7   │         │ reads ℓ\* from v13   │        │                    │
    │                  │         │   cert_config        │        │                    │
    └──────────────────┘         └──────────────────────┘        └────────────────────┘

★ sha256 anchor (D3): engine_inputs/MANIFEST.json::data_labels_provenance
                    ≡ deployment_prior/summary.json::data_labels_provenance
                    ≡ live data/labels/{labels,raters}.csv sha256
```

**Note that the engine and deployment Σ are different fits**: engine consumes
the 6×6 Sigma_l matched-mean-CS-approximated to K=7; deployment consumes a
14×14 PI block fit. Documented in code, not yet gated by tests. See §4.G5.

---

## 4. Critical findings — integrity and data-leakage gaps

The four gaps the test suite does not currently catch. Each is a TIER-1
action in §6.

### 4.G1 — Replay/calibration cohort overlap (HIGHEST severity)

**Convergent across:** Pipeline G1, Biostat G1 ("CV-top-14 panel selection"), EEG Gap 6 ("circular CV pool").

**The finding.** The 21-rater per-candidate replay cohort (`data/replay/rater_replay_summary.csv`; `docs/PHASE7_REPLAY_HEADLINE.md:91-111`) is the headline cohort for the +43 % to +1154 % replay-vs-Bernoulli claim. By direct join against the calibration intermediates:

- **16 of 21** replay raters are in the 29-rater Q2-locked candidate pool from which the CV-top-14 expert panels are drawn (verified by canonical-name join against `data/engine_inputs/cross_domain_rater_matrix.q2locked.csv`).
- **8–9 of the 14** raters in each domain's CV-Youden expert panel are also in the 21-rater replay cohort. Per-domain panel overlap (read from `pipeline/reference_calibration/youden_ell_star.json::panels`):

| Domain | overlap (panel ∩ replay) / panel size |
|---|---|
| sz | 9 / 14 |
| lpd | 9 / 14 |
| gpd | 8 / 14 |
| lrda | 9 / 14 |
| grda | 9 / 14 |
| iic | 8 / 14 |

- The 70/30 split inside `pipeline/run_unified_calibration.py:step_g` (`EXPERT_TRAIN_FRAC = 0.70`) is **over the σ-pool**, not over rater identity — raters in TRAIN can appear in the replay cohort. The 30 % held-out half is a σ slice for σ\* selection, not a rater-identity hold-out for downstream verification.

**Why it matters for Nature Medicine.** The headline replay-vs-Bernoulli effect-size estimate uses the same individuals' σ̂ / θ̂ to set ℓ\* AND to measure replay performance. Nature Medicine reviewer 2 will ask "show me ℓ\* held out across raters and the replay finding on the disjoint set"; the current pipeline cannot produce a clean answer. The magnitude of confounding is unquantified.

**What would catch it.** (a) A test asserting `len(set(panel_names) & set(replay_cohort_names)) == 0` for each domain (zero by construction would force a re-design); or — more pragmatically — (b) a leakage-aware sensitivity rerun: with these 8–9 raters per domain removed from the calibration AND ℓ\* recomputed, what is the new replay-vs-Bernoulli contrast? If materially smaller, the headline collapses and needs reframing.

**TIER-1 action T1.2 (1–2 days):** run (b). Add `pipeline/replay/leakage_audited_replay.py` that re-runs `step_f` (CV-top-14 Youden) with the 21-rater replay cohort masked out of the candidate pool, regenerates per-domain ℓ\*, then re-evaluates the replay headline. Report the contrast next to the current headline. **This is the single most reviewer-defining action in the audit.**

**Documentation gap.** `docs/OPEN_DECISIONS.md:102-134` flags "ℓ\* held-out independent-panel re-fit has not been run in this repo" as 🟡 PARTIAL — but does not explicitly call out the calibration ∩ replay overlap. Should be promoted to a numbered open item.

### 4.G2 — CV-top-14 two-stage selection without selection-penalty (MEDIUM)

**Convergent across:** Pipeline G2, Biostat G3, Psychom #2.

`pipeline/reference_calibration/run_youden_calibration.py:139-150` builds each domain's expert panel by ranking candidates on their cross-other-5-domains mean ℓ, then computes Youden J on the held-out domain. Non-circular in the sense the held-out domain's own ℓ is not used for selection — but it is still a **two-stage estimator with no penalty**. The 29-candidate pool itself is non-random (most-prolific cross-domain candidates), and the bootstrap CI (`_run_calibration` lines 117-135) resamples *within* the post-selection top-14 and the non-expert pool — it does NOT resample the top-14 selection itself or the 29-candidate pool selection.

**Numerical impact** (`pipeline/reference_calibration/youden_ell_star.json::results`): Youden J 0.78–0.87 across IIIC; CI half-widths 0.06–0.12. These are post-selection CIs; honest CIs (with selection-resampling) would be wider, plausibly substantially.

**TIER-2 action T2.3 (med effort):** Modify `run_youden_calibration.py` to add a "wild-CV" bootstrap that re-samples the 29-candidate selection AND the top-14 within. Report two CI columns: `ci_low_within_panel` and `ci_low_with_selection`. The first remains the v1.0 ship value (matching the byte-md5 invariant); the second is the supplement-table honest CI.

### 4.G3 — D3 anchor proves input lineage but not fit-output lineage (MEDIUM-HIGH)

**Unique finding:** Pipeline G3.

The D3 sha256 chain (`tests/test_phase5_engine_inputs.py:91-116`) proves `engine_inputs/MANIFEST.json::data_labels_provenance.sha256` ≡ `deployment_prior/summary.json::data_labels_provenance.sha256` ≡ live `labels.csv` + `raters.csv` sha256. This proves they share the same *input* labels file. It does NOT prove:

1. `sdt_fits.csv` (the 14,214 rows) was computed from the same labels.csv version that `freeze_deployment_prior.py` consumed.
2. The per-task Rasch outputs at `data/labels/fits/{task}/cases.csv` (which `freeze_deployment_prior.py:65-74` reads) are bit-identical to the in-repo Rasch outputs at `pipeline/_calib_work/data/prepared/{task}_fit/{c,l}.csv` produced by `pipeline/run_unified_calibration.py:step_b` — these are separate fit invocations at different wall-clock times.
3. `Sigma_l_fitted.npy` (repo root, 6×6) was fit on this corpus at all — it is a frozen 15-rater-era artifact (`docs/INVARIANT_AUDIT.md` §9; `data/engine_inputs/README.md:56-60`).

**Failure mode.** A contributor regenerates `data/labels/fits/` (PI Rasch) without re-running `pipeline/run_unified_calibration.py:step_b` → engine_inputs and deployment_prior outputs silently diverge while D3 still passes (both still sha256-match labels.csv).

**TIER-2 action T2.11 (low):** Extend `scripts/build_engine_inputs.py::_d3_anchor` to also record sha256 of `data/labels/fits/{task}/cases.csv` (PI side) and `pipeline/_calib_work/data/prepared/{task}_fit/{c,l}.csv` (in-repo side); chain in `tests/test_phase5_engine_inputs.py`.

### 4.G4 — cert_config v13 ℓ\* values not pinned numerically in any test (HIGH integrity gap)

**Unique finding:** Pipeline G4.

The seven shipped `ell_star_unified_v13` values in `calibration/cert_config.yaml:58-112` are the **clinical decision boundary for every PASS/FAIL verdict** the deployment emits. None of the 282 tests pin them numerically. Tests check shape (7 tasks present; `tests/test_phase3_calibration.py:205-213`), `config_version == 13`, "wired correctly" via `load_deployment` returning whatever the YAML says, and a "v13 ≠ PI legacy" inequality — but no test asserts the *actual numerical values*.

**Failure mode.** A contributor edits `calibration/cert_config.yaml` (hand-maintained per `data/engine_inputs/README.md:62-64`) to nudge any of the 7 ℓ\* values; all 282 tests still pass; the deployment now certifies under different thresholds with **no detection at all**.

**TIER-1 action T1.1 (30 minutes):** Add to `tests/test_phase3_calibration.py`:

```python
def test_v13_ell_star_pinned():
    blk = _load_cfg()["ell_star_unified_v13"]["tasks"]
    expected = {
        "sparcnet_sz":    0.45504292981835925,
        "sparcnet_lpd":   0.5337430687087749,
        "sparcnet_gpd":   0.3297002787536504,
        "sparcnet_lrda":  0.4792595265871899,
        "sparcnet_grda":  0.48645101728803,
        "sparcnet_iic":   0.4418131962513707,
        "combined_spike": 0.2542662861051924,
    }
    for task, exp in expected.items():
        got = blk[task]["ell_star"]
        assert abs(got - exp) < 1e-12, (
            f"{task} ell* drifted from v13 calibration: {got} vs {exp} — "
            f"regenerate cert_config v13 AND update this test if intentional"
        )
```

The expected dict are the values currently in v13 (read off `cert_config.yaml`); pin them to 1e-12. The "regenerate AND update" comment is the social contract — the test forces a deliberate, paired edit instead of a silent drift.

### 4.G5 — Engine-side Mode-A uses K=6 matched-mean CS approximation to K=7 (HIGH for Paper-1)

**Unique finding:** Pipeline G5.

The Paper-1 Mode-A SMC + MCMC engine reads the hierarchical prior from `engine_paths.SIGMA_L` (symlinked `engine/Sigma_l_fitted.npy` → repo root → 6×6 frozen 15-rater era; `data/engine_inputs/README.md:56-60`). For K=7 it uses `scripts/run_phase4_simstudy.py:130-136`'s "matched-mean compound-symmetry approximation" — a single off-diagonal r computed by averaging the 6×6 Corr_l off-diagonals and applying it uniformly in K×K.

The **deployment** (Laplace + EKF) reads Σ from `data/deployment_prior/Sigma.csv` — built by `deployment/freeze_deployment_prior.py:65-74` from a fresh per-block fit on `data/labels/fits_hier_block`. This is a **different** Σ (3 free parameters: `r_iiic=0.334`, `r_cross=0.447`, `ρ=0`).

**The two engines therefore use different priors.** The replay headline (`docs/PHASE7_REPLAY_HEADLINE.md:99-111`) compares replay vs Bernoulli under Mode-A on the *matched-mean K=7 approximation*; the deployment headline (same document, top half) uses the *block-fitted K=7 Σ*. A reviewer asking "are these the same prior?" gets "no."

**TIER-2 action T2.7 (med-medium effort):** Re-fit Sigma_l at K=7 from `data/labels/fits_hier_block` (the same block fit `freeze_deployment_prior.py` uses); ship as `Sigma_l_fitted_k7.npy`; update `engine/engine_paths.py:14-17` to consume it; regenerate Paper-1 figures; verify Phase-2 byte-equivalence tests still hold at the new prior. Medium-risk because Phase-2 byte-equivalence assumes the K=6 prior — re-run the Phase-2 validation suite at the new prior and document any deltas in CHANGELOG.

---

## 5. Critical findings — methodological and reporting gaps

Nature Medicine non-negotiables identified across the 5 audits. Each TIER-1 or
TIER-2 in §6.

### 5.B1 — Headline D6 finding has no formal hypothesis test (HIGH)

**Convergent across:** Biostat G2/R1, TRIPOD Reviewer 3.

`docs/PHASE7_REPLAY_HEADLINE.md:209-212` explicitly defers: "Cross-cohort statistical test (replay vs bernoulli paired-t / rank-test across the 14,823 cells)... a formal hypothesis-test layer is a follow-on analysis if a reviewer asks for it." Reviewer 2 *will* ask.

**TIER-1 action T1.5 (1–2 days):** Add `pipeline/replay/statistical_tests.py`:
- Mixed-effects model: `verdict ~ arm + (1|rater) + (1|task) + (arm|rater)`; report log-OR + 95 % CI.
- Paired McNemar per task on PASS verdict.
- Wilcoxon signed-rank on per-cell AUROC differences per task; Hodges-Lehmann estimator with 95 % CI.
- Multiple-comparison adjustment table (Šidák, BH-FDR).

Inputs: `results/replay/replay_per_task.csv` (already produced).

### 5.B2 — No frequentist calibration of probabilistic outputs (HIGH; TRIPOD+AI mandatory)

**Convergent across:** Biostat G7/R2, TRIPOD #15 + Rec 2, Psychom Rec 7.

No Brier score, calibration-in-the-large, expected calibration error, calibration slope, reliability diagrams anywhere in the codebase (`grep -r "brier|hosmer|calibration_curve|expected calibration"` finds only Bonferroni for SBC bins). The deployment engine emits `expected_p(mu, Sigma_post, k, s)` at `deployment/simulate_test.py:178-194`, but the predicted response probability is **never validated against observed response frequencies** on the replay cohort.

This is TRIPOD+AI Item 15 (Calibration Assessment) — currently MISSING.

**TIER-1 action T1.4 (4–6 hours):** Add `pipeline/replay/calibration_metrics.py`:
- Per-task reliability diagram (10 deciles of predicted `expected_p` vs observed Y on the 14,823 replay cells).
- Brier score per task.
- Calibration-in-the-large (intercept of `logit(Φ(η_pred)) ~ Y`).
- Calibration slope.
- Integrated Calibration Index (Austin–Steyerberg).
- Hosmer–Lemeshow optional.

Produce one figure per task (7) and one summary table.

### 5.B3 — No multiple-testing correction across 7 tasks × 21 raters (HIGH)

**Unique finding:** Biostat G8/R3.

With 7 per-task PASS/FAIL gates at `pass_p=0.95 / fail_p=0.05`, per-candidate FWER is `1 − 0.95⁷ ≈ 0.30`. No FWER/FDR control is documented in code or text. The `mode_b_legacy.stop_thresh_sidak: 0.9915` at `cert_config.yaml:45` is for the *retired* Mode-B path; the live deployment doesn't use it.

**TIER-1 action T1.6 (0.5 day):** Either (a) introduce a `pass_p_family = 0.9927` (Šidák-7) for any single-candidate roll-up reporting; OR (b) more defensibly given the working "per-task certificates, no roll-up" policy (OPEN_DECISIONS #1), document explicitly in `docs/PHASE7_REPLAY_HEADLINE.md` and the manuscript Methods that **the per-task policy is the FWER-resolving choice** — each task is its own certificate; downstream consumers (credentialing boards, study sites) can apply their own roll-up rule with their own α. Add a new `docs/MULTIPLE_TESTING.md` formalising this.

### 5.B4 — Subgroup / fairness analyses absent (HIGH; TRIPOD+AI mandatory)

**Convergent across:** TRIPOD #19/Rec 3, EEG Rec 5/6.

No stratified analysis by expertise tier, by patient demographic, by site, or by years-EEG-experience. The deployment replay headline reports aggregated PASS rates only. **For Nature Medicine in 2024–2026, this is a hard ask under the model-card / TRIPOD+AI / Liu 2024 requirement set.**

What can be done on existing data (no new acquisition):

- **Per-task PASS rate by rater expertise tier** (expert / experienced / borderline / novice / other / unknown): re-aggregate `results/replay/replay_per_task.csv` joined on `raters.csv:expertise_level`. Eli's raters.csv has 174 expert / 163 experienced / 47 borderline / 1,244 novice / 740 unknown / 465 other / 2,473 NaN. The expert-tier ordering should be the expected ordering of PASS rates.
- **Per-task PASS rate by self-reported `years_eeg`** for the 289 raters with that field populated.
- **Spike PASS rate by patient age band** (pediatric ≤18, adult 18–64, elderly ≥65): use Bonobo + SN1:sn1 segments with age (n=15,670).
- **Spike PASS rate by patient sex** (Bonobo only, n=3,966): 1,985 F vs 1,981 M.
- **Replay-vs-Bernoulli gap by rater affiliation cluster** (MGH-cluster vs other) for the 1,671 raters with affiliation.

**TIER-1 action T1.7 (2–3 hours):** Run these five on the existing `results/replay/replay_per_task.csv`. Produce Table-3 + Figure-4 of the manuscript.

What cannot be done without new acquisition (this is §5.E2):

- Race / ethnicity fairness — no race field anywhere.
- Comorbidity / etiology fairness — no etiology field.
- ICU vs EMU vs outpatient stratification — no setting field.
- Site-level fairness — no `site_id` in `segments.csv`.

These become EXPLICIT limitation statements (TRIPOD+AI "Data not collected") — see T1.10.

### 5.B5 — Decision-curve analysis absent (Nature Medicine clinical-AI standard)

**Convergent across:** TRIPOD #20/Rec 4, Psychom Rec 6, Psychom #7.

Vickers & Elkin (Med Decis Making 2006) / Vickers et al. (BMJ 2019) net benefit
is now expected for any clinical-AI / decision-support paper. CORTEX certifies
at Youden-J (implicit symmetric cost) — but FP and FN have very different
clinical consequences. False-positive "seizure" → AED (low risk); false-negative
seizure → missed status (high risk); false-positive "spike" → epilepsy diagnosis
+ driving restriction (high stakes); false-negative "spike" → low stakes.

**TIER-1 action T1.8 (6 hours):** Two complementary analyses:

1. **Threshold-level DCA:** for each task, compute net benefit at `pass_p ∈ {0.80, 0.85, 0.90, 0.95, 0.99}` × `fail_p ∈ {0.01, 0.05, 0.10, 0.15, 0.20}` over the 21-rater replay arm; report net benefit under asymmetric cost (mis-certify novice as expert ≫ mis-certify expert as novice). Use the existing deployment harness with sandboxed config overrides — `pass_p=0.95 / fail_p=0.05` remains the v1.0 ship value.

2. **Youden-J vs cost-weighted threshold:** report `ℓ*_α = arg max α·Se + (1-α)·(1-Sp)` for `α ∈ {0.3, 0.5, 0.7}` per task next to the Youden-optimal `ℓ*_J=0.5`. Quantifies the asymmetric-cost shift.

### 5.B6 — Lapse-sensitivity sweep never run on unified corpus (LOW cost, HIGH value)

**Convergent across:** Psychom Rec 1, Biostat R8.

`scripts/run_lapse_sensitivity.py` exists with the Wichmann–Hill λ_true ∈ {0.01, 0.025, 0.05, 0.10} grid and a coded acceptance criterion (|AUROC excess bias| ≤ 0.02). `results/phase2_validation/` is empty — **the sweep has not been run on the unified corpus.** The "λ=0.025 is robust" claim has no current empirical support in this repo.

Wichmann & Hill (Perception & Psychophysics 2001) Fig 8 showed lapse mis-estimation of 0.01–0.05 induces 4–10 % slope bias in the SDT fit; every psychophysics methods paper since includes this sweep.

**TIER-1 action T1.3 (minutes):** Run `python scripts/run_lapse_sensitivity.py` on the unified corpus. Quote `worst_excess_bias` in Methods. This is the single most likely reviewer-1-round ask.

### 5.B7 — Pre-registration absent and explicitly acknowledged as absent

**Unique finding:** TRIPOD #4 / Rec 1.

`docs/PHASE7_CLOSEOUT.md:243-271` openly admits: "team did NOT explicitly pre-register numeric tolerances before the sub-7.3 run". Phase-7 sub-5 carved out four "implicit bounds" retrospectively. **Nature Medicine red flag.**

**TIER-1 action T1.9 (1 day):** Retroactively register the v1.0 acceptance protocol on OSF using TRIPOD-AI + SPIRIT-AI templates, citing 2026-05-18 as the Phase-7 design lock and 2026-05-19 as the headline run. **Be explicit** that this is retrospective. Then commit prospectively to the CORTEX live-deployment cohort (which is what the click-to-run app is FOR) as a ClinicalTrials.gov registration BEFORE candidate enrollment — that becomes the formal external prospective arm.

### 5.B8 — NUTS R̂ > 1.05 in 4/6 IIIC tasks (MEDIUM-HIGH for any Bayesian reviewer)

**Unique finding:** Biostat G5/R5.

NUTS R̂ for s_j: sz 1.13, gpd 1.10, grda 1.05, lrda 1.06 (all > 1.05) — only iic and lpd below. From `calibration/joint/{sz,gpd,grda,lrda}_nuts_summary.json`. Per Vehtari et al. 2021, R̂ > 1.01 is grounds for not trusting posterior; R̂ > 1.05 is materially non-converged. ell and t are fine. Subsample n=80,000, 400 warmup, 400 samples, 2 chains is **light**. The downstream SVI vs NUTS variance-calibration κ may be biased by NUTS not having converged on these tasks.

**TIER-2 action T2.4 (compute-heavy):** Re-run `pipeline/joint_calibration/fit_joint_iiic.py --method nuts` with `--warmup 1500 --samples 1500 --chains 4` per task on the 4 R̂ > 1.05 tasks (sz, gpd, grda, lrda); larger subsample or full corpus. Re-derive variance_calibration κ from the converged NUTS. Document in `tests/test_phase35_variance_calibration.py` an R̂ target (1.01 / 1.05 documented; current state of 4/6 above 1.05 disclosed and tied to compute-budget decision; new state asserts below 1.05).

### 5.B9 — AUROC interpretation risk

**Unique finding:** Biostat G4.

The "AUROC 0.81–0.96 range" in `data/deployment_prior/summary.json:53-87` is the **expert-vs-non-expert AUROC** (`deployment/freeze_deployment_prior.py:106-114`, Mann-Whitney over `is_expert` vs `r["ell_mean"]`) — i.e. how well the fitted ℓ̂ separates experts from non-experts. It is **not** classifier AUROC against ground-truth labels. The two are different quantities (and tell different stories); the former is the correct quantity for Youden threshold-setting on ℓ\*; the latter is what most clinical-AI readers expect when they see "AUROC."

**TIER-1 action (manuscript-level):** In the methods + every figure caption + every results table, label this quantity **unambiguously**: "expert-vs-non-expert AUROC on fitted ℓ̂" (which is the threshold-setting AUROC). Provide the *classifier* AUROC (predicted P(Y=1) vs Y on the replay cohort) as a separate quantity in the calibration analysis (T1.4 / §5.B2).

### 5.B10 — Spike task is a fold (combined J=0.366 vs binary-only J≈0.63)

**Convergent across:** Psychom #8, Biostat I5, EEG Gap 7.

`calibration/CALIBRATION_PROVENANCE.md:18-22, 234-245`: `combined_spike` folds the binary SN1 spike-detection task with the Centaur-IED 6-class spike-subtype panel into a single binary "IED vs benign". Compressed task earned Youden J = 0.366 vs J ≈ 0.63 for binary-only. The Phase-3.5 un-fold (`cert_config.yaml:303-310`) honestly documents this. **For Nature Medicine, this is a non-trivial task-definition concession** — a candidate excellent at binary spike but poor at IED-vs-vertex-wave discrimination would be miscertified.

**TIER-2 action T2.6 (low to med):** Either (a) report separate spike-binary certificate (J=0.63) and spike-IED-vs-benign certificate, OR (b) keep `combined_spike` as the single task but explain its broader clinical interpretation in the manuscript. Renaming `combined_spike` → `clean_spike` in cert_config v14 if (a) is chosen. EEG agent's Rec 8 + Psychom #8 lean toward (a).

### 5.B11 — Bias-warning channel `|t_k| > tol` absent

**Convergent across:** OPEN_DECISIONS #4 (already noted), Biostat I7/R7, Psychom Rec 10.

A candidate with `t_k ≫ 0` is calling discharges too liberally (high FP); `t_k ≪ 0` too conservatively (high FN). Current deployment only certifies on `ℓ_k > ℓ*_k`; a strongly-biased rater may PASS on ℓ while being clinically miscalibrated. Reference repo's `T_TOL = 0.20` ≈ 9.3 pp deviation.

**TIER-2 action T2.5 (3 hours):** Implement per-task posterior median of `t_k` in `deployment/simulate_test.py` output schema; surface as a warning column in `replay_per_task.csv`. Documented tolerance `Config.t_tol = 0.20` (Phase-6 invariant audit re-affirmed value).

---

## 6. Critical findings — corpus / data gaps

These are bounding the manuscript's claim space. They cannot be closed by code; they require either documentation honesty or new acquisition.

### 6.E1 — All five datasets trace to MGH/BIDMC/Harvard/Yale cluster (HIGH)

**Unique finding:** EEG Gap 1.

Every label stream is from a tightly-coupled MGH/BIDMC/Harvard-affiliated network: SN1 + Bonobo + SpikeEd are MGH-PI-curated; SPARCNET 50K is the Hong/Jing SPARCNET consortium (MGH-BIDMC-Yale core); pd_rda_profiler is internal; Centaur 2025 is the same lab's spinoff/contest platform; Kong 2025 is the same author group. IRBs (`data/SENSITIVE.md:14-18`): BIDMC 2016P000058 and MGH 2013P001024 — both Harvard.

**Implication.** Every Σ_l, every ℓ\*, every σ\* in the prior is a Boston-network estimate. Nature Medicine reviewer 2: "show that ℓ\* on a clean external site replicates the same PASS/FAIL boundary."

**TIER-2 / TIER-3 actions T2.18, T3.1, T3.2:** §7 below ranks external cohort options (Option B expand n=4→n≥10 with non-PI affiliations; Option C 1 non-PI US academic site with fresh signals; Option D international; Option G ABCN pool).

### 6.E2 — Patient demographics nearly absent (HIGH; bounds fairness)

**Unique finding:** EEG Gap 2.

Of 95,327 segments:
- Sex: 3,966 (4.2 %) — Bonobo only.
- Age: 15,670 (16.4 %) — Bonobo + SN1:sn1 only.
- Race / ethnicity: **0**.
- Comorbidity / etiology: **0**.
- Clinical context (ICU / EMU / outpatient): **0**.
- Encephalopathy grade: **0**.
- MRI / structural lesion: **0**.

**Implication.** Forecloses race-stratified ℓ\*, age-stratified AUROC, sex-stratified Σ_l, etiology-stratified PASS rate. Nature Medicine has demanded fairness analyses for every clinical-AI submission since 2024.

**TIER-2 action T2.2 (HIGH effort; highest leverage):** Backfill `segments.csv` from BDSP source HDF5/CSV records via `pat_key`/`mrn`. The MGH-BIDMC corpora DO carry race in their underlying clinical records under the original IRBs. Add columns: `pat_age` (everywhere), `sex`, `race`, `ethnicity`, `clinical_setting`, `primary_dx`, `encephalopathy_grade`. Document missingness honestly. **This unlocks T1.7 (subgroup) for race/etiology.** IRB amendment to BIDMC/MGH likely required; feasible within current PHI-governance posture (D9 — the EEG itself stays external; demographic fields can be added to the de-identified `segments.csv` if approved).

### 6.E3 — Pediatric-heavy spike substrate vs adult-ICU deployment framing

**Unique finding:** EEG Gap 3.

Where age is recorded (Bonobo + SN1:sn1; n=15,670):
- Median age: 16–21 y.
- **51 % pediatric (< 18 y).** 30 % adult < 65 y; 19 % elderly ≥ 65 y.

The spike-skill paper substrate (`docs/PHASE7_REPLAY_HEADLINE.md` headline; spike bank n=17,346) is dominated by SN1:sn1 (12,801 segs from 1,063 patients) and Bonobo (3,966 segs from 1,450 patients). If the deployment is framed for adult-ICU IIIC monitoring, this is a **training/use mismatch** — pediatric spike morphology differs (more focal, more state-dependent, more benign-variant overlap).

**TIER-1 action T1.7 (subgroup analysis):** Spike PASS rate by patient age band on the existing data. **TIER-2 action T2.2 (demographic backfill):** Then stratify on race/etiology too. **Manuscript-level:** either reframe spike task as "any-age spike detection" OR explicitly cite the pediatric-heavy substrate as a limitation requiring adult-cohort external validation.

### 6.E4 — No site_id in segments.csv

**Unique finding:** EEG Gap 4.

SPARCNET 2023 reported 11 sites; `segments.csv` carries no `site_id` column. `affiliation` lives only in `raters.csv` (1,671 raters with affiliation; concentrated to MGH / Yale / Johns Hopkins / Hartford). Without per-segment site identifiers, no site-disjoint cross-validation, no site-effect fixed-effect adjustment.

**TIER-2 action T2.10 / T2.16:** Backfill `site_id` from BDSP S3 path patterns (the morgoth1 BIDS path contains contributor codes); run site-disjoint ℓ\* cross-validation; report per-task ℓ\* drift across sites.

### 6.E5 — Cross-dataset label-comparability never formally validated

**Unique finding:** EEG Gap 10.

Five datasets used different annotation protocols: SPARCNET 6-class IIIC consensus; pd_rda_profiler 8-feature curated; Centaur 2025 7-class with BIPD/BIRDS; Kong 2025 6-class DiagnosUs; SN1 binary spike consensus. The corpus assumes `pattern_class='lpd'` from a SPARCNET expert and `pattern_class='lpd'` from a Centaur novice and `pattern_class='lpd'` from a Kong crowd rater are exchangeable. No anchor-segments, no bridge-rater table, no per-dataset label-noise estimate.

**TIER-2 action T2.12 (med effort):** Anchor-segment study using the 2 cross-domain experts who already span multiple datasets (Aaron Struck, M. Brandon Westover). Identify ~100 anchor segments scored by ≥3 of the 4 protocols (SPARCNET / Centaur / Kong / pd_rda); compute pairwise κ per IIIC class. If κ < 0.5 on any class, the cross-dataset union assumption needs explicit modeling.

### 6.E6 — Centaur IED panel (5,000 segments, 643 raters) not in K=7 banks

**Unique finding:** EEG Gap 9.

The 5,000 IED-panel segments (`centaur_2025_ied`, 167,503 labels from 643 raters) are present in `labels.csv` but **none feed into the K=7 banks** (verified: 0 of 5,000 IED seg_ids appear in the spike bank). The CORTEX spike task therefore does not test BETS / vertex-wave / POSTS / wickets variants that an ICU/EMU reader actually faces. Either undocumented gap or implicit Paper-2 carry.

**TIER-3 action T3.5:** Either (a) extend to K=8 spike-variant task (full recalibration — expensive); OR (b) document explicitly in `data/DATA_PROVENANCE.md` why the IED panel is excluded from v1.0. (a) is Paper-2 scope.

### 6.E7 — Spike-only / IIIC-only rater pool fragmentation

**Unique finding:** EEG Gap 8.

2,872 spike-only raters; 2,047 IIIC-only raters; **only 322 with both tasks.** SN1 has 2,447 NaN-expertise raters (the SpikeEd crowd, never resolved). Kong 2025: 907 self-reported novices, 130 experienced, only 10 self-reported experts. Per-rater density: median 54 segs/IIIC task, 69 spike. The 14,823-cell per-task cohort is dominated by low-bank raters — ≈50 % of decisions are REFER from bank exhaustion (`docs/PHASE7_REPLAY_HEADLINE.md:165-173`), not engine indecision.

**TIER-2 action T2 (reporting):** Promote the bank-density caveat to main-text figure. Stratify PASS/FAIL/REFER by per-rater bank size {10, 50, 100}. Already present in design doc; lift to main text.

---

## 7. External cohort options ranked

Eli's full-scope mandate makes this central. Per EEG agent, ranked by Nature
Medicine credibility-per-effort.

| Option | What | Feasibility | n | Methodological fit | Time | Cred | Recommendation |
|---|---|---|---|---|---|---|---|
| **A** | Hold out existing n=4 Centaur gold panel | trivial (done) | 4 × 5000 | poor — same lab, same substrate; D7 already 4/6 disjoint CIs | done | **low** | **Not sufficient alone** |
| **B** | Expand n=4 → n≥10 with non-PI experts | med | 10–15 × 5000 | strong if from UCSF/Mayo/Cleveland/Penn/Hopkins-EEG | 3–6 mo; honoraria $5–10K/expert | **med-high** | **MUST for v1.0** |
| **C** | 1 non-PI US site with fresh signals + 5–10 external raters | med-low | ~5K–10K labels | gold-standard; external signals AND external raters | 4–9 mo (new IRB + DUA) | **highest** | **SHOULD for v1.0 if timeline permits; otherwise Paper-2** |
| **D** | International (EU / Japan / Latin America) | low-med | 0.5K–2K cases × 5–15 raters | strong — answers "Western-trained generalises?" | 12–18 mo | **very high** | **Paper-2 (ILAE branding)** |
| **E** | BDSP SPARCNET 2.0 patient-disjoint subset | high (already accessible) | varies | limited if same Jing lab; useful for patient-disjoint | weeks | **medium** | **Cheap supplementary; include** |
| **F** | Nascimento RCT / SpikeEd | already internal | n/a | not external | – | n/a | **Not eligible as external arm** |
| **G** | AAN / ABCN cooperative pool | high (interest exists) | candidate panel | direct deployment-validation | ~12 mo | **high** | **Paper-2 deployment paper** |

**Ranked TIER-1/2 for v1.0:**
1. **B** (expand the n=4 panel via non-PI experts) — MUST.
2. **C** (1 non-PI US academic site fresh signals) — SHOULD if timeline; T2.18.
3. **E** (SPARCNET 2.0 supplement) — recommend include.

Defensible Nature Medicine framing if (B) is the only one in by submission:
position the n=4 Centaur as an *independent expert anchor*, the expanded
n≥10 (Option B) as the v1.0 external panel, the prospective CORTEX
live-deployment cohort as the formal external arm being run during paper
revision.

---

## 8. Convergence matrix — what each agent flagged

The matrix below shows where the 5 PhD audits agreed (a finding flagged by 2+ agents has the highest reviewer-attack credibility).

| Finding | Biostat | Psychom | EEG | TRIPOD | Pipeline |
|---|:---:|:---:|:---:|:---:|:---:|
| External validation absent | ✓ | ✓ | ✓ | ✓ | – |
| Pre-registration absent | – | – | – | ✓ | – |
| Calibration metrics missing | ✓ | ✓ | – | ✓ | – |
| Decision-curve analysis | – | ✓ | – | ✓ | – |
| Subgroup / fairness | – | – | ✓ | ✓ | – |
| Multiple-testing FWER | ✓ | – | – | – | – |
| Lapse-sensitivity sweep | ✓ | ✓ | – | – | – |
| Bias-warning channel | ✓ | ✓ | – | ✓ | – |
| Spike fold disambiguation | ✓ | ✓ | ✓ | – | – |
| D7 expand n=4 panel | ✓ | ✓ | ✓ | – | ✓ |
| Σ block-structure | ✓ | ✓ | – | – | ✓ (different angle) |
| CV-top-14 selection bias | ✓ | ✓ | ✓ | – | ✓ |
| **Replay/calibration overlap (G1)** | ✓ | – | – | – | **✓ (primary)** |
| **cert_config v13 unpinned (G4)** | – | – | – | – | **✓** |
| **K=6→K=7 engine prior CS approx** | – | – | – | – | **✓** |
| NUTS R̂ > 1.05 in 4/6 tasks | ✓ | – | – | – | – |
| AUROC interpretation risk | ✓ | – | – | – | – |
| 21-rater 90% expert | ✓ | – | (implied) | – | – |
| No race/ethnicity data | – | – | ✓ | – | – |
| All-Harvard cluster | – | – | ✓ | – | – |
| No site_id | – | – | ✓ | – | – |
| Centaur IED panel not in K=7 | – | – | ✓ | – | – |
| Cross-dataset label-comparability | – | – | ✓ | – | – |

The four findings unique to a single agent but highest-severity:

- **Pipeline G1 (replay/calibration overlap)** — most consequential single finding in the audit.
- **Pipeline G4 (cert_config unpinned)** — cheapest fix, highest engineering severity.
- **Pipeline G5 (engine vs deployment Σ different)** — Paper-1 figure-regeneration risk.
- **Biostat G5 (NUTS R̂ > 1.05)** — Bayesian reviewer ask.

---

## 9. Recommended sequencing

Three sprints sized to a typical clinical-AI submission cycle.

### Sprint 1 — Pre-submission (4–6 weeks, ≤ ~3 weeks of focused effort)

Cost: low; D-decision risk: none. Closes all four integrity gaps in §4 and three
of six methodology gaps in §5.

| # | Title | Effort | Deliverable | D-risk |
|---|---|---|---|---|
| **T1.1** | Pin v13 ℓ\* numerically in `tests/test_phase3_calibration.py` | 30 min | New test passing | None |
| **T1.2** | Leakage-aware ℓ\* sensitivity rerun (replay-disjoint calibration) | 1–2 days | `pipeline/replay/leakage_audited_replay.py` + supplement table | Possibly shifts the headline contrast |
| **T1.3** | Run `scripts/run_lapse_sensitivity.py` on unified corpus | minutes | `results/phase2_validation/lapse_sensitivity.{csv,md}` | None |
| **T1.4** | Brier + calibration-in-the-large + reliability diagrams + ECE per task | 4–6 hours | `pipeline/replay/calibration_metrics.py` + Figure F2 + Table T5 | None |
| **T1.5** | Formal hypothesis test for replay-vs-Bernoulli | 1–2 days | `pipeline/replay/statistical_tests.py` + supplement | None |
| **T1.6** | Multiple-testing / FWER analysis | 0.5 day | `docs/MULTIPLE_TESTING.md` | None |
| **T1.7** | Subgroup analyses on existing data (expertise tier, age band, sex) | 2–3 hours | Table T4 + Figure F4 | None |
| **T1.8** | Decision-curve analysis (DCA) net benefit grid | 6 hours | Figure F5 + supplement table | None |
| **T1.9** | OSF retroactive pre-registration | 1 day | OSF DOI + citation in manuscript | None |
| **T1.10** | TRIPOD+AI compliance supplementary table | 0.5 day | Supplementary file | None |
| **T1.11** | Manuscript Methods + Results scaffold | 1–2 weeks | Draft for co-authors | None |

**Total: ~3–4 weeks of focused effort.** All preserve D1–D9 invariants. Closes
the two highest-severity integrity gaps (G1, G4) and four of six TRIPOD+AI
non-negotiables.

### Sprint 2 — Revision-ready (6–12 weeks)

Cost: medium; D-decision risk: low to medium (mostly localised). Addresses
Sprint-1-leftover methodology gaps and the strongest external-cohort option that
can land before submission.

| # | Title | Effort | D-risk |
|---|---|---|---|
| T2.1 | Expand expert panel from n=4 to n≥10 with non-PI affiliations (EEG Option B) | 3–6 months | None — additive |
| T2.2 | Backfill patient demographics from BDSP source records (IRB amendment) | high | None — additive |
| T2.3 | Nested-CV for ℓ\* with panel selection in outer loop | med | Could shift ℓ\* CIs |
| T2.4 | NUTS R̂ fix (4/6 tasks); re-derive κ | compute-heavy | None |
| T2.5 | Bias-warning channel `|t_k| > tol` (closes OPEN_DECISIONS #4) | 3 hours | Adds deployment field; drift-guard |
| T2.6 | Spike task disambiguation (binary vs combined-IED) | med | K=7 → K=8 if separated; full recalibration |
| T2.7 | Re-fit Sigma_l at K=7 from PI block fit; retire CS approx | med | Paper-1 figures regenerate |
| T2.8 | K=7 factor decomposition (PCA + parallel analysis) | low | None |
| T2.9 | Probit-native Rasch refit as supplement | med | None (supplement) |
| T2.10 | Site-disjoint CV (cohort-disjoint folds) | med | Could shift ℓ\* |
| T2.11 | Chain intermediate sha256 through MANIFEST | low | None |
| T2.12 | Cross-dataset label-comparability bridge study (κ across protocols) | med | None |
| T2.13 | Pin sha256 of `Sigma_l_fitted.npy` + `sdt_fits.legacy_oldcorpus.csv` | 5 min | None |
| T2.14 | `make reproduce` end-to-end target | low | None |
| T2.15 | BLAS pinning in environment.yml | low | None |
| T2.16 | D7 site-partition independent-panel refit | med | None |
| T2.17 | Code + data availability statement | 0.5 day | None |
| T2.18 | External US academic site (UCSF/Mayo/Cleveland/Penn) with fresh signals (EEG Option C) | 4–9 months | None |

### Sprint 3 — Post-acceptance / Paper-2 (Paper-2 timeline)

Tier-3 items: international validation, ABCN pool, demographic hierarchical
model, K=8 spike-variant extension, ordinal expertise model, MIN_FIT_TRIALS
sensitivity, SBC replicate boost, replay-level lapse sweep.

---

## 10. Manuscript scaffolding

Per TRIPOD agent. Concrete section plan + figure list. Captures the audit
recommendations directly into the paper.

### Methods (8 subsections)

1. **Study design.** Prospective derivation + internal validation + retrospective deployment-OC via real-rater replay on n=21 prior raters with prior-recorded annotations from 2,115,793 labeled responses; click-to-run prospective deployment via CORTEX. Cite OSF pre-registration (T1.9). Note retrospective scope.
2. **Data sources.** SN1 (BIDMC IRB 2016P000058), SPARCNET-50K (MGH IRB 2013P001024 + consortium), Centaur 2025, Kong 2025, pd_rda_profiler — each with date range, n_segments, n_raters. **Table 1** here.
3. **Model.** λ-lapse probit IRT; single likelihood definition (cite `engine/core.py:21`); two-engine architecture (SMC + MCMC vs Laplace + EKF). Cite `λ = 0.025` single source. 7 tasks: combined_spike + 6 IIIC.
4. **Calibration.** CV-top-14 two-stage Youden; cert_config v13; ℓ\* per task with bootstrap 95 % CI. Expert membership = `raters.csv:expertise_level`. 70/30 expert split.
5. **Stopping rule.** `pass_p = 0.95, fail_p = 0.05, N_max_per_task = 120, N_min_per_task = 10`. Per-task PASS/FAIL/REFER verdicts. No single roll-up. FWER discussion (T1.6).
6. **Validation.** SBC; SPARCNET ICC; real-rater replay on 21-rater cohort + 14,823 per-task cells. Subgroup analysis by expertise tier + age band (T1.7).
7. **Statistical analysis.** AUROC + 95 % CI (label as expert-vs-non-expert AUROC); reliability diagrams + Brier + CITL + ECE (T1.4); DCA net benefit (T1.8); formal replay-vs-Bernoulli tests (T1.5).
8. **Code + data availability.** `data/curated_banks/*.json` aggregate signals; rater identities `rater_<sha256[:12]>`; raw EEG external (D9); IRB-covered subset behind DUA.

### Results (7 subsections)

- **R1 Cohort + flow** (Table 1 + Figure F7 CONSORT-style). 5,306 raters → 21 fully-typed candidates → 14,823 per-task cells.
- **R2 Internal validation** (Figure F1). SBC mean-rank, coverage, SPARCNET ICC.
- **R3 Discrimination + calibration** (Table T2 + Figure F2). Per-task AUROC, Youden J, calibration slope, Brier, reliability diagram.
- **R4 Deployment OC** (Figure F3). PASS/FAIL/REFER per task; replay vs Bernoulli; +43 % to +1154 % headline.
- **R5 Subgroup** (Table T4 + Figure F4 — NEW). Expertise-tier-stratified; age-band-stratified.
- **R6 Decision curve** (Figure F5 — NEW). Net benefit grid.
- **R7 Failure modes** (NEW). 0/21 all_pass dissection; bank-exhaustion vs engine indecision.

### Discussion (5 subsections)

- D1 Headline.
- D2 Comparison to standard of care (ABCN).
- D3 Limitations (small expert panel; pediatric-heavy spike substrate; demographics absent; D7 4/6 disjoint; retrospective pre-reg; external = Centaur n=4).
- D4 Clinical implications (per-task feedback for credentialing; complementary to ABCN; not replacement).
- D5 Future directions (prospective CORTEX live deployment; multi-site external; bias-warning channel).

### Figures

| # | Figure | Data ready? |
|---|---|---|
| F1 | SBC + coverage panel at K=7 | Yes (regenerate from test outputs) |
| F2 | AUROC + reliability diagram per task | T1.4 |
| F3 | Deployment OC: PASS/FAIL/REFER per task, replay vs Bernoulli | Yes (results/replay/) |
| F4 | Subgroup PASS rates by expertise tier + age band — NEW | T1.7 |
| F5 | Decision-curve / net-benefit grid — NEW | T1.8 |
| F6 | Example HW(AUROC) trajectory | Yes (results/phase1_figures/) |
| F7 | CONSORT-style flow: 5,306 → 21 | Yes (re-aggregate) |

### Tables

| # | Table | Data ready? |
|---|---|---|
| T1 | Cohort baseline (n, tier, board cert, years EEG, affiliation, RCT arm) | Aggregable |
| T2 | Per-task discriminability + threshold (ℓ\*, σ\*, J, 95 % CI) | Yes |
| T3 | Deployment headline: replay vs Bernoulli PASS/FAIL/REFER + AUROC | Yes |
| T4 | Subgroup-stratified verdicts — NEW | T1.7 |
| T5 | Calibration summary: Brier + CITL + slope per task — NEW | T1.4 |
| T6 | TRIPOD+AI checklist as Supplementary | T1.10 |

---

## 11. Killer review-defenses

Anticipated reviewer-2 attacks + the defense you should have ready. Adapted from
TRIPOD agent's analysis.

1. **"n=21 is too small for Nature Medicine."**
   The 21 is the per-candidate cohort. The per-task cohort is 14,823 cells across 1,000–2,200 raters per task. 29,646 deployment sessions across both arms. n=21 is the maximally-typed subset for cross-validating all 7 verdicts simultaneously. Show F7 (CONSORT flow).

2. **"Conjunctive all-pass = 0/21 invalidates the system."**
   Per OPEN_DECISIONS #1: 0/21 reaching `all_pass` is the conservative stopping rule interacting with 7 conjunctive gates; empirically degenerate, NOT a sign of unskilled raters (most achieve PASS on 4–6 of 7). Working v1.0 policy is per-task certificates. The certificate is the per-task verdict, NOT `all_pass`. T1.6 (`docs/MULTIPLE_TESTING.md`) formalises why.

3. **"Bernoulli under-predicts skill by +43–1154 %. Why trust calibration?"**
   This is the FAVOURABLE direction — real raters MORE skilled and MORE coherent than the fitted-θ Bernoulli twin. Two equivalent framings (reliability: per-seg responses not exchangeable iid; skill: raters more often above threshold than fitted θ predicts) both honestly point in favour. The deployment will be MORE lenient than OC tables predict, not less. Exception (`other` slightly worse) explained by the `{bipd, birds, other}` collapse.

4. **"Pre-registration is post-hoc. Data dredging?"**
   Agreed candidly: pre-registration was post-hoc. Retrospective registration documents what was locked at the 2026-05-18 design lock + 2026-05-19 headline run (T1.9). The four implicit bounds in `docs/PHASE7_CLOSEOUT.md:243-271` (engine drift-guard, both engines run, honest pilot caveats, no regression) were satisfied. CORTEX live-deployment cohort registered prospectively.

5. **"D7 has 2/6 IIIC CIs overlapping with n=4 gold. Not reproducibility."**
   D7 was re-scoped to directional / ordinal consistency, NOT tight-point reproduction; the n=4 limitation is explicit. Spearman ρ(gold, CV) = +0.66; 5/6 gold ℓ\* > CV ℓ\* (gold panel stricter, directionally consistent). Larger external panel (T2.1: n≥10 Option B) closes this; CORTEX live deployment is the prospective remediation.

6. **"No external validation. Reject."**
   Centaur n=4 gold panel is an independent expert anchor; T2.1 expands to n≥10 with non-PI affiliations; CORTEX live deployment is the formal prospective external arm. If editor insists on a larger external cohort pre-acceptance, that becomes a paper-revision cycle.

7. **"Where is the DCA? Calibration plot? Brier?"**
   Closed by T1.4 + T1.8 before submission.

8. **"70/30 expert TRAIN/VAL split is unusual. Why not k-fold?"**
   Byte-verbatim from the reference fitter `pipeline/reference_calibration/youden_sigma_star_ref.py:19`; unchanged from the published methodology paper. CV-top-14 two-stage Youden layered on top.

9. **"Two engines = inconsistency."**
   D1 two-engine architecture shares the single likelihood definition; differ only in inference machinery. Phase 4.6-C's "+0.0145 pass-share-of-decisive leniency shift" is the quantified, signed-off characteristic.

10. **"Code + data availability is opaque."**
    Code = post-acceptance public fork after anonymizer (`data/SENSITIVE.md:170-194`). Data = `data/curated_banks/*.json` aggregate signals (no raw EEG; D9 enforced). Rater identities replaced with `rater_<sha256[:12]>`. All integer-keyed result CSVs ship as-is.

---

## 12. Cross-document links

- `data/DATA_PROVENANCE.md` — corpus derivation chain.
- `docs/OPEN_DECISIONS.md` — Phase-8 acceptance review register. **Should be updated** with the Nature Medicine deltas identified here (planned in next task).
- `docs/INVARIANT_AUDIT.md` — Phase-6 5-PASS / 4-deviation invariants.
- `docs/PHASE7_REPLAY_HEADLINE.md` — D6 finding; lines 209–212 explicitly defer the formal-test layer (T1.5 closes).
- `docs/PHASE7_REPLAY_DESIGN.md` — replay harness design; cohort definitions.
- `docs/PHASE7_CLOSEOUT.md:243–271` — Phase-7 acknowledgement that pre-registration is post-hoc (T1.9 closes).
- `calibration/CALIBRATION_PROVENANCE.md` — D7 honest n=4 re-scoping.
- `data/SENSITIVE.md` — PHI governance + anonymizer spec.
- `data/engine_inputs/README.md` + `MANIFEST.json` — Phase-5 D3 anchor.
- `CLAUDE.md` — D1–D9 reviewer-grade context.
- `README.md` — user-facing entry-point usage.
