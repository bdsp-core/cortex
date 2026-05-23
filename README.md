# ILAE Skill Certification (unified)

Unified repository for the ILAE epileptiform-discharge skill
certification system: **two engines, one shared data + calibration
layer.**

  * **Paper-1 engine** (`engine/`) — SMC + MCMC adaptive testing for
    the Multi-AUROC Precision Protocol (Mode-A). A per-examinee
    Bayesian adaptive test across K=7 tasks (spike + 6 IIIC tasks);
    skill is reported as per-domain AUROC with credible interval;
    session stops when `max_k halfwidth_0.95(AUROC_k) < δ`.
  * **Deployment engine** (`deployment/`) — Laplace + EKF clinical
    deployment runtime. Same likelihood definition as Paper-1
    (`λ + (1 − 2λ)·Φ`, λ = 0.025), bit-faithfully ported from the PI
    deployment scripts and re-frozen on the unified K=7 corpus.

Both engines read the same `data/` + `calibration/` layer; neither
imports the other (D1 module boundary).

Status: **v1.0.0-rc1 candidate** — Phases 0 → 7 SHIPPED; Phase 8
(this packaging + docs pass) in progress. See `CHANGELOG.md` for the
phase-by-phase trail and `docs/PHASE7_CLOSEOUT.md` for the scientific
gate.

Repository policy: **private until journal acceptance**. PHI
inventory: `data/SENSITIVE.md`. IRB 2016P000058 (BIDMC), 2013P001024
(MGH).

## Quickstart

Python 3.11.9 required (see `.python-version`). The engine's
bit-exact-reproducibility contract requires single-thread BLAS at
runtime (`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`); `conftest.py`
sets these automatically for `pytest`.

```sh
python -m venv .venv
.venv/bin/pip install -e ".[test]"      # add ,calibration for Phase-3.5
.venv/bin/pytest                          # 282 passed / 1 xfailed expected
```

Console entry points:

| CLI | Module | Phase | Purpose |
|---|---|---|---|
| `ilae-deploy` | `deployment.cli:main` | Phase 4.7 | Clinical-deployment pipeline `freeze → simulate → plot` at K=7. |
| `ilae-paper` | `bridge.run_multi_auroc_bridge:main` | Phase 1 (F1.3) | Multi-AUROC Precision Protocol bridge for Paper-1 Mode-A runs. |
| `ilae-calibrate` | `pipeline.run_unified_calibration:main` | Phase 3 | Unified non-circular CV Youden calibration (recompute on PI corpus, K=7). |

Run `ilae-deploy --help`, `ilae-paper --help`, `ilae-calibrate --help`
for usage.

## CORTEX internal test

`scripts/eeg_bank_viewer.py` (the CORTEX viewer) plus `session_controller.py`,
`cortex_engine_inputs.py`, and `cortex_storage.py` are a self-contained,
take-the-test build of the IIIC adaptive certification for internal review:
a lab member runs it, answers an adaptive question sequence driven by the
SMC engine, and the per-session results upload to a Dropbox folder. Build
the distributable bundle with:

```sh
python scripts/build_internal_test_zip.py     # → dist/cortex-internal-test.zip
```

Result delivery is configured in `cortex_config.yaml` (see
`docs/CORTEX_DROPBOX_SETUP.md`); the bundle's own `README_CORTEX_TEST.md` is
the test-taker setup + run guide.

## Repository layout

```
ilae-skill-certification-unified/
├── README.md  CLAUDE.md  CHANGELOG.md   # this file; reviewer-facing context; phase trail
├── pyproject.toml  environment.yml  requirements.txt
├── cert_config.yaml                      # v13 production calibration (K=7)
├── Sigma_l_fitted.npy                    # legacy K=6 empirical Corr_l (engine reads via symlink)
│
├── engine/                               # Paper-1 SMC + MCMC engine (hardened)
│   ├── core.py  core_mcmc.py  core_mcmc_brute_k.py  engine_mode_b.py
│   ├── auroc.py  diagnostics.py  engine_paths.py
│   └── variants/                         # PI methodological variants on hardened likelihood
│
├── deployment/                           # Clinical-deployment runtime (Laplace/EKF)
│   ├── simulate_test.py  run_deployment_sim.py  freeze_deployment_prior.py
│   ├── cli.py  plot_deploy.py            # K-agnostic; renders 5 figures at K=7
│   ├── deployment_config.yaml            # shipping stopping-rule contract
│   └── replay/                           # Phase 7.3 strict-A real-rater replay (D6)
│
├── pipeline/                             # Data ingest + fitting + calibration orchestration
│   ├── ingest_*.py  build_*.py  fit_2pl_probit*.py
│   ├── reference_calibration/            # byte-verbatim reference fitters (Rasch + per-rater probit-lapse + CV Youden)
│   ├── joint_calibration/                # Phase 3.5 joint hierarchical s_j (NumPyro NUTS, calibration-stage only)
│   ├── replay/                           # Phase 7.3 unified replay driver
│   └── run_unified_calibration.py        # ilae-calibrate target (Phase 3 orchestrator)
│
├── calibration/                          # v13 production calibration outputs
│   ├── cert_config.yaml  youden_ell_star.json
│   └── CALIBRATION_PROVENANCE.md
│
├── bridge/                               # Reproducibility / audit-trail layer; Mode-A entry
│   ├── run_multi_auroc_bridge.py         # ilae-paper target
│   └── _common.py
│
├── data/
│   ├── labels/                           # Canonical PI corpus (D3 single source of truth)
│   ├── deployment_prior/                 # K=7 frozen Σ + ℓ* + case_bank + figures
│   ├── engine_inputs/                    # Derived: sdt_fits.csv + cross_domain_rater_matrix.csv
│   ├── curated_banks/                    # K=6 SPARCNET item-bank signals (Phase 7.4-A vendored)
│   ├── replay/                           # Phase 7.3 strict-A rater bank (gitignored, regenerable)
│   ├── DATA_PROVENANCE.md  SENSITIVE.md
│
├── scripts/                              # Validation / experiment harnesses (Phase-2 + Phase-7)
├── tests/                                # 282-test suite (+ 1 xfailed)
├── experiments/                          # Reproducible experiment configs
└── docs/                                 # Phase docs + design audits
    ├── PHASE7_CLOSEOUT.md                # Phase-7 scientific gate sign-off
    ├── PHASE7_REPLAY_HEADLINE.md         # D6 real-rater replay v1.0-blocker findings
    ├── PHASE7_PAPER1_FIGURES.md          # Paper-1 + Tier-2 OC K=7 figures
    ├── DEPLOYMENT_INTEGRATION.md         # Phase 4 integration verdict
    ├── DATA_UNIFICATION_ANALYSIS.md      # Phase 3.5 joint s_j + engine SBC results
    ├── INVARIANT_AUDIT.md                # Phase 6 reference-truth checklist
    ├── OPEN_DECISIONS.md                 # Phase 8 open shipping decisions
    └── …
```

## Two-engine architecture (D1)

| | Paper-1 engine | Deployment engine |
|---|---|---|
| Algorithm | SMC + MCMC rejuvenation | Laplace approximation + EKF |
| Item selection | Global-EV (A-optimal posterior-variance reduction) | EV-driven with task-level caps |
| Stopping rule | `max_k HW₀.₉₅(AUROCₖ) < δ` (Mode-A) | `pass_p ≥ 0.95 / fail_p ≤ 0.05`, `N_min=60`, `N_max=500`, `N_max_per_task=120`, `N_min_per_task=10` (Mode-B-style PASS/FAIL/REFER per task) |
| Module | `engine/` | `deployment/` |
| Entry CLI | `ilae-paper` | `ilae-deploy` |
| Use | Paper-1 simulation study + validation | Clinical deployment at K=7 |

Both consume the same calibration (`calibration/cert_config.yaml`
v13) and the same data (`data/labels/`); the likelihood is **one
definition** (`λ + (1 − 2λ)·Φ`, λ = 0.025; `engine/core.py:21`).

## Reference-truth invariants (per Phase 6)

Audited in `docs/INVARIANT_AUDIT.md` (5 PASS, 4 signed-off
deviations). Hard pins:

  * `LAPSE_RATE = 0.025` — single source at `engine/core.py:21`,
    shared by Mode-A SMC, Mode-B, deployment runtime, and the
    reference fitter (`LAMBDA = 0.025` in
    `pipeline/reference_calibration/fit_sdt_per_domain.py`).
  * `LOGIT_TO_PROBIT = 1.0 / 1.7` — at
    `pipeline/reference_calibration/fit_sdt_per_domain.py:38`;
    runtime-asserted in `pipeline/run_unified_calibration.py:201`.
  * **Expert split = 70/30** for σ\*/ℓ\* TRAIN/VAL (no leakage);
    non-expert split = 50/50 (`EXPERT_TRAIN_FRAC = 0.70` at
    `pipeline/reference_calibration/youden_sigma_star_ref.py:19`).
    The earlier "50/50" CLAUDE.md comment was stale; corrected.
  * The two `fit_sdt_per_domain.py` copies (in
    `pipeline/reference_calibration/` and `pipeline/_calib_work/src/`)
    are byte-equivalent (md5 `6b90d59dcd0aaa878f9a52802254044b`);
    gated by `tests/test_phase6_invariants.py`.
  * Expert membership = `data/labels/raters.csv:expertise_level`
    (NOT the legacy `EXPERTS` Python set or
    `gold_standard_raters.yaml`; those references in the plan/CLAUDE.md
    were stale — corrected per `docs/INVARIANT_AUDIT.md` §7).

## Reproducibility

The engine is bit-exact reproducible across machines with the same
numpy + scipy + BLAS configuration **under single-thread BLAS**.
`tests/test_parallel_determinism.py` asserts serial == parallel
bitwise; `tests/test_phase35_*.py` gates the s_sd-propagation
back-compat at `s_sd → 0`.

Phase 3.5 calibration uses NumPyro + JAX NUTS (the
`[calibration]` optional dep) which is calibration-stage-only — the
engine runtime never imports JAX/NumPyro, preserving the BLAS
contract. JAX determinism (PRNGKey + XLA) is documented separately
in `calibration/CALIBRATION_PROVENANCE.md`.

## Sensitive data + IRB

See `data/SENSITIVE.md` for the PHI inventory, IRB coverage, and the
hash-at-acceptance plan (the anonymizer
`scripts/anonymize_rater_data.py` is spec'd and runs at journal
acceptance). The raw EEG source (`SN1_combined_v2.h5`) is **never
vendored**: per D9, it stays external (sibling
`ilae-skill-certification-test-main` repo or equivalent); retrieval
path is documented in `data/SENSITIVE.md`.

## Where to look for what

| Question | Doc |
|---|---|
| What was decided and why? | `../UNIFIED_REPO_MERGE_PLAN.md` (D1–D9, phased plan) |
| Phase-by-phase trail | `CHANGELOG.md` |
| Is the engine calibrated? | `docs/PHASE7_CLOSEOUT.md`; `docs/DATA_UNIFICATION_ANALYSIS.md` §8 |
| How is real-rater replay built? | `docs/PHASE7_REPLAY_HEADLINE.md` |
| What are the Paper-1 figures? | `results/phase1_figures/` + `docs/PHASE7_PAPER1_FIGURES.md` |
| What still needs deciding for v1.0? | `docs/OPEN_DECISIONS.md` |
| What's the deployment contract? | `deployment/deployment_config.yaml` + `docs/DEPLOYMENT_INTEGRATION.md` |
| What's PHI / IRB? | `data/SENSITIVE.md` |
| What's the calibration provenance? | `calibration/CALIBRATION_PROVENANCE.md` + `data/DATA_PROVENANCE.md` |
| Reference-truth invariants | `docs/INVARIANT_AUDIT.md` |

## Contributing

See `CLAUDE.md` for reviewer-grade context and the documented
contributor conventions (incremental + regression-gated edits;
component-by-component ports against PI baseline; gate on
PI-output-equivalence + the 282-test suite). The repo policy is
private until journal acceptance; external contributions follow the
acceptance hash step.
