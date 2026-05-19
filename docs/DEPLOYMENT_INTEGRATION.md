# Phase 4 — Deployment Integration: close-out & provenance

Status: **Phase 4.1 → 4.6-C COMPLETE** (4.7 = `ilae-deploy` CLI, the
last remaining sub-step). This is the *documented attribution summary*
(decision 2026-05-19): each deliberate change to the deployment runtime
was **isolated and signed off in its own gated sub-step**; this doc
records that attribution + the definitive verdict on the one finding
carried across sub-steps. There is intentionally **no single
"deployment changed by X%" mega-delta** — the changes are not
jointly-isolatable (different K / population definition / RNG regime),
which is exactly why each was isolated separately.

The PI clinical-deployment engine (`simulate_test.py` Laplace/EKF +
`run_deployment_sim.py` + `freeze_deployment_prior.py` +
`fit_2pl_probit*.py`) was merged onto the unified corpus + the
reference-faithful v13 calibration under a strict incremental,
regression-gated discipline (full suite green before *every* commit;
two-step gate = prove port fidelity, then layer each intended change
with a signed-off delta).

## Per-sub-step decisions & signed-off deltas

| Sub-step | What | Gate / signed-off outcome | Commit |
|---|---|---|---|
| **Phase 4.1** | Faithful **path-only** port of PI `simulate_test`/`run_deployment_sim` → `deployment/` (hardcoded `/Users/…ROOT` → `engine_paths` self-locating shim) | Bit-faithful vs the pristine PI baseline (0/1200 decision cells diverged; numerics within LAPACK round-off) | `818bcf3` |
| **Phase 4.2** | Externalise PI `TestConfig` → `deployment/deployment_config.yaml` (auditable clinical pass/fail contract; strict `from_yaml`) | Bit-unchanged vs 4.1; drift-guard (YAML == canonical defaults) | `26025fa` |
| **Phase 4.3** | Likelihood hardening: **one** definition shared with Paper-1 — spike-paper Eq. 2 λ-lapse `p=λ+(1−2λ)Φ(η)`, λ imported from engine `core` | λ=0 reduces to PI bare-probit IRLS to ~2e-16; controlled signed-off delta (population-matched) | `6a07349` |
| **Phase 4.3b** | Multi-seed MC strengthening (separate systematic effect from sampling noise) | **Finding** (not noise): a small systematic pass-share-of-decisive leniency shift + a systematic REFER increase. Carried forward, honestly flagged | `54d66ac` |
| **Phase 4.4-A** | Deployment runtime made **K-agnostic** (derive K/DIM from Σ.shape; authoritative task list parsed from Σ slot names) | Bit-identical at K=6 (pre/post hardened seed=0 md5 MATCH); zero behavioural change | `80cd410` |
| **Phase 4.4-B** | Faithful port `fit_2pl_probit*`→`pipeline/`, `freeze`→`deployment/`; delete degenerate `iic=OR`; real erratum-correct `other` (`value∈{other,bipd,birds}`) | Faithful-port diff = only documented changes; `Y_other`=204,163 = v13 `sparcnet_iic` (the plan's 79,383 was stale pre-merge — reconciled, not silently swapped) | `37693cb` |
| **Phase 4.5** | Deployment ℓ* → **single v13 lineage** (`cert_config.yaml` `ell_star_unified_v13`), retiring PI's `ell_thresholds` Youden lineage (D2) | Isolated ℓ*-lineage delta = **large, expected, signed off** (34.2% cells; v13 ℓ* materially lower on spike/seizure) — the deliberate D2 adoption of the validated calibration, not a regression. + a coupling bug caught by the full-suite gate and fixed (4.3/4.3b studies pinned to PI `ell_thresholds`) | `3c1fea5` |
| **Phase 4.6-A** | Re-fit `fit_2pl_probit`+`fit_2pl_probit_hier --variant block`@**K=7** on the unified corpus → re-freeze (Σ **14×14**, 7-task bank/ell); **clean-sn1 spike** (v13-consistent); PI K=6 baseline **archived** + studies repointed; **D3** `data/labels` sha256 provenance | K=7 v13 wiring live (`other`←`sparcnet_iic`=0.4418); drift-guard default==artifact; archived studies reproduce PI behaviour; D3 sha256 == live corpus | `8939e81` |
| **Phase 4.6-B** | **RNG decoupled** in `run_deployment_sim` (per-candidate `SeedSequence([seed,cand_id]).spawn(2)` → independent θ-gen / Y-draw streams) | **Decoupling guarantee proven**: θ-population bit-identical under a perturbed stopping config (the 4.3 confound fixed); deterministic; decisions still respond to config. Canonical K=7 sim regenerated | `0b84cbe` |
| **Phase 4.6-C** | **Definitive** 4.3b re-assessment on the fully-shipped config | see verdict below | *(this commit)* |

## D3 — derived-artifact lineage

`data/deployment_prior/summary.json` now records
`data_labels_provenance` = sha256 of `data/labels/{labels,raters}.csv`
(verified == live corpus; auto-reproduced on any re-freeze). This
anchors the deployment artifact to the **same unified corpus**
`engine_inputs` derives from (cross-ref `data/DATA_PROVENANCE.md` +
`data/engine_inputs/MANIFEST.json`). `ell_thresholds.csv` in the frozen
artifact is **PI-Youden legacy provenance only** — `load_deployment`
consumes the v13 cert_config ℓ* (single lineage, D2/4.5), not it.

## DEFINITIVE verdict — the carried-forward 4.3b finding

The λ-lapse pass-share-of-decisive leniency shift was measured three
times, on three independent configs (each isolating different
variables), via a population-matched comparison (4.3b/4.5 controlled
dual-engine; 4.6-C native λ-vs-λ=0 on the decoupled shipped engine —
the 4.2 engine is K=6-hardcoded so the λ=0 counterfactual, proven @4.3
≡ PI bare-probit to ~2e-16, is the only K=7-viable no-lapse arm, which
the 4.6-B decoupling made clean):

| Config | pass-share shift | significance |
|---|---|---|
| 4.3b — PI K=6, PI `ell_thresholds` | +0.01448 ± 0.0028 | ~5·SE |
| 4.5 — K=6, v13 ℓ* | +0.01568 ± 0.0031 | ~5·SE |
| **4.6-C — shipped K=7** (v13, re-frozen Σ, decoupled, clean-spike) | **+0.01444 ± 0.0030** | **~4.8·SE** |

**It PERSISTS, essentially invariant (~+0.0145), across all three.**
This is **not noise** and **not a config artifact** — it is a *stable,
quantified, reproducible property of the reference-correct λ-lapse
likelihood interacting with the decision boundary*. On the shipped
system the lapse also (as designed) makes the adaptive test more
**conservative**: ΔP(refer)=+0.0619±0.0031 (the expected direction,
stronger at K=7); ΔP(fail)=−0.039 drops ~1.7× more than ΔP(pass)=−0.023
(the slight leniency among *decisive* verdicts: under lapse the extreme
negative evidence that drives FAIL is attenuated more than the
positive evidence near the boundary).

## Known characteristic (NOT a bug) — disposition

The λ-lapse mixture is the **validated, reference-correct** model
(spike-paper Eq. 2; shared one definition with Paper-1; PI's bare-
probit was the numerical approximation it replaced). Relative to that
approximation, on the shipped clinical deployment it:

1. **increases REFER by ~6 pp** (more conservative — the intended,
   safe direction; borderline candidates get referred, not mis-decided);
2. **shifts the pass-share of *decisive* verdicts by ~+1.4 pp**
   (slightly lenient: proportionally fewer FAIL among decided cases),
   small, stable, and fully quantified above.

This is **recorded as a known characteristic of the shipped system**,
not a defect. No further re-assessment is owed — 4.6-C IS the shipped
config; the finding has its definitive verdict. Any future change to
ℓ*, the corpus, or the stopping rule should re-cite this
characterisation. Honest residual caveats unchanged from the
calibration phases (small expert panels, D7 n=4 gold, SVI-vs-NUTS
s_mean r≈0.72–0.78) live in `docs/DATA_UNIFICATION_ANALYSIS.md`.

## Artifacts (reproducible)

- `pipeline/deployment_delta/phase4_3_lapse_delta.py`,
  `phase4_3_multiseed_mc.py`, `phase4_5_v13_ell_delta.py` — historical
  K=6 studies, pinned to `data/deployment_prior/_pi_baseline_frozen/`
  (reproducible as "under PI ℓ*/PI population").
- `pipeline/deployment_delta/phase4_6c_shipped_reassess.py` →
  `deployment/phase4_6c_shipped_reassess.json` — the definitive
  shipped-K=7 verdict.
- `tests/test_phase4_port_fidelity.py` + `test_phase4_4b_ports.py` —
  the full Phase-4 regression suite (gated green before every commit).
