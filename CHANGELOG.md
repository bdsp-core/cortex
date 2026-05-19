# CHANGELOG — ilae-skill-certification (unified)

Consolidated 2026-05-15 (F3.5).  Supersedes the per-day `STATE_LOG_*.md`
and per-wave `CHANGES_W*.md` files (archived under
`docs/historical/`).  Forward-chronological; the box below is the
fast-path orientation for a new contributor.

---

## ▶ UNIFIED MERGE (2026-05-18) — Phases 0–3.5

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
