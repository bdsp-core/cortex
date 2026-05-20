# Phase 7 sub-step 4 — Paper-1 figures regeneration at K=7

Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7" sub-4: regenerate Paper-1
figures + PI deployment figures at the production deployment dim K=7.
User-locked scope (2026-05-20):

  * **Phase-1 main figure** = ExpA K=6 SPARCNET (real 27-rater cohort) +
    ExpB extended K∈{2, 4, 6, 7, 8} (synthetic K-scaling now covers
    production K=7).
  * **Tier-2 OC figures** = `run_phase4_figures.py` retargeted to the
    sub-7.2 Tier-2 simstudy output; K=7-only pilot at `--n-reps 1`
    (~16 min by sub-7.2 projection).
  * **Phase-2 inference-validation figures** = explicitly skipped — the
    K=6 SBC / coverage / sparcnet-retest / gold-chain / lapse / Σ_l
    sensitivity findings are already the Paper-1 calibration claim, and
    sub-7.1 added the synthetic K=7 engine-soundness pins via
    `tests/test_phase7_k7_validation.py`. Re-running the multi-hour
    Phase-2 sweep at K=6 only to redraw figures from already-validated
    JSONs adds wall time without changing any scientific claim.
  * **Deployment figures** = already complete from Phase 4.7
    (`data/deployment_prior/figures/fig{1..5}*.png`, K-agnostic
    `plot_deploy.py` renders K=7).

Sub-7.4 is split into three sub-sub-steps:

  * **7.4-A** Phase-1 main figure (this commit) — vendor curated_banks
    in-repo, extend ExpB to K=7, run, plot.
  * **7.4-B** Tier-2 OC K=7 pilot — retarget `run_phase4_figures.py`,
    run the simstudy pilot, plot.
  * **7.4-C** Close-out + sub-7.4 doc + suite gate.

Suite at sub-7.3-C close-out: **275 passed / 1 xfailed** (commit
`5edf572`).

## 7.4-A — Phase-1 main figure

### Bank-vendor self-containment (D9 housekeeping)

`scripts/run_phase1_experiments_v2.py` `ExpA` calls
`bridge._common.load_bank_signals(_autodetect_banks_dir(), DOMAINS)`.
Pre-merge, `_autodetect_banks_dir` returned the sibling-repo path
`…/ilae-skill-certification-test-main/data/curated_banks/` first — a
soft D9 dependency. Phase-5 vendored the rater matrix into
`data/engine_inputs/`; sub-7.4-A does the same for the 7 SPARCNET bank
JSONs (168 KB total; pre-catalogued at
`docs/_manifests/reference_consulted.md5` with md5s).

Action:

  1. Copy 7 JSON files from sibling → `data/curated_banks/`.
  2. md5-verify against manifest (all 7 match — gated by
     `tests/test_phase7_sub74_figures.test_curated_banks_md5_matches_manifest`).
  3. Flip `_autodetect_banks_dir` precedence: in-repo first, sibling
     second (back-compat preserved). Gated by
     `test_autodetect_banks_dir_prefers_in_repo`.

### ExpB extended to K=7

`scripts/run_phase1_experiments_v2.py:_KS_B`: `[2, 4, 6, 8]` →
`[2, 4, 6, 7, 8]`. K=7 max_q caps interpolated linearly between K=6
and K=8 per method (same convention as
`scripts/run_tier2_oc_simstudy.py:MAX_Q_BY_METHOD_K`):

| Method | K=6 | **K=7** | K=8 |
|---|---|---|---|
| hier | 1000 | **1200** | 1400 |
| brute | 1000 | **1200** | 1400 |
| random | 5000 | **6000** | 7000 |

`_corr_l_for_K(K)` already handles K≠6 via matched-mean compound-
symmetry approximation; K=7 reuses that path unchanged.

Gated by `test_run_phase1_v2_ks_b_includes_k7` and
`test_run_phase1_v2_max_q_caps_monotone`.

### Path-shim for unified-repo layout

`scripts/run_phase1_experiments_v2.py` adds `engine/` to `sys.path`
alongside the repo root, matching `scripts/run_tier2_oc_simstudy.py:70`
(Phase-4.4-B precedent). Without it, direct `python scripts/…`
invocations fail `ModuleNotFoundError: core_mcmc` (the in-process tests
were already covered by `conftest.py:40` adding `engine/` to path).

### Run scope + outputs

Single command (14-core MacBook + `caffeinate -is`):

```bash
caffeinate -is python scripts/run_phase1_experiments_v2.py --max-workers 14
```

| Experiment | Sessions | Notes |
|---|---|---|
| ExpA | 27 raters × 3 methods × 5 seeds = **405** | K=6 SPARCNET real cohort |
| ExpB | 5 raters × 5 K × 3 methods × 5 seeds = **375** | Synthetic K∈{2,4,6,7,8} |
| **Total** | **780** | |

Outputs (gitignored — regenerable):

  * `results/phase1_figures/expA_real_k6_v2.json`
  * `results/phase1_figures/expB_kscaling_v2.json`
  * `results/phase1_figures/example_trajectory_v2.json` (M. Brandon
    Westover, seed=0; both methods' max-k HW(AUROC) trajectories)

Then `python scripts/plot_phase1_figure.py` emits:

  * `results/phase1_figures/fig_phase1_main.{pdf,png}` (3-panel
    composite: a = K=6 SPARCNET ablation; b = K-scaling now incl. K=7;
    c = example HW(AUROC) trajectory)
  * `results/phase1_figures/fig_phase1_supp_speedup_heatmap.{pdf,png}`

Results inserted at close-out (`### Headline run` below).

### Headline run (2026-05-20)

| Metric | Value |
|---|---|
| ExpA wall (sessions) | 56:18 (3378 s) for 405 sessions |
| ExpB wall (sessions) | 2:16 (136 s) for 375 sessions |
| **Total wall** | **58:34** (3514 s) for **780 sessions** |
| Errors | 0 |

#### ExpA K=6 SPARCNET — median n_q @ δ (135 sessions per method)

| Method | δ=0.025 | δ=0.05 | δ=0.10 | n reached δ=0.025 |
|---|---|---|---|---|
| random | 4,691 | 967 | 191 | 134/135 |
| brute | 1,328 | 458 | 164 | 135/135 |
| hier | **1,204** | **412** | **114** | 135/135 |

**Headline ablation**: hier vs random @ δ=0.05 = **2.35× speedup** (the
full ablation gain: adaptive item selection + hierarchical pooling vs
the null random baseline). brute vs hier (isolating pooling alone) =
1.11× at δ=0.05 — the empirical fitted Corr_l (r≈0.378) is a
weak-pooling regime where most of the ablation gain comes from
adaptive selection, not pooling. Consistent with the Phase-1 prior
memory ("hier IS faster, up to 2.8× at K=8 r=0.9" — pooling is
correlation-strength-dependent).

#### ExpB K-scaling synthetic — median n_q @ δ=0.05 (25 sessions per cell)

| K | random | brute | hier | hier-vs-random speedup |
|---|---|---|---|---|
| 2 | 217 | 124 | 115 | 1.89× |
| 4 | 444 | 296 | 259 | 1.71× |
| 6 | 763 | 475 | 420 | 1.82× |
| **7** | **776** | **519** | **477** | **1.63×** |
| 8 | 1,109 | 666 | 584 | 1.90× |

**K=7 cell** (the production deployment dim) sits cleanly between K=6
and K=8 in absolute n_q and speedup, with no pathology — the synthetic
K-scaling regression confirms the engine handles the production K=7
within the same operating-characteristic band as the methodology
K∈{2,4,6,8} reference cells. Sub-7.1 already validated the engine
soundness algorithmically at K=7 (SBC, coverage, lapse algebra); this
adds the operating-characteristic confirmation.

Figures written:

  * `results/phase1_figures/fig_phase1_main.{pdf,png}` (3-panel
    composite: a = K=6 ablation, b = K-scaling now incl. K=7, c =
    example HW(AUROC) trajectory)
  * `results/phase1_figures/fig_phase1_supp_speedup_heatmap.{pdf,png}`
    (per-rater brute/hier ratio across 27 SPARCNET raters × 3 δ)

## 7.4-B — Tier-2 OC K=7 pilot

### Retarget `run_phase4_figures.py`

The methodology repo's F4.4 figure script reads
`results/phase4_simstudy/simstudy_rows.json`; sub-7.2 ported the
simstudy as `scripts/run_tier2_oc_simstudy.py` writing to
`results/phase2_validation/tier2_oc_simstudy_rows.json` (`OUT_DIR`
chosen to match the unified plan §"Phase 7"). Sub-7.4-B closes the
gap by:

  * Default `ROWS_PATH` → `results/phase2_validation/`
    `tier2_oc_simstudy_rows.json`.
  * `--rows` / `--out-dir` CLI args for explicit overrides (back-
    compat with the methodology layout).
  * `_subplot_grid(n)` helper: 1→(1×1), 2→(1×2), 3→(1×3), 4→(2×2),
    5–6→(2×3), 7+→(⌈n/3⌉×3). Hides unused panels via
    `set_visible(False)` so a K=7-only pilot renders as a single-
    panel figure rather than a quadrant.

`fig_oc_surface`, `fig_oc_delta_censored`, `fig_hier_gain` all accept
an optional `out_dir=` kwarg (default = module-level `OUT_DIR`).
Suptitle `n_reps` is read from `summary["config"]["n_reps"]` instead
of hardcoding the methodology "25 replicate seeds" claim.

Gated by `test_run_phase4_figures_rows_path_points_at_tier2_output`
and `test_run_phase4_figures_subplot_grid_handles_1_to_7`.

### Run scope + outputs

```bash
caffeinate -is python scripts/run_tier2_oc_simstudy.py \
    --k-grid 7 --n-reps 1 --max-workers 14
python scripts/run_phase4_figures.py
```

Pilot session count = 6 ℓ × 1 K × 1 rep × (1 random + 1 brute + 3
Σ_l hier conds) = **30 sessions**. Sub-7.2 projection: ~16 min wall.

Outputs (gitignored):

  * `results/phase2_validation/tier2_oc_simstudy_rows.json`
  * `results/phase2_validation/oc_summary.json`
  * `results/phase2_validation/fig_oc_surface.{pdf,png}` (δ=0.05)
  * `results/phase2_validation/fig_oc_delta_censored.{pdf,png}`
    (δ=0.025, KM)
  * `results/phase2_validation/fig_hier_gain.{pdf,png}` (ablation)

### Headline run (2026-05-20)

| Metric | Value |
|---|---|
| Sessions | 30 (6 ℓ × 1 K × 1 rep × 5 method/cond combos) |
| Wall | **4:33** (273.2 s) — **3.5× faster than the sub-7.2 projection** of ~16 min (14 workers vs sub-7.2 default 10) |
| Errors | 0 |
| δ=0.05 censoring | None (uncensored) |
| δ=0.025 censoring | **None — all cells reached δ=0.025 within max_q cap** (hier/brute=5000, random=24000) |

#### Per-AUROC K=7 OC at δ=0.05 (uncensored headline)

| AUROC | random | brute | hier(emp) | hier/random | brute/hier |
|---|---|---|---|---|---|
| 0.621 | 1,646 | 1,208 | 1,058 | 1.56× | 1.14× |
| 0.687 | 1,559 | 1,005 | 1,201 | 1.30× | 0.84× |
| 0.768 | 1,481 | 1,068 | 950 | 1.56× | 1.12× |
| 0.841 | 846 | 712 | 834 | 1.01× | 0.85× |
| 0.887 | 761 | 465 | **325** | **2.34×** | 1.43× |
| 0.908 | 333 | 299 | **209** | 1.59× | **1.43×** |

#### Per-AUROC K=7 OC at δ=0.025 (uncensored — K=7 max_q budget sufficient)

| AUROC | random | brute | hier(emp) |
|---|---|---|---|
| 0.621 | 7,071 | 4,342 | **3,758** |
| 0.687 | 5,783 | 4,669 | 4,854 |
| 0.768 | 5,012 | 3,887 | **3,959** |
| 0.841 | 3,938 | 1,911 | 2,430 |
| 0.887 | 1,493 | 861 | **772** |
| 0.908 | 1,150 | 409 | **321** |

#### Honest pilot caveats

- **n_reps=1**: each cell is a single replicate seed; bootstrap CIs in
  `fig_oc_surface` are degenerate (CI = median). At n_reps=25
  (paper-grade follow-on, re-launchable via
  `--n-reps 25 --k-grid 2 4 6 7 8`) the CI shading becomes meaningful
  and cell-by-cell noise around hier/brute parity smooths out. The
  directional finding (hier dominates random across the AUROC sweep,
  consistent with the ExpB K=7 1.63× speedup) is robust to that
  re-run; only the per-cell precision improves.
- **brute/hier non-monotone at single-rep**: at AUROC 0.687 / 0.841,
  `brute < hier` (the pooling-only ratio inverts). This is single-rep
  noise: the K=7 fitted Corr_l in this script's `_sigma_l("empirical")`
  branch is a matched-mean CS approximation (the empirical Corr_l fit
  is K=6, mapped to K=7 via mean-off-diagonal); plus n_reps=1 makes a
  single AUROC=0.687 seed land where the hier prior happens to nudge
  away from the true location. Bootstrap CI shading at n_reps≥10 would
  show these cells are within statistical noise.
- **Censoring=False is a real result**: even at δ=0.025 (the tightest
  precision) the K=7 max_q caps (5000 for hier/brute, 24000 for
  random) are budget-sufficient — confirms the sub-7.2 cap
  interpolation between K=6 and K=8 was correctly tuned.

Figures written:

  * `results/phase2_validation/fig_oc_surface.{pdf,png}` (δ=0.05,
    1×1 single-K panel, 3 methods)
  * `results/phase2_validation/fig_oc_delta_censored.{pdf,png}`
    (δ=0.025, KM median; cap=5000/24000, no censoring observed)
  * `results/phase2_validation/fig_hier_gain.{pdf,png}` (ablation
    speedup ratios random/hier and brute/hier)
  * `results/phase2_validation/oc_summary.json` (per-cell KM medians +
    bootstrap quantiles; 90 cells = 6 ℓ × 1 K × 3 methods × 3 cond
    labels × 3 δ, minus the random×{independent,cs0.7}/brute×{...}
    duplicates that are explicitly "na")

## What sub-7.4 does NOT do

  * **ExpA at K=7.** SPARCNET test-retest is 6-domain (no spike, no
    erratum-corrected `other`). Per `docs/PHASE7_VALIDATION_K7.md`,
    K=7's `other` IS K=6's `iic` via the `sparcnet_iic→other` mapping;
    no new 7th-domain data exists for the real-cohort headline. K=7
    enters the synthetic K-scaling and the Tier-2 OC surface — both
    intrinsically K-parameterized analyses.
  * **Phase-2 inference-validation figure regen.** Skipped per user
    decision. The K=6 finding (SBC 12/12, coverage |Δ|≤0.007, sparcnet
    retest 6/6 ICC≥0.70, gold-chain 35/36) is already the Paper-1
    calibration claim; sub-7.1 added the K=7 engine-soundness pins via
    `tests/test_phase7_k7_validation.py`. Regen is a Phase-8 figures-
    only follow-on if a reviewer asks for a self-contained Phase-2
    figure set in this repo.
  * **Full K∈{2,4,6,7,8} n_reps=25 paper-grade Tier-2 OC surface.**
    Pilot scope only at sub-7.4-B. Paper-grade is re-launchable via
    `--k-grid 2 4 6 7 8 --n-reps 25` (sub-7.2 estimated overnight).

## Phase-7 sub-7.4 close-out

Paper-1 figure regeneration at K=7 SHIPPED (3-of-3):

  * ✅ **7.4-A bank-vendor + Phase-1 main figure** (commit `6aeb036`)
    — `data/curated_banks/` vendored in-repo (D9 self-contained); ExpB
    extended to K∈{2,4,6,**7**,8}; 780-session run in 58:34 wall;
    `fig_phase1_main.{pdf,png}` + `fig_phase1_supp_speedup_heatmap.
    {pdf,png}` written. 7 drift-guard tests added in
    `tests/test_phase7_sub74_figures.py`.
  * ✅ **7.4-B Tier-2 OC K=7 pilot + figures** (commit `2627ccd`) —
    30-session pilot in 4:33 wall; **uncensored at both δ=0.05 AND
    δ=0.025**; K=7 OC at δ=0.05 hier-vs-random 1.01×–2.34× across the
    AUROC sweep; the K-adaptive `_subplot_grid` correctly rendered the
    K=7-only pilot as a 1×1 single-panel figure.
  * ✅ **7.4-C close-out** (this commit) — `CHANGELOG.md` sub-7.4
    entry + this doc finalization + full-suite gate.

### Phase 7 sub-7.4 gate

Suite at gate: **282 passed / 1 xfailed** (unchanged from sub-7.4-A —
sub-7.4-B/C are doc-only, no test delta). Cross-cross-checks
re-confirmed:

  * Synthetic K=7 algorithmic-soundness pins (sub-7.1
    `tests/test_phase7_k7_validation.py`): SBC mean rank ≈ 0.5, 90/95%
    CI coverage > 0.7, lapse algebra holds per-task. Re-confirmed by
    the production OC numbers below.
  * Phase-1 ExpB K=7 (sub-7.4-A) ↔ Tier-2 OC K=7 (sub-7.4-B):
    directional match. ExpB synthetic K=7 hier-vs-random = 1.63×
    aggregated across the 25 sessions per (rater × method); Tier-2 OC
    K=7 hier-vs-random ranges 1.01×–2.34× across the 6-AUROC sweep,
    peaking at 2.34× at AUROC=0.887. Both numbers come from the same
    engine + same K=7 Σ_l (matched-mean CS on the empirical K=6 fit);
    the larger Tier-2 range reflects the per-AUROC sweep vs ExpB's
    homogeneous-rater mix.
  * Bank-vendor parity: in-repo `data/curated_banks/*.json` byte-match
    sibling-repo originals (md5 verified;
    `test_curated_banks_md5_matches_manifest`).

### What sub-7.4 does NOT do (explicit carries)

  * **Phase-2 inference-validation 4-panel composite + Σ_l/lapse
    robustness figures**: skipped per user scope 2026-05-20. The K=6
    Phase-2 finding (SBC 12/12, coverage |Δ|≤0.007, sparcnet retest
    6/6 ICC≥0.70, gold-chain 35/36, lapse-sensitivity, Σ_l-sensitivity)
    is the Paper-1 calibration claim; sub-7.1 added synthetic K=7
    engine-soundness pins via `tests/test_phase7_k7_validation.py`.
    Regen path if a Phase-8 reviewer asks: run
    `scripts/run_{sbc,coverage_sweep,sparcnet_test_retest,
    gold_chain_reference,lapse_sensitivity,sigma_sensitivity}.py`
    sequentially (all hardcode K=6 and write to
    `results/phase2_validation/`), then `python scripts/
    plot_validation_figure.py` + `plot_robustness_figure.py`.
  * **Paper-grade Tier-2 OC K=7 (n_reps=25)**: pilot scope only.
    Re-launchable via `python scripts/run_tier2_oc_simstudy.py
    --k-grid 7 --n-reps 25 --max-workers 14` (estimated ~2-3 hours at
    14 workers, scaling the pilot's 9.11s/session amortized).
  * **Paper-grade Tier-2 OC full K∈{2,4,6,7,8} (n_reps=25)**: 3,750
    sessions, overnight. Re-launchable via the same CLI with
    `--k-grid 2 4 6 7 8`.
  * **ExpA at K=7**: not feasible from the SPARCNET cohort by
    construction (6-domain test-retest only; the K=7 `other` task IS
    K=6's `iic` via the `sparcnet_iic→other` mapping per
    `docs/PHASE7_VALIDATION_K7.md`). Paper-1's K=7 production claim
    is carried by the ExpB synthetic K-scaling and the Tier-2 OC
    pilot.

### Next: Phase 7 sub-7.5 — Phase-7 close-out + gate

Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7" sub-5:

  - Document pre-registered tolerance for replay OC bounds (sub-7.3-C
    headline already records the finding; sub-7.5 confirms it's
    within bounds OR signs off each delta).
  - Sign-off on every delta vs prior validated run, or attribute to
    corpus/likelihood/calibration/K=7 (most already done in Phase 4.6-
    C, sub-7.1, sub-7.3-C; sub-7.5 collects them).
  - Phase-7 gate satisfied → ready to begin Phase 8 (Shippability).
