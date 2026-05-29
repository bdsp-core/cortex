# Does the cross-task skill correlation let the multi-task test stop earlier?

**Short answer: no — not at our fitted correlation, and not for an aggregate
pass/fail decision in general.** The idea is reasonable but the math works
against it for this target. Adopt the `ideal-test` gray-zone sequential rule on
its own merits; do **not** re-architect stopping around the correlation.

Run `run_experiment.py` then `make_plots.py` (and `r_sweep.py`) to regenerate
(outputs land in `out/`, which is gitignored because per-rater rows key on
clinician name = PHI).

## Setup

- K = 6 IIIC tasks (seizure, lpd, gpd, lrda, grda, other) — what the web test
  certifies. Engine = the validated `deployment/simulate_test.py` machinery
  (joint Laplace/EKF probit, λ-lapse likelihood).
- Stopping rule = ideal-test gray-zone Bayesian SPRT mapped σ→ℓ:
  PASS iff P(ℓ ≥ ℓ*+δ)≥0.95, FAIL iff P(ℓ ≤ ℓ*−δ)≥0.95, δ=0.10; else refer.
- Two priors, paired per candidate: **correlated** (fitted Σ, cross-task
  r_ℓ=0.37) vs **independent** (Σ block-diagonalised to each task's 2×2 block =
  tasks in isolation).
- Two decision targets: **per-task** (all 6 must resolve) and **aggregate**
  (one verdict on the mean margin m = mean_k(ℓ_k − ℓ*_k)).
- Candidates ("real data"): 812 clinicians fitted across all 6 IIIC tasks
  (`data/labels/fits_hier_block`), 800 population draws ~ N(0,Σ), and a
  concordant skill sweep. Responses generated from each candidate's true θ.

## Results (real raters, N=812)

| metric | correlated | independent | takeaway |
|---|---|---|---|
| aggregate Q-to-decision (median) | 403 | 480 | faster in **only 34%** of raters; **median saving 0** |
| aggregate decision accuracy | 0.91 | **0.96** | correlation is **less** accurate |
| over-confident on borderline raters | **64%** | 52% | correlation over-decides |
| per-task Q-to-resolve (median) | 344 | 352 | wash (~1%) |
| per-task accuracy | 0.95 | 0.96 | wash |

Population draws (N=800) agree: aggregate faster in only 25%, accuracy 0.976
(corr) vs 0.990 (indep).

## Correlation-strength sweep (prior = truth, aggregate target)

Sweeping a clean exchangeable Σ(r) with the population drawn at the same r:

| r | mean Q corr | mean Q indep | acc corr | acc indep |
|---|---|---|---|---|
| 0.00 | 381 | 358 | 0.97 | 0.99 |
| 0.20 | 337 | 338 | 0.99 | 0.99 |
| 0.37 (fitted) | 342 | 329 | 1.00 | 1.00 |
| 0.55 | 316 | 302 | 0.97 | 0.99 |
| 0.75 | 302 | 286 | 0.98 | 0.99 |
| 0.90 | 298 | 307 | **0.93** | 0.99 |

Independent is as-fast-or-faster up to r≈0.75. Correlation only edges ahead at
r≈0.90 — and there its accuracy falls to 0.93 vs 0.99. Our fitted r is 0.37.

## Why (it's principled, not a bug)

1. **Weak transfer.** A question on task A reduces task B's posterior variance
   by ~r² of a direct B question. At r=0.37 that's ~14% — too little to skip
   probing B.
2. **The aggregate fights correlation.** m is an *average*. Positive
   correlation *raises* Var(mean) (correlated quantities don't self-average), so
   the correlated model starts *less* certain about the aggregate and needs
   *more* evidence — the opposite of "stop earlier." The independent model's
   point estimate of the mean is unbiased and robust, so ignoring correlation
   costs nothing for this target.
3. **Extrapolation error.** Exploiting r aggressively (low per-task floor) lets
   the model infer unprobed tasks from probed ones; for *discordant* candidates
   (strong on some tasks, weak on others — common among real raters) this
   shrinks estimates toward each other and produces wrong/over-confident
   verdicts, lowering accuracy.

The dramatic single-case wins (a uniformly-weak candidate failed overall in ~18
vs ~566 questions) are real but confined to *clearly* pass/fail, *concordant*
candidates — rare, because the test population sits near the cut (fig5: benefit
only at clear skill levels; fig3: per-rater saving ≈0 across all candidates).

## Recommendation

- **Adopt the gray-zone SPRT** (`ideal-test`) for the per-task verdicts — it
  cleanly *refers* borderline candidates instead of forcing a call, which is a
  genuine improvement over the current hard P≥0.95 / ≤0.05 threshold.
- **Keep the correlated joint posterior for the aggregate _readout_** (the web
  test's "probability of eventually passing/failing") — it's a coherent display
  quantity — but **do not** use the correlation to stop the test early. Stop on
  per-task resolution as now.
- To shorten the test, the effective lever is **item informativeness / better
  banks**, not the prior correlation.

## Figures (out/)

- `fig1_aggregate_savings.png` — paired scatter + ECDFs (correlated ≈ independent)
- `fig2_calibration.png` — correlated slightly worse calibrated
- `fig3_concordance.png` — per-rater saving ≈0 vs concordance
- `fig4_pertask_vs_aggregate.png` — both targets ~overlap at r=0.37
- `fig5_concordant_sweep.png` — benefit only at clear pass/fail skill levels
- `fig6_r_sweep.png` — speed/accuracy vs correlation strength
