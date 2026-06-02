# Bank-composition quality sweep (n_raters × expert filter)

**Question.** What `(n_min_raters, min_n_expert)` filter on the question bank
maximizes the number of questions available while preserving — or improving —
the test's accuracy and statistical power?

**Harness.** `sim_v1_3_5/run_bank_quality_sweep.py`, 2,400 synthetic sessions
(24 filters × 2 noise models × 5 skill levels × 10 reps), AD6 ship params
(N_MIN=20, ALPHA=0.25, R*=0.30, Z=2.0) + the v1.3.5 consecutive cap=12,
48-way parallel, ~21 min wall. Runs on signals only (no EEG): each filter
builds a `K7EngineInputs` pool straight from `data/labels/segment_signals.csv`,
each session draws a fixed 350-segment bank (300 IIIC + 50 spike) from that
pool, so the **filter varies pool QUALITY while session size is held constant**.

**Two noise models.**
- **Model A** (engine's own view): rater answers to the bank's `s_mean` treated
  as truth. Measures efficiency / power only.
- **Model B** (realistic): each segment's TRUE difficulty = `s_mean + N(0, s_sd)`
  — the bank value is a noisy *estimate*; the rater answers to the true value,
  the engine works from `s_mean`. Measures ACCURACY under estimation noise.
  This is the model the recommendation is read from.

**Power metric.** Resolution rate among raters who SHOULD resolve — clearly
above or below the cut scores (skill ∈ {−0.5, 0, 0.8, 1.2}). The borderline
cohort (skill 0.4, sitting on the cut) is excluded because REFER is the correct
outcome there, so counting it would conflate "resolved a clear rater" with
"wrongly resolved a borderline one."

## Result (Model B, collapsed over the expert filter)

| `n_raters ≥` | pool | power (clear-rater resolution) | max false-verdict | skill err |
|---|---|---|---|---|
| 1 | 89,138 (2.3×) | 41% | 5.4% | 0.36 |
| 2 | 80,706 (2.1×) | 49% | 8.7% | 0.36 |
| 3 | 61,489 (1.6×) | 49% | 7.7% | 0.36 |
| **5 (current)** | **38,632** | **61%** | **7.5%** | **0.34** |
| 8 | 37,034 (0.96×) | **66%** | 7.3% | 0.35 |
| 10 | 34,781 (0.90×) | **69%** | 7.5% | 0.34 |

## Findings

1. **Power rises monotonically with `n_raters`, with a significant knee at `≥5`**
   (49% → 61%, ~3σ at 160 clear sessions/level). The current bar is well-placed.
2. **`n_raters ≥ 8` and `≥ 10` genuinely beat `≥ 5` on power** (66% / 69% vs
   61%) at trivial pool cost (37k / 35k vs 38.6k). Nearly every ≥5-rater
   segment also has ≥8, so the pool barely shrinks while signals tighten. The
   `≥8` gain is directionally clear but soft (within ~4% sampling noise; the
   knee at `≥5` is the hard result).
3. **Accuracy is filter-insensitive.** False-PASS/FAIL on clear raters sits
   ~5–9% across every filter — the filter moves *resolution* (power), not the
   error rate.
4. **The expert-tier filter buys nothing consistent.** `n_expert ≥ 1/2/3` shows
   no monotonic benefit in either noise model; it only shrinks the pool. Do not
   impose it.
5. **The size lever is real but costly.** Relaxing to `≥3` (61k, 1.6× the bank)
   drops power ~12 points; `≥1` (89k, 2.3×) drops it ~20. The binding limiter is
   signal-estimation noise (`s_sd`), which only more raters-per-segment fixes
   (a Path-B labeling lever), not filtering.
6. **Model B costs ~15–20 resolution points vs Model A** at every filter — the
   inherent price of estimation noise. Stricter filters recover part of it,
   which is exactly why power tracks `n_raters`.

## Recommendation (accuracy-first, then size)

- **Best power at ~no cost: `n_raters ≥ 8`, no expert filter** — same ~37k pool
  as today, +5 resolution points over `≥5`. A free upgrade.
- **For a larger marketable bank: `n_raters ≥ 3`** (61k, 1.6×) is the most
  defensible relaxation — 49% resolution, accuracy intact. Below that power
  degrades sharply.
- **Do not filter on expert tier.**

Raw per-cell results: `results/sim_v1_3_5/bank_quality_rows.csv`.
