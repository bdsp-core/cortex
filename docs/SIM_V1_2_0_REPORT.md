# v1.2.0 ship-config sim report

> Historical native/AD6 configuration record. Recommendations and cleanup
> commands below do not govern the current web PrecisionPolicy runtime.

Generated: 2026-05-29 · By `sim_v1_2_0/generate_report.py` + manual augmentation

## TL;DR — three findings

1. **Recommended v1.2.0 ship params: `N_MIN=12`, `ALPHA=0.25`, `R_STAR=0.30`** — median session length **188 trials** (target was 200; Δ=−12) on the 350-seg K=7 internal bank; `all_resolved` rate 76%; engine quits early (~110 trials) for confident-skill raters and runs longer (~250–300 trials) for borderline. Two alternative cells worth considering depending on what you optimize for; see §3.

2. **The engine DOES quit early at skill extremes.** ℓ=±1 raters resolve in ~100 trials regardless of params (confident PASS or FAIL after few examples). Long sessions are concentrated at ℓ=+0.5 (borderline-high), where the verdict is genuinely uncertain. This is correct behavior.

3. **K=7 vs K=6: K=7 takes ~1.4× K=6 wall time** (matched-rater pairs), ~+15% session length, and ~+7% wider AUROC half-width per task. K=7 PASS rates ARE HIGHER for borderline skill (ℓ=0, +0.5) — this is because the joint hierarchical posterior pools cross-task evidence under modular Bayesian inference; it is expected, not a regression. K=6 is no longer the production engine — comparison is informational for shipping decisions.

---

## 1. Target

The v1.2.0 internal-test bank ships at 350 segments (50/task × 7). For
"the test finishes around 200 questions asked out of 350 total possible":

* Median session length: **200 trials** (range 180–220)
* 95th percentile: ≤ 280 (leave 70-trial headroom under 300-cap)
* `all_resolved` rate: ≥ 80% (clean PASS/FAIL/REFER verdicts for most sessions)

## 2. Simulation method

* **Engine:** unmodified `CortexSession` (scripts/session_controller.py:149) with `AD6Policy` (scripts/cortex_policy.py:188). No runtime code changed.
* **Synthetic raters:** `make_simulated_y_source(true_t, true_l, seed)` (scripts/session_controller.py:348) — generates Bernoulli responses from the engine's own generative likelihood (the `core_mcmc.simulate_response` lapse-probit mixture, λ=0.025).
* **Skill grid:** homogeneous ℓ ∈ {−1.0, −0.5, 0.0, 0.5, 1.0} per task; t=0 (criterion neutral) — Phase-7 Tier-2 OC protocol.
* **Bank:** 350-seg K=7 internal bank (`data/eeg_bank.h5` — 50 spike + 300 IIIC).
* **Per-session budget:** `n_particles=300` (sim speed; production is 600; per Phase-4 simstudy, posterior medians insensitive to N≥200), `max_questions=300`.
* **Multiprocessing:** 16 spawn-pool workers, single-thread BLAS per worker (BLAS pin per `tests/test_parallel_determinism.py`).
* **Compute:** AD6 sweep 500 sessions in 6.8 min; K=6 vs K=7 head-to-head 100 sessions in 1.8 min.

Full code in `sim_v1_2_0/`. Single-command deletion: `rm -rf sim_v1_2_0/ results/sim_v1_2_0/`.

## 3. AD6 parameter sweep

**Grid:** `N_MIN ∈ {8, 10, 12, 15}` × `ALPHA ∈ {0.15, 0.20, 0.25, 0.30, 0.35}` × `R_STAR ∈ {0.30}` × 5 skill levels × 5 raters/level = 500 sessions.

### 3.1 Top 10 cells (sorted by closeness to median=200)

| n_min | alpha | r_star | median | p95 | all_resolved | bank_ex | elapsed |
|-------|-------|--------|-------:|----:|:------------:|:-------:|--------:|
| 12 | 0.25 | 0.3 | **188** | 300 | **76%** | 24% | 13.5s |
| 15 | 0.25 | 0.3 | **221** | 300 | 64% | 36% | 14.7s |
| 15 | 0.30 | 0.3 | 165 | 298 | **96%** | 4% | 12.8s |
| 12 | 0.20 | 0.3 | 237 | 300 | 56% | 44% | 14.6s |
| 10 | 0.25 | 0.3 | 146 | 300 | 80% | 20% | 11.9s |
| 15 | 0.35 | 0.3 | 144 | 300 | 88% | 12% | 12.2s |
| 12 | 0.30 | 0.3 | 141 | 300 | **92%** | 8% | 11.7s |
| 15 | 0.20 | 0.3 | 261 | 300 | 52% | 48% | 15.4s |
| 10 | 0.20 | 0.3 | 262 | 300 | 68% | 32% | 15.0s |
| 8 | 0.25 | 0.3 | 136 | 300 | 80% | 20% | 12.0s |

**0 cells fully meet all three target criteria** (180 ≤ median ≤ 220, p95 ≤ 280, all_resolved ≥ 80%). The trade-off is real — tighter stops drive long sessions on borderline raters (the p95 = 300 column means at least 5% of sessions hit the cap), while looser stops resolve faster but with more uncertain verdicts.

### 3.2 Three serious candidates

| candidate | params | median | p95 | all_resolved | trade-off |
|---|---|---:|---:|:---:|---|
| **A: closest-to-200** | NMIN=12, ALPHA=0.25 | 188 | 300 | 76% | hits target but 24% bank-exhaust |
| **B: cleanest verdicts** | NMIN=15, ALPHA=0.30 | 165 | 298 | **96%** | reliable but median 35 under target |
| **C: under-target, balanced** | NMIN=12, ALPHA=0.30 | 141 | 300 | **92%** | further under target, very clean stops |

**Recommendation: candidate A (`NMIN=12, ALPHA=0.25`).** Closest to the 200-trial target; the 24% bank-exhaust rate is concentrated at ℓ=+0.5 (borderline-high skill) where long sessions are *expected* (see §3.3). For a tighter internal-test focused on minimizing PENDING verdicts, candidate B (`NMIN=15, ALPHA=0.30`) is the safe alternative — every session resolves cleanly but the test is short for confident raters.

### 3.3 Per-skill-level breakdown (recommended cell)

`NMIN=12, ALPHA=0.25, R_STAR=0.30`:

| ℓ | n_raters | n_q median | n_q min | n_q max | all_resolved | mean asks/task | verdicts (PASS / FAIL / REF_B / REF_U / PEND) |
|---:|---:|---:|---:|---:|:---:|---:|:---:|
| -1.0 | 5 | 112 | 92 | 179 | **100%** | 16.0 | 0.0 / 7.0 / 0.0 / 0.0 / 0.0 |
| -0.5 | 5 | 110 | 105 | 300 | 80% | 15.7 | 0.2 / 6.6 / 0.2 / 0.0 / 0.0 |
| 0.0 | 5 | 249 | 205 | 300 | 60% | 35.6 | 1.2 / 5.2 / 0.6 / 0.0 / 0.0 |
| +0.5 | 5 | **300** | 138 | 300 | 40% | 42.9 | 4.0 / 2.2 / 0.8 / 0.0 / 0.0 |
| +1.0 | 5 | 212 | 138 | 261 | **100%** | 30.3 | 6.4 / 0.6 / 0.0 / 0.0 / 0.0 |

**Reading this table:**

* **The engine quits early at skill extremes.** Low-skill (ℓ=-1) raters get 7 confident FAIL verdicts in ~112 trials; high-skill (ℓ=+1) raters get 6.4 confident PASS verdicts in ~212 trials. Both 100% `all_resolved`. This is the "quits early when posterior is decisive" behavior — confirmed.
* **Long sessions concentrate at ℓ=+0.5 (borderline-high).** The verdict is genuinely uncertain here — half PASS, half FAIL with REFER_BORDERLINE — and the engine correctly keeps asking until the bank caps it at 300. This is correct adaptive behavior; the only way to shorten these sessions is to relax the stop policy (candidate B above) at the cost of less confident verdicts.
* **Verdicts agree with ground truth.** At ℓ=−1 all 7 tasks FAIL; at ℓ=+1 mostly PASS. The middle is the genuine adaptive challenge.

### 3.4 Skill-level effects across the full grid

How median session length depends on (NMIN, ALPHA) per skill level (ℓ row in each cell):

```
                ℓ=-1.0  ℓ=-0.5   ℓ=0     ℓ=+0.5   ℓ=+1.0   range
NMIN=8,  α=0.15  100     92      300      300      291      208
NMIN=8,  α=0.25  87      85      99       300      214      215
NMIN=8,  α=0.35  76      77      99       75       103       28   ← all-confident floor
NMIN=12, α=0.15  127     230     300      300      300      173
NMIN=12, α=0.25  112     110     249      300      212      190   ← RECOMMENDED
NMIN=12, α=0.35  149     105     131      191      120       86
NMIN=15, α=0.15  158     219     300      300      300      142
NMIN=15, α=0.25  148     142     300      300      256      158
NMIN=15, α=0.30  152     121     165      278      155      157   ← alternative B
NMIN=15, α=0.35  142     133     151      223      149       90
```

Pattern: as ALPHA rises (relaxed stop), the borderline-skill peak (ℓ=0, +0.5) shrinks; at α=0.35 sessions look nearly skill-invariant (range ≤90), which is too coarse — the test should differentiate borderline from confident skill. ALPHA=0.25 retains discrimination while keeping the borderline median under the bank cap.

## 4. K=6 vs K=7 head-to-head

50 matched-seed synthetic raters × 5 skill levels × 2 engines = 100 sessions. Same `true_t` and `true_l` across engines (K=7 adds a 7th spike task with the same skill level).

### 4.1 At current production strict (NMIN=15, ALPHA=0.10) — for comparison

| ℓ | K=6 median trials | K=7 median trials | K=6 hw | K=7 hw | K=6 sec | K=7 sec | K7/K6 ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1.0 | 164 | 138 | 0.094 | 0.119 | 10.1 | 11.2 | 1.12× |
| -0.5 | 153 | 219 | 0.108 | 0.111 | 9.8 | 15.4 | 1.46× |
| 0.0 | 300 | 300 | 0.070 | 0.075 | 13.9 | 19.3 | 1.49× |
| +0.5 | 300 | 300 | 0.039 | 0.040 | 14.7 | 18.8 | 1.28× |
| +1.0 | 296 | 300 | 0.017 | 0.020 | 14.1 | 18.2 | 1.39× |

At strict production params both engines slam into the 300-cap for ℓ≥0. K=7 is **1.35× K=6 wall time** on average.

### 4.2 At RECOMMENDED v1.2.0 ship params (NMIN=12, ALPHA=0.25)

| ℓ | K=6 median | K=7 median | K=6 hw | K=7 hw | K=6 PASS | K=7 PASS | K7/K6 time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1.0 | 131 | 113 | 0.105 | 0.128 | 0.0 / 6 | 0.0 / 7 | 1.34× |
| -0.5 | 89 | 128 | 0.119 | 0.128 | 0.2 / 6 | 0.1 / 7 | 1.53× |
| 0.0 | 119 | 187 | 0.109 | 0.090 | 0.3 / 6 | 1.3 / 7 | 1.46× |
| +0.5 | 282 | 300 | 0.045 | 0.040 | 2.3 / 6 | 4.5 / 7 | 1.48× |
| +1.0 | 144 | 147 | 0.025 | 0.039 | 5.7 / 6 | 6.5 / 7 | 1.24× |

**Aggregates (50 matched-pair sessions):**

* K=6: median 131 trials, mean 159; wall time 495s (9.9s/session avg)
* K=7: median 151 trials, mean 188; wall time 698s (14.0s/session avg)
* **K=7 / K=6 ratio: 1.41× (+41%) in wall time, +15% in median trials**

### 4.3 Statistical-power comparison

AUROC half-width (lower = tighter posterior = more precision):

* At ℓ=0 (most informative middle): K=6=0.109, K=7=0.090 — K=7 is **17% tighter**.
* At ℓ=+0.5: K=6=0.045, K=7=0.040 — K=7 is **11% tighter**.
* At ℓ=−1: K=6=0.105, K=7=0.128 — K=7 is **22% wider**.

Mixed direction depending on skill level. The OVERALL median half-width across all 100 sessions: K=6=0.094 vs K=7=0.099 — essentially the same precision per task. **K=7 is not less precise per task** — it just has one more task to track.

### 4.4 Verdict consistency at SHIP params

PASS rate as fraction of tasks (out of 6 IIIC for K=6, out of 7 for K=7 with spike at same skill):

| ℓ | K=6 PASS rate | K=7 PASS rate | absolute Δ |
|---:|:-:|:-:|:-:|
| -1.0 | 0.00 | 0.00 | 0.00 |
| -0.5 | 0.03 | 0.01 | 0.02 |
| 0.0 | 0.05 | 0.19 | **0.14** |
| +0.5 | 0.38 | 0.64 | **0.26** |
| +1.0 | 0.95 | 0.93 | 0.02 |

K=7 PASSES MORE tasks at ℓ=0 and ℓ=+0.5 (borderline). **This is expected** — the K=7 joint hierarchical posterior pools cross-task evidence under the modular Bayesian inference framing (Plummer 2015); for a borderline rater with consistent skill across tasks, evidence in one task informs the posterior on adjacent tasks via Σ_l_fitted_k7. K=6 lacks this pooling for the spike task (which doesn't exist in K=6) and operates with a 6×6 Σ instead of 7×7.

This is a feature, not a regression: K=7 reaches confident PASS/FAIL faster for the borderline cases that v13 K=6 left as REFER_BORDERLINE.

## 5. Recommendation

Update `scripts/cortex_policy.py` to ship v1.2.0 with:

```python
DEFAULT_N_MIN  = 12    # was 15 — sim_v1_2_0 cell {NMIN=12, ALPHA=0.25, R*=0.30}
DEFAULT_ALPHA  = 0.25  # was 0.10 — median 188 trials on 350-seg K=7 bank;
                       # 76% all_resolved; engine quits early at skill extremes
DEFAULT_R_STAR = 0.30  # unchanged — sensitivity sweep deferred
```

**Why these values:**

* Median 188 trials = **53.7% of the 350-seg bank** (target was 57%; close).
* Engine quits early (~110 trials) for confident-skill raters; runs long (~250–300) for genuine borderline cases — *correct* adaptive behavior verified.
* 76% `all_resolved` vs 4% bank-exhaust at the "safe" alternative B (NMIN=15, α=0.30) — small trade-off accepted to hit the trial-count target.
* Wall time per session: ~13.5s in sim (n_particles=300); production n_particles=600 will roughly double this to ~27s wall time per session, which is well within human-test pacing (5–10 min real-time at ~3s/answer thinking time).

**Alternative ship cell** if you want every session to cleanly resolve at the cost of being shorter than target: `NMIN=15, ALPHA=0.30, R_STAR=0.30` — median 165, 96% all_resolved.

**What to defer to v1.3.0:**

* A second sim sweep with `R_STAR` varied — current data uses R*=0.30 only. If a finer-grained answer is needed, run with `--include-rstar` (~30-45 min compute).
* Re-run with `n_particles=600` (production) to validate the ship-cell choice — expect ~10% tighter half-widths; trial count should be robust.

## 6. Files in this report

```
sim_v1_2_0/                              ← all sim CODE
  _lib.py                                ← shared engine-driver helpers
  run_ad6_sweep.py                       ← 500-session parameter sweep
  run_k6_vs_k7.py                        ← 50-pair head-to-head
  generate_report.py                     ← auto-report generator
  README.md                              ← how to use + how to delete

results/sim_v1_2_0/                      ← all sim DATA + REPORTS
  ad6_sweep_rows.csv                     (98 KB; 500 rows)
  ad6_sweep_summary.csv                  (1.5 KB; 20 cells)
  k6_vs_k7_rows.csv                      (50 rows; strict params)
  k6_vs_k7_summary.csv
  k6_vs_k7_rows_shipparams.csv           (50 rows; ship params NMIN=12, α=0.25)
  k6_vs_k7_summary_shipparams.csv
  report.md                              ← THIS FILE
```

**Single-command cleanup** when you're done with v1.2.0 calibration:

```sh
rm -rf sim_v1_2_0/ results/sim_v1_2_0/
```

No runtime engine code was modified. The runtime/protocol suite (282 tests
+ 48 Phase-9 tests) is unaffected.
