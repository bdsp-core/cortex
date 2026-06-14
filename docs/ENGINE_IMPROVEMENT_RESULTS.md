# Engine Statistical-Power Improvement — Results Ledger

> Running record of the sequential engine/policy improvement program (Eli
> direction, 2026-06-10). Each step: a drift-guarded, default-OFF change,
> validated against the OC baseline through the **live** `CortexSession +
> AD6Policy`. See `docs/PROJECT_CORPUS.md` §14 (latest entry) for context and
> `docs/HOW_THE_TEST_WORKS.md` for the algorithm.
>
> **Validation instrument:** `sim_v1_3_5/run_oc_validation.py` — synthetic
> raters of known true skill `ℓ = ℓ*_k + offset` run through the live engine.
> Metrics: per-offset P(PASS/FAIL/REFER), clear-case false-PASS / false-FAIL
> (|offset| ≥ 0.30), median session length, % capped.
>
> **Shipped config under test:** N=600, ESS-frac 0.5, N_MH=15,
> AD6 N_MIN=20 / R\*=0.30 / α=0.05 / Z=2.0. Harness MAX_Q=300.
> Iterate at `--alphas 0.05 --per-tasks 60 --n-per-cell 30` (270 sessions).
> Same flags+npc ⇒ identical seeds ⇒ paired comparison vs baseline.

---

## Step 0 — Baseline (UNMODIFIED shipped engine)

Run: `python sim_v1_3_5/run_oc_validation.py --n-per-cell 30 --procs 44 --alphas 0.05 --per-tasks 60`
(270 sessions, 549 s wall on 44 cores ≈ 2.0 s/session wall).

| offset (ℓ−ℓ\*) | PASS% | FAIL% | REFER% | q_med | capped% |
|---:|---:|---:|---:|---:|---:|
| −1.00 |  0.0 | 97.6 |  2.4 | 171 | 16.7 |
| −0.60 |  0.5 | 71.9 | 27.6 | 300 | 90.0 |
| −0.30 |  1.0 | 30.0 | 69.0 | 300 | 100.0 |
| −0.15 |  2.4 | 18.6 | 79.0 | 300 | 100.0 |
| +0.00 |  4.3 |  6.7 | 89.0 | 300 | 100.0 |
| +0.15 |  5.7 |  3.3 | 91.0 | 300 | 100.0 |
| +0.30 | 15.7 |  0.5 | 83.8 | 300 | 100.0 |
| +0.60 | 37.1 |  0.0 | 62.9 | 300 | 100.0 |
| +1.00 | 51.0 |  0.0 | 49.0 | 300 | 96.7 |

- **Clear-case false-PASS = 0.48%**, **false-FAIL = 0.16%** (error control already strong).
- **Resolution is the deficiency:** +0.6 candidate PASSes only 37% (63% REFER).
- Per-domain @ +0.6 (PASS%/REFER%): spike 17/83 · sz 60/40 · lpd 30/70 · gpd 40/60 ·
  lrda 30/70 · grda 33/67 · iic 50/50. **Spike is the structurally weak task.**

**Headline:** the engine over-refers; the goal is to convert REFER→PASS for
genuinely-skilled candidates **without** raising false-PASS above ~0.5%.

---
## Step 1 — Rao-Blackwellized π_k

### Step 1a — kernel smoothing (NEGATIVE RESULT, rejected)

First attempt: `g_i = Φ((ℓ_i−ℓ*)/h)`, `h = pi_smooth·sd_post·ESS^(−1/5)`,
`π_k = Σ w_i g_i`. Run at `pi_smooth=1.0` (identical seeds → paired vs Step 0):

| offset | PASS% base→s1a | REFER% base→s1a |
|---:|---|---|
| +0.30 | 15.7 → 14.3 | 83.8 → 85.2 |
| +0.60 | 37.1 → **25.7** | 62.9 → 74.3 |
| +1.00 | 51.0 → 47.6 | 49.0 → 52.4 |

false-PASS 0.48%→**0.00%**, false-FAIL 0.16%→0.16%. **Resolution got WORSE.**
**Why:** kernel smoothing convolves the cloud with `N(0,h²)`, *inflating* the
effective posterior variance, which pulls tail probabilities toward 0.5. For a
skilled candidate (true π≈0.97) this biases π *down* ⇒ harder to PASS. The mcse
reduction was real (false-PASS→0) but the first-order π bias dominates. **The
kernel adds variance; we need to reduce estimator noise WITHOUT adding variance.
Rejected.**

### Step 1b — Gaussian-tail RB (the correct estimator)

`π_k = Φ((μ_post−ℓ*)/sd_post)` (uses the actual posterior SD — no inflation),
delta-method `mcse_k = φ(d)·√((1+d²/2)/ESS)`, `d=(μ_post−ℓ*)/sd_post`. Unit-tested:
byte-identical default, `mcse_g < mcse_hard` in the tail, matches the closed-form
tail on a Gaussian cloud (6 tests). OC run (`--pi-estimator gaussian`, paired seeds):

| offset | PASS% base→s1b | REFER% base→s1b |
|---:|---|---|
| +0.30 | 15.7 → 8.1  | 83.8 → 91.0 |
| +0.60 | 37.1 → **21.0** | 62.9 → 79.0 |
| +1.00 | 51.0 → 41.4 | 49.0 → 58.6 |

false-PASS 0.48%→0.16%, false-FAIL 0.16%→0.32%. **Resolution got WORSE again.**
**Why:** the ℓ-posterior is RIGHT-SKEWED for skilled raters (skill = e^ℓ; the data
bound ℓ from below, leaving a fat upper tail). The hard indicator counts that fat
tail's actual mass above ℓ*; the Gaussian approx understates it ⇒ lower π ⇒ harder
to PASS. The `Z·mcse` buffer is only ~0.018 of the bar, so no estimator refinement
can overcome the first-order change in π itself.

### Step 1 — CONCLUSION (confirmed negative result)

Resolution at +0.6 ranks: **hard/shipped 37% > kernel 26% > gaussian 21%.** The
shipped hard indicator is already best; both Rao-Blackwell variants reduce
resolution. **The π_k estimator is NOT a lever** — this rigorously corroborates
the corpus Finding A (the operating frontier is *information-bound*, not
statistic-bound). The resolution deficiency is purely informational ⇒ the real
lever is **item selection (Step 3)**, which adds information per question.

**Action taken:** the engine/policy change was **REVERTED** — `cortex_policy.py`
restored byte-pristine (AD6 + instrument-freeze tests green), the harness
`--pi-estimator` flag and the unit test removed. Kept: this documented finding +
the reusable harness tooling (`--alphas`/`--per-tasks` scope flags + the
progress/ETA reporter). No shipped behavior changed.

## Step 2 — N-particle sweep (calibration / precision / latency)

Context from Step 1: posterior WIDTH is set by the data+prior, not the particle
count, so N is a **calibration/precision** lever (coverage, mcse), not a
resolution lever. Goal: pick a production N trading coverage gain vs GUI latency.

### Latency (live K=7 `CortexSession`, full 700-seg bank, 150 q/session)

`python sim_v1_3_5/bench_n_particles.py --budget 150 --ns 600,1200,2400`

| N | mean ms/q | p50 ms/q | p95 ms/q | select% |
|---:|---:|---:|---:|---:|
| 600  | 119.5 | 11.0 | 441.1 | 91.8 |
| 1200 | 247.9 | 18.1 | 863.0 | 94.1 |
| 2400 | 558.9 | 33.8 | 1845.6 | 96.1 |

- Cost ≈ linear in N (≈2× per doubling). **Even N=2400's p95 (1.85 s) is fine**
  for a GUI where the human answers in seconds — latency is NOT a blocker.
- **Item selection is 92–96% of per-question cost** (full-bank EV scan over
  ~600 IIIC candidates × N particles) — the session uses full-grid `choose_item`
  (no `n_subsample`). Directly relevant to Step 3.
- mean ≫ p50: the cheap spike-phase questions (small bank, ~11 ms) pull the
  median down; the expensive IIIC-phase questions set the mean/p95.

### Coverage / SBC (AUROC CI empirical coverage)

`run_coverage_validation.py --N-particles {600,1200,2400} --n-raters 200 --max-q 200`
(K=6 Mode-A — the validated SBC instrument; draws from the Corr_l prior, runs to
a fixed 200-q budget, measures fraction of true AUROC inside each credible band):

| N | 50% | 80% | 90% | 95% |
|---:|---:|---:|---:|---:|
| 600  | 0.472 | 0.771 | 0.887 | **0.932** |
| 1200 | 0.479 | 0.786 | 0.882 | **0.946** |
| 2400 | 0.496 | 0.805 | 0.900 | **0.940** |

(MC SE ≈ 0.006 per cell: 200 raters × 6 domains.)

### Step 2 — CONCLUSION + recommendation

- N=600 mildly **under-covers** (95% CI = 0.932, ~3 SE low) — matches the corpus
  "sub-0.95 at N=600 = finite-particle MC error" finding. **N=1200 restores
  nominal** (0.946); N=2400 adds only marginal gain (95% tied with 1200 within MC
  noise) at 2× latency.
- mcse ∝ 1/√N ⇒ doubling N cuts the buffer ~29%, moving the effective PASS bar
  only ~0.968→0.963 — **N is a calibration lever, not a resolution lever**
  (consistent with Step 1).
- **Recommendation: N=1200** — nominal CI calibration (more trustworthy Type-1/2
  control) at comfortable latency (mean 248 ms, p95 863 ms ≪ human answer time).
  **NOT flipped unilaterally:** changing shipped `N_PARTICLES` (session_controller.py:72)
  alters the instrument + bit-exact reproducibility and trips `instrument_freeze`
  — Eli/PI sign-off required. Caveat: coverage measured on the K=6 Mode-A
  instrument (the trusted SBC harness); the calibration property generalizes to
  the K=7 live AD6 path but absolute numbers are K=6.

## Step 3 — Decision-aligned item selection

Shipped selector = A-optimal: minimise expected `Σ_k Var(θ_k) + Σ_k Var(ℓ_k)`
(all 14 traits). The verdict depends only on `π_k = P(ℓ_k>ℓ*_k)`, so half the
budget (the 7 biases θ) and the skills of already-decided tasks are "wasted".

### Step 3a — indicator-variance objective (NEGATIVE, rejected)

Minimise expected `Σ_{k∈pending} π_k(1−π_k)` (`_expected_decision_loss_vec`).
Full OC (npc=30, paired seeds):

| offset | PASS% base→3a | REFER% base→3a |
|---:|---|---|
| +0.30 | 15.7 → 1.4  | 83.8 → 98.6 |
| +0.60 | 37.1 → **7.6** | 62.9 → 92.4 |
| +1.00 | 51.0 → 38.6 | 49.0 → 61.4 |

false-PASS 0.48%→0.16%, false-FAIL 0.16%→0.00%. **Resolution collapsed.**
**Mechanism — the π=0.5 barrier:** π(1−π) is symmetric, maximal at π=0.5. The
prior pass-mass is π≈Φ(−ℓ*)≈0.35 (*below* 0.5); a skilled rater's π must climb
0.35→0.97 *through* the 0.5 max. A myopic minimiser of π(1−π) RESISTS pushing π
through 0.5 (it transiently raises the objective) ⇒ stalls below → REFER. FAILs
(π:0.35→0, no barrier) resolve fine — exactly the observed asymmetry (PASS side
collapses, FAIL side intact; worse at moderate skill, less-bad at +1.0 where the
strong signal overcomes the barrier). **Indicator/entropy acquisition has a
0.5 barrier — wrong objective for sequential resolution. Rejected.**

### Step 3b — ℓ-only variance objective

Keep variance reduction (no 0.5 barrier — tightens the ℓ-posterior so π flows to
the truth) but DROP the nuisance θ: minimise expected `Σ_k Var(ℓ_k)` only. Should
match-or-beat the shipped trace for resolution by not spending budget on bias.

Full OC (npc=30, paired seeds):

| offset | PASS% base→3b | REFER% base→3b |
|---:|---|---|
| +0.30 | 15.7 → 14.3 | 83.8 → 84.3 |
| +0.60 | 37.1 → **29.0** | 62.9 → 70.5 |
| +1.00 | 51.0 → 46.7 | 49.0 → 52.4 |

false-PASS 0.48%→0.00%, false-FAIL 0.16%→**0.95%**. **Slightly WORSE.**
**Mechanism — θ–ℓ coupling:** in the likelihood `z = e^ℓ(s+θ)`, skill and bias
are entangled; you cannot pin ℓ without co-estimating θ. Dropping the θ-variance
term starves the bias estimate, which inflates effective ℓ-uncertainty. The θ
effort the shipped A-optimal trace spends is NOT wasted — it buys ℓ-precision
through the coupling.

### Step 3 — CONCLUSION (shipped selector is near-optimal)

Resolution at +0.6: **shipped A-optimal 37% > ℓ-only 29% > indicator 7.6%.** Both
principled alternatives LOSE to the shipped selector. **Item-selection
reorganization is NOT the resolution lever** — the shipped A-optimal design is
already near-optimal for this coupled SDT problem. Combined with Steps 1–2, this
rigorously establishes (by direct OC experiment on the live engine) that
**resolution is information/calibration-bound, not an engine-math problem**
(confirms corpus Finding A). The remaining levers are genuinely PI
calibration/policy (per-pattern ℓ*, α, all-7 → tiered, more questions/bank), not
engine internals.

**Code status:** the Step-3 surface (`objective` on `choose_item`,
`selection_objective` on `CortexSession`, `_expected_decision_loss_vec`,
`include_theta`) is **default-OFF and byte-identical** when unused — proven by 29
regression tests incl. `instrument_freeze`, plus the choose_item drift guards.
Kept as a drift-guarded, reproducible research hook (matches the estimation-ladder
precedent; tagged for §17 publication-cleanup). Shipped behavior UNCHANGED.

## Overall conclusion (Steps 0–3)

The CORTEX engine **math is already well-tuned**. Error control is excellent
(false-PASS ~0.5%, false-FAIL ~0.2%). The deficiency is **resolution** (skilled
raters over-REFER), and it is **information/calibration-bound**: neither the π_k
estimator (Step 1), more particles (Step 2), nor item-selection objective
(Step 3) moves it. The path to production resolution is the **PI
calibration/policy decisions** (corpus §11 / §15): per-pattern ℓ* recalibrated to
achievable expert performance, the PASS α, and replacing all-7-clean-PASS with a
tiered/per-pattern certification.

## Production-config validation — shipped vs v15 (definitive)

Submission-grade OC: current shipped (v14 ℓ*, corr_l, N=600) vs proposed **v15**
(credentialed ℓ*, corr_t, N=1200), both at **MAX_Q=500, npc=100** (bank-limited to
~420 q at per_task=60), zero-bias. CSVs: `results/calibration_study/oc_prod_{shipped,v15}.csv`.

| metric (clear-skill) | shipped | **v15** | p |
|---|---|---|---|
| PASS@+0.30 | 29.1% [25.9–32.6] | **48.1% [44.5–51.8]** | 3e−13 |
| PASS@+0.60 | 55.4% [51.7–59.1] | **76.9% [73.6–79.8]** | 3e−17 |
| PASS@+1.00 | 71.6% [68.1–74.8] | **87.0% [84.3–89.3]** | 1e−12 |
| intrinsic false-PASS (vs own cut) | 1.14% [0.77–1.69] | **0.29% [0.11–0.73]** | ↓ |
| intrinsic false-FAIL | 1.14% [0.77–1.69] | **0.43% [0.24–0.80]** | ↓ |
| q-to-resolution @+1.0 (median, n resolved/100) | 276 (6) | **215 (22)** | faster |

**v15 dominates on every axis: +15–21pp resolution (all p<1e−12), LOWER false-PASS
AND false-FAIL (N=1200 calibration + well-placed credentialed cut), and faster
resolution.** (Wilson 95% CIs; errors scored vs each arm's OWN decision cut.)
⚠ Notes: bank-limited at ~420 q (the live 700-seg bank would lift both arms'
absolute resolution); shipped here reads higher than earlier runs (55 vs 37 @+0.6)
only because MAX_Q=500 (≈420 effective) > the earlier MAX_Q=300; zero-bias (the
no-collapse-under-realistic-bias result is established separately). **This is the
manuscript-grade evidence: the proposed v15 instrument is unambiguously better
than the current shipped one.** Adoption remains the post-pilot re-freeze
(`docs/POST_PILOT_PRODUCTION_INSTRUMENT.md`), gated on pilot closure + sign-off.

### Overnight v15-rigor campaign (decomposition + robustness + coverage)

Full report: `results/overnight_v15/FINDINGS.md` (8 jobs, ~8.75 h). Headlines:

**Ablation — Δ(PASS@+0.6) vs shipped (npc=100, zero-bias):** **+ℓ\* = +17.4pp**
(the resolution lever, ~80% of the gain; also lowers both errors), **+N1200 =
+2.7pp** (mainly calibration/error-control: false-PASS 1.14→0.19%), **+corr_t =
−0.4pp** (resolution-neutral zero-bias; lowers false-PASS, +5pp @ high skill);
**FULL v15 = +21.4pp** (mild super-additive), intrinsic false-PASS 0.29% /
false-FAIL 0.43%.

**Realistic-bias robustness (correct-sign θ≈−1.7):** ~12–15pp cost for BOTH
instruments (real, NO collapse); **v15 keeps +18.4pp over shipped** (62.0% vs
43.6% @+0.6), errors <0.5%.

**Live-bank absolute resolution** (per_task=100 ≈ 700-seg live bank): **v15 =
62/88/94% PASS @ +0.3/+0.6/+1.0** (bank-limited per_task=60 understated it).

**SBC coverage at production N:** **N=1200 achieves NOMINAL** (95% CI coverage
0.956 vs N=600's 0.932; N=2400 = 0.950, no further gain) ⇒ N=1200 = calibration
sweet spot, credible intervals calibrated.

**Net:** v15 is decomposed, robustness-checked, and calibration-validated —
~88–94% resolution for skilled raters at false-PASS 0.29% / false-FAIL 0.43%,
robust to examinee bias, calibrated at N=1200. Remaining (supervised):
K=7+corr_t coverage; **real-pilot-response replay** (external-validity
circularity-breaker); pilot θ for the EB-centering upside.

## Step 4 — Bias-block Σ_θ refit

**Finding:** the engine FITS a bias correlation `Corr_t` but IGNORES it —
`session_controller.py:434` reused the SKILL `Corr_l` for the θ-block (historical
"Sigma_t=Sigma_l symmetry"). The empirical θ is also strongly non-zero (IIIC mean
~+1.7, 60–73% CLIPPED at the +2 fit bound, SD ~0.4) vs the prior N(0,unit). θ is a
nuisance (verdict depends on ℓ), so the lever is efficiency via better bias pooling.

Three variants, default-OFF + byte-identical (`CortexSession(bias_prior=…,
sigma_t_override=…, t_prior_mean=…)`; engine `t_prior_mean` mirrors `l_prior_mean`):

| variant | examinee θ | PASS@+0.6 | false-FAIL | verdict |
|---|---|---:|---:|---|
| baseline (corr_l) | 0 | 37.1% | 0.16% | shipped |
| **corr_t** | 0 | **41.0%** | 0.95% | **sig. gain (p=0.008)** |
| EB (Σ_t+μ_t) | 0 | 10.5% | 3.02% | craters |
| baseline (corr_l) | ~pop | 5.7% | 1.27% | — |
| EB (Σ_t+μ_t) | ~pop | 4.8% | 0.63% | no gain (p=0.83) |

**corr_t (fitted bias correlation, zero-mean):** significant resolution gain
(p=0.008), no significant error change, no bias-location assumption, uses the
matrix the engine already computes. **→ the Step-4 recommendation** (confirm at
larger n; adopt post-pilot with the recalibration).

**Full-EB (corr_t + fitted scale + population mean): REJECTED** (sandbox
`sandbox_bias_prior/`). It craters when examinees are unbiased and gives NO
resolution gain even on its ideal population. Its value hinged on the unknowable
"examinees ≈ rater population" assumption + the censored (clipped) μ_t. Per the
pre-set rule, fall back to corr_t-only. The engine `t_prior_mean` hook + sandbox
are retained default-OFF (byte-identical, drift-guarded) to RE-TEST with real
pilot bias data; §17 cleanup candidate.

**⚠ MAJOR new risk (pilot-gated):** if real examinees are biased like the training
raters (θ~+1.7), **resolution COLLAPSES ~7× (37%→5.7% @ +0.6, p≈0) regardless of
prior** — saturated high-bias responses carry little ℓ-information. This dwarfs the
bias-prior choice, would undercut the zero-bias resolution claims in Steps 0–3 +
the recalibration, and is tied to the +2 θ-clipping (possibly a fit artifact). The
in-flight pilot's per-rater θ must be checked FIRST. If examinees are materially
biased, the instrument needs more questions / a bias-aware design — a separate,
higher-priority thread than the bias prior.

### Step 4 — θ-REALISM CORRECTION (supersedes the rejection + risk above)

A θ-realism audit found the per-rater fitter (`fit_sdt_per_domain.py:76`)
hard-bounds θ∈[−2,+2] (IIIC 60–71% censored at +2) AND uses the OPPOSITE sign
convention to the engine: fitter `z=(c−t)/σ` vs engine `z=e^ℓ(s+θ)` ⇒
**θ_engine = −θ_fit**. Real engine-θ: spike +0.48 (small, fine), **IIIC ≈ −1.7**
(large, structural — the one-vs-rest base rate). **The Step-4 EB/population
experiments above used the WRONG SIGN (+1.7), which over-call-saturates responses
and falsely collapsed resolution.** Re-run with the CORRECT sign:

| prior (REALISTIC bias, θ_engine≈−1.7) | PASS@+0.6 | false-FAIL |
|---|---:|---:|
| shipped (corr_l) | 39.5% | 0.32% |
| corr_t | 40.5% | 1.27% |
| **EB (correct-sign, μ=−1.7)** | **52.9%** | 0.48% |

- **Resolution-collapse REFUTED:** shipped under realistic bias 39.5% ≈ zero-bias
  37.1% (p=0.69). The real structural bias preserves the discriminating segments;
  the earlier 5.7% was a sign artifact. **Steps 0–3 + recalibration claims HOLD.**
- **EB rejection was sign-error-driven and is REVERSED to conditional:**
  correctly-centered EB **helps +13pp under realistic bias** (52.9 vs 39.5,
  p=0.008) but craters under zero-bias. corr_t helps under zero-bias (p=0.008),
  neutral under realistic (p=0.92).
- **Net: corr_t-only remains the SAFE default** (robust to the unknown examinee
  bias). A correctly-centered θ prior is a **pilot-gated upside** — revisit IF the
  pilot shows examinees are biased, centering on the PILOT's examinee θ (not the
  censored training fits). Spike fine throughout. The one engine-side recommendation that stands
is **N=1200** (Step 2) for better-calibrated Type-1/2 control, pending Eli/PI
sign-off.

## Calibration-lever demonstration (the REAL resolution path)

Having shown the engine can't move resolution, this quantifies the PI
calibration/policy levers on the LIVE engine (synthetic raters, true ℓ fixed
relative to the ORIGINAL cut; the lever shifts only the DECISION cut / α). All
α=0.05 / per_task=60 / npc=30, vs the Step-0 baseline.

| Lever | PASS@+0.3 | PASS@+0.6 | PASS@+1.0 | false-PASS | false-FAIL | q_med@+0.6 |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** (α=.05, ℓ*+0) | 15.7 | 37.1 | 51.0 | 0.48% | 0.16% | 300 |
| α=0.25 (ℓ*+0) | 74.8 | 88.1 | 92.9 | **6.83%** | **7.62%** | 206 |
| **ℓ*−0.3** (α=.05) | 59.5 | 86.7 | 91.0 | 2.06% | **0.00%** | 264 |
| ℓ*−0.6 (α=.05) | 91.9 | 95.7 | 97.1 | **12.54%** | 0.00% | 191 |

Per-domain PASS% @ +0.6 (clear-pass): baseline spike 7 / IIIC 3–17 →
**ℓ*−0.3** spike 50 / IIIC 87–100. Findings:

1. **Resolution is fully recoverable by calibration** (37%→87–96% @ +0.6) — the
   engine levers (Steps 1–3) could not; **proves resolution was
   calibration-bound.**
2. **Levers are NOT equivalent.** α loosens PASS *and* FAIL symmetrically ⇒
   false-FAIL jumps to 7.6% (bad). Lowering ℓ* is **asymmetric** — only the PASS
   side moves ⇒ false-FAIL stays **0%**, false-PASS rises modestly. **Lowering ℓ*
   dominates loosening α** for this goal.
3. **ℓ*−0.3 @ α=0.05 is the sweet spot:** PASS@+0.6 37%→87%, false-PASS ~2%,
   false-FAIL 0%, faster sessions (q_med 300→264). ℓ*−0.6 over-shoots
   (false-PASS 12.5%).
4. **Spike stays weak** in every arm (50–70% vs 90–100% IIIC) ⇒ needs a
   spike-specific remedy (even-lower spike ℓ*, more spike questions, or
   REFER-not-FAIL) — confirms §11/§15 Finding C.

⚠ Caveats: synthetic raters on the engine's own generative model; uniform ℓ*
shift is a DEMONSTRATION — the real fix is a **per-pattern** ℓ* recalibrated from
an expert cohort (§11). "false-PASS" is scored vs the ORIGINAL cut, so the ℓ*−0.3
arm's 2% is slightly inflated (some "clear-fail" raters are borderline vs the
NEW cut). MAX_Q=300 here (live=500). n=30/cell (Wilson CI ≈ ±a few %).

## PI-facing recommendation (Step 5)

The engine math is sound and should NOT be the focus before production. To hit
the resolution + error targets:
1. **Recalibrate ℓ* per pattern** from an expert cohort, targeting achievable
   expert AUROC (the demonstration says ~−0.3 in log-skill is roughly the right
   magnitude for IIIC; spike needs more). This is the highest-value action and
   keeps false-FAIL ≈ 0.
2. **Keep α at 0.05** (do NOT loosen — it inflates false-FAIL).
3. **Give spike a dedicated remedy** (lower spike ℓ*, longer spike section, or
   REFER-not-FAIL on spike).
4. **Replace all-7-clean-PASS** with per-pattern / tiered certification (§11).
5. **N=1200** for better-calibrated intervals (Step 2), pending sign-off.
6. A real tiered human pilot remains the ultimate validation (§11 / §15).

## Expert-panel recalibration study (items 1 & 3) — NON-DESTRUCTIVE

Premise (Eli): a panel of *certified world-class experts* may give a better /
more rigorous ℓ* than the shipped `ell_star_unified_v14`, which selects its
14-expert panel **by data-driven skill** (top-14 by cross-task mean ℓ from a
29-rater pool) — *not* by credentials. Script:
`pipeline/reference_calibration/expert_panel_recalibration_study.py` (reuses the
byte-pinned Youden machinery; swaps only the panel-selection rule; writes only to
`results/calibration_study/`; **changes nothing shipped** — adopt only if proven
better, else ignore = revert). P0_cv14 reproduces the shipped ℓ*/J EXACTLY
(validation). Credentialed panels: Super8 (8 foundational epileptologists),
Bonobo (15 held-out experts), union (23), all from `raters.csv:groups`.

| task | shipped (P0) ℓ*/J | credentialed ℓ*/J | Δℓ* |
|---|---|---|---|
| spike | 0.325 / 0.649 | **Super8** 0.238 / **0.693** | −0.087 |
| sz | 0.256 / 0.734 | S8∪Bon 0.155 / 0.669 | −0.101 |
| lpd | 0.534 / 0.866 | S8∪Bon 0.186 / 0.732 | **−0.348** |
| gpd | 0.330 / 0.822 | S8∪Bon 0.257 / 0.804 | −0.073 |
| lrda | 0.479 / 0.807 | S8∪Bon 0.321 / 0.783 | −0.158 |
| grda | 0.486 / 0.779 | S8∪Bon 0.354 / 0.761 | −0.133 |
| iic | 0.442 / 0.814 | S8∪Bon 0.304 / 0.780 | −0.138 |

**Findings:**
1. Credentialed panels give a **0.07–0.35 LOWER ℓ*** everywhere — the shipped
   skill-maximal top-14 sits mechanically high; a credentialed (exogenous,
   non-circular) panel is the gold-standard "expert" definition for a regulated
   instrument AND yields a more achievable cut. This is the **principled
   justification** for the resolution-recovering lower ℓ* the lever-demo found.
2. **Spike (item 3): Super8 is strictly better** — higher J (0.649→0.693), lower
   ℓ* (0.325→0.238), and full 8/8 credentialed coverage vs the shipped panel's
   9/14 (5 picked on IIIC skill alone). Clean win.
3. IIIC: J is modestly lower (0.78–0.80 vs 0.81–0.87) but ℓ* is far more
   achievable; the J drop reflects a less-elite (more representative) panel, not
   worse calibration. Defensibility + achievability favor credentialed.

Recommended set (spike←Super8, IIIC←S8∪Bonobo):
`results/calibration_study/recommended_ell_star_final.json` (the study's `recommended_ell_star.json` was deleted 2026-06-11; this OC table used its PRE-TRIM lpd ℓ\* .186 — the surviving file holds the robust-trimmed .306; canonical reproducible source `ell_star_v15.json`). **Downstream OC proof
(does the credentialed ℓ* recover resolution at controlled false-PASS):**

OC with `--ell-star-json recommended_ell_star_final.json` (decision cut = credentialed
per-pattern ℓ*; true skill stays on the shipped ℓ*; paired seeds vs baseline):

| offset | PASS% base→cred | false-PASS | false-FAIL |
|---:|---|---|---|
| +0.30 | 15.7 → 37.6 | — | — |
| +0.60 | 37.1 → **67.6** | — | — |
| +1.00 | 51.0 → **77.6** | — | — |
| clear-case | — | 0.48% → **1.75%** | 0.16% → **0.00%** |

**Proof:** the credentialed per-pattern ℓ* **nearly doubles** skilled-rater
resolution (37→68% @ +0.6; 51→78% @ +1.0), **zeroes false-FAIL**, and keeps
false-PASS < 2% (an UPPER bound — false-PASS is scored vs the ORIGINAL cut, so
some "clear-fail" raters are competent vs the new credentialed cut). Less
aggressive than the uniform ℓ*−0.3 demo (87% @ +0.6, fP 2.1%) because the
per-pattern shifts average ~−0.15; but it is **principled** (the cut where
credentialed world-class experts separate from the field) rather than an
arbitrary offset. **Statistically + methodologically better than the shipped
skill-selected calibration, and the defensible choice for the regulated
submission.**

### Calibration study — CONCLUSION + adoption gate

**Item 1 answered: YES**, a credentialed panel produces a better ℓ* (more
rigorous, non-circular, more achievable; ~2× resolution at fP<2%, fF=0).
**Item 3 answered:** spike's Super-8 panel is strictly better (J↑, ℓ*↓, full
coverage). Recommended set: `results/calibration_study/recommended_ell_star_final.json`.
**Nothing shipped changed.** Adopting it is a POST-PILOT instrument change (new
`ell_star_unified_v15` block + re-freeze), gated on PI sign-off + pilot closure —
see `docs/POST_PILOT_PRODUCTION_INSTRUMENT.md`. Revert path: discard the study
outputs; the shipped v14 calibration is untouched. ### Rigor analysis (R1/R2/R3 — `pipeline/reference_calibration/panel_rigor_analysis.py`)

**R1 — paired cluster-bootstrap ΔJ (credentialed − shipped P0), 2000 reps:**

| task | panel | ΔJ | 95% CI | verdict |
|---|---|---:|---|---|
| spike | Super8 | +0.045 | [−0.086,+0.192] | comparable-or-better (NS) |
| sz | S8∪Bon | −0.082 | [−0.187,−0.007] | sig. lower J |
| lpd | S8∪Bon | −0.102 | [−0.159,−0.008] | sig. lower J |
| gpd | S8∪Bon | −0.013 | [−0.049,+0.022] | comparable (NS) |
| lrda | S8∪Bon | −0.024 | [−0.040,+0.007] | ~comparable |
| grda | S8∪Bon | −0.018 | [−0.040,+0.003] | ~comparable |
| iic | S8∪Bon | −0.024 | [−0.065,+0.010] | comparable (NS) |

**R1b — stability (ℓ* bootstrap-CI width):** spike Super8 0.110 (vs P0 0.241 — MORE
stable); most IIIC comparable; ⚠ **lpd credentialed CI width 0.588 (vs P0 0.126) —
HIGHLY UNSTABLE.** Cause: among credentialed experts lpd skill is heterogeneous
(one ℓ=−0.08 outlier, a gap, then a 0.5–0.84 cluster) so the Youden cut jumps —
i.e. even world-class experts genuinely vary on lpd. **Do NOT adopt lpd=0.186
as-is.**

**R2 — clean error scoring (credentialed OC, vs the DECISION cut not the original):**
intrinsic **false-PASS 0.77%** (3/390, Wilson [0.26%,2.24%]), **false-FAIL 0.14%**
— the 1.75% was an overstatement (raters competent vs the lowered cut were scored
as fails vs the original cut). So the credentialed calibration is **~2× resolution
at fP 0.77% / fF 0.14%** — well-controlled.

**R3 — per-pattern vs uniform (data-decided):** Super8-everywhere fails (n=6 for
IIIC, lpd ℓ* goes negative); union-everywhere works for IIIC but spike J 0.610 <
Super8's 0.693. **Per-pattern (spike←Super8, IIIC←union) is data-superior** — the
spike-specific remedy (item 3) is genuinely warranted, not just narrative.

**Net:** the credentialed recalibration is rigorously justified for spike (strictly
better) and most IIIC (comparable J, far more achievable ℓ*, fP<1%), with **lpd
flagged for a robust re-derivation** and three PI/opinion calls pending
(lpd handling, per-pattern-vs-uniform narrative, the competence-standard choice).
Re-derive on the FINAL panel + production OC (N=1200) at adoption (post-pilot).

### FINAL validated recalibration (PI-confirmed 2026-06-10)

PI decisions: competence standard = credentialed-vs-field separation; per-pattern
(spike←Super-8, IIIC←Super-8∪Bonobo); lpd robust re-derivation. Final set
(`results/calibration_study/recommended_ell_star_final.json`): spike .238, sz .155,
**lpd .306 (robust)**, gpd .257, lrda .321, grda .354, iic .304.

Definitive OC (paired vs baseline, N=600):

| offset | PASS% base→final | 
|---:|---|
| +0.30 | 15.7 → 35.7 |
| +0.60 | 37.1 → **66.2** |
| +1.00 | 51.0 → **77.1** |

**Intrinsic (vs decision cut) false-PASS 0.71%** (3/420, Wilson [0.24,2.08]),
**false-FAIL 0.14%.** The robust lpd (0.306) preserved the resolution gain
(67.6→66.2% @ +0.6) while being stable. Per-domain @ +0.6: spike 23%, IIIC 67–80%
— **spike improved (~7–17%→23%) but stays the weakest** even with the Super-8 panel
+ lower cut ⇒ spike's residual gap is structural (binary task); a fuller spike
remedy (more spike questions / spike-section length / REFER-not-FAIL) may be
warranted ON TOP of the recalibration, but that is a separate instrument-policy
item. **Recalibration RIGOROUSLY COMPLETE + justified.** Adoption = post-pilot
`ell_star_unified_v15` + re-freeze (with a final N=1200 re-derivation +
pilot data) — see `docs/POST_PILOT_PRODUCTION_INSTRUMENT.md`. Nothing shipped
changed; revert = discard `results/calibration_study/`.





