# EXTSET_SAP_M13 — Statistical Analysis Plan for the EXTSET in-scratch release

**Status:** EXECUTED (M13.3, 2026-06-12). All prespecified endpoints met:

| Endpoint | Result | Verdict |
|---|---|---|
| P1 (task2 ΔELPD vs M0) | +0.0006…+0.0137/read; 13/14 frames CI>0; M3 heavy-tail dominant | PASS |
| P1b (task1 ΔELPD vs M0) | M2, +0.0049 [+0.0037,+0.0062]; fitted λ < engine's 0.025 | PASS |
| P2 (task2 replay validity) | ρ=0.481, p=1.3e-19, n=315 (per-domain 0.67–0.83, Holm-sig) | PASS |
| P2b (task1 replay validity) | ρ=0.868, p=6e-99, n=321 | PASS |
| P3 (real-link stress) | inside envelope; ≈wellspec on all campaign metrics | PASS |
| P4 (cross-domain skill) | ρ=0.571, p=4e-13, n=136; disattenuated 0.771 | PASS |

Deferred secondaries (explicit): M5 softmax; nonstationarity refit; DS/senior
gold sensitivities; shipped-vs-LOO delta table. Gated: W3.2 dynamics (OQ8).
Findings F40–F44; artifacts `data_extset_{link,replay,xdomain,stress}.npz`,
fig9–fig12. Original plan below, unchanged.

---

**Plan as drafted:** v2 (2026-06-12). v1 covered the task2 (K=6) release; v2 adds
the task1/domain1 reads + signals (arrived 2026-06-12), the SIGNAL-CIRCULARITY
finding (F40) and its leave-one-user-out protocol, and workstream W4
(cross-domain skill correlation). Fits start when OQ8 (feedback) lands; W1/W2/W4
are not gated on it. Gold policy (OQ9) RESOLVED: expert ≥3/4 consensus primary.

**Scope anchor (binding, per M11.2c + D18):** EXTSET supports (a) observation-model
grounding, (b) measurement-layer convergent validity, (c) population-prior
grounding. It does NOT validate the closed-loop trainer (no adaptive placement;
feedback status unknown pending OQ8). Trainer-efficacy claims remain pilot-gated.

**Data perimeter (v2):**
- task2: 125,860 valid reads × 699 users × 5,000 cases, 6-class forced choice,
  signals for domains 2–7, 4-expert panel + source gold, timestamps + Qscore.
- task1: 167,503 reads × 643 raters × 5,000 cases, BINARY domain1/non-domain1,
  `s_mean/s_sd_domain1` for all 5,000 segs, source gold ONLY (no expert panel),
  NO timestamps/Qscore. Reads reconcile exactly with the signals-file votes
  (yes/total match 1.000) and with `extset_users` totals up to a snapshot gap
  (+6,438 reads / +40 raters vs the users-table counts; users table is older).
- Dual-contest cohort: 275 users in both; 147 with ≥20 reads in both.
- Truth: task2 = expert ≥3/4 consensus (3,789 cases; sensitivities §1);
  task1 = source gold (only available truth; binary; prevalence 0.298 by case,
  0.239 read-weighted).

---

## 0. The signal-circularity protocol (F40 — governs every analysis below)

**Finding:** `s_mean_k` is not an independent instrument measurement; it is a
crowd-consensus score ≈ affine-probit of the smoothed vote share
(`s ≈ 1.10·Φ⁻¹((yes+0.78)/(n+7.5)) + 1.12`, R²≈0.966 on task1; corr with
probit vote-share 0.93–0.96 on task2 EXTSET segs, 0.80 on the bank's own
domain1 pool; partial corr given gold 0.95). On EXTSET segments the votes ARE
the EXTSET reads (+ 4 experts on task2; F35) — so the shipped signal contains
every analyzed user's own responses. Quantified leverage: per-read LOO signal
shift |Δs| median 0.016, p90 0.048, max 0.41; mean shift is SELF-CONFIRMING
(+0.026 toward the user's own response; mechanically guaranteed > 0).

**Protocol (mandatory for all confirmatory fits):**
1. **Primary signal = exact leave-one-user-out consensus score**, computed from
   the read-level data: `s^(−u)_k = c·Φ⁻¹((yes−y_u·1[u voted]+a)/(n−1[u voted]+a+b)) + d`
   with (a,b,c,d) FIXED at the task1-recovered values above (transparent,
   exactly reproducible; tier-weighting tested and rejected — ΔR² ≈ 0.0005).
   On task2, the 4 expert votes stay in (they are not analyzed subjects).
2. **Sensitivity: shipped `s_mean`** (the engine's deployed values) — reported
   alongside; the F40 self-shift bounds the expected inflation.
3. `s_sd` is NOT pure vote-count noise (delta-method R² < 0; corr 0.84) —
   treated as given metadata, kept as-is in both variants (one-vote removal
   perturbs it <2% at median n≈30).
4. Residual 0.19-sd of the shipped-s map is unexplained → OQ13 (exact pipeline
   formula) to the data team; does not block the LOO-primary design.
5. Paper framing: item evidence is crowd-calibrated (standard psychometric
   calibration-sample logic) — VALID provided the calibration excludes the
   analyzed individual; that is exactly what the LOO signal does.

## 1. Gold-standard policy (OQ9 — RESOLVED 2026-06-12)

- **task2 primary:** expert ≥3/4 consensus (3,789 cases / 93,185 reads).
  Sensitivities: (A) Dawid–Skene soft gold (4 experts + source); (B) source
  gold; (C) designated senior expert (pending identification).
- **task1 primary:** source gold (no panel exists). Mitigation of single-source
  noise: (i) report the consensus-vs-source disagreement rate from task2 (8.4%
  on clean cases, F38) as the plausible gold-noise scale; (ii) lapse term in
  the link absorbs symmetric label noise; (iii) exclude no cases.
- Robustness criterion: P1–P4 sign-stable across sensitivities.

## 2. W1 — Real observation-model fit (rung-3 closure)

### 2.1 Two frames
- **W1a (task2, K=6):** 6-class forced choice; marginal one-vs-rest models
  M0–M4 + joint softmax M5 (ladder unchanged from v1, now on LOO signals).
- **W1b (task1, K=1, NEW):** binary detection — the engine's response frame
  EXACTLY (no one-vs-rest approximation). Model ladder M0–M4 applies verbatim;
  M5 is N/A. This is the cleanest engine-likelihood test in the dataset and
  the largest single read set (167,503).

### 2.2 Model ladder (per-domain; hierarchical, crossed user × case effects)
| Model | Form | Tests |
|---|---|---|
| M0 | P(y=1) = λ + (1−2λ)Φ(e^{ℓ_{uk}}(s_k + θ_{uk})), λ=0.025 FIXED | shipped likelihood verbatim |
| M1 | + free symmetric λ | lapse magnitude |
| M2 | + asymmetric (λ_fa, λ_miss) | zoo `AsymmetricLapseLearner` axis |
| M3 | M2 with slope-matched t_ν link | zoo `HeavyTailLinkLearner` axis |
| M4 | M2 + s_sd smearing (GH quadrature) | F30/`smear_w` physics on real data |
| M5 | joint softmax over s_2..s_7 (+uniform-lapse mixture) — task2 only | forced-choice dependence |

### 2.3 Holdout discipline
- Split by CASE (never by read), 50/50, stratified by gold class and (task2)
  consensus level; model selection on the fit half; ONE prespecified
  comparison re-run on the held-out half per frame (P1 = task2, P1b = task1).
- Metric: held-out per-read log-loss/ELPD, cluster-bootstrap SE by user
  (10,000 resamples). Secondary: 5-fold CV over USERS (new-user transfer).
- CRN/frozen seeds; both signal variants (§0) run on identical splits.

### 2.4 Identifiability & bias controls
λ-vs-skill-floor: report profile likelihood + fixed-λ sensitivity (decile span
0.026→0.852 on task1 gives strong curvature identification). Gold noise:
consensus primary + DS sensitivity (task2); λ absorbs symmetric noise (task1).
Nonstationarity: task2 refit on second-half-per-user reads (F36); task1 has no
timestamps → fit assumes exchangeability within user (stated limitation).
Selection: case order ≈ random (task2 verified; task1 unverifiable without
timestamps — listed as limitation, mitigated by case fixed-structure via s).

### 2.5 Output → rung-3 stress (P3)
`RealLinkLearner` zoo member (responds via fitted M5/M2 population, learns via
R–W); re-run the M12 campaign (CRN, shipped + D19-hardened configs) with the
domain1 arm now REAL-LINK-parameterized (it was the F19/D16 problem domain —
highest-value stress). Claim: real link inside/outside the F28–F31 envelope.

## 3. W2 — Measurement-layer validation at scale (rung-5)

Static per-domain TaskFilter replay (F34 protocol, smear_w per D19, LOO
signals), per user per domain → ℓ̂_{u,k}.
- **P2 (primary, task2):** Spearman(mean_k ℓ̂_{u,k}, clean-gold accuracy_u),
  n=315 (≥20 clean-gold reads), α=0.001. Power >0.999 at ρ≥0.3.
- **P2b (task1):** same with source-gold accuracy, n=321, α=0.001 — the K=1
  anchor-domain validation the 3-session F34 could only sketch (n=3 → n=321).
- Secondaries (Holm within family): per-domain ρ_k (7 tests now); ℓ̂ vs Qscore
  (task2 only; pending OQ10); known-groups tier AUC (29 expert vs 135 novice,
  task2; task1 tiers via users table); SBC-on-fitted-link calibration loop;
  F34-pathology (skill+bias explains chance) prevalence at n≈640 replays.

## 4. W4 — Cross-domain skill structure (NEW; grounds the hierarchical prior)

The 147 dual-contest users (≥20 reads both) give per-user real skill vectors
spanning domain1 × domains2–7. Estimate the cross-domain skill correlation
matrix (accuracy-based, then ℓ̂-based from W2 replays; Spearman + disattenuated
via split-half reliability). Output: empirical check of the engine's Σ_l
hierarchical prior (currently fitted from cert_config artifacts) — reported as
correlation-structure grounding, with n=147 precision (CI half-width ≈0.14 at
ρ=0.5). Also: per-tier skill distributions for the v15 realistic-bias prior.

## 5. W3 — Empirically-grounded simulation population (unchanged from v1)

Population (ℓ,θ) fits from W1 → re-run trainer benchmarks; dynamics anchoring
GATED on OQ8 (task2 only — task1 has no time axis). The pooled-curve
survivorship trap (F36) applies to every longitudinal claim.

## 6. Pre-specified endpoints (v2)

| ID | Endpoint | n | α / report |
|---|---|---|---|
| P1 | task2 held-out ΔELPD/read, best-marginal vs M0 | 93k reads | est.+CI |
| P1b | task1 held-out ΔELPD/read, best vs M0 | 167.5k reads | est.+CI |
| P2 | task2 Spearman(ℓ̂, accuracy), users ≥20 | 315 | 0.001 |
| P2b | task1 Spearman(ℓ̂, accuracy), users ≥20 | 321 | 0.001 |
| P3 | FG-rate + trials-to-declare shift under RealLinkLearner (hardened cfg) | 30 seeds CRN | exact binomial CI |
| P4 | dual-contest skill correlation (domain1 vs mean domains2–7) | 147 | 0.001 |
| S-family | per-domain ρ_k (7), Qscore ρ, tier AUC, λ-asymmetry, t-ν, M5-vs-M2 ELPD, shipped-vs-LOO signal deltas | — | Holm per family |

## 7. Power & precision summary (v2 additions)

| Quantity | n | Precision |
|---|---|---|
| task1 λ_fa, λ_miss | 167.5k reads, floor deciles at 0.026 | SE ≲ 0.003 — sharpest lapse test available |
| P1b ELPD | 83k held-out reads | per-read SE ~1e-3 scale |
| P2b | 321 users | CI half-width ≈0.10; power >0.999 for ρ≥0.3 |
| P4 | 147 users | half-width ≈0.14 at ρ=0.5; detects ρ≥0.27 at α=0.001 |
| F40 self-shift bias bound | analytic + LOO vs shipped delta | reported exactly |

## 8. Publication map (v2)

Methods paper: rung-3 closure now anchored on BOTH frames (binary K=1 — the
engine's exact likelihood — and multiclass K=6), n≈640 total replay validation,
cross-domain correlation grounding (W4), signal-circularity protocol (§0) as a
methodological contribution in itself (crowd-calibrated item banks are common;
the LOO-consensus correction + leverage quantification is reusable). Boundary
unchanged: efficacy claims pilot-gated (D18 language).

## 9. Execution order

1. `extset_adapter.py` + `test_extset.py` — loaders, gold policies, LOO-signal
   construction (F40 guard: refuse shipped-s as primary; refuse bank-plurality
   truth on EXTSET segs), reconciliation asserts (vote==read match).
2. `study_extset_link.py` — W1a/W1b ladders + holdout → `figures/data_extset_link.npz`.
3. `RealLinkLearner` + rung-3 campaign arm.
4. `study_extset_replay.py` — W2 (both tasks) → `figures/data_extset_replay.npz`.
5. `study_extset_xdomain.py` — W4 → `figures/data_extset_xdomain.npz`.
6. W3 population refits + benchmark reruns; `make_extset_figures.py`.
7. PROJECT_MEMORY consolidation; claim re-pinning.
