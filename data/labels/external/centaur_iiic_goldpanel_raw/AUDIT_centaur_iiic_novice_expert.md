# Audit — Centaur-IIIC novice + expert label release

**Date:** 2026-05-16  **Auditor:** Claude (engine/stats)
**Files:** `centaur_iiic_novice_survey_results.csv`,
`centaur_iiic_novice_labels.csv`, `centaur_iiic_expert_labels.xlsx`
(all under `data/labels/`, added 2026-05-15)

---

## 0. Headline

These three files are **the data the unified README flagged as missing**
(Known Caveat #2: *"IIIC crowd data is missing … reportedly held by
Tianyu and not yet integrated."*). They are a Centaur-style IIIC
**novice crowd + 4-expert gold-standard panel** over a shared,
perfectly-keyed **5,000-case image bank** (`task2_images_v3`,
7-class IIIC taxonomy). This is the first real **novice cohort with a
gold standard** available to the project and is directly usable as the
Paper-1 real-data validation set for the Multi-AUROC protocol.

---

## 1. File-by-file inventory

### 1a. `centaur_iiic_expert_labels.xlsx` — gold-standard panel
- 1 sheet, **5,000 rows × 6 cols**: `case_id`, `origin`,
  `label_mbw`, `label_cal`, `label_matt`, `label_tianyu`.
- 4 experts, **each labels all 5,000 cases** (complete panel, no
  missingness). `mbw` = M. Brandon Westover (project PI / MGH-BDSP
  collaborator); `cal`, `matt`, `tianyu` = the other three.
- 7-class IIIC taxonomy, identical class set across all 4 experts:
  `{seizure, lpd, gpd, lrda, grda, bipd, birds}`.

### 1b. `centaur_iiic_novice_labels.csv` — crowd reads
- **125,865 rows × 8 cols**: `Case ID`, `User ID`, `Read ID`,
  `Labeling State`, `Qscore`, `User Label`, `Origin`,
  `Response Submitted At`. (Note: BOM on first header — strip `﻿`.)
- **5,000 unique cases · 699 unique novice users · 125,865 reads**
  (Read ID is the unique row key).
- `Labeling State`: 125,860 `Gold Standard`, **5 `Canceled`** (drop).
- `User Label` is single-quoted (`'seizure'`); strip quotes. Same
  7-class set as experts.
- Coverage: median **23 raters/case** (min 5, max 85); median **22
  labels/user** (min 1, max 4,991 — one near-complete account).
- Window: 2025-07-23 → 2025-09-18.

### 1c. `centaur_iiic_novice_survey_results.csv` — demographics
- **295 rows × 4 cols**: `user_id`, `response`, `created_at`, `name`.
- `response` is a 4-field comma-joined string →
  `[consent="OK", role, epileptology specialization, EEG experience]`.
- Two survey instruments: `eeg_survey_5290` (197), `_5291` (98).
- Cohort is genuinely novice: role mostly **medical student (127) /
  other (94) / resident (28) / nurse (28)**; **252/295 not specialized
  in epileptology**; EEG experience **None (174) / 0–1 yr (85)**.

---

## 2. Relational integrity (joins)

| Join | Result |
|---|---|
| novice `Case ID` ↔ expert `case_id` | **5,000 ↔ 5,000, exact 1:1**, zero orphans either side |
| `Origin` ↔ `origin` (image path) | 5,000/5,000 identical — redundant secondary key, confirms join |
| survey `user_id` ↔ novice `User ID` | **only 176 of 699 labelers surveyed (25%)**; 523 labelers have no demographics; 119 surveyed users never labeled |

**Integrity verdict:** case/origin keys are pristine — expert gold and
novice reads join cleanly with no cleaning required. **Demographic
coverage is the one structural gap:** any survey-stratified result
generalizes from a 25% (non-random — self-selected responders) slice.

---

## 3. Expert gold-standard quality (the certification ceiling)

Inter-expert agreement is **moderate, not high** — the documented
reality of IIIC scoring:

| pair | raw agree | Cohen κ |
|---|---|---|
| mbw–cal | 0.727 | **0.673** |
| cal–tianyu | 0.669 | 0.602 |
| mbw–tianyu | 0.664 | 0.595 |
| matt–tianyu | 0.638 | 0.558 |
| mbw–matt | 0.603 | 0.522 |
| cal–matt | 0.585 | **0.495** |

- **42.5% unanimous (4/4)**, **75.4% ≥3/4**, **88.3% have a unique
  plurality**; **10.9% tied**, **0.8% all-four-different**.
- `matt` is the consistent low-agreement outlier (κ 0.50–0.56 vs
  others' 0.60–0.67).
- Expert-plurality class prevalence is balanced across the 5 main
  classes (lpd 996, gpd 956, grda 877, lrda 837, seizure 828) but
  **birds (346) and bipd (160) are rare**.

**Implication:** the gold standard is itself a ~κ≈0.55 noisy label.
This sets an **irreducible AUROC ceiling** below 1.0 and *must be
propagated*, not ignored. Two defensible policies (decision needed —
see §7): (a) calibrate only on the **≥3/4-consensus subset** (clean
gold, ~75% of cases), or (b) keep all cases with a soft/weighted gold
and let the probit-lapse model absorb gold noise via its lapse term.

---

## 4. Novice skill signal — and why it matters for the OC surface

**Per-novice 7-class accuracy vs expert-plurality gold (clean-gold
cases, 692 novices):** mean **0.180**, median 0.167, sd 0.166,
p90 0.375 (7-class chance ≈ 0.14). Genuinely novice, wide spread.

**Per-class one-vs-rest pooled AUROC** (the natural map onto our
K-domain Multi-AUROC framework — each IIIC class = one "domain" ℓ_k):

| class | gold N | prevalence | pooled AUROC |
|---|---:|---:|---:|
| birds | 12,459 | 10.0% | 0.718 |
| seizure | 18,963 | 15.2% | 0.710 |
| grda | 20,048 | 16.1% | 0.688 |
| lrda | 19,198 | 15.4% | 0.636 |
| gpd | 22,975 | 18.4% | 0.626 |
| lpd | 23,235 | 18.6% | 0.600 |
| bipd | 8,003 | 6.4% | 0.591 |

Per-novice **seizure** one-vs-rest AUROC (372 estimable, ≥20 gold
labels): mean **0.594**, sd 0.127, p10–p90 ≈ **0.44–0.76**.

> **This is the key Paper-1 result-in-waiting.** The real novice
> per-domain AUROC band (≈0.59–0.72) sits squarely in the **low end of
> the synthetic `L_GRID`** (AUROC 0.62–0.91, ℓ∈[−1.5,1.0]). Real
> novices empirically *are* the slow-to-certify, high-censoring regime
> the F4.2 surface predicted — the same AUROC≈0.6–0.7 band where the
> δ=0.025 censoring caveat bites. The synthetic grid's lower bound is
> now empirically justified rather than assumed.

**Estimability for per-rater latent-skill fitting:** ≥20 clean-gold
labels → **349** novices; ≥30 → 278; ≥50 → 202; ≥100 → 122. A solid
200–350-rater real cohort for the Multi-AUROC engine.

---

## 5. `Qscore` — an in-platform competence proxy

Range 0–90, integer, mean 40.3. **corr(per-user mean Qscore, per-user
accuracy) = 0.873** (378 users ≥20 labels); row-level mean Qscore is
51.9 on correct reads vs 32.5 on incorrect. Not monotone within a user
over time → it is a **rolling/windowed Centaur competence score, not a
cumulative counter**.

**Value:** an *independent, platform-native skill measurement* — ideal
**convergent-validity check** for the latent ℓ̂ our protocol estimates
(does our Bayesian skill posterior track the platform's own score?).
Its exact formula/window should be confirmed before citing it (§7).

---

## 6. Taxonomy reconciliation vs the existing corpus

Centaur-IIIC is **7-class**; the unified `sparcnet50K` corpus is
**6-class** (`…,other`). The mapping is clean: Centaur **splits
`other` into `{bipd, birds}`** (jointly 10.0% of expert-gold cases);
the other five classes are 1:1. So:

- **Cross-corpus / reuse of SPARCNET SDT fits:** collapse
  `{bipd,birds}→other` → exact 6-class compatibility.
- **Native IIIC certification exam (recommended for Paper 2):** keep
  all 7 classes — Centaur is the *richer, more clinically current*
  taxonomy (BIRDs and BIPD are distinct, exam-relevant entities).

---

## 7. Open questions for you

1. **Gold-standard policy?** ≥3/4-consensus subset (clean, ~75% of
   cases) vs all-cases-with-soft-gold vs a designated senior oracle
   (mbw/Westover). This is the single biggest downstream decision — it
   sets σ* calibration and the AUROC ceiling.
2. **`Qscore` definition?** Is it the Centaur platform's own scoring
   (rolling accuracy? Brier-like?)? Knowing the formula lets us use it
   as a clean external validator for ℓ̂.
3. **7-class native or collapse to 6** for the certification exam? (I
   recommend native 7 for Paper 2, with a 6-class bridge for any
   SPARCNET-corpus comparison.)
4. **Survey:** is the 25% demographic coverage final, or is more survey
   data coming? Which instrument (5290 vs 5291) is canonical, and do
   they differ in items?
5. **Physical-recording overlap:** could any of these 5,000
   `task2_event*` images derive from the same EEGs as `sparcnet50K`
   segments (leakage risk if the same recording trains *and* certifies)?
6. **Minor:** drop the 5 `Canceled` rows; confirm `Qscore==0` is a
   true floor (cold-start) not a missing sentinel.

---

## 8. Impact on Paper 1 (methodology / validation)

- **Supplies the missing real-data arm.** Phase-1 F-rig had only 27
  fixed SPARCNET raters. This adds **~200–350 real novices with a
  gold standard** → the OC surface stops being purely synthetic.
- **Direct framework fit:** K=7 IIIC classes → one-vs-rest per-domain
  ℓ_k is exactly the Multi-AUROC K-domain object the engine already
  models. No re-architecture.
- **Empirically anchors the synthetic grid.** Real novice per-domain
  AUROC (0.59–0.72) validates `L_GRID`'s low end and gives the
  censoring caveat (§F4.4) a real-world referent: novices genuinely
  live in the slow/high-censoring regime.
- **IIIC-native priors:** an IIIC `Corr_l` (cross-class skill
  correlation) can be fitted here to replace the SPARCNET-derived
  `Sigma_l_fitted.npy` (K=6) for an IIIC exam — the *correct* prior.
- **Real σ\* / Youden calibration** becomes possible on a real novice
  distribution instead of SPARCNET transfer.
- **Honest caveats to carry into the manuscript:** noisy gold
  (κ≈0.55), 25% demographic coverage, class imbalance (bipd 6%).

## 9. Impact on Paper 2 (Nature Medicine — deployed 7-domain exam)

- This *is* the operational substrate: a real **5,000-item IIIC bank
  with expert gold + a demonstrated novice population + a platform
  competence metric (Qscore)** for convergent validation.
- The expert-panel disagreement is itself a Paper-2 argument:
  certification must be defined against a **consensus**, not a single
  oracle — reinforces the "objective certification" thesis and the
  Bayesian treatment of gold-label noise.
- BIRDs/BIPD rarity → the adaptive item-selection (F4.1 EV) must
  guarantee minimum exposure of rare domains so every certified
  examinee is actually tested on them.
- Qscore gives a built-in external yardstick for any deployed-exam
  claim that the protocol's ℓ̂ reflects real competence.
