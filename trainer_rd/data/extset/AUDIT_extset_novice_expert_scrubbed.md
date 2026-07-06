# Audit — ExtSet novice + expert label release

**Files:** `extset_novice_survey_results.csv`,
`extset_novice_labels_scrubbed.csv`, `extset_expert_labels_scrubbed.xlsx`
---
IMPORTANT: This live data set does not contain the domain1 questions. So please be mindful of this note. 
---

## 1. File-by-file inventory

### 1a. `extset_expert_labels_scrubbed.csv` — gold-standard panel
- 1 sheet, **5,000 rows × 6 cols**: `case_id`, `origin`,
  `label_expert1`, `label_cal`, `label_matt`, `label_tianyu`.
- 4 experts, **each labels all 5,000 cases** (complete panel, no
  missingness).
- 7-class  taxonomy, identical class set across all 4 experts:
  `{domain2, domain3, domain4, domain5, domain6, domain7}`.

### 1b. `extset_novice_labels_scrubbed.csv` — crowd reads
- **125,865 rows × 8 cols**: `Case ID`, `User ID`, `Read ID`,
  `Labeling State`, `Qscore`, `User Label`, `Origin`,
  `Response Submitted At`. (Note: BOM on first header — strip `﻿`.)
- **5,000 unique cases · 699 unique novice users · 125,865 reads**
  (Read ID is the unique row key).
- `Labeling State`: 125,860 `Gold Standard`, **5 `Canceled`** (drop).
- `User Label` is single-quoted (`'domain2'`); strip quotes. Same
  7-class set as experts.
- Coverage: median **23 raters/case** (min 5, max 85); median **22
  labels/user** (min 1, max 4,991 — one near-complete account).
- Window: 2025-07-23 → 2025-09-18.

### 1c. `extset_novice_survey_results_scrubbed.csv` — demographics
- **295 rows × 4 cols**: `user_id`, `response`, `created_at`, `name`.
- `response` is a 4-field comma-joined string →
  `[consent="OK", role, epileptology specialization, signal experience]`.
- Two survey instruments: `signal_survey_5290` (197), `_5291` (98).
- Cohort is genuinely novice: role mostly **student (127) /
  other (94) / resident (28) / sub student(28)**; **252/295 not specialized
  in signal/domain identification**; signal experience **None (174) / 0–1 yr (85)**.

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
reality of  scoring:

| pair | raw agree | Cohen κ |
|---|---|---|
| expert1–expert2 | 0.727 | **0.673** |
| expert2-expert4 | 0.669 | 0.602 |
| expert1–expert4 | 0.664 | 0.595 |
| expert3-expert4 | 0.638 | 0.558 |
| expert1–expert3 | 0.603 | 0.522 |
| expert2–expert3 | 0.585 | **0.495** |

- **42.5% unanimous (4/4)**, **75.4% ≥3/4**, **88.3% have a unique
  plurality**; **10.9% tied**, **0.8% all-four-different**.
- `matt` is the consistent low-agreement outlier (κ 0.50–0.56 vs
  others' 0.60–0.67).
- Expert-plurality class prevalence is balanced across the 5 main
  classes (domain3 996, domain4 956, domain6 877, domain5 837, domain2 828) but
  **domain7 (346) and domain7 (160) are rare**.

**Implication:** the gold standard is itself a ~κ≈0.55 noisy label.
(a) calibrate only on the **≥3/4-consensus subset** (clean
gold, ~75% of cases), or (b) keep all cases with a soft/weighted gold
and let the probit-lapse model absorb gold noise via its lapse term.

---

## 4. Novice skill signal — and why it matters for the OC surface

**Per-novice 7-class accuracy vs expert-plurality gold (clean-gold
cases, 692 novices):** mean **0.180**, median 0.167, sd 0.166,
p90 0.375 (7-class chance ≈ 0.14). Genuinely novice, wide spread.

**Per-class one-vs-rest pooled AUROC** (the natural map onto our
K-domain Multi-AUROC framework — each  class = one "domain" ℓ_k):

| class | gold N | prevalence | pooled AUROC |
|---|---:|---:|---:|
| domain7 | 12,459 | 10.0% | 0.718 |
| domain2 | 18,963 | 15.2% | 0.710 |
| domain6 | 20,048 | 16.1% | 0.688 |
| domain5 | 19,198 | 15.4% | 0.636 |
| domain4 | 22,975 | 18.4% | 0.626 |
| domain3 | 23,235 | 18.6% | 0.600 |
| domain7 | 8,003 | 6.4% | 0.591 |

Per-novice **domain2** one-vs-rest AUROC (372 estimable, ≥20 gold
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
over time → it is a **rolling/windowed extset competence score, not a
cumulative counter**.

**Value:** an *independent, platform-native skill measurement* — ideal
**convergent-validity check** for the latent ℓ̂ our protocol estimates
(does our Bayesian skill posterior track the platform's own score?).
Its exact formula/window should be confirmed before citing it (§7).


---

## 7. Open questions for LLM to resolve

1. **Gold-standard policy?** ≥3/4-consensus subset (clean, ~75% of
   cases) vs all-cases-with-soft-gold vs a designated senior oracle
   (expert1/Westover). This is the single biggest downstream decision — it
   sets σ* calibration and the AUROC ceiling.
2. **`Qscore` definition?** Is it the extset platform's own scoring
   (rolling accuracy? Brier-like?)? Knowing the formula lets us use it
   as a clean external validator for ℓ̂.
4. **Survey:** is the 25% demographic coverage final, or is more survey
   data coming? Which instrument (5290 vs 5291) is canonical, and do
   they differ in items?
6. **Minor:** drop the 5 `Canceled` rows; confirm `Qscore==0` is a
   true floor (cold-start) not a missing sentinel.

---


## 9. Impact on Paper 2 (Nature Medicine — deployed 7-domain exam)

- This *is* the operational substrate: a real **5,000-item  bank
  with expert gold + a demonstrated novice population + a platform
  competence metric (Qscore)** for convergent validation.
- The expert-panel disagreement is itself a Paper-2 argument:
  certification must be defined against a **consensus**, not a single
  oracle — reinforces the "objective certification" thesis and the
  Bayesian treatment of gold-label noise.
- Qscore gives a built-in external yardstick for any deployed-exam
  claim that the protocol's ℓ̂ reflects real competence.
