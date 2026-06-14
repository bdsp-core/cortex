# Post-Pilot Production Instrument — staged changes (NOT YET ACTIVE)

> Two evidence-backed instrument changes are **staged** for the production
> re-freeze that happens **after the in-flight tiered human pilot cohort closes**.
> They are NOT applied to the live code now, because the pilot must be collected
> on a bit-identical instrument (`instrument_freeze` pins the v1.3.6 paper-grade
> config: n_particles=600, ℓ*=`ell_star_unified_v14`). Flipping either mid-cohort
> would break that guarantee. Full evidence: `docs/ENGINE_IMPROVEMENT_RESULTS.md`.
>
> **Trigger:** pilot cohort finalized → PI sign-off → apply both → re-freeze →
> tag the new production instrument (proposed `v2.0` / `ell_star_unified_v15`).

---

## Change 1 — N_PARTICLES 600 → 1200 (Step 2 evidence)

**Why:** N=600 mildly under-covers (AUROC 95% CI empirical 0.932, ~3 SE low);
**N=1200 restores nominal** (0.946) ⇒ better-calibrated Type-1/2 control. Latency
stays fine (mean 248 ms, p95 0.86 s/question ≪ human answer time). N=2400 adds
only marginal gain at 2× cost. N is a calibration lever, not a resolution lever.

**Apply:**
1. `scripts/session_controller.py:72` → `N_PARTICLES = 1200` (remove the STAGED note).
2. Re-run `python scripts/freeze_instrument_v1_3_6.py` (regenerate the freeze
   manifest with the new params) — or create a v2.0 freeze.
3. `tests/test_instrument_freeze.py::test_freeze_records_paper_grade_config` →
   update `assert m["params"]["n_particles"] == 1200`.
4. Re-run the cortex suite; confirm green.

**Revert:** set N_PARTICLES back to 600, re-freeze, restore the test assertion.

---

## Change 2 — ℓ* recalibration to a CREDENTIALED expert panel (items 1 & 3)

**Why:** the shipped `ell_star_unified_v14` selects its 14-expert panel BY
DATA-DRIVEN SKILL (top-14 by cross-task mean ℓ), which sits mechanically high. A
**credentialed world-class panel** (Super-8 / Bonobo, from `raters.csv:groups`)
is exogenous, non-circular, the gold-standard "expert" definition for a regulated
instrument, AND yields a 0.07–0.35 lower, more-achievable cut. OC proof: the
credentialed per-pattern ℓ* **nearly doubles** skilled-rater resolution
(PASS@+0.6 37%→68%, @+1.0 51%→78%), **zeroes false-FAIL**, holds false-PASS < 2%.
Spike's Super-8 panel is strictly better (J 0.649→0.693, ℓ* 0.325→0.238, full
8/8 coverage vs the shipped 9/14).

**FINAL recommended set** (`results/calibration_study/recommended_ell_star_final.json`),
PI-confirmed 2026-06-10: per-pattern (spike←Super-8, IIIC←Super-8 ∪ Bonobo),
robust lpd:

| task | shipped v14 | credentialed v15 | Δ | basis |
|---|---:|---:|---:|---|
| spike | 0.3251 | **0.2383** | −0.087 | Super-8 (J 0.693 > shipped 0.649; full coverage) |
| sz | 0.2560 | **0.1548** | −0.101 | S8∪Bonobo |
| lpd | 0.5337 | **0.3059** | −0.228 | S8∪Bonobo, robust mild-trim (CI 0.59→0.23) |
| gpd | 0.3297 | **0.2570** | −0.073 | S8∪Bonobo |
| lrda | 0.4793 | **0.3214** | −0.158 | S8∪Bonobo |
| grda | 0.4865 | **0.3540** | −0.133 | S8∪Bonobo |
| iic | 0.4418 | **0.3042** | −0.138 | S8∪Bonobo |

**PI decisions (2026-06-10):** competence standard = credentialed-expert-vs-field
separation; per-pattern methodology; lpd via robust re-derivation.

**Rigor (DONE — `panel_rigor_analysis.py`, `expert_panel_recalibration_study.py`):**
- Bootstrap ΔJ: spike comparable-or-better (NS, more stable); gpd/iic comparable;
  sz/lpd sig.-lower-J but far more achievable.
- lpd was unstable (bootstrap CI width 0.588) → robust mild-trim (drop 2
  credentialed experts with lpd ℓ ≤ 0.19, within the non-expert range) → 0.306,
  CI 0.233. Sensitivity: lpd ∈ [0.31, 0.50] by trim depth; 0.306 = achievable end.
- Clean OC error scoring (vs decision cut): intrinsic false-PASS 0.77% / false-FAIL
  0.14% (the credentialed cut ~doubles resolution at well-controlled error).

**Still pending before adoption:** re-derive on the final panel + production OC at
N=1200 (adoption-time); fold into the all-7 → tiered certification framework
(item 4); incorporate the real tiered pilot data when it returns.

**Apply:** emit an `ell_star_unified_v15` block into `calibration/cert_config.yaml`
(generator: extend `pipeline/joint_calibration/emit_cert_config_v14.py`), point
`cortex_policy_k7.load_ell_star_k7(block_name=...)` at v15, re-freeze, update the
freeze's `ell_star_block`. **Revert:** point the loader back at v14; v14 block is
never deleted (audit chain).

---

## Change 3 — θ-block prior: corr_l → corr_t (Step 4)

**Why:** the engine fits a bias correlation `Corr_t` but the shipped wiring reuses
the SKILL correlation `Corr_l` for the θ-block (`session_controller.py:434`). Using
the fitted bias correlation gives a **significant** resolution/efficiency gain
(PASS@+1.0 51→64%, Fisher p=0.008; PASS@+0.6 37→41%) with no significant error
change, zero-mean (no examinee-bias assumption). The full empirical-Bayes variant
(corr_t + fitted scale + population mean) was sandbox-tested and **REJECTED** (it
craters when examinees are unbiased and gives no gain even on its ideal population);
adopt corr_t-only.

**Apply:** `session_controller.py:434` already supports it via
`CortexSession(bias_prior="corr_t")` — flip the default (or set the live session's
`bias_prior="corr_t"`). Re-freeze the instrument (changes the θ prior ⇒ a new
instrument version). **Revert:** `bias_prior="corr_l"` (the shipped default; the
option is byte-identical when set to corr_l).

**Before adoption:** confirm the resolution gain + the (non-significant at n=30)
false-FAIL trend at larger n with the production OC at N=1200.

**⚠ Pilot dependency (shared with the whole resolution story):** the OC evidence
above simulates UNBIASED examinees (θ=0). The Step-4 2×2 found that if real
examinees are biased like the training raters (θ~+1.7), resolution collapses ~7×
regardless of prior. **Check the pilot's per-rater θ before finalizing any
resolution claim** (corr_t, ℓ* recalibration, N).

## Reproduce the evidence
- Calibration study: `python pipeline/reference_calibration/expert_panel_recalibration_study.py`
- Coverage/latency (Step 2): `scripts/run_coverage_validation.py --N-particles {600,1200}`;
  `sim_v1_3_5/bench_n_particles.py`
- OC (credentialed ℓ*): `sim_v1_3_5/run_oc_validation.py --alphas 0.05 --per-tasks 60
  --n-per-cell 30 --ell-star-json results/calibration_study/recommended_ell_star_final.json`
