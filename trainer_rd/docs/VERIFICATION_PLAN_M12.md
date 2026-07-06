# M12 — Verification-ladder completion plan (autonomous run, 2026-06-11)

Executes PROJECT_MEMORY §3D. Goal: finish rungs 3 (mis-specification envelope) and 4
(calibration/OC) and the in-scratch part of rung 5 (real-session measurement replay),
with smoke tests between steps. Constraint: everything stays in /data/eli-work/scratch
(⇒ the EXTSET real-response-link fit named in §3D is OUT OF SCOPE here — data lives
outside scratch; documented as the one remaining rung-3 axis).

## Steps (each = code → smoke → full campaign in background → record findings)

- **V0** Baseline regression: full existing suite re-run (background, /tmp/baseline_suite.log).
- **V1** `misspec_learners.py` + `test_misspec.py` — the mis-specified learner zoo.
  Each member deviates from the assumed model (learner_sim soft R–W) on ONE named axis:
  - dynamics family: `PowerLawLearner` (law-of-practice σ curve), `PlateauLearner`
    (stagewise/insight drops), `DriftingCeilingLearner` (non-stationary σ_∞),
    `MomentumCriterionLearner` (EMA-integrated prediction error; temporal correlation)
  - observation model: `HeavyTailLinkLearner` (t₃ link, slope-matched at 0),
    `AsymmetricLapseLearner` (λ_fa=0.08 ≠ λ_miss=0.01 — bias-mimicking, uncorrectable
    by criterion training), `FatigueLearner` (λ grows with trial — state-dependent)
  - adversarial: `AntiLearner` (sign-flipped criterion update, static σ>σ*),
    `CarelessLearner` (λ=0.35, ×0.25 rates), static-below-cut (false-grad null)
- **V2** `study_misspec.py` — rung-3 campaign. tier2/staircase85/random × zoo × 30 seeds,
  CRN pool (task domain3, pool 600, budget 400, Npart 400). Plus a heterogeneity arm:
  ~100 population draws (α_t, α_σ ±~3–4×, σ0, t0, σ_∞/σ*, λ, rule) vs fixed filter.
  Metrics per run: trials-to-TRUE-mastery; trials-to-DECLARED mastery (shipped D16
  hybrid gate via TaskModePolicy); **false graduation** = declared while true pool
  accuracy A < A_bar (A = E_pool[p_correct] under the learner's TRUE response fn incl.
  s_sd smearing; A_bar = same for probit ℓ=ℓ*, t=0 — the performance projection that
  stays meaningful under link misspec); tracking RMSE(ℓ,t) and 90% PIT coverage
  (u = weighted CDF at truth ∈ [.05,.95]).
- **V3** `study_gate_oc.py` — rung-4 OC: full operating characteristic of the D16 gate
  over ceiling-to-cut margin Δ = ℓ_∞−ℓ* ∈ {−.2,−.1,−.05,0,+.05,+.1,+.2,+.3} × 40 seeds:
  P(declare) with Wilson CIs, time-to-declare, A at declaration. Extends F19's 2 points.
- **V4** `study_train_sbc.py` — rung-4 SBC of the TRAINING filter in closed loop:
  truth ~ filter prior, well-specified dynamics, tier2 selection (valid under sequential
  design), PIT ranks for ℓ and θ at n ∈ {10,50,150,300}, 200 reps, KS + coverage table.
- **V5** `study_real_replay.py` — rung-5 (in-scratch part): replay ALL 3 real sessions
  through the filter; convergent validity (ℓ̂ vs empirical accuracy ordering across
  raters 0.23/0.52/0.57), careless-certifies-nothing regression.
- **V6** Synthesis: fig8 misspec envelope; PROJECT_MEMORY findings F28+, §3D rung
  statuses, §6 status-log M12 entry; final report.

## Stop conditions
User-mandated: stop at 90% session usage / 70% weekly usage (not directly observable —
proxy: keep token use lean, prefer background compute). Otherwise run to completion.

## Status (update as steps complete)
- [x] V0 baseline regression (all green) and final regression (13 files, ~145 checks, green)
- [x] V1 zoo + smoke (test_misspec.py, 20 checks)
- [x] V2 campaign → F28/F29 (data_misspec_campaign.npz); V2b re-cert backstop → F32
      (data_recert_backstop.npz); V2c mitigations → F31 (data_misspec_mitig{,2}.npz)
- [x] V3 OC → F29 (data_gate_oc{,2}.npz)
- [x] V4 SBC → F30 found+fixed, F33 (data_train_sbc{,_smearw}.npz)
- [x] V5 replay → F34 (data_real_replay.npz)
- [x] V6 synthesis: fig8_misspec_envelope, PROJECT_MEMORY §3 F28–F34 + §3D statuses +
      D18/D19 + §6 M12 entry

All findings and numbers are recorded in PROJECT_MEMORY.md (normative); this file is
the execution log only.
