# PILOT_SAP_M14 — Pre-registered design + power analysis for the staged trainer pilot

**Status:** DRAFT for PI/IRB review (M14, 2026-06-12). All effect-size inputs
are EMPIRICAL (EXTSET-fitted population F43; D19-hardened trainer; CRN
campaigns in `figures/data_pilot_power.npz`, `data_extset_shadow.npz`).
Language per D18 throughout: **the trainer recommends; certification decides.**

## 1. Why a pilot is still required (and only a pilot)

The EXTSET program (M13) closed every pre-pilot rung: the observation model
is fitted from 293k real reads (F43), the measurement layer is validated on
636 real raters (P2/P2b; disattenuated ρ = 0.65–1.00), the trainer's operating
characteristics are preserved under the real link (F44), and the cross-domain
skill manifold is empirically grounded (P4). The residual that NO existing
data can validate (§3C/M11.2c): feedback-driven dynamics + the adaptive
selection's causal effect on learning. That is precisely a controlled pilot.

## 2. Staged design (C11; gates between stages)

**Stage 0 — SHADOW (no human contact, done in-scratch and repeatable live):**
the trainer runs silently against real labeling streams. In-scratch result:
real contest serving sat ~1.7σ̂ from the trainer's optimal difficulty;
median expected-learning-weight uplift **×6.0 (IQR 5.6–6.4, n=434 raters)**
— large placement headroom. Live shadow gate: belief-update latency, item
supply, and instrumentation checks pass on ≥95% of streams.

**Stage 1 — supervised cohort (consented, single-arm, n≈10):** real adaptive
placement + veridical feedback, observers present. Gate: no harm signals,
engagement (completion ≥80%), measurement sanity (ℓ̂ trajectories within the
fitted-population envelope).

**Stage 2 — randomized pilot (the efficacy test):** parallel-arm RCT,
tier2 trainer vs staircase85 (strongest conventional adaptive control).
Random serving is NOT an arm (×4.4 effect, 63% censoring — unethical waste
of participant time once superiority over the stronger control is testable).

## 3. Stage-2 endpoints

- **Primary:** trials to TRUE mastery, defined as PASSING the independent
  eval re-certification (D18 — certification decides), analyzed as
  log(trials) (two-sample Welch t on the log scale; masterers within the
  600-trial budget), with the mastery-rate difference as co-primary gate.
- **Secondary:** declared-vs-recert agreement (F32 replication in vivo);
  ℓ̂ trajectory vs Qscore-analog convergence; per-domain transfer (W4
  manifold prediction); dropout/engagement.

## 4. Empirical power (from `data_pilot_power.npz`, 60 CRN seeds/arm)

Simulated under the EXTSET-fitted population, hardened config, budget 400:

| arm | mastered | median trials | log-mean ± sd |
|---|---|---|---|
| tier2 | 60/60 | **45** | 3.78 ± 0.30 |
| staircase85 | 60/60 | 77 | 4.27 ± 0.56 |
| random | 22/60 | 247 (censored 63%) | 5.26 ± 0.61 |

Effect tier2 vs staircase85: δ = 0.495 log-trials (×1.64 faster), pooled
σ = 0.448.

| α | power | N/arm | enroll (15% dropout) |
|---|---|---|---|
| 0.05 | 0.8 | 13 | 16 |
| 0.05 | 0.9 | 18 | 22 |
| 0.005 | 0.8 | 22 | 26 |
| **0.005** | **0.9** | **28** | **33** |

**Recommended: 33 enrolled per arm (66 total) → α=0.005, power 0.9.** A
strict-α two-arm pilot of ~70 consented raters is sufficient. Sensitivity:
if the realized effect is half the simulated δ, N/arm=110 — the Stage-1
cohort's observed trajectories update this table before Stage-2 enrollment
(pre-specified interim re-estimation, blinded to arm).

## 5. Analysis plan & guardrails

- Intention-to-treat on randomized raters; censoring handled by the
  co-primary mastery-rate comparison + log-rank sensitivity.
- All measurement uses the LOO-consensus signal protocol (F40) — no
  self-vote leakage into item evidence.
- Stopping: O'Brien–Fleming-style interim at 50% information for harm/futility
  only (no early efficacy claim at pilot scale).
- The pilot validates the METHOD's efficacy; deployment claims remain gated
  on the separate production-bar checklist (§3D two-bars).

## 6. Instrumentation (from the shadow study)

Per trial: served seg_id, s/s_sd, belief snapshot (ℓ̂, θ̂, sds), mode,
E[w] of served + argmax-E[w] counterfactual, response, RT, feedback shown,
gate state; per session: seed ledger (D9), re-cert verdicts. Everything the
shadow replay logs today — the live system must log the same.
