# PHASE3_AND_PORT.md — Phase-3 methodology spec + main-repo port checklist

> Step 9 of `INTEGRATION_PLAN.md` (docs only). Read `PROJECT_MEMORY.md` first.
> The Phase-2 prototype (Steps 0–8) is complete and green (8 steps, ~90 checks).
> This file specifies the Phase-3 modeling work and the path to production.

---

## A. Phase-3 methodology spec

### A1. Offline hierarchical fitting of the dynamics parameters (D7 payoff)
**Goal.** Replace the literature-placeholder (α_t, α_σ, τ_σ, q_t, q_σ) with
per-learner posteriors fit from logged trial data; keep σ_∞ at the data-grounded
expert ceiling (§2B) as a strong prior, refined per learner.

- **Data.** The `TrialRecord` stream (Step 1 schema) across the `LearnerLedger`:
  (s, s_sd, y, y*, mode, feedback, RT, absolute epoch) per task per session.
- **Model.** A hierarchical state-space model: per-learner (α_t, α_σ, σ_∞) drawn
  from a cohort hyper-prior; the transition kernel of `learner_sim`/`training_filter`
  is the within-learner likelihood. Fit by particle-MCMC / SMC² or Stan-style HMC
  on the marginal likelihood (the static params are exactly the ones a plain
  particle filter can't estimate online — F4 — so they're fit offline here).
- **Identifiability guardrail.** Re-run the Step-4 gate on the fitted params;
  D14 says priors should lean conservative (under-estimate rates) because the
  filter degrades under over-estimation, not under-estimation (F17).
- **Deliverable.** A cohort hyper-prior + per-learner posterior that the Phase-2
  `LearnerParams` defaults are replaced by; `bank_adapter`-style loader for it.

### A2. LT1 — two-timescale consolidated trait (long-term skill→max, bias→0)
**Goal.** Model the *persistent* trait (σ̄, t̄) that LT1 targets, distinct from the
within-session state.

- **Structure (memory §3A).** Slow consolidated trait (σ̄_k, t̄_k) per task; each
  session's within-session state starts near (σ̄, t̄) and partially consolidates
  back at session close: σ̄ ← σ̄ + κ·(σ_session_end − σ̄). The trainer's long-run
  objective is σ̄ → σ_∞ (expert ceiling, §2B) and |t̄| → 0.
- **Estimable from** the `LearnerLedger` (session-end posteriors over calendar
  time) — the data structure already emitted in Phase 2.
- **Mastery** stays the cert gate σ*_k = exp(−ℓ*_k); since σ_∞ < σ* in every
  domain (§2B), the LT1 target clears the bar.

### A3. LT2 — Ebbinghaus forgetting + spaced-review optimization
**Goal.** Replace the SM-2 default (Step 5) with a fitted decay model.

- **Decay kernel.** Between-session retrievability R(Δt) = exp(−Δt/S) (or a power
  law), with stability S growing under successful spaced retrieval. Drops into the
  `training_filter.propagate_gap(Δt)` call site (D9 hook already wired) and the
  `RetentionScheduler` interface (`due`, `update_on_retrieval`, already abstract).
- **Fit** S-dynamics from cross-session retrieval outcomes in the ledger; the
  optimal review time is then *derived* (review when predicted R hits a target),
  not the hardcoded SM-2 interval.
- **Validation.** The Step-7 benchmark's post-gap retention metric upgrades from
  "process-noise drift" to the fitted decay; compare schedules on retention@7/30d.

### A4. RT observation channel (F10)
Production logs `reaction_time_ms` (heavy-tailed; careless-rater median 152 ms,
deliberate median 8 s). Add an RT likelihood to the filter as an auxiliary
observation on fluency φ: lognormal + an outlier/lapse component (NOT Gaussian).
RT is a fluency proxy (perceptual-learning methods) — informs retirement and the fatigue monitor.

### A5. λ(fatigue) from the free lapse probe (F15)
Retention-mode very-easy items are near-pure lapse probes (P(error) ≈ λ). Their
within-session error rate is a fatigue signal; model λ_k(φ) where φ accumulates
over a session. The probe flag is already emitted on TrialRecords.

### A6. Cross-task transfer (OQ4 / F8)
Phase 2 keeps tasks independent (D2). If pilot data shows transfer (training task
j improves task k), add cross-task terms to the transition kernel — but only with
fitted, identified parameters (re-run the Step-4 gate with the coupled model).

### A7. Eval-side decision-aligned selection (F16) — parallel track
The production `choose_item` already exposes `ell_star`/`decision_tasks` (seen in
`session_controller_general.py`) — a decision-aligned objective the scratch engine
copy lacks (R6). Worth a dedicated ticket: minimize expected Σ π_k(1−π_k) over
PENDING tasks instead of total trait variance; the biggest eval-side efficiency
lever (HOW_THE_TEST_WORKS §5 "REMEMBER THIS").

---

## B. Port checklist → `the main repo`

Scratch module → production touchpoint. Validate parity after each.

| Scratch module | Production target | Notes |
|---|---|---|
| `bridge_conventions.py` | new `engine/trainer_conventions.py` | Assert `LAPSE_RATE` matches the engine's; this is the canonical σ=exp(−ℓ), t=−θ source. |
| `auroc.py` (vendored) | DROP — use the repo's real `auroc` module | Scratch stub only; repo version wins. |
| `bank_adapter.py` | wrap the repo's `engine_inputs_k7.as_engine_arrays()` | Production already assembles per-task banks; reuse it. Keep the label-reconstruction + feedback-safe/exclusion logic (D11/D12). Prefer a definitive domain1 hard label if the repo has one (§5A residual). |
| `training_seed.py` | new `trainer/handoff.py` | `SessionResult` (controller) already carries the cloud, history, verdicts, served seg_ids — map fields directly; add timestamps/ledger (D9). |
| `learner_sim.py` | `trainer/sim/` (research only) | Simulator stays research code; not shipped. |
| `training_filter.py` | new `engine/trainer_filter.py` | Reuses the engine's likelihood; the ONLY new numerics is `propagate` (no MH, F1). Bit-parity test vs the scratch filter on a fixed trace. |
| `trainer_policy.py` | new `trainer/policy.py` + a `Session`-like loop | Mirror `session_controller.py`'s structure (active-domain filtering, consec cap, telemetry). Reuse the TerminationPolicy pattern for graduation (AD6 on filtered posterior, F14). |
| `trainer_greedy.py` | `trainer/policy_greedy.py` (ablation) | Comparator/regression only. |
| `benchmark_trainer.py`, `study_identifiability.py` | `trainer/studies/` | Research/CI campaigns. |
| `pipeline_demo.py` | the sibling trainer app's integration test | The seamless chain template (plan §12 next-step 3: ~15 min/day sibling app). |

**Pre-port gates:**
1. Engine-drift reconciliation (R6): the scratch `core_mcmc_general.choose_item`
   lacks the `ell_star`/`decision_tasks` kwargs the production controller passes.
   Port against the CURRENT production engine, not the scratch copy.
2. `AD6Policy` needs public accessors for verdicts/diagnostics (M1 note) — the
   handoff reads `_verdicts`/`_last_diag` today.
3. BLAS single-thread + seeded determinism for any parity assertions.
4. Re-run the full scratch smoke suite (`test_step0..8`) as the acceptance bar
   before porting each piece.

---

## C. Open items carried to PI / Phase 3
- **OQ3** (PI): re-cert warm-start from trainer posterior vs fresh prior
  (psychometric integrity). Demo uses fresh-prior + exclude-seen (conservative).
- **OQ6** (PI): bias tolerance t* and per-task product mastery targets. Prototype
  assumes t*=0.30; skill targets from config σ*.
- **§5A residual**: definitive per-segment domain1 hard label (votes give ~92%).
- **Source memos** `learning-theory1.md`/`learning-theory2.md` not yet provided —
  may sharpen the A1 priors.
