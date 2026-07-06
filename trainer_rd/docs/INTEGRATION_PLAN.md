# INTEGRATION_PLAN.md — Sequential build plan: eval → trainer pipeline

> Companion to `PROJECT_MEMORY.md` (read that first — conventions in its §2 are
> normative). Scope per Decision D4: prototype here in /data/eli-work/scratch.
> Every step ends with a **smoke test** (must pass before the next step) and a
> **memory checkpoint** (update `PROJECT_MEMORY.md` §6, prune stale info per its §0).
>
> Working rule: each step is small enough to verify independently; if a smoke test
> fails, fix within the step — do not carry debt forward.

---

## Step 0 — Conventions bridge module ✅ DONE 2026-06-10 (checkpoint M0)
**Goal.** One canonical place for parameter conversions and difficulty constants, so
the sign-flip bug (F3) and the 1.04 constant (F2) can never silently re-enter.

**Build:** `bridge_conventions.py`
- `engine_to_plan(theta, ell) -> (sigma, t)` and inverse (`t = −θ`, `σ = exp(−ℓ)`).
- `difficulty_multiplier(target_acc, lapse=0.025)` = Φ⁻¹((a−λ)/(1−2λ)); constants
  `WILSON_OPT_ACC = 0.8413`, default multiplier ≈ 1.077.
- `auroc_from_ell` (re-implement the one-liner; `auroc.py` is absent in scratch).
- Stub/vendor shims for missing imports so the engine file can be imported
  (`auroc.py` minimal implementation).

**Smoke test:** `test_step0.py` — round-trip conversions over random grids; verify
`P(yes)` computed both ways agrees to 1e-12; verify accuracy at the returned
multiplier equals target within 1e-9; verify AUROC(0)=0.8413.

**Memory checkpoint M0:** record constants actually shipped; delete §2 caveats that
the module now enforces.

---

## Step 1 — Trial telemetry & `TrainingSeed` handoff artifact ✅ DONE 2026-06-10 (checkpoint M1)
**Goal.** The seam between eval and trainer (F9, F10, D1, D2).

**Build:** `training_seed.py`
- `TrainingSeed` dataclass + `.npz`/JSON save/load: per-task marginal clouds
  (θ_k, ℓ_k, w) split from the joint cloud (D2), full history (k, s, y, s_sd,
  seg_id, RT placeholder), AD6 verdicts + diagnostics, posterior summaries,
  ℓ*_k cut-scores, eval-seen seg_ids.
- **D9 hooks (LT1/LT2):** absolute wall-clock timestamps + session-ID on every
  trial record and on the seed itself; a cross-session **learner ledger**
  (append-only sequence of seed references over calendar time) so the future
  two-timescale consolidated-trait model (memory §3A) has its data structure
  from day one.
- `build_seed_from_state(state, policy_diag, inflate=1.5)` — variance inflation
  applied per task by rescaling particle deviations about the weighted mean (D1).
- Trial-record schema for the TRAINER side too (mode, fb timestamp, RT,
  answer-changes) — defined now, used from Step 5 on.

**Smoke test:** `test_step1.py` — run a small simulated eval session
(`run_session_mcmc_auroc`, K=3, synthetic bank, `capture_posterior=True`), build a
seed, save → reload → arrays byte-equal; inflated SDs = 1.5× originals (tol 1e-9);
weights renormalized; eval-seen seg_ids complete.

**Memory checkpoint M1:** record seed schema version; raise OQ2 (eval-segment reuse)
if an answer is now needed.

---

## Step 2 — Learner simulator (the transition kernel T) ✅ DONE 2026-06-10 (M2)
**Goal.** Executable ground truth for everything downstream.

**Build:** `learner_sim.py`
- Soft-RW criterion dynamics (plan Eq. tdyn), log-σ relaxation with the
  85%-weight `w(s,θ)` using the Step-0 multiplier and feedback gate (Eq. sdyn),
  process noises q_t, q_σ; lapse-mixture response sampling (reuse engine form).
- Hard-rule variant (Eq. deltahard) and a "no-learning" static variant.
- Config dataclass for (α_t, α_σ, σ_∞, τ_σ, q_t, q_σ). **σ_∞ defaults are the
  data-grounded per-domain expert ceilings exp(−expert_ℓ_mean) from memory §2B
  (D10); mastery σ\* from config `sigma_star`.** α_t/α_σ/τ_σ/q's stay
  literature-placeholder (D7) until pilot data.
- Skill-mode difficulty uses `bridge_conventions.SKILL_MODE_MULTIPLIER` (1.0772, D6).

**Smoke test:** `test_step2.py` — (a) with w=f=1, fitted exponential to mean σ_k
trajectory recovers τ_σ within 10% over 200 sims; (b) under balanced veridical
feedback, E[t_k] → 0 and stays; (c) hard vs soft rule trajectories agree in mean
within MC error (the unbiasedness identity); (d) zero process noise + zero rates ⇒
static learner bit-stable.

**Memory checkpoint M2:** record default dynamics params used and τ recovery stats.

---

## Step 2.5 — Bank adapter (real 89k bank + label file) — NEW (M1.5, D13) ✅ DONE 2026-06-10 (M2.5)
**Goal.** One validated bridge from the real CSVs to per-task candidate arrays;
reimplements the absent `engine_inputs_k7.as_engine_arrays()`.

**Build:** `bank_adapter.py`
- Load `segment_signals_general.csv` → per-task (seg_id, s_mean, s_sd); drop NaN-candidacy cells.
- Load `segment_labels_general.csv` → per-(seg,task) y\* + margin: multiclass
  y\*=(plurality==task), margin from plurality/asked-task vote fraction; domain1
  y\*=(domain1_pos_frac>0.65), margin=|pos_frac−0.65| scaled (D11).
- Filters: `exclude_segids` (eval-seen, D12); `coherent_only` / `min_margin`
  (drop label–signal-conflicted + low-confidence for feedback trials, D11).
- `candidates(task, ...)` → arrays (signals, sds, segids, y_star, margin) in the
  shape `choose_item(bank_signals, bank_sds, bank_segids)` consumes.
- Constants from memory §2B exposed: `ELL_STAR`, `SIGMA_STAR`, `SIGMA_INF` per task.

**Smoke test:** `test_step2_5.py` — candidacy counts match the known per-task totals
(domain1 19,332; others 69,806); label reconstruction of the 3 real sessions'
served labels ≥97% overall; exclusion actually removes eval-seen + low-margin segs;
returned arrays are parallel and finite; conflicted-fraction on a random sample is
≈25–30% (sanity).

**Memory checkpoint M2.5:** record realized counts + label-match rate.

---

## Step 3 — Training filter (bootstrap SMC with propagation) ✅ DONE 2026-06-10 (M3)
**Goal.** The belief-update engine for training (F1, D5), seeded from Step 1.

**Build:** `training_filter.py`
- Per-task 2-D cloud (θ_k, ℓ_k): `reweight()` (reuse engine lapse likelihood with
  s_sd attenuation), **new** `propagate(s, y_star, fb)` sampling from T with the
  D7-fixed dynamics params, multinomial resample when ESS < 0.5·N. **No MH.**
- `steady_state_sd()` utility: long-run filtered SD under no-information trials —
  the variance floor (F5).
- **D9 hook (LT2):** `propagate_gap(dt_seconds)` — between-session decay kernel
  applied when a session opens after a gap. Identity transform in Phase 2 (plus
  optional variance widening), but the call site exists so the fitted Ebbinghaus
  kernel (memory §3A) drops in without filter surgery.

**Smoke test:** `test_step3.py` — (a) track a Step-2 simulated drifting learner:
posterior mean RMSE bounded, 95% credible coverage of true (σ_k, t_k) in [90, 99]%
over 100 replicate sessions; (b) static learner + zero process noise: filtered
posterior matches the eval engine's static posterior on the same data (moments
within MC tolerance); (c) measure and record the variance floor; assert the
planned δ_σ mastery threshold sits above it; (d) **real-session fixture (R8):** feed
the careless rater (acc 0.23) and confirm the filter does NOT reach mastery on any
task and recovers near-chance/strong-bias state. Uses real banks via Step 2.5 with
realistic s_sd.

**Memory checkpoint M3:** record variance floor numbers; resolve OQ5 (inflation
factor) using coverage results; if degeneracy was observed, record the D5 revision.

---

## Step 4 — Identifiability gate (hard gate — do not pass on failure) ✅ PASSED 2026-06-10 (M4)
**Goal.** Verify (σ_k, t_k) trajectories are jointly recoverable while BOTH move
(F11), with dynamics params fixed at wrong-by-±2× values (F13 stress).

**Build:** `study_identifiability.py` — simulation campaign: grid over true
(α_t, τ_σ, σ_0, t_0), trainer-like item streams, filter from Step 3; recovery
metrics (trajectory RMSE, terminal-state coverage, σ-vs-t confusion check:
correlation of estimation errors).

**Pass criteria (pre-registered):** coverage ≥ 90% nominal; |error correlation|
between σ̂ and t̂ ≤ 0.5; misspecified-α degradation graceful (no divergence).

**Memory checkpoint M4:** record pass/fail + plots/tables paths. On failure: revisit
q's, item placement diversity, or session length — and update D5/D7 before Step 5.

---

## Step 5 — Tier-2 policy + cross-task scheduler ✅ DONE 2026-06-10 (M5)
**Goal.** The plan's mode-conditional policy (plan §8.2/§10) made K-task-real (F8, D3).

**Build:** `trainer_policy.py`
- Mode gate per task: bias-correction / skill-building / retention, with cut-offs
  t*, σ*_k = exp(−ℓ*_k) (F14) and posterior-SD multiplier c.
- Bias mode: items near t̂_k with **running label-balance constraint** (F6b).
- Skill mode: |s − t̂_k| ≈ 1.077·σ̂_k (D6), morphology rotation, criterion-vs-boundary
  guard (F7).
- Retention mode: spacing over mastered bins behind a **decay-model interface**
  (D9/LT2): `due(bin, now) -> bool` + `update_on_retrieval(bin, outcome, now)`.
  SM-2 is the default Phase-2 instance; a fitted retrievability/stability model
  (R(Δt)=exp(−Δt/S), memory §3A) replaces it in Phase 3 without policy changes.
  Retention reward **eligibility-gated** on `due` (F6a).
- Mastery/graduation: AD6-style π_k − Z·mcse ≥ 1−α on the FILTERED posterior, plus
  floor-aware SD gate from M3 (F5, F14).
- `scheduler.py` (or same module): deficiency-weighted interleave across tasks (D3),
  deficiency score per OQ1 (decide at this step; record in memory).
- Per-trial telemetry logging per Step-1 schema, incl. RT field + easy-item lapse
  probe counter (F15).

**Smoke test:** `test_step5.py` — unit: mode gating on synthetic posteriors; label
balance maintained within tolerance over 500 bias-mode trials; spacing schedule
fires at expected intervals. Integration: simulated learner (Step 2) reaches
graduation on all K tasks within a finite trial budget; no mode oscillation
thrashing (mode switches bounded).

**Memory checkpoint M5:** resolve OQ1; record default mode thresholds; prune any
§3 findings now structurally prevented by code.

---

## Step 6 — Tier-1 greedy + ablations ✅ DONE 2026-06-10 (M6)
**Goal.** The Eq.-(Qgreedy) one-step policy as comparator, with the F6 guards.

**Build:** `trainer_greedy.py` — per-particle expected-reward scoring over the
candidate bank (closed form under soft rule; y-dependent form for the hard
ablation), β-weights config; WITH and WITHOUT the balance/eligibility constraints
(the unconstrained variant exists only to demonstrate the exploit, as a regression
test).

**Smoke test:** `test_step6.py` — greedy beats random item choice on one-step
expected reward (sanity); unconstrained variant demonstrably exploits retention
spam / one-sided labels (assert detection); constrained variant does not.

**Memory checkpoint M6.**

---

## Step 7 — Benchmark campaign ✅ DONE 2026-06-10 (M7)
**Goal.** Plan §12 benchmarks, plus misspecification stress (F13).

**Build:** `benchmark_trainer.py` — policies {tier-2, tier-1 constrained, pure
85%-staircase, measurement-optimal placement (the "wrong objective" baseline),
random} × learners {soft, hard, α off ±2×, non-exponential σ-curve} × ≥50 seeds.
Metrics: trials to σ ≤ σ*, trials to |t| ≤ t*, held-out post-test at unseen
difficulty bins, retention proxy after simulated gap (process-noise-only drift).
Report means + bootstrap CIs.

**Smoke test:** small-N pilot run completes end-to-end; tier-2 ≥ staircase on
trials-to-mastery on the well-specified learner (if not, investigate before the
full campaign — this is the headline claim).

**Memory checkpoint M7:** record the results table; this is the evidence base for
porting to the main repo.

---

## Step 8 — End-to-end seamless pipeline demo ✅ DONE 2026-06-10 (M8)
**Goal.** The user's stated objective: take the test → results inform the trainer.

**Build:** `pipeline_demo.py` — (1) simulated candidate takes the eval
(`run_session_mcmc_auroc` + AD6 verdicts); (2) `TrainingSeed` built (D1/D2);
(3) N daily training sessions under the Step-5 policy with the Step-2 learner
responding AND actually learning; (4) simulated re-certification; (5) one
report (JSON + printed summary) showing the full telemetry chain: question
progression, correct/incorrect patterns, skill/bias trajectories, RT fields,
mode decisions, graduation, re-test verdict flips FAIL/REFER → PASS.

**Smoke test:** a mid-skill simulated candidate who initially FAILs ≥2 tasks PASSes
them after training; total wall-clock sane; report self-contained.

**Memory checkpoint M8:** raise OQ3 (warm-start re-cert) for PI; final stale purge.

---

## Step 9 — Phase-3 prep & port checklist (docs only) ✅ DONE 2026-06-10 (M9)
- Spec for offline hierarchical fitting of (α_t, α_σ, σ_∞, q's) from logged trial
  records (D7 payoff); RT observation-channel design; λ(fatigue) from the F15 probe;
  cross-task transfer model sketch (OQ4).
- **LT1/LT2 methodology spec (memory §3A):** the two-timescale consolidated-trait
  model (session state vs slow trait; consolidation update at session close) and
  the Ebbinghaus decay kernel `T_gap` with retrievability/stability dynamics —
  including how both are fitted from the learner ledger (Step 1) and how the
  derived spacing schedule replaces SM-2 via the Step-5 interface.
- Port checklist to `the main repo`: which scratch modules map to which
  engine touchpoints (`update`/`choose_item`/`session_controller`/policy registry),
  incl. the eval-side decision-aligned selection lever (F16) as a separate ticket.
- Final memory consolidation.

---

### Standing rules
- Reproducibility: every sim/test takes an explicit seed; single-thread BLAS if
  bit-exactness is asserted.
- Never edit the five source artifacts in place except `PROJECT_MEMORY.md` and this
  plan; new code goes in new modules.
- A step's smoke test lives in `test_step<N>.py` and must be re-runnable with
  `python3 test_step<N>.py` (no framework dependency assumed).
