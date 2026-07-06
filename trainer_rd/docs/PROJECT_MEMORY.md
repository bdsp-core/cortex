# PROJECT_MEMORY.md — Eval→Training Pipeline (unified memory)

> **Purpose.** Single source of truth for the project connecting the
> adaptive evaluation engine (SMC measurement of skill/bias) to the POMDP-based
> learning/training algorithm. Later agents and humans should read THIS FILE FIRST.
> Every claim about the code is checkable against the files in this directory.
>
> **Last consolidated:** 2026-07-05 (checkpoint M27 — **the
> pre-submission completion loop (user-directed; a legitimate D43
> stopping-rule reopen, category 1)**, `docs/M27_TERMINAL.md`: **F90**
> terminal-confirmation semantics (confirmed tasks → retention layer +
> anytime-valid stale e-gate; consolidated review schedule — the
> bin-proliferation finding is port-relevant) validated opt-in with an
> anchored-confirm harness: pair completion +20pp (rates 256@65% →
> 212@85%) at a 50%-undetected-regression staleness blind spot ⇒ **D44
> default OFF**, ratification filed as ADVISOR_BRIEF ask #7 (it is the
> D18 semantics taken seriously — needs scheduled re-certs). **F91**
> contact-triggered onboarding (demo ramp until the F83 e-process
> certifies + trainability-guarded Gate-4 handoff): pre-contact
> censored@30% → 148@70%, wellspec no-harm (100% rate, +14 med),
> careless FG 0 ⇒ **D45: `CONTACT_ONBOARD=True` ships in the sandbox**.
> **F92** λ(time-on-task) CLOSED by real data (within-sitting easy-read
> accuracy RISES +8.7pp/h — learning dominates; not sign-identifiable
> observationally; F79 guard stays the operational answer). **F93**
> domain-pluggability interface (`training/domain.py`: ItemBank /
> ArrayBank / Domain + `v15_domain()`; toy domain graduates through the
> unchanged stack) + `docs/ARCHITECTURE.md` + EXTSET release →
> `data/extset/`. Suite 21 files / 299 checks. M26.1 —
> **SUBMISSION-FROZEN, D43:
> the algorithm is ready for advisor review; the gate + stopping rule
> live in `docs/SUBMISSION_CRITERIA.md` — check it BEFORE starting any
> new algorithm loop.** Prior checkpoint M26 — **the queued
> evidence-adaptive declared-task boundary hazard (M25 §6 item 1)
> executed as one PECR loop; honest verdict: validated OPT-IN, sandbox
> default UNCHANGED** (`docs/M26_HAZARD.md`): **F89** Beta posterior-mean
> decay c0/(c0+n) of Gate 4's (ε_b, w_share) for declared tasks
> (`declared_hazard_scale` + `AdaptiveBoundaryHazard` π-credit lifecycle,
> `training/mixture_filter.py`; sandbox wiring behind `HAZARD_ADAPT`,
> False = bit-identical M25). Validation `study_m26_hazard` (fixed-hazard
> baselines replicate M25 arm Q2 bit-for-bit): the completion-tail gain
> exists exactly where the F85-ii shape exists — rates BOTH 300@55% →
> 275@60% (c2f25) / 266@65% (c5f25) / 255@75% (geometric), first
> declarations bit-unchanged, jumper/b-trap unchanged — BUT arm G shows
> the gain is COUPLED to a sharp-margin (ℓ*−0.15) regression blind spot:
> post-regression false-mastery flicker fixed +0 vs +14/+71/+100 trials
> (non-monotone middle), while careless/static FG (6/20 declared → 0/20
> CONFIRMED, identical) and deep-regression detection are unchanged at
> every pin ⇒ **D42: `HAZARD_ADAPT` stays False** (D28: an FG-adjacent
> persistence channel outranks a +5pp lateness win at the safe pin);
> best-priced pin (c0=2, floor 0.25) recorded for field ablations;
> primary successor re-queued = TERMINAL-CONFIRMATION semantics
> (retention layer owns confirmed tasks — removes the re-polish tax
> without weakening the hazard). Housekeeping: `archive/` convention
> established (first entry `sbc_run.log`); `TRAJECTORY_CADENCE_PLAN.md`
> added to §1 (was unreferenced). Suite 20 files/287 checks. Prior
> checkpoint M25 — **the three M24
> selection successors executed as one PECR loop; three honest verdicts,
> no default changes** (`docs/M25_SELECTION_V2.md`; advisor-facing
> summary `docs/ADVISOR_BRIEF.md`): **F86 EP-v2** (confirmation-lifecycle
> lane + exploration floor) implemented + validated but ENDPOINT-DECISIVE
> NO — the Gate-4 hazard's ~28-trials/session re-polish of every declared
> task is allocator-invariant, Gate-1 dominates the BOTH-declared
> endpoint (300@55%/240@70% vs censored@20%/5%), sandbox
> `PROGRESS_ALLOC` stays False (D41); successor moves to the HAZARD
> level (evidence-adaptive boundary hazard for declared tasks). **F87**
> item-level tier-3 rollout placement (mixture-honest, herding-safe): no
> dominance ⇒ default-off, the F84→F87 placement line CLOSED — plug-in
> 85%-rule placement stands under the hazard-widened belief. **F88**
> sharp-margin gate: information bound derived (δ≈0.03 ⇒ ~1.4k probes /
> ~600 trials for BF 20 at ℓ*−0.15); instrumentation shows FGs are
> mid-session marginal π crossings with trainability 0.92 and a quiet
> e-gate; all three conjuncts REJECTED (refractory/trainability/probe
> densification); **the shipped D33 confirmation lifecycle is the
> filter** (declared 6/20 → CONFIRMED 1/20) ⇒ FG claims move to
> CONFIRMED semantics (D41). Plus per-tester trajectory MP4s
> (`viz/make_user_videos.py`) and `docs/ADVISOR_BRIEF.md` (the C4
> sign-off vehicle). Suite 19 files/273 checks. Prior checkpoint
> M24 — **question
> selection by expected progress** (`docs/M24_SELECTION.md`): the
> delivered-value audit put mean per-trial training value at 0.48 of
> ideal; the stratum-aware selection machinery shipped OPT-IN
> (`expected_progress_score`/`expected_progress_rate`, F84-ii
> static-stratum fix) and BOTH default flips were correctly blocked —
> **F85 EP allocation** wins first-declarations decisively (134–184 →
> 72–86 trials, sunk trials halved) but F85-ii (post-boundary re-polish
> loop starving recovering tasks; unfavorable-not-significant sharp-margin
> FG direction 27/60 vs 22/60) defers the sandbox default; **F84
> placement** is a six-variant mixed result (tail vote, calibration
> externality — M10 mirror restored, model-injected hazard width;
> full-run near-parity with a jump/slow scenario split). Successors: EP
> coupled to the confirmation lifecycle + exploration floor
> (BOTH-declared endpoint); item-level tier-3 rollout. Suite 18
> files/258 checks. Same-day prior checkpoint
> M23 — **seven-profile
> regime-shift analysis; the boundary-shift stack shipped**
> (`docs/M23_REGIME_ANALYSIS.md`): the 2026-07-03 sessions (A-s3, C-s2,
> E×3, F, G×2) show acquisition as a DISCRETE regime shift at sitting
> boundaries (E: d′ 0.8→3.0 across a 64 s restart) that the OU kernel
> tracks a full session late while BMA trainability sits poisoned;
> onboarding is a no-contact guessing state, and careless collapse has a
> 5×-margin RT signature. Fixes F80–F83 / D38–D39, all opt-in
> bit-identical: **Gate 4** `boundary_shift` (σ-dominant state jump +
> fixed-share ceiling re-mix; E-like jump lag 93.5→15.2 trials) with the
> sandbox declaration floor re-measured 0.23→0.33 (F81-iii); **Gate 3b**
> RT floor 500 ms/k=3; shadow **contact e-process** (KT/0.55, anytime
> α=0.05). PECR rejections recorded: per-trial DMA forgetting (F81-ii,
> declaration 100%→30%) and the stage-1 hyperparameter pin (arm G).
> Validated by `study_m23_regime`; suite 17 files/247 checks. Prior
> checkpoint M22 — **four-participant
> profile pilot analyzed; the three protocol gates shipped**
> (`docs/M22_PARTICIPANT_GATES.md`): USER-A learned cleanly; B/C/D exposed
> three mechanisms — F77 worst-first allocation starves near-mastery tasks
> while trainability collapses (+ F77-ii finishing-eligibility deadlock at
> the mixture variance floor), F78 shadow flags don't stop the bleeding
> (USER-D: 10 post-flag trials, then quit), F79 late-session fatigue lands
> in the persistent belief. Gates: trainability-discounted deficiency
> (`trainability_floor=0.25`) + `finish_sd_tol=1.25`; consistency-pause;
> differenced fatigue guard (δ=0.35). Decisions D36–D37; engine changes
> opt-in bit-identical; validated by `study_participant_gates`; suite 16
> files/235 checks green. Prior checkpoint M21 — **first two human
> tester sessions analyzed end-to-end** (`docs/M21_TWO_TESTER_ANALYSIS.md`):
> the two testers ran back-to-back on ONE belief state — findings
> F72 (identity must key state; trainability pollution + criterion
> mis-attribution, invisible to all shipped monitors), F73 (render gain
> bounded achievable σ at 0.606 vs bars 0.736/0.857 — recalibrated to
> 0.85), F74 (10% of served trials rendered evidence opposing feedback —
> label-consistent draws now), F75 (novice onboarding served ~50%-acc
> items vs the 84% target — opt-in conservative-quantile placement),
> F76 (shadow ConsistencyMonitor OCs: corruption tripwire, not an
> identity oracle). Decisions D34 (profiles + shadow monitor) / D35
> (stimulus contract). Sandbox now takes `--user` per tester; old state
> archived; suite 16 files/219 checks green. Prior checkpoint M20 — **the
> manual-tester SANDBOX is OPERATIONAL** (`sandbox/`, F71): the user trains as the
> learner via `python3 -m sandbox.session`, full validated stack +
> wall-clock GapAnchor + the D33 post-gap confirmation rule (C2
> discharged), production-shaped telemetry, resumable state, analytics via
> `sandbox.report`; verified by test_sandbox (11 checks) + an 800-trial
> robot campaign (archived); suite 15 files/193 checks green. Prior
> checkpoint M19 — queue executed +
> **TESTER-READINESS VERDICT: CONDITIONAL GO** — shadow-mode testers NOW;
> supervised feedback-driven cohort after 4 named conditions (see M19 §6
> row); public release stays the last confirmatory rung. Key findings:
> **F70** K=7 full-protocol capstone (scheduler deadlock found+fixed via
> opt-in finish-first D32; CONSOLIDATION is load-bearing — a named pilot
> endpoint; measure-or-leave-alone gap re-anchoring 2.1× throughput;
> pilot scopes K=1–3); **F67** domain1 narrow-margin handoff protocol;
> **F69** drift-aware shrinkage settles the deployment estimator. Prior
> checkpoint M18 — the M17 queue
> executed: **F64/D30** the static_below residual was the D16 MEAN-SKILL
> branch (11/11 attribution; disabled for mixture graduation → capstone
> adversarial FG **1/90**, F66); **D29** F62-anchored priors adopted;
> **F65** hierarchical refit on 2,040 real half-fits — per-user rates ARE
> identifiable (r≈.48), personalization pays held-out (+.008/read,
> reversing the F58 sim), odd/even drift-aware shrinkage queued; **D31**
> pipeline v15-coherent end-to-end (all 3 demo tasks now flip to PASS).
> Next queue in the M18 §6 row. Prior checkpoint M17 — user ratified the
> decision list (D25–D28) and the iteration executed it end-to-end:
> **defaults flipped** (exact kernel; v15 trainer cuts; derived 0-bias
> band), **F60** zero-bias theorem (OQ6 resolved: anchored serving ⇒
> E[t∞]=0, stationary floor validated ≤3%, alternation load-bearing),
> **F61** `GapAnchor` measurement-based retention (D27: stale-mastered
> 71→0, ℓ̂ error −80% vs power-law truth; answers the update-cadence
> question), **F62** first REAL-DATA dynamics priors (1,158 EXTSET fits:
> α_t 0.097 — the 0.2 default over-estimates ×2; median ceiling below every
> v15 cut), **F63** capstone zoo OC under the new stack (careless/anti FG
> 0/30 with e-gate; static_below residual 11/30 → D18 re-cert still owns
> certification; λ locked = calibration-benign). Suite green: 14 files,
> 182 checks. Priorities normative: FG ≥ lateness > speed > retention
> (D28). Next queue in the M17 §6 row. Prior checkpoint M16 — same-day
> iteration on
> M15: **F57** exact-kernel closed-loop SBC NOMINAL at every checkpoint —
> F33 CLOSED, D21 evidence complete (incl. benchmark re-pin: tier2 Δ+0.0);
> **D23 wired** into `TrainerPolicy(probe_every=)`; **F58** Phase-3 offline
> dynamics fitter (`dynamics_fit.py`) train/test-validated — population
> params recovered, personalized fits overfit ⇒ hierarchical shrinkage
> mandatory, fitted params HALVE declaration lateness 41→21.5 trials;
> **F59** one-step evidence flat in dynamics params ⇒ power pilot analyses
> on decision-level outcomes. Audit doc Part III. Next: hierarchical refit,
> fitted-ℓ_∞→mixture-center study, LT2 multi-session fitting. Prior
> checkpoint M15 — user-requested deep
> POMDP math audit EXECUTED end-to-end: audit Part I (findings MA-1…MA-10)
> + seven-step hardening Part II, both in `docs/AUDIT_POMDP_MATH.md`;
> memory findings F53–F56, decisions D20–D24; full suite green incl. new
> `test_audit_fixes` (15 checks). Highlights — the EXACT conditional
> transition kernel (`TaskFilter(exact_kernel=True)`, 21-node GH) restores
> nominal filter calibration (cov_t@90 0.63→0.91 at better RMSE, F53;
> recommended port default D21 with gates re-tuned via
> `recommended_sd_floor`); the σ∞-MIXTURE belief (`mixture_filter.py`)
> replaces the F28 prophecy with an honest trainability posterior
> P(ℓ∞>ℓ*|data) (careless FG 20/20→0/20, static_below 20/20→7/20, D22);
> cut-state certification probes + anytime-valid e-process gate shipped as
> library (F55/D23); F25 re-adjudicated — tier-3's hard-rule advantage
> VANISHES under the exact kernel (F56, tier-2 default reinforced); F22
> criterion-feedback gating FIXED as default (D20). Remaining M15 wiring:
> TrainerPolicy probe/e-gate integration + the D21 re-pin campaign, both at
> port. Prior checkpoint M14 — "Nature-gap"
> publication-hardening program **COMPLETE in-scratch**: all Tier-A/B/C
> analyses closed, findings F45–F52. Highlights — A1: composed marginals beat
> the single-skill softmax even with a per-user lapse dim (F49); A2: Bayesian
> refit puts contest-rater lapses an order below the engine's 0.025 (P=0.9999)
> and a proper R̂-gated SBC calibrates 5/6 globals — F51's blanket SBC failure
> was an under-convergence artifact, only σ_θ is residually under-estimated
> (F51/F52); B8: the K=7 re-cert replicates F32's protective direction but is
> REFER-dominated at the cut, which is the instrument's OC behavior not a
> pathology (F50). Figures fig15–20 via `make_m14_figures`. **Remaining work is
> all OUTSIDE scratch-analysis: PI-gated decisions (D18/D19/D16, F26/F27, OQ6,
> v15 consumer switch), the pilot (`PILOT_SAP_M14.md`, gated on OQ8), and the
> production port incl. the decisive K=7+corr_t coverage run.** Prior checkpoint
> M13 — EXTSET scrubbed release
> LANDED IN SCRATCH: 125,860 real reads × 699 users × 5,000 task2 cases with
> full engine signals + 4-expert panel; findings F35–F38 (incl. F35 bank-vote
> leakage), inventory §3E, analysis plan `EXTSET_SAP_M13.md`. Prior checkpoint
> M12 — verification-ladder
> campaign: rung-3 mis-specification envelope + rung-4 OC/SBC + rung-5
> in-scratch real replay COMPLETE; findings F28–F34, mitigations implemented
> opt-in; see §3D, §6 M12, `VERIFICATION_PLAN_M12.md`. Prior consolidation
> M10 — post-audit fix round:
> review findings F18–F22 fixed and verified, production-faithful s_sd physics
> (F20) threaded through filter/policy/benchmark/eval, noise-aware item
> selection (D15), hybrid mastery gate (D16), due-driven retention (D17), full
> suite green (~93 checks). Benchmark headline claims RE-BASED under realistic
> physics — see §6 M10. Remaining: Phase-3 modeling + production port — see
> `PHASE3_AND_PORT.md`).

---

## 0. Maintenance protocol (read before editing)

1. **Checkpoints.** After every plan step (see `INTEGRATION_PLAN.md`), append results
   to §6 (Status log) and update any section the step invalidated.
2. **Stale deletion.** When a finding/decision is superseded, DELETE the old text and
   leave a one-line tombstone in §7 ("superseded: <what> → <what>, <date>"). Do not
   accumulate dead prose. Tombstones older than two checkpoints get deleted too.
3. **Dating.** Every entry in §4 (Decisions) and §6 (Status log) carries a date.
4. **Numbers are normative.** The constants in §2 were verified numerically on
   2026-06-10. If you change the lapse rate or the accuracy target, recompute them
   and update §2 in place.

---

## 1. System map (what lives where)

| File | Role |
|---|---|
| `learning_algorithm_plan.md` | LaTeX design memo for the **trainer**: POMDP formulation, transition kernels (Rescorla–Wagner criterion dynamics, exponential σ relaxation), reward, 3 approximation tiers, 3-mode policy. Reviewed 2026-06-10; findings in §3. |
| `HOW_THE_TEST_WORKS.md` | Walkthrough of the **production evaluation test**: probit-lapse SDT model, joint SMC posterior over 2K traits, A-optimal item selection, AD6 verdict rule. |
| `core_mcmc_general.py` | The SMC engine: prior, reweight (`update`), ESS-triggered multinomial resample + MH rejuvenation (`resample_and_rejuvenate`), A-optimal `choose_item`, Mode-A session driver `run_session_mcmc_auroc` (with `capture_posterior` trajectory output). |
| `policy_general.py` | `AD6Policy` (π_k / mcse_k / R_k gates, monotonic verdict lock, REFER split), `DeltaStopPolicy` (legacy), `NoStopPolicy`. |
| `policy_k7_general.py` | K=7 wiring of AD6 (cut-score loader; AD6Policy itself is K-agnostic). |
| `INTEGRATION_PLAN.md` | The sequential build plan (steps, smoke tests, memory checkpoints). |
| `bridge_conventions.py` | **Step 0 (DONE).** Canonical parameter conversions (σ=exp(−ℓ), t=−θ), lapse-corrected difficulty multipliers, AUROC mappings, σ*=exp(−ℓ*) mastery mapping. The ONLY place these may be defined. |
| `auroc.py` | Vendored minimal stand-in for the main repo's auroc module (the 3 names `core_mcmc_general.py` imports). Main-repo version wins at port time. |
| `test_step0.py` | Step-0 smoke test (23 checks, incl. engine end-to-end with the stub). Re-run with `python3 test_step0.py`. |
| `training_seed.py` | **Step 1 (DONE).** `TrainingSeed` (eval→trainer handoff: joint cloud raw + inflated, history incl. seg_id/RT/epoch/post_decision, AD6 verdicts+diagnostics, ℓ*), `build_seed_from_state`, `inflate_cloud` (D1), `TrialRecord` (trainer-side per-trial schema, F10/D9), `LearnerLedger` (append-only cross-session sequence — the LT1/LT2 data structure). All latents stored in ENGINE coords. `.npz`, no pickle. |
| `test_step1.py` | Step-1 smoke test (18 checks) — runs a real 200+-trial mini eval session (engine loop + AD6) and verifies the full handoff. |
| `learner_sim.py` | **Step 2 (DONE).** `Learner` + `LearnerParams` — the transition kernel T (soft/hard/static R–W criterion dynamics + log-σ relaxation toward σ_∞, 85%-weight, feedback gate; **M15 D20: the criterion δ is f-gated too — F22 fixed**). State in PLAN coords; `engine_state()` bridges. Cross-task independent (D2). |
| `test_step2.py` | Step-2 smoke (8 checks): τ_σ recovery, |t|→0 under balanced feedback, hard≈soft expected dynamics, static stability, σ→σ_∞ no-overshoot, f=0 freeze. |
| `bank_adapter.py` | **Step 2.5 (DONE; reconstructed M9.1 after file loss, user-ratified M10).** `BankAdapter` over the real 89k bank + label file → per-task `TaskCandidates` (seg_id, s_mean, s_sd, y\*, margin, coherent). Exclusions: eval-seen (D12), feedback-safe coherent+margin (D11). Exposes ELL_STAR/SIGMA_STAR/SIGMA_INF (§2B). **M10 domain validation:** per-domain label match vs production `target_present` = 1.000 on all multiclass domains in both labeled real sessions (only deviations: domain1 0.92 — the known pos_frac-rule residual §5A — and 1/186 domain7); all served seg_ids resolve in pools; pool y\*=1 base rates 0.07–0.48; served-item y\* rates 0.3–0.6 (A-optimal selector preference, not a pool defect); s_mean ranges/s_sd medians match §2B. |
| `test_step2_5.py` | Step-2.5 smoke (9 checks): candidacy counts, 98% label reconstruction vs real session, exclusion filters, σ_∞<σ* invariant. |
| `training_filter.py` | **Step 3 (DONE; upgraded M10).** `TaskFilter` — per-task bootstrap SMC (reweight → SYSTEMATIC resample → propagate; NO MH, F1/D5). Propagation mirrors `learner_sim` T; hard rule conditions on the OBSERVED y (F21); soft-rule δ uses the s_sd-attenuated E[p] (F20). `is_mastered` = AD6 on filtered posterior, floor-aware (F14/F5). `propagate_gap` LT2 hook (D9). `steady_state_sd(s_sd=…)` (information-balanced floor; 0.152±0.002 synthetic, 0.179 under real-bank s_sd). `filter_from_seed_task` (D2 handoff). **M15:** `exact_kernel=True` exact conditional kernel (F53/D21; dedicated 21-node GH rule; raises the floor — θ-SD 0.153→0.222, ℓ-SD 0.114→0.155 at s_sd=0.85 like-for-like ⇒ re-tune gates via new `recommended_sd_floor`); `reweight` returns the prequential evidence increment (mixture hook); honest `pass_mass` MCSE (min(ESS, distinct-lineage n_anc)); criterion channel f-gated (D20). **M23 (F80):** opt-in contaminated-innovation jump noise (`jump_eps/jump_kappa`, shared per-particle regime indicator, defaults bit-identical) + `boundary_jump(eps, sd, theta_scale)` session-boundary regime-shift shock (σ-dominant). |
| `test_step3.py` | Step-3 smoke (7 checks): 96% credible coverage of a drifting learner, static-filter==grid-posterior, variance floor, real careless-rater certifies nothing (R8). |
| `study_identifiability.py` | **Step 4.** Joint-motion recovery campaign (grid × ±2× rate misspec): coverage, RMSE, σ/t error-confusion correlation, posterior coupling. |
| `test_step4.py` | **Step 4 HARD GATE (PASS, 6 checks).** Identifiability holds: err-corr ≤0.17, coverage 0.91, graceful misspecification. |
| `trainer_policy.py` | **Step 5 (DONE; upgraded M10).** Tier-2 policy: `TaskModePolicy` (bias/skill/retention gating; F6b label-balance, F7 boundary guard; **noise-aware selection D15**: skill = `expected_skill_weight` argmax, bias = `bias_correction_score` cloud argmax with 4096-item cap; **hybrid graduation D16**: AD6 pass-mass OR trailing-mean-skill), `RetentionScheduler` (SM-2 default behind the LT2 decay interface, caller-injectable for unit consistency), `DeficiencyScheduler` (D3/OQ1 + consec cap R7), `TrainerPolicy` orchestrator (**due-driven retention serving D17**, transition-only bin registration, F15 lapse-probe flag, TrialRecord logging). t* assumption flagged (OQ6). **M15:** `cert_probe_score` (Fisher info about ℓ at the CUT state — scale-parameter optimum z̃≈±1.35, F55) + `EProcessGate` (anytime-valid below-bar refutation monitor, Ville α-control; D23) + `at_bar_accuracy`. **M16: probes/e-gate WIRED into `TrainerPolicy(probe_every=, egate_alpha=)`** (opt-in; probe override on per-task cadence, e-gate blocks declaration while `elevated`; default None bit-identical). **M17 (F60/OQ6):** `bias_stationary_sd`/`derived_t_star` — the derived σ̂-adaptive 0-bias band is now the DEFAULT gate bias condition (`ModeThresholds.t_star=None`; a float restores the legacy fixed tolerance). **M21 (F75):** `ModeThresholds.skill_sigma_z` — opt-in conservative-quantile skill placement σ_place = σ̂·exp(z·max(sd_ℓ−sd_floor,0)) for novice onboarding; 0.0 default bit-identical. **M22 (F77/F78):** `DeficiencyScheduler(trainability_floor=)` — effective deficiency × (floor + (1−floor)·P(trainable)), opt-in; `finish_sd_tol=` finishing-ELIGIBILITY sd tolerance (declaration keeps the strict floor); `TrainerPolicy.suspended` + `pick(exclude=)` in-session task suspension (the consistency-pause hook). All default-off bit-identical. **M24 (F84/F85):** `expected_progress_score` (stratum-aware tier-1 Q over mixtures; static strata score 0), `expected_progress_rate` (bar-referenced EP per trial), `ModeThresholds.progress_placement` (default-off EXPERIMENTAL, F84 negative result), `DeficiencyScheduler(progress_alloc=, progress_s_sd=)` (D40: opt-in everywhere; sandbox DEFERRED by F85-ii). **M25 (F86/F87/F88):** `DeficiencyScheduler(progress_lifecycle=, refinish_threshold=, explore_every=)` EP-v2 confirmation-lifecycle lane + exploration floor (+ `mark_declared`/`ever_declared`); `ModeThresholds.rollout_placement` (+`rollout_H/L/nro/short`) item-level tier-3 placement; `ModeThresholds.boundary_refractory`/`min_trainability` sharp-margin declaration conjuncts (+ `TaskModePolicy.note_boundary`). All default-off bit-identical. **M27 (F90):** `TrainerPolicy(terminal_confirmation=, maint_probes=, stale_alpha=)` — terminal tasks leave the training rotation (retention + at-bar maintenance probes only; ONE consolidated (k,0) review schedule — transition registration is exempted for terminal tasks, the bin-proliferation fix), stale-mastery `EProcessGate` revokes on a Ville crossing (`mark_terminal(now=)`/`note_session_open`/`terminal_events`). Default-off bit-identical; D44 keeps the sandbox off pending advisor ask #7. |
| `test_step5.py` | Step-5 smoke (15 checks): mode gating, label balance, spacing, scheduler, + integration (trainee graduates 3/3 tasks in 159 trials, no thrashing). |
| `trainer_greedy.py` | **Step 6 (DONE).** Tier-1 myopic policy (`greedy_select`, Eq. Qgreedy expected-reward scoring) as the Tier-2 comparator; constrained (label-balance + retention eligibility) vs unconstrained variants. The unconstrained variant exists to witness the F6 exploits as regressions. **M24 (F85):** `_expected_reward(bar_ell=)` opt-in bar-referenced σ-credit. |
| `test_step6.py` | Step-6 smoke (6 checks): greedy beats random on Q; F6b base-rate exploit (unconstrained excursion 24 vs constrained 1); F6a retention spam (ungated picks easy item, gated does not). |
| `benchmark_trainer.py` | **Step 7 (DONE).** Single-task trials-to-mastery campaign: 5 policies (tier2/tier1/staircase85/measure_opt/random) × {soft,hard} × rate-misspec × seeds. `run_benchmark(pool_size=…)`. |
| `test_step7.py` | Step-7 pilot (5 checks, 18s): reproduces the campaign ordering. |
| `pipeline_demo.py` | **Step 8 (DONE).** `run_pipeline()` — the full seamless chain: eval (engine+AD6) → `TrainingSeed` handoff → N training sessions (Tier-2, real bank, LT2 gap) → re-cert. `PipelineReport` captures verdict progression, skill trajectories, flips. Eval pool subsampled (`eval_pool`) for tractability. |
| `test_step8.py` | Step-8 capstone (7 checks, 57s): candidate FAILs 3/3 → trains → re-certifies PASS 3/3. |
| `PHASE3_AND_PORT.md` | **Step 9 (DONE).** Phase-3 methodology spec (dynamics fitting, LT1 two-timescale trait, LT2 Ebbinghaus, RT channel, fatigue, transfer, eval-side F16) + main-repo port checklist + pre-port gates. |
| `segment_signals_general.csv` | **REAL bank, 89,138 segments.** Per-(seg,task) `s_mean_{code}`/`s_sd_{code}` (code∈domain1..7). Candidacy per task: domain1 19,332; others 69,806. s_sd medians 0.6–0.98 (NOT negligible — must thread through filter+selection). Plus rater-tier vote counts. |
| `segment_labels_general.csv` | **Ground-truth/vote source, 95,327 segs.** Reconstructs production `target_present` at 97.2% (multiclass `plurality` 100%; domain1 `domain1_pos_frac > ~0.65` ≈92%). Margin (`plurality_frac`, \|pos_frac−thr\|) = the coherence/confidence used to exclude trap/ambiguous items from feedback (D11). |
| `Sigma_l_fitted_k7_general.npy` | dict: `Corr_l`/`Corr_t` (unit-diag — the engine prior; Var_prior=1 ✓), `Sigma_l`/`Sigma_t` (fitted cov, diag 0.5–2.1 = cross-rater skill spread), domains, n_raters=1949. |
| `cert_config_general.yaml` | v14 cut-scores ℓ*_k + per-task `sigma_star`(=exp(−ℓ*) ✓), `expert_ell_mean`, `non_expert_ell_mean`, youden_J, CIs. The empirical-Bayes prior gold mine (D10). Also Mode-A/Mode-B legacy config. |
| `deployment_yaml_general.yaml` | Mode-B/deployment stopping contract: N_min 60 total, per-task 10–120, pass_p 0.95 / fail_p 0.05. Distinct from AD6Policy (N_MIN 20, α 0.05) — trainer mastery reuses AD6 (F14). |
| `session_controller_general.py` | **REFERENCE ONLY — not runnable here** (anonymization mangled dunders `_init_`/`_file_`; imports `core_mcmc`/`engine_inputs_k7`/`policy_k7`/auroc names absent in scratch). Shows the production loop: domain1-first sectioning, consec-same-domain cap=5, extended-collection snapshot freeze, and a **decision-aligned `choose_item(ell_star=, decision_tasks=)`** the scratch engine copy LACKS (R6 drift). |
| 3× `*_trials_anonymized.csv` + `*_summary_*` | Real sessions (William Smith acc 0.23 careless/RT-median-152ms; a specialist acc 0.52; an Experienced rater acc 0.57). Capture `reaction_time_ms`, `answer_changes`, `n_interactions`, `select_ms` → **RT is logged in production (R1, resolves F10)**. Test fixtures for Steps 3/8 (R8). NB schema drift across sessions (22 vs 25 cols). |
| `beautiful_figure_example_general.py` | **Aesthetic reference** for all project figures: sans-serif 20 pt, low-alpha major+minor grids below data, purple/grey/teal palette (#9671bd/#7e7e7e/#77b5b6 + dark edges), neutral-grey reference lines, frameless top legends, PNG+PDF+SVG export. |
| `viz_style.py` | **M10.1.** Shared style module codifying the reference aesthetic (palette constants, `use_style`/`style_ax`/`save_fig`/`top_legend`, mode colours). All figures/video import it. |
| `make_figures.py` | **M10.1.** Five publication figures → `figures/` (PNG 300 dpi + PDF + SVG), simulation data cached as npz: fig1 benchmark CIs (30-seed raw `data_benchmark.npz`), fig2 F17 misspec asymmetry + mechanism, fig3 F5 variance floor (+Riccati line, both s_sd regimes), fig4 F23 attractor (3 placement schemes), fig5 sandbox outcomes (F18 retention designs, F19 gate detection — recorded values). Re-runnable: `python3 make_figures.py`. |
| `make_param_video.py` | **M10.1.** Second MP4 (`figures/param_evolution_demo.mp4`, 28 s, 1080p): skill ℓ and bias t evolution per training question on ONE shared y-axis, three policies overlaid (Tier-2, Tier-1 greedy, random; colour = policy, solid = skill, dashed = bias), 15 paired seeds mean ±1 SD, ★ mean first true-mastery, benchmark physics (domain3 real bank, pool 1000, F20 stimulus noise). Trajectory data cached `figures/data_param_evolution.npz`. Headline frames: Tier-2 mastery ≈ 46 q, Tier-1 ≈ 47 q; random clears neither gate in 300 q — its bias line never enters the ±t* band (the F23 midpoint attractor visible live). Plots TRUE learner parameters (not filter estimates) by design. |
| `make_demo_video.py` | **M10.1.** Animated MP4 demo (`figures/learning_algorithm_demo.mp4`, 1080p, ~7 fps): the REAL pipeline modules run eval → 2 training sessions (Tier-2, real bank, LT2 gap) → re-cert with per-trial trace; panels = belief cloud in (t, σ) with mastery zone, active psychometric curve truth-vs-estimate with served item ±s_sd, skill ℓ̂±SD vs cut-scores, bias t̂ vs tolerance band, posterior-SD panel, mode strip. Demo run (seed0=23): eval FAIL/FAIL/REFER → 156 train trials → re-cert PASS/PASS/REFER. MP4 written via the bundled `imageio-ffmpeg` binary (no system ffmpeg on this box). |
| `v15_simulation_spec.txt` | **M11 source of truth for v15.** Full-precision ℓ\*/expert/non-expert means, Corr_t + Corr_l verbatim, n_particles semantics (600 frozen / 1200 staged — PARTICLE count, not question budget; stopping rule unchanged), realistic-bias prior (mu_t, Sigma_t), provenance sha256s, domain3 robust-trim, "bank/labels UNCHANGED", "t\*/mastery targets DO NOT EXIST (OQ6 open)". Caveat: all v15 OC evidence is on θ=0 examinees. |
| `instrument_v15.py` | **M11.** Versioned-instrument accessor: `instrument("v14"\|"v15")` → cuts, σ\*, credentialed ceilings, prior blocks (v14 = Corr_l both; v15 = Corr_t t-block), n_particles; `for_tasks()` K-subset views; `draw_examinee_theta()` realistic-bias examinee draws (μ_t from YAML block `eb_examinee_bias_prior_v15`, Σ_t from npy, +2 clip). bank_adapter module constants stay v14 (live frozen). |
| `test_v15.py` | M11 smoke (21 checks): YAML block complete/precise, v13/v14 untouched, corr_t swap real + PD, gap widening, bias-prior draws, eval-harness opt-in plumbing + default-path regression. |
| `study_v15_oc.py` | **M11.** v15 operating-characteristic study: K=7 full-instrument evals (max_q 420, pool 800/task) × arms {v14, v15, v15cuts600 ablation} × offsets {−0.3, +0.3, +0.6, +1.0} × {zero, realistic} bias, multiprocessing, checkpointed → `figures/data_v15_oc.npz`. |
| `study_v15_sbc.py` | **M11.** SBC-style coverage study: truth ~ the engine's own hierarchical prior (Corr_l/Corr_t) → K=7 AD6 eval → equal-tailed 50/80/90/95% ℓ-CI coverage at 600 vs 1200 particles → `figures/data_v15_sbc.npz`. Result = F27. |
| `make_v15_figures.py` | **M11.** fig6 (v15 resolution + realistic-bias cost + A1 ablation marker) and fig7 (tier-3 benchmark) from the cached study npz files, viz_style aesthetic, PNG+PDF+SVG. |
| `trainer_rollout.py` | **M11 — Tier-3 (plan §6.3 "method 3" MC rollout).** `tier3_select`: tier-1-constrained shortlist (F6b) → batched H-step rollouts (L per candidate, n_ro-particle beliefs, π0 = tier-2 boundary-centered/dual-mode rule honoring F23) → argmax mean discounted return. **CRN (common random numbers) across candidates is LOAD-BEARING**: without it per-candidate signal ≈ rollout-noise SE and the argmax is a coin flip (caught by test_tier3 check 4). ~47 ms/trial at H=6, L=8, n_ro=96, shortlist 12. **M25 (F87):** `rollout_item_q` — item-level rollout value for skill-mode placement, mixture-honest (`_pooled_truth_arrays`: per-stratum σ_∞, static mass frozen incl. process noise); interior belief stays the base-dynamics pooled approximation (drives π0 only). |
| `test_tier3.py` | M11 smoke (9 checks): pool lookup, π0 mode gate/alternation, determinism, F6b constraint, separable lookahead pick, real-bank graduation < budget at < 100 ms/trial. |
| `VERIFICATION_PLAN_M12.md` | **M12.** The verification-ladder execution plan (V0–V6) + step status. |
| `EXTSET_SAP_M13.md` | **M13.** Pre-registration-style statistical analysis plan for the EXTSET release (W1 link fit, W2 replay validation, W3 population grounding; endpoints P1–P3). |
| `viz/make_extset_figures.py` + `viz/make_extset_videos.py` | **M13.3.** Publication aesthetics (beautiful-example conventions): fig9–fig14 statics + 2 films (`extset_measurement_replay.mp4` — filter converging on 3 real raters; `extset_reallink_training.mp4` — trainer vs the fitted real link, declaration vs true-mastery markers). All in `figures/`. |
| `viz/make_m14_figures.py` | **M14.** The "Nature-gap" result figures from the M14 study caches, same viz_style aesthetic (PNG 600 dpi + PDF + SVG): fig15 A1 softmax-vs-composed model comparison (F49), fig16 A2 Bayesian posterior + proper SBC rank-uniformity (F51/F52; σ_θ the lone residual), fig17 A3 specification curve (F45, 42 branches all positive), fig18 A4 disattenuated validity (F46), fig19 C9 shadow-mode placement uplift ×6 (F47), fig20 C10 pilot power + required N/arm (F48). Run: `python3 -m viz.make_m14_figures` (skips panels with missing caches). |
| `data/extset/` (`*_scrubbed.csv` + `AUDIT_extset_novice_expert_scrubbed.md`; **moved from the scratch root at M27** — paths resolve via `training/extset_adapter._root`) | **M13.** The EXTSET release: reads, expert panel, crosswalks, users/survey, contest maps. Inventory + verified structure: §3E; findings F35–F38. |
| `misspec_learners.py` | **M12/V1.** Mis-specified learner zoo (rung 3): one named deviation axis per class — dynamics family (`PowerLawLearner`, `PlateauLearner`, `DriftingCeilingLearner`, `MomentumCriterionLearner`), observation model (`HeavyTailLinkLearner` t₃ slope-matched, `AsymmetricLapseLearner` λ_fa≠λ_miss, `FatigueLearner` λ(n)), adversarial (`AntiLearner`, `careless_learner` λ=0.35, `static_below_cut`). Plus the graduation ground truths: `pool_accuracy` (GH-smeared true response fn) and `accuracy_bar(σ*, t*)` (worst designed-acceptable learner). |
| `test_misspec.py` | M12 smoke (20 checks): GH vs closed form, axis-by-axis deviation sanity. |
| `study_misspec.py` | **M12/V2.** Rung-3 campaign harness: zoo × {tier2, staircase85, random} × 30 seeds + 120-draw heterogeneity arm; metrics n_true/n_decl/A_decl/FG-perf/FG-latent/PIT coverage/RMSE. `run_one(..., p_static=, evidence_z=, smear_w=, fp_override=, stop_at_decl=)` is the shared loop for V2b/V2c/V3. → `figures/data_misspec_campaign.npz`. |
| `study_misspec_mitig.py` | **M12/V2c.** Mitigation matrix: shipped / evidence / staticmix / both (+ `--extra`: smearw, hardened3) × zoo. → `data_misspec_mitig{,2}.npz`. |
| `study_gate_oc.py` | **M12/V3.** Graduation OC over ceiling-to-cut Δ ∈ [−.2,+.3] × arms {shipped, oracle, hardened(3)} × 40 seeds. → `data_gate_oc{,2}.npz`. |
| `study_train_sbc.py` | **M12/V4.** Closed-loop SBC of the training filter (truth ~ own prior, tier2 selection, PIT at n∈{10,50,150,300}, 200 reps). → `data_train_sbc{,_smearw}.npz`. |
| `study_recert_backstop.py` | **M12/V2b.** End-to-end safety: trainer declaration → REAL K=1 eval re-cert (engine choose_item/update + AD6) of the frozen learner via its TRUE response fn. → `data_recert_backstop.npz`. |
| `study_real_replay.py` | **M12/V5.** All 3 real sessions replayed through the TaskFilter (static + dynamics arms): convergent validity, production-ℓ̂ cross-val, careless-certifies-nothing, F28 real-data exposure. → `data_real_replay.npz`. |
| `make_m12_figures.py` | **M12/V6.** fig8 (OC curves / zoo FG / SBC coverage / declared-vs-true timing). |
| `mixture_filter.py` | **M15 (D22/F54).** `SigmaInfMixtureFilter` — Bayesian model averaging over a J=7 ceiling grid (D10-centered, τ=0.30, optional static stratum) with prequential evidence weights; TaskFilter-compatible API (drop-in for the policy stack) + `trainability(ℓ*)` recommendation statistic. F4-safe by construction (no static parameter lives on a particle). Parameter justifications in the module docstring. **M21:** `predictive_p(s, s_sd)` posterior-predictive over the pooled cloud (feeds the ConsistencyMonitor null). **M23 (F80/F81):** `boundary_shift(eps, sd, theta_scale, w_share)` — state jump + FIXED-SHARE ceiling re-mix toward the prior (exact Bayes for a hidden-Markov ceiling; sandbox Gate 4); opt-in per-trial DMA `forget=` (F81-ii: REJECTED for serving — ablations only). **M26 (F89):** `declared_hazard_scale` (Beta posterior-mean Gate-4 decay c0/(c0+n) for declared tasks; linear-in-ρ ⇒ plug-in exact; floor + geometric ablation) + `AdaptiveBoundaryHazard` (per-task ever/n/pending lifecycle; π-credit at opens — the strict gate is unreachable as the credit event because its sd conjunct is hazard-injected; π<0.5 collapse resets; never-declared tasks always pay the pinned hazard). Callers scale (ε_b, w_share); never scaling is bit-identical. D42: sandbox default OFF. |
| `studies/study_exact_kernel_audit.py` | **M15 (F53).** Open-loop paired kernel comparison — the audit headline (cov_t@90 0.631→0.906, cov_ℓ 0.771→0.913, RMSE also better; width column shows shipped's ~40% overconfidence). `--smoke` ≈ 90 s. |
| `studies/study_audit_hardening.py` | **M15 (F54/F55).** Graduation-honesty campaign: {shipped, exact, mix, mix+probe} × {wellspec, static_below, careless} × 20 seeds × 400 trials, real domain3 pool, D16 gate. Full table in docs/AUDIT_POMDP_MATH.md Part II step 5. |
| `tests/test_audit_fixes.py` | M15/M16 smoke (20 checks): f-gating freezes, kernel s_sd=0 bit-identity + study equivalence, learn-aware Q, honest MCSE, F28 declaration contrast (shipped 3/3 vs mixture 0/3), e-gate type-I/power, probe optimum, λ-multiplier, D23 wiring cadence/blocking, fitter evidence direction (F59 flatness noted). |
| `dynamics_fit.py` | **M16 (F58/F59).** Phase-3 offline dynamics estimator: `filter_evidence` (prequential log-evidence via the filter's reweight normalizer; CRN seed ⇒ deterministic objective; `score_from=` for temporal splits; M17: optional 5th trial column = feedback flag for mixed-feedback logs) + `fit_learner` (Nelder–Mead penalized ML of (log α_t, log α_σ, ℓ_∞); q/ρ held fixed per F4/D7; weak N(x0,1.5²) log-space penalty; exact kernel so fits don't absorb kernel bias). |
| `gap_anchor.py` | **M17 (D27/F61).** `GapAnchor` — measurement-based retention re-anchoring: session-open mixture over retention fractions toward personal baselines, 4–16-trial info-optimal anchor block (prequential BMA), collapse + per-person stability S update (through-origin LS on −log r̂ vs Δt). Answers the update-cadence question (D28). |
| `studies/study_gap_anchor.py` | **M17 (F61).** Multi-session validation vs identity/widen under power-law truth: stale-mastered 71/11→0, post-gap ℓ̂ error −80%. → `data_gap_anchor.npz`. |
| `studies/study_extset_dynamics.py` | **M17 (F62, OQ8=yes).** Real-data dynamics anchoring: per-(user×domain) fits on task2 reads (LOO signals F40, consensus y\* OQ9, non-consensus reads fb=0), Pool(42). → `data_extset_dynamics.npz`. |
| `studies/study_m17_capstone.py` | **M17 (F63); re-pinned M18 (F66).** Zoo graduation OC under the default stack (exact kernel + σ∞-mixture v15-anchored + probes/e-gate + derived 0-bias band + v15 cuts; M18: + anchored rates D29, meanskill off D30) vs point comparator, 7 members × 30 seeds, Pool(42). → `data_m17_capstone.npz`. |
| `studies/study_m18_hardening.py` | **M18 (F64).** static_below hardening ablation with GATE-BRANCH ATTRIBUTION (τ / meanskill / probe-density arms) — the study that isolated the mean-skill leak. → `data_m18_hardening.npz`. |
| `studies/study_hier_shrink.py` | **M18 (F65); M19-1 drift-aware refit (F69).** Hierarchical EB shrinkage on the real EXTSET fits: split-half reliabilities, moment-deconvolved Σ_pop, held-out pop/raw/shrunk comparison; V̄ now from odd/even interleaved splits (drift-free). → `data_hier_shrink.npz` (M18 temporal-V̄ record), `data_hier_shrink_oe.npz` (M19). |
| `gap_anchor.py::MixtureGapAnchor` | **M19-3 (F68).** Gap re-anchoring for `SigmaInfMixtureFilter`: product (ceiling × retention) cells, retention marginalized at close, in-place strata mutation, shared personal S. |
| `studies/study_k7_protocol.py` | **M19-4 (F70).** THE product simulation: K=7, full TrainerPolicy orchestration + per-task mixtures + MixtureGapAnchor + open-ended 40-trial sessions with calendar gaps + power-law forgetting + non-trainable tasks; fullstack vs identity-gap arms. → `data_k7_protocol.npz`. |
| `sandbox/` | **M20 (F71/D33); M21 (F72–F74/D34–D35); M22 (F77–F79/D36–D37); M23 (F80–F83/D38–D39).** The manual-tester shadow protocol — the delivery-vehicle prototype: `session.py` (CLI, human + robot responders, **`--user` per-tester profiles**), `protocol.py` (gap anchors, D33 confirmation, full-stack serving, JSONL telemetry incl. per-trial `consistency` shadow stats; **M22 gates: trainability-discounted allocation + finish tolerance, consistency-pause on flag, differenced fatigue guard**), `stimulus.py` (latent-monotone perceptual task; **label-consistent truncated s_real draw**; gain 0.85 after the F73 recalibration), `state_io.py` (npz+json resume; render-param + profile stamps, resume blocked across recalibration), `report.py` (analytics + trajectory figure, `--user`), `contact.py` (**M23**: ContactMonitor KT-numerator contact e-process + rt_floor_breach), README. **M23 gates:** Gate 4 boundary_shift at every session open (post-confirm ordering load-bearing; declaration sd_floor 0.33 per F81-iii), Gate 3b RT floor (500 ms/k=3), shadow contact telemetry. Verified by `tests/test_sandbox.py` (26 checks) + `tests/test_m23.py` (16) + archived robot campaigns. **M26 (F89):** declaration-lifecycle hazard state tracked always (`meta['hazard_lifecycle']`, π-credit at opens, confirm/revoke hooks, `hazard_scale` events); the SCALE is applied to Gate 4 only under `HAZARD_ADAPT` — False (D42) is bit-identical M25. **M27 (F91, D45 — ON):** contact-triggered onboarding — demonstration ramp (`_demo_trial`, easiest label-alternating feedback-safe items) for tasks without EVER-certified contact (`meta['contact_certified']`), trainability-guarded Gate-4 handoff shift at first certification (`contact_handoff` events); F90 wiring behind `TERMINAL_CONFIRMATION=False` (stale gates persist in `meta['stale_gates']`, revocation un-confirms). |
| `training/consistency.py` | **M21 (F72/D34).** `ConsistencyMonitor` — shadow learner-consistency changepoint monitor: sliding-window (W=20) GLR of recent (s, s_sd, y) against the belief's OWN per-trial predictive (`SigmaInfMixtureFilter.predictive_p`), probit-lapse alternative on a (σ′, t′) grid; auxiliary log-RT shift z reported, never a trigger. Log-only by design — a flag means "verify who is at the keyboard", not "reset beliefs". Threshold pinned by `study_learner_switch` arm A. Wired shadow into `sandbox/protocol.py` (per-trial telemetry + `consistency_flag` events + json persistence). |
| `studies/study_learner_switch.py` | **M21 (F72/F75).** Validation campaign: A stationary false-alarm calibration (pins the monitor threshold), B switch-detection power (guesser→competent / reverse / mild), C replay of the REAL two-tester logs, D pollution cost (fresh vs guesser-polluted state), E F75 onboarding-placement ablation. → `figures/data_learner_switch.npz`. |
| `studies/study_participant_gates.py` | **M22 (F77–F79).** Gate validation on a headless two-task full-stack harness (TrainerPolicy + mixtures + monitors over the real bank): G1 allocation (trainable + static pairs × {legacy, discounted}), G2 flag-pause damage containment, G3 fatigue-guard FA calibration + FatigueLearner power + REAL four-participant replay. → `figures/data_participant_gates.npz`. |
| `studies/study_m23_regime.py` | **M23 (F80–F83).** Regime-shift study: arm A hyperparameter fit by pooled prequential evidence on the 14 real (user, task) sequences (CRN + leave-one-user-out); A2 real-log breakout lag; B synthetic E-like jump + well-specified control; C FG safety on the M22 Harness2 production stack; D floors; G the harness-realism re-pin (2-task serving rate; the F81-iii floor re-measurement); E RT-floor calibration; F contact e-process validity/power. → `figures/data_m23_regime.npz`. |
| `studies/study_m24_selection.py` | **M24 (F84/F85).** Selection study on the production stack (real bank, Gate 4, floor 0.33): P placement OC (4 learner scenarios × 3 rules), H criterion-herding safety, Q allocation OC (3 rosters × 3 rules), S guardrails (FG zoo + 2-task lateness) for {M23, M24 shipping, M24 experimental}. → `figures/data_m24_selection.npz`. |
| `studies/study_m25_selection.py` | **M25 (F86/F87).** The M24-successor study: arm Q2 EP-v2 allocation on the BOTH-declared endpoint (3 rosters × {gate1, ep, epv2nf, epv2}, 8 sessions), arm R item-level rollout placement (4 scenarios × {z1, roll} + herding H2), arm S2 guardrails for the shipping candidates. Arms selectable/mergeable (`--arms q2,r,s2`). → `figures/data_m25_selection.npz`. |
| `studies/study_sharp_margin.py` | **M25 (F88).** The M23/M24 sharp-margin FG open item: X1 instrumentation of static ℓ*−{0.10,0.15,0.25} false declarations (trials-since-boundary, trainability, π, e-gate at declaration), X2 declaration-conjunct mitigations ({refr12, tr50, both, probe2} + careless guardrail), X3 well-spec lateness pricing. → `figures/data_sharp_margin.npz`. |
| `studies/study_m26_hazard.py` | **M26 (F89).** The adaptive-hazard campaign (D33-style lifecycle harness `run_sessions_h`): arm A BOTH-declared endpoint (M25 rosters × 5 pins; fixed baselines replicate M25 arm Q2 bit-for-bit), arm G true-regression guardrail ({deep ℓ*−0.405, sharp ℓ*−0.15} × 5 pins, regression at the session-5 boundary), arm F FG zoo (static15/careless). Arms selectable (`--arms a,g,f`). → `figures/data_m26_hazard.npz`. Verified by `tests/test_m26.py` (14 checks). |
| `training/domain.py` | **M27 (F93).** The DOMAIN pluggability seam: `ItemBank` (the candidates contract), `ArrayBank` (in-memory ItemBank with BankAdapter's exact filter semantics — the new-domain plug-in path), `Domain` (tasks + cuts + assumed dynamics + bank + belief-stack config; `fresh_filters()`/`trainer()` factories), `v15_domain()` (the shipped instrument as the reference construction). Additive only; toy-domain end-to-end graduation pinned by test_m27 check 11. |
| `docs/ARCHITECTURE.md` | **M27.** Committee-facing architecture: layer/dependency map, math↔code correspondence table (SMC/BMA/HMM hazard/e-processes/POMDP tiers/retention), the domain plug-in guide, governing conventions, reviewer reading order. START HERE for a first read of the codebase. |
| `studies/study_m27_terminal.py` | **M27 (F90).** Terminal-confirmation campaign on the ANCHORED lifecycle harness (`run_sessions_t` — 8-probe anchor blocks before D33 confirm checks; baselines re-pinned): arm T endpoint (3 rosters × {base, term0, term3}), arm R regression-after-confirmation ({deep, sharp} × arms), arm Z FG zoo + lock-in. → `figures/data_m27_terminal.npz`. |
| `studies/study_m27_onboard.py` | **M27 (F91).** Contact-onboarding component study (K=1): {base, shift, demo, demoshift} × {contact60, wellspec, careless} — ramp vs guarded-handoff decomposition. → `figures/data_m27_onboard.npz`. |
| `studies/study_lambda_tot.py` | **M27 (F92).** λ(time-on-task) on the real EXTSET reads: per-user sittings, margin-stratified within-sitting accuracy slopes, user-level bootstrap. Verdict: learning dominates (easy +1.45e-3/min), no Phase-3 λ(t) term. → `figures/data_lambda_tot.npz`. |
| `docs/M27_TERMINAL.md` | **M27.** The F90/F91/F92/F93 analysis + PECR record (incl. the bin-proliferation port lesson and the anchored-harness re-pins). READ THIS FIRST for anything M27. Verified by `tests/test_m27.py` (12 checks). |
| `docs/FINAL_REPORT.md` | **M27.** The pre-submission report for the user: bottom line, what M27 changed, the seven advisor asks, stated limitations, tree state, and the response plan per advisor outcome. The user reads THIS before sending the package. |
| `viz/make_user_videos.py` | **M25.** Per-tester MP4s — the learning algorithm in action on the REAL sandbox telemetry (`figures/learning_trajectory_USER-*.mp4`): per-question belief skill ℓ̂±SD vs the v15 bars and bias t̂±SD vs the F60 band, session boundaries w/ gaps, D33 lifecycle events, consistency flags, serving-mode strip. Re-runnable: `python3 -m viz.make_user_videos [USER-A ...]`. |
| `docs/ADVISOR_BRIEF.md` | **M25.** Self-contained advisor-facing summary (system, estimation choices, evidence ladder, honest OCs, decision asks, reproducibility) — the C4 sign-off vehicle. |
| `docs/TRAJECTORY_CADENCE_PLAN.md` | **(2026-06-22; added to this map at M26 — was unreferenced.)** LOCKED data/cadence contract between the trainer stack and the deployed dashboard's per-domain evolution curves: one point per (domain, session) at session finalize; anchor (`eval`/`recert`) vs interim (`train`) point kinds; interim points are the RUNNING posterior (seeded once from the eval cloud), never a per-session re-fit. |
| `archive/` | **M26.** Stale-file archive (move, never delete; mirrors the `sandbox/state/archive-*` convention): every move gets a dated row in `archive/README.md`. Grep code AND this map before archiving anything. First entry: `sbc_run.log` (M14 console capture; results live in `data_extset_sbc.npz`/F51). |
| `docs/M26_HAZARD.md` | **M26.** The F89 derivation (Beta hazard-learning, credit-rule honesty), the A/G/F campaign tables, and the D42 PECR record incl. the coupled gain/blind-spot finding. READ THIS FIRST for anything M26. |
| `docs/SUBMISSION_CRITERIA.md` | **M26.1 (D43).** The submission gate S1–S8 (all MET 2026-07-05) + the STOPPING RULE: the algorithm is SUBMISSION-FROZEN; new PECR loops open only on advisor requests, new human data, a regression, or a gate unblocking — NOT on further optimization ideas absent new data (the M24–M26 three-NOs convergence record is the justification). Parked queue recorded in §3. CHECK THIS BEFORE STARTING ANY NEW ALGORITHM LOOP. |
| `docs/M25_SELECTION_V2.md` | **M25.** The EP-v2/rollout/sharp-margin analysis + PECR record. READ THIS FIRST for anything M25. |
| `docs/M24_SELECTION.md` | **M24.** The selection-math analysis (delivered-value audit, plan-Eq.-greedy mapping), the F85 derivation + validation, and the F84 six-variant negative result with its three mechanisms. READ THIS FIRST for anything M24. |
| `docs/M23_REGIME_ANALYSIS.md` | **M23.** Seven-profile analysis (per-session d′/probit tables, guessing-vs-reversal adjudication, calibration audit) + the F80–F83 derivations (jump kernel, fixed-share HMM ceiling, RT floor, KT contact e-process) + the full PECR record incl. the two rejections. READ THIS FIRST for anything M23. |
| `docs/M22_PARTICIPANT_GATES.md` | **M22.** Four-participant pilot analysis (per-user d′/RT/quarter tables, allocation/starvation mechanics, post-flag damage audit) + the F77–F79 gate derivations. READ THIS FIRST for anything M22. |
| `docs/M21_TWO_TESTER_ANALYSIS.md` | **M21.** Full quantitative analysis of the first two human tester sessions (d′ tables vs label AND vs percept, RT identity signal, ideal-observer render bound, contradiction-trial audit) + the F72–F75 derivations. READ THIS FIRST for anything M21. |
| `studies/study_phase3_fit.py` | **M16 (F58).** Train/test validation of the fitter: heterogeneous cohort (biased sub-skill starts — smoke lesson: belief-prior starts leave α_t unidentified), cross-learner (24/16) + temporal (150/150) splits; A recovery-vs-length, B held-out evidence + trainability classification, C soft/hard rule-ID (F12), D closed-loop declaration quality. → `data_phase3_fit.npz`. |
| `docs/AUDIT_POMDP_MATH.md` | **M15.** The math audit: Part I (findings MA-1…MA-10, verified-sound list) + Part II (implementation log of the seven-step action order with every parameter estimation justified). READ THIS FIRST for anything M15. |
| `docs/REPO_RELOCATION_PLAN.md` | **(2026-07-06.)** The gated plan (G0–G6) for re-homing this WHOLE project as `<main repo>/trainer_rd/` — verified-facts table, collision map, the 5 repo-side edits, ratification defaults R1–R4, rollback story. RELOCATION ONLY (not "the port"); D43-compliant, zero algorithm changes. Status: plan written, execution pending user ratification. |
| *(main repo, not present here)* | `the main repo` — production home. Scratch prototypes get ported there after validation (Decision D4). |

Still missing-but-referenced in scratch: `core_mcmc_brute_k.py` (only engine
"brute"/"random" methods), `diagnostics.py` (only optional diag_callback),
`engine_inputs_k7.py` (the `as_engine_arrays()`

---

## 2. Parameter conventions & verified constants (CRITICAL — read before writing any bridge code)

**Two different parameterizations of the SAME observation model:**

| Quantity | Trainer plan (`learning_algorithm_plan.md`) | Eval engine (`core_mcmc_general.py`) | Bridge |
|---|---|---|---|
| Latent decision variable | `z = (s − t)/σ` | `z = exp(ℓ)·(s + θ)` | `t = −θ`, `σ = exp(−ℓ)` |
| Skill | `1/σ` | `exp(ℓ)` | same number |
| Bias/criterion | `t` (subtracts from s) | `θ` (adds to s) | **SIGN FLIP** |
| Response model | `(1−2λ)Φ(z) + λ` | `λ + (1−2λ)Φ(z)` | identical |
| Lapse `λ` | nominally in state φ, held constant | `LAPSE_RATE = 0.025` 🔒 locked | fixed 0.025 |

⚠️ A naive bridge that forgets `t = −θ` reverses the direction of every
bias-correction update. Round-trip conversion tests are mandatory (plan Step 0).

**Difficulty-placement constants under λ = 0.025** (verified numerically 2026-06-10,
`scipy`, see analysis transcript):

| Target | `|s − t|/σ` multiplier |
|---|---|
| Plan's Eq. (eq:eightyfive) value | **1.04 — WRONG for λ=0.025**: gives 83.3% accuracy, and 1.04 is actually the *no-lapse* 85% point Φ⁻¹(0.85)=1.0364 |
| 85% accuracy, lapse-corrected | 1.1189584 |
| Wilson et al. optimum (error Φ(−1)=15.87% ⇒ accuracy Φ(1)=0.8413447), lapse-corrected | **1.0772256** = `SKILL_MODE_MULTIPLIER` (D6 default for skill mode) |
| Wilson optimum, no lapse | exactly 1.000 |

Formula: `multiplier = Φ⁻¹((a_target − λ)/(1 − 2λ))`.
**Executable source of truth: `bridge_conventions.py`** (Step 0) — do not re-derive
these constants elsewhere; import them. `test_step0.py` pins all values above.

**Other live eval constants:** N_PARTICLES=600, ESS_THRESHOLD_FRAC=0.5, N_MH_STEPS=15,
proposal_scale=2.38/√(2K), N_MIN=20, R*=0.30, α=0.05, Z=2.0, MAX_Q=500.
Cut-scores ℓ*_k (v14): .325/.256/.534/.330/.479/.486/.442 (AUROC bars ≈0.87–0.89).
Trainer mastery targets per task: `σ*_k = exp(−ℓ*_k)`, i.e. mastery ⇔ cert PASS bar.
AUROC(ℓ) = Φ(√2/√(exp(−2ℓ)+1)); ℓ=0 ⇒ AUROC 0.841 (not chance).

---

## 3. Review findings on `learning_algorithm_plan.md` (2026-06-10)

Ordered by severity. "Plan §x" = section of the LaTeX memo.

- **F1 — MH rejuvenation breaks under learning dynamics (plan §4.2 understates the change).**
  The plan claims "the only engine change required is the addition of the propagation
  step." False: `mh_rejuvenate` recomputes likelihood by replaying the FULL history
  against the particle's *current static* position (`_log_lik_history`). Once θ
  evolves, the posterior is over *trajectories* and that replay targets the wrong
  distribution. **Resolution:** trainer filter = bootstrap SMC (reweight → propagate →
  multinomial resample on low ESS, NO MH). Process noise (q_t, q_σ) supplies particle
  diversity. Escalate to fixed-lag/Storvik/Liu–West only if degeneracy observed.
- **F2 — 85%-rule constant numerically wrong.** See §2. Use 1.077 (Wilson optimum,
  lapse-corrected) and parameterize the target accuracy. Also ties to the open
  calibration note in `HOW_THE_TEST_WORKS.md` §1.1.
- **F3 — Sign convention collision** (`t = −θ`). See §2. Bridge module first.
- **F4 — Online estimation of per-learner static dynamics params (α_t, α_σ, σ_∞, q's)
  is not solved by the proposed filter** — static parameters in a state-space model
  suffer path degeneracy in particle filters. **Resolution (Phase 2):** FIX them at
  empirical-Bayes prior means; fit hierarchically OFFLINE from logged trial data
  (Phase 3). Online learning-rate estimation is out of scope until then.
  **M1.5 update:** the σ_∞ component of these priors is now data-grounded (§2B, D10);
  α_t/α_σ/q's remain literature-placeholder until pilot training data exists.
- **F5 — Process noise puts a FLOOR on posterior precision; the mastery criterion can
  be unreachable.** With q_σ > 0 the filtered SD(ℓ) converges to a steady state; if the
  mastery gate `SD(σ) ≤ δ_σ` is set below that floor, no learner ever graduates.
  Measure the steady-state variance in simulation (plan Step 3) and set δ_σ (and any
  R-gate analog) above it. **MEASURED (M3):** with q_t=0.05, q_σ=0.03 the
  information-balanced steady-state floor is **ℓ-SD ≈ 0.154, θ-SD ≈ 0.132** (NOT pure
  diffusion — `steady_state_sd` tracks a learner pinned at σ_∞ under informative
  trials). Mastery `sd_floor` is set to `max(1.5×floor, 0.15)` ≈ 0.23 in ℓ-SD. Note:
  an early version of `steady_state_sd` measured propagate-only diffusion (no
  steady state, SD→√k·q); fixed to the information-balanced regime.
- **F6 — Tier-1 greedy has two exploitable degeneracies:** (a) retention bonus
  ret_k ∈ {0,1} is spammable — must be eligibility-gated (counts only when the
  sub-category is actually DUE per the spacing schedule); (b) the t-term reward is
  maximized by one-sided label selection = implicit base-rate manipulation. Plan §9
  permits base-rate influence but tier-2 demands sign balance — inconsistent. Add an
  explicit running-label-balance constraint to ANY tier and document the permitted
  base-rate envelope.
- **F7 — Skill-mode placement `|s − t̂| ≈ c·σ̂` is relative to the criterion, not the
  true category boundary (s=0).** When |t̂| ≳ c·σ̂, items between 0 and t̂ have
  ground-truth labels conflicting with criterion side. Mode ordering (bias first)
  mostly protects this; add an explicit guard in skill mode.
- **F8 — The plan is single-task; production is K tasks.** Resolved by D2 + D3 below:
  per-task 2-D filters + a deficiency-weighted interleave scheduler (a NEW component
  the plan does not specify).
- **F9 — No handoff artifact is defined anywhere.** Defined here as `TrainingSeed`
  (plan Step 1): per-task marginal particle clouds (θ_k, ℓ_k, w), full history
  (k, s, y, s_sd, seg_id), verdicts + AD6 diagnostics (π, R, mcse, n_per_task),
  posterior summaries, cut-scores, per-trial RT/telemetry, eval-seen seg_ids.
  Under extended-data-collection mode, seed from the FULL-data posterior (better
  measurement), while verdicts remain the byte-frozen snapshot.
- **F10 — Response time is a first-class fluency signal (perceptual-learning methods).** ~~RT lives (if
  anywhere) at the GUI layer.~~ **RESOLVED M1.5 (R1):** production already logs
  `reaction_time_ms`, `answer_changes`, `n_interactions`, `select_ms` per trial (see
  real `*_trials_anonymized.csv`). RT is heavy-tailed with careless-rater outliers
  (one session RT-median 152 ms / acc 0.23; another median 8 s) ⇒ the Phase-3 RT model
  must be robust (lognormal + outlier/lapse component), not Gaussian. TrialRecord
  already carries rt_ms + answer_changes (Step 1); add n_interactions/select_ms.
- **F11 — Identifiability under joint (σ_k, t_k) motion.** **RESOLVED M4 (PASS).**
  Hard gate `test_step4.py`: across-replicate σ/t error-confusion |corr| ≤ **0.17**
  (criterion ≤0.5) in every cell, posterior |corr(θ,ℓ)| ≤0.13, correct-spec terminal
  coverage **0.91**, RMSE <0.20. σ and t are cleanly disentangled under joint motion.
  **New finding F17 (asymmetry):** the filter is robust to UNDER-estimating learning
  rates (0.5× → 0.97 coverage, conservative) but degrades under OVER-estimating
  (2.0× → 0.57 coverage; no divergence, RMSE still bounded). ⇒ **D14: empirical-Bayes
  rate priors (α_t, α_σ) should lean conservative (slightly low).**
- **F12 — Soft vs hard prediction-error rule:** same expected dynamics
  (Eq. eq:hard-unbiased), different variance; distinguishable only from real feedback
  trials. Keep hard rule as the standing ablation. Note the soft form implies the
  trainer's expected reward is independent of the learner's actual response y —
  a strong model commitment to validate empirically.
- **F13 — Validation circularity risk:** benchmarking the policy on simulated learners
  that follow the assumed dynamics flatters the policy. Stress-test under
  misspecification (hard-rule learner, wrong α by ±2×, non-exponential σ curve).
- **F14 — Mastery should literally reuse the AD6 machinery** on the filtered posterior:
  graduate task k when π_k − Z·mcse_k ≥ 1−α against ℓ*_k plus an information/floor
  gate per F5. This makes train→re-certify seamless in both directions.
- **F15 — Free lapse/fatigue probe:** retention-mode very-easy items are near-pure
  lapse probes (P(error) ≈ λ). Track their error rate within-session as a fatigue
  monitor; feeds a future λ_k(φ) model. (Original improvement, not in plan.)
- **F16 — Eval-side efficiency lever (from `HOW_THE_TEST_WORKS.md` §5, "REMEMBER
  THIS"):** the A-optimal objective wastes budget on variance that moves no verdict;
  a decision-aligned objective (target π_k uncertainty) is the biggest eval-side win.
  Parallel track to trainer work; same Q-style scoring skeleton could serve both.

**M9.1/M10 review findings (2026-06-10) and their resolutions:**

- **F18 (FIXED M10) — RETENTION mode was structurally unreachable.**
  `DeficiencyScheduler.pick` skipped mastered tasks while `choose_mode` returned
  RETENTION only for mastered tasks ⇒ 0 retention trials ever served (verified:
  600-trial run, 3 bins due, none served). Sandbox `exp_retention` compared
  designs (12 reps, 6 sessions × 70 trials, 30% between-session relaxation
  toward baseline): **(A) status quo: final skill deficit ℓ*−ℓ = +0.32, belief
  staleness ℓ̂−ℓ = +0.57; (B) due-driven interleave: deficit +0.016, staleness
  +0.42, 39 review trials, first-mastery 194 (vs 219 — no starvation); (C)
  blind periodic: +0.26 (barely helps).** ⇒ D17: due-driven (B) implemented in
  `TrainerPolicy.step`. Two interaction bugs fixed with it: review bins now
  register only at the mastery TRANSITION (per-trial registration proliferated
  bins → permanently-due queue starving unmastered tasks), and the
  `RetentionScheduler` is caller-injectable so interval units match the
  caller's `now` clock (trial-index vs epoch-seconds inconsistency). Verified:
  235 retention trials served over 600, bins bounded. Residual for Phase 3:
  belief staleness ≈ +0.4 even under (B) — `propagate_gap` widens variance but
  does not decay the mean; the fitted LT2 kernel must.
- **F19 (FIXED M10) — pass-mass mastery gate cannot fire for small
  ceiling-to-cut gaps.** Steady-state pass mass at the expert ceiling
  Φ((ℓ_∞−ℓ*)/floor) with the real-bank floor (ℓ-SD 0.179 under s_sd≈0.85
  attenuation; 0.152±0.002 at s_sd=0, both ≈ analytic Riccati prediction):
  only domain2 (0.98) and domain4 (0.96) clear 0.95. Sandbox `exp_gating`
  (20 reps × 300 trials, at-ceiling vs 0.15-below-cut learners): **G0
  first-crossing / G1 dwell-5 / G2 avg-π NEVER detect domain1's ceiling (0/20);
  G3 mean-skill gate (trailing-20 mean ℓ̂ − 1.645·SE_autocorr ≥ ℓ*) detects
  100% at median 37 trials; false-graduation 0.00 for ALL variants at −0.15
  below cut** (the repeated-look type-I risk is mild at a meaningful margin;
  the real failure mode was never-detect). ⇒ D16: **hybrid gate** (pass-mass
  OR mean-skill) in `TaskModePolicy.is_mastered` — fast in clear cases,
  detects small-gap ceilings, union FPR 0 observed. OQ7 thereby resolved at
  the engineering level (PI ratification of the G3 acceptance semantics still
  worth seeking).
- **F20 (FIXED M10) — s_sd regimes were inconsistent across sim/filter/eval.**
  The simulator responded to s_mean exactly while the filter attenuated
  (~×0.33–0.42 info on the real bank); the demo eval ignored s_sd entirely.
  Now production-faithful end-to-end: harnesses draw s_real ~ N(s_mean, s_sd²);
  `TaskFilter.propagate` uses the attenuated E[p]; the pipeline eval uses the
  engine's Phase-3.5 path (`choose_item(bank_sds=…, return_sd=True)` +
  `update(s_sd=…)`). **This changed the benchmark physics and re-based the
  Step-7 claims (see §6 M10)** — and exposed that ITEM SELECTION must be
  noise-aware too (D15).
- **F21 (FIXED M10) — hard-rule propagate discarded the observed response.**
  Exact conditional kernel δ = y_obs − y* now used for every particle when y
  is known (`propagate(..., y=)`); per-particle Bernoulli resampling retained
  only as the y-unknown fallback. Measured: hard-rule tracking RMSE(t)
  0.304 → 0.262 (−13.7%), coverage 0.80 → 0.82, RMSE(σ) unchanged.
- **F22 (FIXED M15 — D20) — the criterion update δ = p − y* fired even when
  feedback=False** in both `learner_sim` and the filter propagate (only
  σ-learning was f-gated) — informationally impossible (needs y\* without
  feedback) and the main suspect for the F34-i replay inflation. M15 f-gates
  the criterion channel in both files; all feedback=True paths bit-identical
  (f=1.0 multiplication is exact), full suite green.
- **Filter MC quality (M10):** multinomial → systematic resampling in
  `TaskFilter.maybe_resample`: RMS error of the posterior mean vs an exact
  grid posterior fell 33% (θ) / 22% (ℓ) at identical cost (~1.6–2.2× effective
  particles for free). `trainer_greedy` Q is fully vectorized (bit-identical
  argmax, ≥5× faster) — enables production-size candidate pools.
- **F23 (M10, load-bearing control property) — the criterion converges to the
  served stream's midpoint.** Under the plan's R–W dynamics, a balanced-label
  item stream centered at c has drift zero ⟺ t = c (stable attractor): item
  placement does not merely measure or exercise the learner — it SETS the
  long-run criterion. Observed: t̂-centered skill placement herded true t to
  the mode threshold (pinned at ≈0.30, graduation impossible); independent
  per-side argmax served midpoint ≈ +0.19 (bank asymmetry: precise positives
  at ≈+0.96 vs precise negatives at ≈−0.55) and true t parked there.
  ⇒ skill placement centers on s=0 and MIRROR-PAIRS sides (D15). Production
  implication: ANY trainer (including a future Tier-3) must control the
  served-stream midpoint explicitly; it is the de-facto bias setpoint.
- **F24 (M10, dual-control failure mode) — pure-corrective bias serving blinds
  the filter.** Drift-optimal corrective items are information-POOR about t
  (responses near-deterministic; high s_sd attenuation): with corrective-only
  serving, t̂ froze ≈0.3 while true t crossed 0 and overshot (trainer
  correcting blind), and the mastery gate flapped at the tolerance boundary.
  ⇒ bias mode alternates corrective and probe picks; the mastery bias
  condition carries the +0.5·sd_t margin (D15). With mirror-paired skill mode
  + dual-control bias mode: 3/3 graduation in 194 trials with ZERO mode
  switches (vs 1500-trial non-graduation with 200+ switches before the fix).
- **F25 (M11, Tier-3 value boundary) — MC-rollout lookahead pays exactly when
  the dynamics are RESPONSE-CONDITIONAL; otherwise it is an MC-noise tax.**
  30-seed paired campaign (domain3, pool 1000, `figures/data_benchmark_tier3.npz`,
  fig7): SOFT learner (δ = p − y*, y-independent — the plan's §6.1 remark that
  the y-sum collapses): tier3 − tier2 = +2.1 [+0.3,+4.3] (1×), +4.2 [+2.3,+6.4]
  (2×) — lookahead adds nothing the one-step policies don't already see, and
  the rollout argmax noise costs ~2–4 trials. HARD learner (δ = y − y*):
  tier3 − tier2 = **−2.9 [−5.2,−0.8]** (1×; tier3 44.9 best-in-cell), −1.5
  [−4.9,+1.5] (2×, best-in-cell mean) — sampling y inside rollouts captures
  response-dependent risk (overshoot branches) invisible to one-step scoring.
  Both ≫ staircase/measure_opt/random everywhere; tier3 never censors. Cost 47
  ms/trial (vs ~0.5 tier2) at H=6/L=8/n_ro=96/shortlist 12. **Engineering
  load-bearing fact: COMMON RANDOM NUMBERS across candidates are mandatory** —
  per-candidate signal (≈ one step's reward) is the same order as rollout-noise
  SE; without CRN the argmax is a coin flip (test_tier3 check 4 witnesses it).
  ⇒ ship-state recommendation unchanged (tier2 default); tier3 is the Phase-3
  upgrade path gated on fitted dynamics being response-conditional (hard-rule
  evidence, F21) — revisit when pilot fits resolve the rule form.
  **M15 re-adjudication → F56: the hard-rule tier-3 advantage VANISHES under
  the exact kernel** (the lookahead was partly compensating filter
  mis-tracking); tier-2 default reinforced.
- **F26 (M11, v15 lever decomposition) — the claimed "+17pp from ℓ* alone" is
  predominantly the BAR-RECALIBRATION effect, not extra instrument resolution;
  anchoring convention decides which you measure.** Scratch K=7 OC study (168
  verdicts/cell, `figures/data_v15_oc.npz` + `data_v15_abs_anchor.npz`, fig6):
  with raters anchored at each arm's OWN cut+0.6 (per-instrument OC), cuts-alone
  is FLAT (71.4% vs v14 72.0%) and full v15 adds only +6.6pp (corr_t+1200p);
  with raters anchored at the CREDENTIALED bar+0.6 (the claim's implicit
  convention), v14 passes only 49.4% [41.9,56.9] vs v15 78.6% — +29pp, and the
  absolute numbers nearly reproduce the claim table (55.4→76.9 theirs,
  49.4→78.6 ours). Both readings are correct: IF the credentialed cuts are the
  true bar, v14 was mass-failing above-bar raters (miscalibration, the
  recalibration fixes it); the per-anchor view says the machine's resolving
  power barely changed. Realistic examinee bias (θ~N(μ_t,Σ_t)) costs both arms
  ~20pp at +0.6 (v14 72.0→51.8, v15 78.6→56.0) — the spec's θ-gating caveat is
  real and first-order. Intrinsic errors at our n: false-PASS v14 0.6%/v15 2.4%
  (4 vs 1 of 168, Wilson CIs overlap), false-FAIL ≤0.6% both.
- **F27 (M11, SBC) — n_particles 600→1200 does NOT restore nominal ℓ-credible-
  interval coverage in the scratch K=7 AD6 harness.** Truth drawn from the
  engine's own prior (Corr_l/Corr_t blocks), 100 raters × 7 intervals per
  config (`figures/data_v15_sbc.npz`): 95% coverage 0.933 (600p) vs 0.919
  (1200p); 80/90% similarly ~0.75–0.87 both. Our 600p matches the spec's
  0.932@95 almost exactly, but the spec's "1200 ⇒ nominal (0.956)" was measured
  on AUROC-CI coverage (K=6 Mode-A) — a DIFFERENT quantity. With truth ~ the
  assumed prior and an exact likelihood match, residual under-coverage is SMC
  approximation error (resample-move/rejuvenation quality, 7-D particle
  impoverishment), which raw particle count does not fix at this scale.
  ⇒ the staged 1200 should NOT be sold internally as "fixes calibration" for
  posterior-ℓ intervals; the K=7+corr_t AUROC-CI replication on the production
  harness remains the decisive check (spec's own "last gap").

**M12 verification-ladder findings (2026-06-11; campaign data in `figures/data_misspec_*.npz`, `data_gate_oc*.npz`, `data_train_sbc*.npz`, `data_recert_backstop.npz`, `data_real_replay.npz`; fig8):**

- **F28 (MAJOR, rung 3) — trainer graduation is dynamics-prior-dominated: under
  mis-specification the belief converges to the model's σ_∞ attractor regardless
  of data, and the D16 gate certifies the model's prophecy.** Mechanism: every
  particle is deterministically pulled toward σ_∞ by `propagate`; under real-bank
  s_sd (~0.85) the per-trial Bernoulli likelihood is too weak to resist, and after
  resampling the cloud holds NO trajectories in which the learner failed to learn.
  Measured (domain3, tier2, 30 seeds): a STATIC sub-cut learner (σ=1.25σ*,
  ℓ=0.311 < cut 0.534) is declared mastered 30/30 at median 47 trials with ℓ̂→0.72
  (= the attractor) and PIT coverage 0.00; a CARELESS learner (λ=0.35, max
  accuracy 0.65 < any bar) 30/30 at median 68 (ℓ̂=0.69 while running 40–70%
  accuracy); an ANTI-learner 30/30. Population-heterogeneity arm (120 draws over
  α's ±~4×, σ0, t0, ceilings, λ, rule): 100% declared, 12.5% below even the
  performance bar, 65% below the cert bar at declaration.
- **F29 (rung 4) — the shipped gate's operating characteristic is FLAT:
  P(declare)=1.00 at EVERY ceiling-to-cut margin Δ ∈ [−0.20,+0.30] (40
  seeds/cell), at a learner-INDEPENDENT median ~47–68 trials for every zoo
  member.** Declared mastery is a property of filter convergence, not the
  learner. An ORACLE arm (filter told the true individual ceiling) reproduces
  F19's step function exactly (0% below cut incl. Δ=0, 100% above, sensible
  timing 117→36) ⇒ the F19 sandbox "false-graduation 0" was CONDITIONAL on a
  well-specified filter — production never has that (σ_∞ prior = population
  ceiling, individual unknown). Important split: for Δ>0 most "FG-latent" is
  premature TIMING (true crossing follows shortly); for Δ≤0 it is verdict error.
  TRAINING efficacy, by contrast, is robust across the envelope: median
  trials-to-TRUE-mastery 40–52 for every genuinely-learning member (power-law
  108, careless 212 — slower but reached); tier2 still beats baselines
  everywhere (staircase85/random rarely or never even fire the gate, lacking
  info-targeted placement).
- **F30 (rung 4, FIXED opt-in) — even WELL-SPECIFIED, the filter over-estimates
  ℓ in the graduation window: closed-loop SBC dips at n=50 (PIT mean 0.368,
  truth in the lower tail 26% vs 10% nominal; KS 0.22).** Cause isolated by
  s_sd=0 control (dip vanishes): `propagate` evaluates the skill-weight at
  w(s_mean) while the learner experiences w(s_real); bank-level smearing dilutes
  the difficulty bump, so assumed progress > actual progress — the F20
  "second-order, folded into process noise" term is FIRST-order for graduation
  timing. Fix: `TaskFilter(smear_w=True)` propagates with E[w(s_real)] (11-node
  GH). SBC after fix: ℓ nominal at every checkpoint (KS ≤0.09, 90%-coverage
  0.90–0.94); declaration shifts 51→63–66 (honest timing); wellspec FG-latent
  11/30→9/30. Default OFF (shipped bit-preserved); ~~RECOMMEND default ON at
  port~~ **superseded M15: the exact kernel (F53/D21) subsumes smear_w (its
  unconditional ℓ-half) and adds the y-conditioning + variance the t-channel
  needed — the port recommendation is now `exact_kernel`, not smear_w.**
- **F31 (mitigations, all opt-in, full suite still green) — two complementary
  hardenings: (a) `TaskFilter(p_static=0.3)` static-hypothesis mixture
  (per-particle learning-active indicator gating the deterministic kernel terms)
  keeps posterior support on "not improving" trajectories: careless 0/30
  declared, anti 2/30, FG-perf→0 zoo-wide, benign FG-latent halved (wellspec
  11→5), coverage restored (powerlaw 0.17→0.70, wellspec 0.86→0.91); cost:
  declaration 51→~85–105, slow learners may not declare in budget (powerlaw
  21/30 — arguably correct waiting-for-evidence). (b) raw-EVIDENCE gate
  (trailing-40 observed correct count must not fall 1.28·SD below the
  borderline-σ* expectation on the SAME served items — raw outcomes the
  dynamics prior cannot fake): blocks gross performance deficits (careless
  30→10 declared, the survivors at median 262 trials AFTER genuine improvement;
  anti 30→10) at ZERO cost to benign members; blind to latent shortfalls (the
  served-item accuracy gap σ=1.25σ* vs σ* is ~0.02 under s_sd — undetectable by
  design). Full stack `hardened3` (+smear_w): benign FG-latent lowest (wellspec
  4/30, t3link 4/30), coverage ≈0.9. **Structural residual: static_below still
  declared 23/30 (med 106) and sub-cut OC false-pass stays 0.68–0.97** —
  tier2's TRAINING-optimal placement is information-poor for the ℓ-vs-ℓ*
  contrast at bank s_sd; no calibration fix changes that (hardened3 OC ≈
  hardened OC). Certification-grade discrimination needs eval-style probe items
  or re-cert (→ D18).
- **F32 (system safety, rung 3 closed end-to-end) — the EVAL re-certification
  backstop catches every false graduation: 0/75 false certs.** Frozen
  at-declaration learners (true response fn, real engine K=1 + AD6): careless
  FAIL 15/15 (eval ℓ̂ −1.54), static_below 0 PASS (10 FAIL/5 REFER, ℓ̂ +0.07),
  anti/asym_lapse/powerlaw 0 PASS. Flip side: GENUINE (wellspec) graduates pass
  immediate re-cert only 2/15 (9 REFER_BORDERLINE) — shipped's premature
  declaration wastes re-cert sessions; the hardened gate's later declaration is
  ALSO the efficiency fix. (Caveat: K=1 re-cert harness; production is K=7.)
- **F33 (rung 4) — θ-interval coverage is persistently ~3–8pp below nominal in
  closed loop (0.82–0.87 @ 90%) at every n, BOTH kernels** — the training-side
  analog of F27 (SMC approximation error, not model misspec). Mild; document,
  don't oversell trainer bias CIs. **M15 update: the cause is identified and
  fixable — it is predominantly the propagate kernel's dropped y↔s_real
  coupling, not irreducible SMC error (F53: exact kernel restores nominal θ
  coverage open-loop). Re-run this SBC with `exact_kernel=True` at port.**
- **F34 (rung 5, in-scratch) — measurement layer grounds on real data:** all 3
  real sessions replayed through the static TaskFilter give mean-ℓ̂ rater
  ordering EXACTLY matching accuracy ordering (acc 0.235/0.524/0.574 → ℓ̂
  −0.54/+0.31/+0.59); cross-validation vs the PRODUCTION engine's recorded
  per-task ℓ̂ (summary CSVs): Spearman 0.803, RMSE 0.486 (n=21 cells); careless
  rater certifies nothing (extends R8 to all sessions). Two caveats: (i) the
  dynamics-ON replay reads +0.256 higher ℓ̂ on identical no-feedback real data
  even with the σ-pull f-gated off — process-noise widening alone degrades
  measurement; (ii) one pathological cell (careless × domain7, 186 trials @
  45%): scratch per-task prior lets the 2-D posterior explain near-chance
  responding as high-skill+huge-bias (ℓ̂ +1.31 vs production −0.70) — the
  engine's JOINT hierarchical prior suppresses this; keep the joint prior at
  port (gate still did not fire). ~~The EXTSET response-link stress named in §3D
  remains OUT OF SCOPE in scratch.~~ SUPERSEDED M13: the scrubbed EXTSET release
  landed in scratch 2026-06-12 (§3E).

**M13 EXTSET-release findings (2026-06-12; data in `data/extset/` since
M27, audit in `AUDIT_extset_novice_expert_scrubbed.md`, inventory in
§3E):**

- **F35 (CRITICAL for any EXTSET validation) — the bank's crowd labels on the
  5,000 task2 EXTSET segments ARE the EXTSET reads:** median(bank `n_votes` −
  EXTSET reads/case) = 4.0 with IQR [4,5] — exactly the EXTSET novice reads
  plus the 4 experts; corr(n_votes, reads/case)=0.58; bank plurality matches
  the NOVICE crowd plurality on 73.2% of these segs and the source gold on only
  71.0%. ⇒ D11's y\* (= bank plurality) is CIRCULAR on these segments: every
  EXTSET analysis must define truth from the expert panel and/or source gold,
  never from bank plurality/margin. (Trainer ops on the rest of the bank
  unaffected, but note the bank's votes are partly THIS novice population.)
- **F36 — the dramatic pooled EXTSET learning curve is survivorship:** pooled
  accuracy by read-index bin climbs 0.27→0.58, but it conflates users (only
  high-volume accounts reach high bins). Within-user split-half gain is
  +2.2pp (first→second half of own reads; Wilcoxon p=5.7e-5; 60% of 183 users
  with ≥60 reads improve) — real but small, consistent with §3C caveat 5.
  Longitudinal claims must be within-user; the pooled curve is quarantined.
- **F37 — real psychometric link structure is clean and the novice band
  empirically grounds the synthetic grid:** P(novice says k | s_mean_k) is
  monotone across signal deciles (domain2: 0.016→0.709; domain5: 0.012→0.445)
  — a real probit-like curve on real reads. Pooled one-vs-rest AUROC on
  ≥3/4-consensus gold: 0.619 (domain3) – 0.739 (domain2). Per-user clean-gold
  accuracy: mean 0.253, sd 0.136 (n=315 users ≥20 reads). The real cohort sits
  in the LOW band the synthetic L_GRID assumed — the grid's lower bound is now
  empirically justified. Qscore convergent proxy verified: corr(mean Qscore,
  accuracy) = 0.840 (378 users ≥20 reads); Qscore=0 is mostly cold-start (63%
  of zero rows in a user's first 10 reads).
- **F38 — the gold standard is itself a κ≈0.5–0.67 noisy panel:** pairwise
  expert κ 0.495–0.673; 42.8% unanimous, 75.8% (3,789 cases) with ≥3/4
  consensus; 11.6% tied. Expert plurality matches the SOURCE gold on 77.1%
  of all cases and 91.6% of the consensus subset. ⇒ gold policy is a real
  analytic decision (OQ9), and panel noise is itself a publication argument:
  certification must be defined against consensus, not a single oracle.
- **F39 (M13 deep-dive, 2026-06-12) — domain1/task1 state in scratch: the
  SCAFFOLDING is fully present, the reads and signals are not.** Present:
  (i) all 5,000 task1 cases + gold (1,490 domain1 / 3,510 non-domain1) in
  `domain1_case_source_map` + segment map; (ii) crosswalk → bank seg_ids
  89201–94200, ingested as `source_dataset=extset_2025_aux` with RESERVED
  domain1 vote columns (`domain1_n_votes`/`domain1_pos_votes`) that are ALL
  ZERO (schema slots awaiting the contest votes); (iii) redundant `subclass`
  gold for 2,480 of them (100% agreement with the map gold); (iv) cohort
  participation totals in `extset_users`: 603 users × **161,065 domain1 reads
  recorded upstream** (mean 267/user, max 5,000 — LARGER than the task2
  contest); (v) domain1-cohort demographics (survey instrument **5290 =
  domain1 survey**, 247 rows; **5291 = domains-2–7 survey**, 139 — instrument
  question from audit §7 RESOLVED; `extset_novice_survey_results` is the
  295-row subset of surveyed users who labeled in task2). Absent and NOT
  derivable in-scratch: the 161,065 read-level responses, and engine signals
  for task1 segs (zero rows in `segment_signals_general.csv`; task1 source
  recordings are disjoint from the bank's signal-extracted segments — only
  2/5,000 `file_map` matches) ⇒ signals require upstream extraction on the
  task1 source windows. Also noted: 354 task2 EXTSET segs predate the contest
  as `corpus_a` bank members (explains the n_votes>reads+4 tail, max 297).
  If released, domain1 reads would be the highest-value addition: largest
  read count, BINARY response (matches the engine frame exactly — no
  one-vs-rest reduction), and it is the anchor domain with the hardest OC
  (F19 ceiling gap, D11 threshold). **RESOLVED M13.1: released 2026-06-12
  (`extset_task1_reads.csv` + `extset_task1_signals.csv`) — see F41.**
- **F40 (CRITICAL, M13.1, 2026-06-12) — `s_mean` is a CROWD-CONSENSUS score,
  not an independent instrument signal; on EXTSET segments it embeds the
  analyzed users' own reads ⇒ naive link fits are self-referential.**
  Evidence: task1 signals ≈ affine-probit of the smoothed vote share
  (s ≈ 1.10·Φ⁻¹((yes+0.78)/(n+7.5)) + 1.12, R²=0.966; tier-weighting adds
  nothing, ΔR²≈0.0005; residual 0.19 sd unexplained → OQ13); the vote columns
  in `extset_task1_signals.csv` reconcile EXACTLY with the read file (match
  1.000); task2 EXTSET segs corr(s_mean, probit vote-share) = 0.93–0.96 where
  votes = EXTSET reads + 4 experts (F35); bank's own domain1 pool corr 0.80
  (different crowd); partial corr given gold 0.95. Quantified self-leverage:
  per-read LOO shift |Δs| median 0.016 / p90 0.048 / max 0.41, mean shift
  +0.026 TOWARD the user's own response (mechanically self-confirming).
  **Protocol (SAP §0, mandatory):** primary analysis signal = exact
  leave-one-user-out consensus score recomputed from read-level votes;
  shipped `s_mean` only as sensitivity; s_sd kept as metadata (NOT
  delta-method vote noise — corr 0.84 but wrong scale). Signals still
  consensus-faithful: AUROC(s_mean→expert consensus) = 0.987–0.997 (task2),
  AUROC(s_mean→source gold) = 0.961 (task1).
- **F42 (M13.2) — the bank's task2 vote columns diverge from the read
  release:** they miss 2,238 of the 125,860 reads (1,651 segs, 1–4 each;
  vote-changes/snapshot drift) while holding ~14.5k extra historical votes
  (F35's corpus_a tail). ⇒ `extset_adapter` builds all vote bases directly
  from reads + the 4-expert panel; bank vote columns are never used.
- **F43 (M13.2, W1 COMPLETE — P1/P1b PASS) — the engine's locked likelihood
  (probit, λ=0.025) is a good but measurably improvable description of real
  reads, and the deviations are characterized:** all 14 frames AIC-select a
  richer ladder member; held-out ΔELPD/read vs M0 = +0.0006…+0.0137 with
  user-cluster CIs excluding 0 in 13/14. **P1b (task1 binary = the engine's
  deployment frame): M2 asymmetric lapse, +0.0049 [+0.0037,+0.0062], with
  fitted lapses BELOW the engine's (λ_fa=0.0024, λ_miss=0.0095 vs 0.025) —
  the engine over-assumes lapse where it actually operates.** Task2 marginal
  frames (d3–d7, LOO): heavy-tail links ν≈1.9–2.7 with LARGE class-specific
  miss-lapse (0.06–0.34) — the one-vs-rest reduction inherits forced-choice
  competition the binary model can't see (motivates the M5 softmax follow-up,
  deferred, prespecified secondary). Population grounding (W3) now empirical
  per frame: e.g. t1 μ_ℓ=−0.303 σ_ℓ=0.434 μ_θ=−1.022 σ_θ=0.231 (engine
  coords; full table in `data_extset_link.npz`). The novice cohort sits
  ~0.3 below the ℓ=0 prior center with strong conservative criterion — v15
  realistic-bias prior and L_GRID lower bound both empirically anchored.
  (Fit detail: one frame's ν ran to ∞ ⇒ probit; reported as M2-equivalent.)
- **F44 (M13.3, P3 PASS — rung 3 CLOSED on the real link): the
  EXTSET-fitted observation model lies INSIDE the characterized robustness
  envelope, near its benign center.** Campaign (tier2, CRN, 30 seeds,
  budget 400): `reallink` (fitted M2 link, zoo start) and `realpop` (fitted
  link + population-drawn starts) are indistinguishable from the
  well-specified control — declared 30/30 each; med trials-to-declare 49/50
  vs 51 (shipped) and 88/88 vs 101 (hardened); FG-perf 0 everywhere;
  FG-latent 12/11 vs 11 (shipped), 5/5 vs 4 (hardened); cov_ell 0.87/0.91 ≈
  control. The F28/F29 graduation-self-assessment weakness is unchanged by
  the real link (it is a placement-information property, as M12 concluded) —
  and D19 hardening helps identically. **Publishable claim: trainer operating
  characteristics are preserved under the observation model fitted from
  167,503 real reads.** Artifact: `data_extset_stress.npz`.
**M14 Tier-A/B/C findings (2026-06-12; plan = user-ratified "additional steps"
list; freeze manifest `PREREG_FREEZE_M14.md` — everything below is labeled
EXPLORATORY or design-input, not confirmatory):**

- **F45 (A3) — specification curve: the validity result is branch-proof.**
  42 specifications (gold policy × signal variant × min-reads × replay
  read-set): task1 ρ ∈ [0.836, 0.904] (6 branches), task2 ρ ∈ [0.346, 0.736]
  (36), every branch positive, no sign flips anywhere. Key economy (verified
  in code): static-rule `propagate()` is a no-op ⇒ y\* never touches ℓ̂, so
  gold branches are post-processing — 26 replays cover the multiverse.
  `data_extset_speccurve.npz`.
- **F46 (A4) — disattenuated validity:** split-half replay reliabilities
  ℓ̂ 0.781 (task1) / 0.776 (task2 composite); accuracy 0.773/0.710 (W4) ⇒
  **P2b disattenuates to 1.00 (capped)** — ℓ̂ and gold-accuracy are
  noise-limited measures of the same construct on the binary frame — and
  P2 to 0.648. (The earlier posterior-SD shortcut was WRONG for composites
  — negative implied reliability; split-half replay is the defensible
  method.) `data_extset_disatten.npz`.
- **F47 (C9) — shadow-mode replay: the trainer's placement headroom on real
  streams is ×6.** Real task1 serving sat mean ≈1.7σ̂ from the trainer's
  optimal difficulty; median per-user expected-learning-weight uplift
  ×6.01 (IQR 5.55–6.39; **n=197 raters ≥50 reads** — corrected 2026-06-13 from
  the artifact + study `MIN_READS=50`; the earlier "434" did not match the saved
  cache; median/IQR unchanged; D19 belief filter, D15
  noise-aware scoring vs the real 19,332-item bank pool).
  `data_extset_shadow.npz` (fig19); instrumentation list → `PILOT_SAP_M14.md` §6.
- **F48 (C10) — empirical pilot power: a ~70-rater two-arm pilot suffices at
  strict α.** realpop × hardened, CRN 60 seeds/arm: tier2 60/60 mastered
  (med 45 trials) vs staircase85 60/60 (med 77; δ=0.495 log-trials, ×1.64,
  σ=0.448) vs random 22/60 (med 247). **N/arm = 28 (α=0.005, power 0.9) →
  enroll 33/arm.** Random arm excluded from the pilot design (ethics: ×4.4,
  63% censoring). `data_pilot_power.npz`; full design `PILOT_SAP_M14.md`
  (staged shadow → supervised cohort → RCT; D18 language).
- **F50 (B8 — F32 replicates at deployment scale, with a new PASS-yield caveat):**
  the REAL K=7 adaptive eval (shipped v14 instrument, joint hierarchical prior —
  the F34-ii config) re-certifying the EXTSET-fitted `realpop` learner at its
  tier-2 trainer-DECLARATION latent state (domain3 = TASK; other six = zero-bias
  fitted-population draws; 30 seeds × {shipped, hardened}, max_q 420, pool 800;
  `data_recert_k7.npz`). **Protective direction holds:** of 16 false-graduate
  declaration states (ℓ̂ < ℓ*=0.534 OR |t̂| > T*=0.3), the K=7 re-cert refuses
  PASS to 14 (shipped 10/11, hardened 4/5) — only 2 FG leaked through as PASS,
  vs F32's 75/75 at the K=1 backstop. So the certification layer still catches
  the trainer's false graduations inside the full instrument, though no longer
  at 100%. **New caveat (invisible at K=1):** the K=7 re-cert is REFER-dominated
  and rarely PASSes even GENUINE graduates (PASS 5/44 — shipped 1/19, hardened
  4/23; 41/60 sessions REFER_BORDERLINE at/below the 420-q cap). With ~420 q
  split across 7 domains (~60/domain), a single domain seldom accrues enough
  evidence to confidently graduate (FAILs resolve early — n_q 146–243; PASS/REFER
  exhaust budget). **This is the instrument's known OC behavior at the cut, not a
  re-cert pathology:** the v15 OC curve (`data_v15_oc.npz`, 264 K=7 sessions)
  resolves 0.970 at +1.0 above the cut but only 0.315 at +0.3, 0.470 at −0.3, and
  0.622 overall — resolution is a steep function of distance from the cut, NOT a
  fixed ~0.98 instrument property (that 0.97 figure is the well-separated +1.0
  population only). Declaration states sit ON the cut by construction (the trainer
  stops the moment believed ℓ crosses ℓ*), so B8's 0.317 resolution (19/60
  definitive) is exactly the offset≈0 zone of that curve — consistent, not a
  regression; the F32 K=1 backstop shows the same (wellspec 0.40, anti 0.27).
  Re-cert as configured is a good GATE but a poor GRADUATOR at
  K=7 ⇒ port flag: genuine re-cert PASS likely needs a per-domain probe block or
  a larger single-domain budget, not just the protective check (refines D18).
  **D19 corroborated:** the hardened arm produced fewer FG declaration states
  (5 vs 11 of 30) — later declaration → fewer false graduates, exactly the
  predicted benefit, now on real-fitted learners at K=7. Closes the M12
  carry-over "K=7 re-cert replication".

- **F49 (A1, FINAL — IMPORTANT model-comparison reversal):** the M5
  single-skill softmax BEATS the composed marginals on equal user-capacity
  smoke subsets (+0.038/read) but LOSES on the full data (M5: −0.021
  [−0.055, +0.002]) — the composed ensemble's 12 user-effect dimensions
  capture real per-user CONFUSION PROFILES (consistent with W4's non-unit
  manifold). M5's fitted lapse is uniform-tiny (λ≈0.02 full) — confirming
  the marginal class-miss-lapses (0.06–0.34) are forced-choice competition,
  not true lapse — but a scalar skill is not a sufficient user model.
  **M5b (skill + per-user lapse, 2-D GH) COMPLETE — does NOT rescue the
  softmax:** ΔELPD/read vs composed = −0.019 [−0.054, +0.004], only +0.002/read
  over M5, and its per-user lapse latent is weakly identified (NM ran to a
  saturated logit ridge, m_lam≈s_lam≈128 → degenerate bimodal lapse). FINAL
  ordering on full data (699 users, 62.7k held-out reads; `data_extset_m5.npz`):
  composed −1.335/read > M5b −1.354 ≳ M5 −1.356 — neither softmax variant beats
  the composed marginals; per-user confusion profiles are real structure (W4),
  a scalar(+lapse) user model is insufficient. smoke→full reversals are
  exactly why nothing from smoke is ever reported.

- **F51 (A2, COMPLETE — Bayesian refit confirms F43; SBC non-pass as run):**
  affine-invariant ensemble MCMC (24 walkers × 500 burn + 1500 kept, GH-marginal
  M2 likelihood, weakly-informative priors μ~N(0,1)/σ~HalfN(1)/λ~Beta(1.2,30)) on
  the t1_loo headline frame. **Posterior:** μ_ℓ −0.315 [−0.358,−0.278], σ_ℓ 0.434
  [0.414,0.456], μ_θ −1.038 [−1.065,−1.008], σ_θ 0.252 [0.220,0.273]; λ_fa 0.0024
  [0.0014,0.0035], λ_miss 0.0079 [0.0008,0.0172]. **P(both λ < engine's 0.025 |
  data) = 0.9999** — the FULL posterior (not just the point fit) puts the contest
  raters' false-alarm/miss lapses an order of magnitude below the engine's assumed
  0.025, upgrading F43 from point-estimate to Bayesian. **SBC (96 replicates):**
  rank-uniformity FAILS 5/6 globals (χ² p = 0.00/0.00/0.00/0.00 on μ_ℓ/σ_ℓ/μ_θ/σ_θ,
  0.01 λ_fa; only λ_miss 0.62). NOT a clean fitter validation — but the two
  explanations (genuine GH-marginal NM+ensemble mis-calibration vs. an
  under-converged-SBC artifact) are CONFOUNDED: the SBC harness deliberately uses
  cheap 300-step chains initialized AT the NM point fit (per the in-code note),
  which under-disperses the posterior and pushes truth toward extreme ranks (the
  classic SBC U-shape). So the non-uniformity does not by itself prove
  mis-calibration. FOLLOW-UP DONE → **F52** (proper SBC resolves it: 5/6 globals
  calibrated; only σ_θ residually under-estimated). `data_extset_bayes.npz`.

- **F52 (A2 follow-up — proper SBC: F51's alarm was MOSTLY a harness artifact;
  only σ_θ residually mis-calibrated):** re-ran SBC with the three fixes the cheap
  harness lacked — 48 walkers (vs 24), 1000+2000 steps (vs 200+300), OVER-DISPERSED
  inits (~2× posterior width, vs a 0.05 ball at the NM fit), and a PER-PARAMETER
  split-R̂ convergence gate (uniformity judged only on replicates that mixed for
  that param). 80 replicates, 60×80-read synthetic datasets; `study_extset_sbc.py`
  → `data_extset_sbc.npz` (84 min). **5/6 globals now PASS rank-uniformity:** μ_ℓ
  p=0.07, σ_ℓ 0.85, μ_θ 0.44, λ_fa 0.62, λ_miss 0.45 (converged 59–71/80 per param)
  — vs the cheap F51 SBC failing 5/6 (all but λ_miss at p≤0.01). So F51's blanket
  failure was dominated by under-converged 300-step chains, NOT fitter
  mis-calibration; the certification-relevant skill globals (μ_ℓ, σ_ℓ) and the
  lapse rates (F43) are CALIBRATED. **Residual:** σ_θ STILL fails (p=0.00) with a
  RIGHT-skewed rank histogram (top-bin pile-up [...,10,17]) ⇒ the fitter
  systematically UNDER-estimates the examinee BIAS-SPREAD σ_θ — a directional bias,
  not the symmetric over-confidence of mere non-convergence. Caveat: the θ-block
  mixes worst (max R̂ μ_θ=5.4; σ_θ R̂ numerically unstable), so σ_θ's failure is
  partly confounded with residual non-convergence — but the skew direction is
  consistent with genuine under-estimation. **Port note:** σ_θ is the spread of the
  bias NUISANCE (not the certified skill); under-estimating it makes the engine
  mildly over-confident about bias homogeneity — conservative-direction, low-stakes
  — flag for a non-centered θ reparameterization / θ-targeted sampler before σ_θ is
  ever relied on as a hard prior.

- **F41 (M13.1) — task1/domain1 release validated end-to-end:** 167,503 reads
  × 643 raters × all 5,000 task1 segs, binary (24.6% say-domain1 vs source-gold
  prevalence 23.9% read-weighted / 29.8% by case), zero (seg,rater) duplicates,
  100% key integrity (segs ↔ crosswalk, raters ↔ `user_id_to_rater_id`);
  +6,438 reads / +40 raters vs the older `extset_users` snapshot totals
  (93.3% of users match exactly). Signals: 5,000/5,000 finite, scale matches
  the bank pool (mean/sd 0.07/1.02 vs 0.07/0.96; s_sd median 0.80 both);
  schema drift: last vote col is `votes_other` (bank: `votes_domain7`); NO
  timestamps/Qscore (no longitudinal or Qscore analyses for K=1). Headline
  psychometrics (LOO signal): P(say domain1|s) 0.026→0.852 by decile;
  per-user accuracy 0.763±0.104 (321 users ≥20 reads, 140 ≥100); pooled
  response AUROC 0.779. Dual-contest cohort: 275 users in both contests,
  147 with ≥20 reads in both ⇒ NEW workstream W4 (empirical cross-domain
  skill correlation — grounds the engine's Σ_l hierarchical prior). SAP v2
  updated to K=7 (W1b binary frame = the engine's exact likelihood test).

**M15 math-audit findings (2026-07-01; full record + derivations in
`docs/AUDIT_POMDP_MATH.md` — Part I is the audit, Part II the implementation
log with parameter justifications; new/changed code rows in §1):**

- **F53 (MAJOR, filter calibration) — the propagate kernel dropped the
  y↔s_real coupling; the EXACT conditional kernel restores nominal coverage.**
  The learner's response y and its state update share the SAME s_real draw,
  so the correct kernel conditions the transition on the observed y:
  E[p|y], E[w|y] means and q² + gain²·Var[·|y] variances (dedicated 21-node
  GH rule — 11 nodes leave up to 2.2e-2 conditional-moment error at
  s_sd/σ∈[1,2], 21 give ≤2e-3 ≈ α_t·err two orders below q_t). Open-loop
  paired experiment (30 seeds, s_sd=0.85, identical streams,
  `study_exact_kernel_audit`): cov_t@90 0.631→0.906, cov_ℓ 0.771→0.913,
  RMSE(t) −0.0085±0.0022, RMSE(ℓ) −0.0071±0.0027 — the shipped t-intervals
  were ~40% too NARROW (overconfidence with worse RMSE, not caution).
  Largely explains F33 (fixable, not irreducible SMC error) and subsumes
  F30's smear_w (= the kernel's unconditional ℓ-half). Adoption consequence —
  the F5 information floor RISES (θ-SD 0.153→0.222, ℓ-SD 0.114→0.155 at
  s_sd=0.85 like-for-like) ⇒ D16 sd_floor must be recomputed via
  `recommended_sd_floor` at adoption. Shipped as `TaskFilter(
  exact_kernel=True)` opt-in, bit-identical at s_sd=0; port default = D21.
- **F54 (MAJOR, graduation honesty) — the σ∞-mixture belief fixes what
  kernel exactness cannot: the F28 prophecy.** `SigmaInfMixtureFilter` =
  exact Bayes over a J=7 ceiling grid with prequential evidence weights
  (valid under adaptive selection by ignorability; immune to F4 path
  degeneracy since no static parameter lives on a particle; p_static is its
  2-point special case). Campaign (20 seeds × 400 trials, real domain3 pool,
  D16 gate, `study_audit_hardening`): shipped falsely declares static_below
  20/20 and careless 20/20 (F28 reproduced); exact kernel alone still 20/20
  (calibration ≠ ceiling knowledge — attribution clean); mixture: careless
  0/20, static_below 7/20 (vs the M12 hardened stack's 23/30), wellspec
  kept at 20/20 with premature-FG 10→2 at median 49→109 trials (honest
  timing, the D19 direction). New output: trainability posterior
  P(ℓ∞>ℓ*|data) as the D18 recommendation statistic. TWO honest caveats:
  (i) the static_below residual 7/20 is STRUCTURAL information poverty — a
  trainability-conjunct does NOT remove it (those seeds' evidence had
  genuinely concentrated high, 0.77–0.99); certification stays with re-cert
  (D18); (ii) mixture discrimination speed depends on served-item PRECISION
  (uniformly high s_sd keeps ceiling hypotheses near-indistinguishable for
  hundreds of trials) — an independent argument for the F55 probes.
- **F55 — cut-state certification probes + anytime-valid e-gate (library,
  D23).** `cert_probe_score` = Fisher information about ℓ at the CUT state
  (σ*, t̂) — placement anchored at the BAR, breaking the belief-placement
  self-confirmation loop; NB ℓ is a SCALE-type parameter (∂z̃/∂ℓ = z̃/att²)
  so the optimum is z̃≈±1.35 under λ=0.025, not z̃≈1. `EProcessGate` =
  mixture Hoeffding e-process on probe outcomes vs at-bar accuracy;
  supermartingale + Ville ⇒ P(false fire | at-or-above bar) ≤ α at EVERY
  stopping time — the honest replacement for the D16 repeated-look
  heuristic. Measured: type-I 0/200, power 200/200 vs δ=0.2 deficits
  (campaign: careless e-gate fired 20/20); near-bar deficits need ~δ⁻²
  probes (Bernoulli information bound — the mixture covers that case via
  the full likelihood).
- **F56 — F25 re-adjudicated under the exact kernel: tier-3's hard-rule
  advantage VANISHES.** Paired CRN (20 seeds, domain3, pool 600): hard rule
  t3−t2 = −1.6 [−6.4,+2.6] (shipped kernel) → **+0.1 [−1.1,+1.3]** (exact);
  soft stays a small noise tax both ways. The F25 lookahead gain was partly
  compensation for belief mis-tracking that the kernel fix removes; honest
  beliefs also collapse the contrast variance ~4×. Tier-2 default
  REINFORCED; tier-3's Phase-3 case must be re-argued from fitted dynamics.
- Minor fixes shipped with M15 (Part II): D20 criterion f-gating (F22
  fixed); MA-9 learn-aware greedy Q (p_static mass earns zero reward);
  MA-6 honest pass_mass MCSE + `recommended_sd_floor` (the 0.23 constant is
  the s_sd=0-era value; real-bank rule gives ≈0.27); MA-4
  `skill_mode_multiplier(λ)` scaffolding (shipped 1.0772 trains at ~85.5%
  under the F51 fitted lapse; λ-correct value 1.01–1.03); MA-5 fold-then-
  smear E[w] ~2× under-count at the boundary (ranking-safe, documented);
  MA-10 plan errata (incl. the §6.1 y-sum collapse being an artifact of the
  approximate kernel — under s_sd>0 even soft-rule dynamics are
  response-conditional).

**M24 findings (2026-07-04, QUESTION SELECTION BY EXPECTED PROGRESS —
full analysis in `docs/M24_SELECTION.md`; validation
`studies/study_m24_selection.py` → `figures/data_m24_selection.npz`):**

- **F84 (posterior-progress PLACEMENT: six variants, three mechanisms,
  MIXED verdict — default stays off).** Delivered-value audit: the
  pipeline served a mean per-trial training value w_true = 0.48 of ideal
  (F80-lag sessions 0.15–0.17). The candidate — skill-mode argmax of the
  stratum-aware posterior expected σ-progress (`expected_progress_score`,
  the exact tier-1 β_σ term under the mixture; the D15 treatment for the
  σ channel) — went through six adversarial variants that isolated three
  mechanisms: (1) TAIL VOTE — the high-gap posterior tail dominates
  E[w·gap]; bar-referencing the placement credit silences the bulk and is
  strictly worse; (2) CALIBRATION EXTERNALITY — a free argmax abandons
  the M10 mirror invariant and the R–W attractor herds the TRUE criterion
  (accuracy/w_true unchanged, declaration 2×) — restoring the mirror
  (final variant) fixes it exactly (|t| end 0.171 vs 0.177);
  (3) MODEL-INJECTED WIDTH — M23 hazards keep the belief wide FOR SAFETY;
  myopic progress hedges against tails the hazard model erases. FULL-RUN
  final variant (20 seeds): wellspec near-parity (49@90% vs 46@100%),
  E-like jump FASTER (134 vs 148), contact 35% vs 30%, slow-wide SLOWER
  (121@70% vs 102@85%) — no dominance ⇒ `progress_placement` default-off
  experimental; successor = item-level tier-3 rollout. **F84-ii:** the
  pooled-cloud tier-1 scorer applied to a mixture credits the static
  stratum (T = identity ⇒ R ≡ 0) and uses the base σ_∞ per stratum —
  fixed in the stratum-aware scorer (the shipped D15 bias path's
  distortion is mild and left as-is). Altitude lesson (twice-confirmed
  with F85-ii): price selection changes end-to-end in
  trials-to-confirmed-mastery for ALL tasks at the real serving rate —
  per-trial training value and single-task declarations both mislead.
- **F85 (allocation by EXPECTED PROGRESS PER TRIAL — validated opt-in;
  sandbox default DEFERRED by F85-ii).**
  EP_k = Σ_j ω_j 1[rule≠static] Σ_i w_ij [β_σ·α_σ·w̄_i·(ℓ∞−ℓ_i)⁺·1[ℓ_i<ℓ*]
  + β_t·min(α_t, (|t_i|−t*)⁺)] (`expected_progress_rate`) ranks tasks;
  finish-first/retention/suspensions/cap unchanged. Subsumes the D36
  discount from first principles (static + at-ceiling mass earn exactly
  0). Production-harness full run (20 seeds, 3 rosters): trainable-task
  declaration 134→86 (B-trap), 150→72 (rates), 184→82 (jumper) with
  sunk/other-task trials roughly halved; careless FG 0/20; well-spec pair
  lateness 99→72. **F85-ii (blocks the default):** (i) post-boundary
  RE-POLISH loop — after a Gate-4 shift the nearly-mastered task's
  re-widened below-bar mass out-ranks the low-trainability recovering
  task (worst-first serves the wide task instead): trainable-pair
  COMPLETION is censored at 241 vs Gate-1's 188/189, the recovering task
  gets ~12 trials/session; (ii) sharp-margin static_below FG direction
  unfavorable though not significant (pooled 60 seeds: 27/60 vs 22/60,
  z≈0.9; one-sided guardrail). Successor queued: EP coupled to the
  confirmation lifecycle (ever-provisional ⇒ finishing need, not EP) +
  exploration floor; endpoint = BOTH-declared. Bar-referencing is
  principled at the ALLOCATION level and harmful at the PLACEMENT level —
  the F84/F85 asymmetry is the checkpoint's core insight. (The pooled
  adjudication also sharpens the M23 open item: the shipped stack
  false-graduates 22/60 static ℓ*−0.15 learners in 240 trials.)
- **M22-gate field verification (8 new sessions):** 0 consistency pauses,
  0 fatigue breaks, 0 flags on engaged sessions; D15 bias mode
  field-validated on USER-C (criterion −0.48 → −0.13 in one session, d′
  preserved — first live confirmation of dual control); A's mastery path
  exercised the full D33 lifecycle (provisional ×2 →
  `all_mastered_or_empty` self-end).

**M22 findings (2026-07-02, the FOUR-PARTICIPANT PROFILE PILOT — full
analysis in `docs/M22_PARTICIPANT_GATES.md`):**

- **F77 (worst-first allocation conflates "far from cut" with "worth
  training"; + the finishing-eligibility crack).** USER-A learned cleanly
  (d′ 1.3→2.1/2.7 across sessions, π→0.81 — the success path works); but
  B/C/D each had their WEAKEST task absorb 60–85% of trials while its
  trainability collapsed (B: domain3 68 trials, d′ flat 0.9, trainability
  0.56→0.15 — while their near-perfect domain2 starved at 6 trials/session
  with π stuck 0.63, just under the 0.70 finishing bar; C: 25 trials into
  d′ −0.33; D: 34 into d′ −0.35). Fix D36: `trainability_floor=0.25`
  deficiency discount (opt-in; sandbox on). Validation G1: earlier/more
  declarations in the fast-collapse guesser regime (median 116 vs 141,
  18/30 vs 16/30), correctly neutral in the slow-collapse and
  both-trainable regimes, FG 0 everywhere. **F77-ii (found by the sim):**
  finishing eligibility `sd_ℓ ≤ sd_floor` deadlocks when the mixture's
  cross-strata variance floor sits a hair above it (0.232 vs 0.23) — the
  residual crack in the F70 deadlock fix; `finish_sd_tol=1.25` relaxes
  ELIGIBILITY only (declaration keeps the strict floor). Residual open
  item: worst-first geometry still lets a floored hopeless task (0.25)
  outrank a π>0.9 task between finishing bursts — expected-progress-per-
  trial allocation is the principled successor.
- **F78 (a shadow flag must at least stop the bleeding).** The monitor's
  first real firing (USER-D, GLR 12.9, 5 events) changed nothing: 10 more
  trials went into the broken task (trainability 0.22→0.04), and the
  participant quit one trial into the next session. D's failure was
  BEHAVIORAL, not perceptual — 0.44 accuracy on EASY items (|s|≥1) while
  d′ +2.7 on the other task in the same interleaved session. Fix D37 Gate
  2: flag ⇒ suspend the task for the session (serving, not inference),
  fresh monitor window next session. Validation G2: post-flag broken-task
  trials 12.9→0.0, end trainability 0.14→0.36, good task unaffected.
- **F79 (late-session fatigue is real, un-modeled, and lands in the
  persistent belief).** Session-quarter accuracy fell to 0.4–0.5 for
  B/C/D (A stable); B's end-of-session ℓ̂ slid +0.23→−0.09 in the last ~10
  trials, saved into state. Fix D37 Gate 3: DIFFERENCED
  realized-vs-predicted deficit guard (trailing-12 vs the session's own
  first-12; δ=0.35, active from trial 24) ends the session early. The
  differencing is load-bearing: the raw level would have cut USER-A's
  fine first session at trial 17 (onboarding belief-optimism); the
  differenced form fires exactly on B s2 (+0.41) and D s1 (+0.37), silent
  on A/C. Stationary FA 5/100 sessions; D's detection margin is thin
  (+0.02) — recorded honestly. Phase-3 successor: λ(time-on-task)
  observation channel fed by the F15 probes.
- **M21-fix field verification:** d′-vs-label ≡ d′-vs-percept for all 8
  participant×task cells (zero label-evidence contradictions, vs 10% in
  M20); profiles kept four independent states; C's strong liberal bias
  (t̂ −0.48, d′ +1.87) queued for bias mode = machinery working.

**M21 findings (2026-07-02, the FIRST TWO HUMAN TESTERS — full analysis in
`docs/M21_TWO_TESTER_ANALYSIS.md`):**

- **F72 (learner identity is a protocol assumption the stack never
  checks).** Two testers ran back-to-back (89.5 s apart) on ONE
  machine-keyed belief state. Tester A: pure guessing (d′ ≈ 0 both tasks,
  40 trials) — an onboarding session. Tester B: genuinely competent at
  domain3 (d′ +1.15 vs label, acc **0.79 vs the percept** — near the
  ideal-observer ceiling for the served items) yet inherited A's
  pessimistic state: domain3 trainability 0.365 at handoff ⇒ the
  deficiency scheduler UNDER-served their strongest task (14 vs 26
  trials) and burned 14 bias-corrective trials on domain2 chasing a
  criterion estimated on tester A (t̂ ran +0.11→+0.63). The switch was
  INVISIBLE to every shipped monitor: e-gate quiet (wrong question),
  GapAnchor closed (gap < 4 h), prequential surprise flat (near-chance
  beliefs make everything unsurprising; a switch TOWARD competence
  LOWERS surprise). What does separate the testers: windowed d′ and
  log-RT (Δ 0.58 nats, Mann-Whitney p = 0.0009). Fixes: D34 (profiles +
  shadow ConsistencyMonitor). Arm-D quantification: after 40 competent
  single-task trials, a guesser-polluted state still shows trainability
  0.54 vs 0.76 fresh (strata weights carry the full history — correct
  Bayes under the violated assumption), while placement/served-accuracy
  recover fully (0.84 vs 0.83).
- **F73 (the render physics bound achievable skill — and the M20 gain put
  the bars near/above that bound).** Ideal-observer σ_eff =
  NOISE·√(1/n_win + 1/n_out)/GAIN; at gain 0.55 σ_eff = 0.606 (empirically
  0.605 through `render()`), vs v15 bars σ* = 0.736/0.857 — a human needed
  71–82% of IDEAL efficiency to bar, and the assumed expert ceilings
  σ_∞ ≈ 0.48–0.52 (the σ∞-mixture grid center!) were physically
  unreachable, structurally forcing "ceiling below cut" for every human.
  Both testers' trainability collapse is partly this. Fix D35: gain 0.85
  (σ_eff 0.392: bars at 46–53% ideal efficiency, ceilings 76–81%);
  test_sandbox now enforces σ_eff ≤ 60% of the tightest bar; resume
  across a recalibration blocked.
- **F74 (label–evidence contradictions at the render layer).** s_real ~
  N(s_mean, s_sd²) was drawn UNCONDITIONED on y\* ⇒ 8/80 served trials
  (10%; 4/26 of tester B's domain2) rendered evidence OPPOSING the
  feedback label — veridical-feedback training of the WRONG mapping,
  violating at the percept level the D11 contract production enforces by
  margin filtering. Fix D35: label-consistent truncated draw
  (deterministic per (seg, session), s_sd physics preserved).
- **F75 (onboarding placement runs hot for novices).** Skill placement
  targets 1.077·σ̂ against the belief-MEAN σ̂ starting at the optimistic
  prior ⇒ both testers spent whole sessions at 47–50% served accuracy
  (design target 84%) while ℓ̂ was still descending at session end. Fix:
  opt-in `ModeThresholds.skill_sigma_z` (place against
  σ̂·exp(z·max(sd_ℓ−sd_floor,0)), annealing to σ̂ at the floor; z=0
  bit-identical default). Arm E (24 seeds, σ0=2.4 novice, 200 trials):
  no cost (first-40 acc 0.761 = 0.761; final true σ 0.520 = 0.519),
  log-σ̂ RMSE 0.262→0.251. Sandbox opts in at z=1 (the human/motivation
  case is the stronger argument).
- **F76 (ConsistencyMonitor operating characteristics;
  `study_learner_switch`, `data_learner_switch.npz`).** Windowed (W=20)
  GLR vs the belief's own predictive, threshold pinned at the 99th pct of
  stationary two-session max (11.30; 95th = 9.35; single-task
  calibration sessions make this conservative for K=2 serving). At that
  1%-FA bar: competent→guesser caught 27/40 within 40 trials (median 19);
  guesser→competent 4/40 (belief adaptation absorbs it); mild σ 1.6→1.1
  0/40; the REAL two-tester switch peaks at GLR 7.0 (≈93rd stationary
  pct) — elevated, not decisive. READ: 20–40 near-chance trials cannot
  decisively identify a person; the monitor is a PROFILE-CORRUPTION
  TRIPWIRE (shadow, D34), profiles are the identity mechanism. Raw
  glr/rt_z logged per trial for post-hoc thresholds.

**M20 findings (2026-07-01, the manual-tester SANDBOX — user-directed):**

- **F71 (the sandbox = the delivery-vehicle prototype; `sandbox/`).**
  Interactive CLI protocol where the USER is the learner: synthetic
  signal-detection stimulus whose difficulty is monotone in the item's
  latent signal (verified; the human's σ/t are genuine in stack units),
  full M15–M19 stack per session (mixtures + probes/e-gates + finish-first
  + derived band + GapAnchor on REAL wall-clock gaps, v15-coherent = C1
  satisfied in-sandbox), production-shaped JSONL telemetry with full
  belief snapshots, resumable npz+json state (no pickle), safe mid-session
  quit, `--reset` archives (never deletes). Robot self-test mode shares
  the exact human code path. VERIFICATION: `tests/test_sandbox.py`
  (11 checks, isolated state) + a 20-session/800-trial robot campaign
  (archived): serve-once clean (800 unique segs), anchors on every daily
  gap, D33 lifecycle exercised (provisional→revoked×4→CONFIRMED on the
  near-band task — the safety rule visibly catching post-gap transients),
  both tasks CONFIRMED, report + trajectory figure + D18 handoff
  recommendation generated. Two bugs found and fixed BY the verification:
  (i) anchor probes bypassed serve-once (deterministic argmax re-served
  items, inflating apparent progress); (ii) the provisional watch only
  checked the SERVED task — a newly-mastered task stops being served, so
  its declaration was unobservable; the watch now sweeps all tasks.
- **D33 (the C2 post-gap confirmation rule, IMPLEMENTED).** Mastery
  lifecycle: training → provisional (gate fires in-session) → CONFIRMED
  only if the gate still holds AFTER the next ≥4h gap and its re-anchor;
  else revoked. Kills the F70 post-gap-transient false graduations at the
  protocol level. Lives in `sandbox/protocol.py` (sandbox-level; port
  candidate for the product).

**M19 iteration findings (2026-07-01, next-pass queue + readiness verdict):**

- **F67 (domain1 narrow-margin protocol).** Empirical margin-vs-floor check
  (20 at-ceiling learners × 600 trials, domain1 pool, mixture stack, D30
  config): **10/20 declare** (median 235 trials); the rest hover at
  π 0.2–0.85 without crossing — the mixture partially rescues v15
  domain1's 0.168 margin (better than F19's "never"), but ~half of genuine
  at-ceiling learners cannot clear the pass-mass gate within 15 sessions.
  **Port protocol for narrow-margin domains (margin/floor ≲ 1.3 — only
  domain1 under v15): hand off to re-cert on a trainability+plateau
  trigger instead of waiting for pass-mass** — the certification
  instrument's A-optimal items resolve what training-optimal placement
  cannot (the F31/D18 information argument, now with a concrete domain).
- **F68 (MixtureGapAnchor, M19-3).** Joint (ceiling × retention) product
  mixture at session open, retention marginalized out at anchor close
  (in-place strata mutation; the D22/D27 division of labor — per-gap
  nuisance vs persistent hypotheses). Unit checks: post-anchor state
  tracking |ℓ̂−ℓ| ≈ 0.05–0.25 (vs 0.33 unanchored); per-gap r̂ stays
  prior-dominated over a short block BY DESIGN (state re-anchoring is the
  load-bearing function; S accumulates retention evidence across gaps).
- **F70 (K=7 open-protocol capstone — three product-level lessons the
  single-task studies could not show; `data_k7_protocol.npz`):**
  (i) **Scheduler starvation deadlock:** worst-first interleave (D3)
  abandons a task as its π rises, so NO task ever clears π−2·mcse ≥ 0.95
  in multi-task service (0/106 declared, both arms, even with forgetting
  OFF). Fix: budgeted FINISH-FIRST phase (D32) — a task within striking
  distance whose ONLY missing condition is the pass-mass margin is served
  to completion (stall-guarded: budget 60 + cooldown; bias-blocked tasks
  stay worst-first — ranking by π alone recreated F24 via scheduling
  interruptions, caught by the step5 integration). OPT-IN
  (`TrainerPolicy(finish_first=True)`); legacy default bit-identical.
  (ii) **Consolidation is load-bearing for the product:** with a FIXED
  3-day forgetting stability, the open protocol's equilibrium skill sits
  below every v15 cut — no one can EVER certify on 40-trial sessions with
  multi-day gaps. Real practice consolidates (§3A LT2); modeled as
  τ_f,k = τ_f0·(1+n_k/40)^1.5 — the exponent is a PILOT ENDPOINT the
  GapAnchor S-estimates will measure, not a fitted fact.
  (iii) **Measure-or-leave-alone re-anchoring:** prior-only gap transforms
  of unprobed tasks are HARMFUL once consolidated (prior assumes more
  forgetting than reality): transforming 6/7 tasks per session → 0/106
  declared vs identity 14/106. Anchoring ONLY the probed task: **30/106
  declared (2.1× identity)**, stale-opens 6, at ~8 probe-trials/session.
  Residual: 7 declaration-instant FGs (23% of declarations) are mostly
  post-gap TRANSIENT dips near the cut (recover with practice; a
  'post-gap confirmation before declaration' rule is the queued fix —
  different failure character from F28 prophecy). PRODUCT IMPLICATION:
  at K=7 a median learner certifies ~2 tasks in 40 sessions under these
  physics — the pilot should scope K=1–3 domains (consistent with the F48
  single-task power design), and full-K certification is a long-horizon
  product journey, not a pilot deliverable.
- **F69 (drift-aware shrinkage CLOSES the F65 refinement).** Odd/even
  interleaved splits give the drift-free noise estimate: V̄ falls from
  [0.57, 0.79, 0.13] (temporal, drift-inflated ×2.4–3.3) to
  [0.23, 0.24, 0.09]; shrink factors rise to B = [0.79, 0.84, 0.47]; and
  held-out shrunk personalization now MATCHES raw (+0.0074±0.0013 vs
  +0.0081±0.0015/read) at smaller SE — robust AND near-optimal, exactly
  the F65 prediction. **Deployment estimator settled: odd/even-calibrated
  EB shrinkage of per-user dynamics fits** (`data_hier_shrink_oe.npz`).

**M18 iteration findings (2026-07-01, "next pass"; decisions D29–D31):**

- **F64 (the static_below residual SOLVED — it was the mean-skill gate).**
  M18 hardening ablation (5 arms × {static_below, wellspec, powerlaw} × 30
  seeds, v15, anchored assumed rates, gate-branch attribution;
  `data_m18_hardening.npz`): in the M17 baseline, **all 11/11 static_below
  false graduations fired via the D16 MEAN-SKILL branch (viaPM=0)** — the
  branch was designed for v14 domain1's sub-floor margin (F19) and, on a
  MIXTURE, reads the strata-dragged mean ℓ̂ before the stratum weights
  resolve (each high-ceiling stratum is a conditional F28 filter, so the
  mixture MEAN inherits the attractor even while the mixture WEIGHTS are
  still honest). Disabling it (`ModeThresholds(meanskill_gate=False)`):
  static FG 11→**1**/30, powerlaw FG 8→1, at moderate cost (wellspec med
  declaration 104→131, 8/30 pushed past the 400-trial budget = later
  sessions under the open protocol; 22/30 within budget). τ=0.45 alone: FG
  4/30 but lateness 175 — dominated. τ=0.45+noms: FG **0**/30 at lateness
  215 and wellspec 13/30 — the FG=0 frontier point, recorded as a PI knob,
  NOT adopted (D28 judgment: 3.3% residual FG into a re-cert layer that
  catches ~90% ≪ doubling every genuine learner's declaration time).
  **D30: mixture-based graduation runs meanskill_gate=False; the branch
  stays available (default True) for point-filter/narrow-margin
  deployments — re-check domain1's v15 margin (0.168) vs the floor before
  port.** Sole remaining static_below FG came via pass-mass (1/30).
- **F65 (hierarchical refit on REAL fits — personalization pays, and the
  shrinkage estimator needs a drift-aware noise term).** 2,040 half-fits
  over 170 users (`data_hier_shrink.npz`): split-half reliabilities
  log α_t 0.47 / log α_σ 0.48 / ℓ_∞ 0.30 — the RATES carry real per-user
  signal on real data (vs the F58 sim where they were unidentified: real
  heterogeneity ≫ sim heterogeneity, echoing F49's per-user structure).
  Held-out second-half evidence vs the population params: **raw
  personalized +0.0081 ± 0.0015/read** — REVERSING F58's sim result
  (personalization helps when user effects are large and logs are long);
  shrunk +0.0053 ± 0.0009 (positive, lower mean, smaller SE). The
  docstring's pre-registered caveat materialized: first/second-half V̄
  absorbs real within-user DRIFT ⇒ B over-shrinks (0.49/0.48/0.22).
  **Queued refinement: odd/even interleaved splits separate noise from
  drift; expect B↑ and shrunk ≥ raw.** Deployment stance meanwhile:
  personalize with shrinkage for robustness (smaller SE = fewer disasters),
  raw for average-case; population μ = (α_t e^−2.11 ≈ 0.12, α_σ e^−3.09 ≈
  0.045, ℓ_∞ 0.128) confirms F62.
- **F66 (M18 final capstone OC — the new operating point).** Zoo OC
  re-pinned under the full M18 config (exact kernel + mixture τ=0.30 +
  probes/e-gate + NO mean-skill (D30) + anchored rates (D29) + derived
  band + v15): **adversarial FG 1/90** (static_below 1/30, careless 0/30
  e-gate 29, anti 0/30 e-gate 26) vs the point comparator's 78/90 — an 11×
  improvement on M17's residual (11/90). Cost profile as chosen: genuine
  learners declare later (wellspec 22/30 within the 400-trial window, med
  131, FG 1; asym_fitted 28/30; powerlaw 11/30 with trainability 0.43 —
  honestly UNRESOLVED rather than falsely declared; open-ended protocol ⇒
  later declarations, not failures). `data_m17_capstone.npz` re-pinned
  (M17 numbers superseded).
- **D31 (pipeline v15 coherence):** `run_pipeline` now uses
  `instrument("v15")` end-to-end — eval cuts + corr_t t-block prior,
  trainer targets, AND re-cert all share one bar (the D25 coherence
  requirement); trainer assumed rates = D29 anchored. test_step8 re-pinned:
  FAIL×3 → 174 train trials → **all 3 tasks flip to PASS** (previously 2
  under the mixed v14/v15 regime).

**M16 iteration findings (2026-07-01, same-day continuation; studies
`study_train_sbc --exact`, `study_phase3_fit`; D23 wiring in
`trainer_policy.py`):**

- **F57 (F33/F53 CLOSED-LOOP CLOSURE) — the exact kernel restores NOMINAL
  coverage under live tier-2 selection at every checkpoint.** 200-rep
  closed-loop SBC (the F30/F33 harness, real-bank pool): shipped kernel
  reproduces the archived record exactly (ℓ n=50 dip cov90 0.82 KS 0.22 =
  F30 unmitigated; θ cov90 0.82–0.88 persistent = F33); exact kernel: ℓ
  cov90 = 0.91/0.90/0.95/0.92 and θ cov90 = 0.90/0.89/0.91/0.94 at
  n=10/50/150/300, all KS ≤ 0.08. `data_train_sbc{,_exact}.npz` re-pinned.
  F33's residual is GONE — trainer bias CIs are now honest; D21's port-flip
  case is complete.
- **F58 (Phase-3 fitter validated on train/test splits, M16-C).**
  `dynamics_fit.fit_learner` (CRN prequential-evidence penalized ML of
  (log α_t, log α_σ, ℓ_∞)) on a 24-train/16-test heterogeneous cohort
  (biased sub-skill starts; truth 1.25–1.6× above the D7 defaults, D14
  world; 300-trial logs from the default-params tier-2 logging trainer).
  Results (`data_phase3_fit.npz`): (i) POPULATION means recovered on all
  three axes — α_t 0.200→0.240 (true 0.320), α_σ 0.060→0.061 (true 0.075),
  ℓ_∞ 0.733→0.829 (true 0.800); (ii) per-learner CEILING identifiable
  (corr 0.71 @ n=150, RMSE 0.22 @ n=300 vs cohort SD 0.21) but per-learner
  RATES weakly identified even at n=300 (corr ≤0.35, prior-pulled — the
  F59 flatness); (iii) PERSONALIZED fits on 150 trials OVERFIT (held-out
  Δ −0.008±0.004/trial vs default; fitted-pop ≈0; oracle +0.003) ⇒ Phase-3
  fitting must be HIERARCHICAL (shrinkage), exactly as the plan intended —
  now with evidence; (iv) rule-ID (F12) 15/20 per learner at 150+150
  trials, median margin 0.72 nats ⇒ per-learner inconclusive, COHORT-level
  decisive (evidence sums across learners); (v) CLOSED-LOOP payoff: fitted
  population params HALVE declaration lateness (median n_decl − n_true
  41 → 21.5 wasted trials; oracle 7) at unchanged learning efficiency
  (true-mastery median ~23 both arms — replicates M10 placement
  robustness; benchmark re-pin under exact kernel likewise: tier2 Δ +0.0,
  tier1 −2.3±0.8, ordering preserved). CAVEAT/design note: fitted ℓ_∞
  should feed the D22 MIXTURE CENTER, not a point filter — the FG pattern
  (fitted-pop 2 vs oracle 0 of 16) shows a higher point-ℓ_∞ strengthens
  the F28 attractor for non-trainable learners; trainability
  classification was underpowered here (15/16 trainable test cohort) —
  F54's campaign remains the proper test of that axis.
**M17 iteration findings (2026-07-01, user-directed continuation; decisions
D25–D28):**

- **F60 (OQ6 RESOLVED — the 0-bias math).** Under BOUNDARY-ANCHORED
  mirror-paired label-alternating serving, the R–W criterion is an AR(1)
  with E[t_∞]=0 (the anchor sets the mean — F23 formalized; placement that
  tracks the criterion has NO restoring force and t random-walks, SD ≈ 3 vs
  ≈ 0.13, reproduced analytically) and stationary floor
  Var_∞ = (q_t²+α_t²Var[p̂])/(κ(2−κ)) + (α_t δ̄)²/(2−κ)², κ = α_t(1−2λ)φ(a/σ_e)/σ_e.
  Strict label ALTERNATION is load-bearing: iid labels would inject α_t²/4
  of diffusion — an order-of-magnitude worse floor (F6b upgraded from
  fairness to control theory). Formula validated vs direct simulation to
  ≤3% across the parameter grid; E[t_∞]≈0 confirmed. Implemented:
  `bias_stationary_sd`/`derived_t_star` (trainer_policy), and the D16 gate's
  bias condition now uses the derived σ̂-adaptive band by DEFAULT
  (ModeThresholds.t_star=None; ≈0.25 for benchmark params vs the old
  assumed 0.30). Zero-bias serving ≈ skill serving (band within ~15% of the
  offset-optimum), so no dedicated mode is needed inside the band.
- **F61 (D27 validated — GapAnchor beats assumed-decay handling under an
  off-model truth).** Multi-session protocol sim (8 × 40-trial sessions,
  gaps 0.5–21 d, TRUE forgetting = power law ± person jitter, deliberately
  off the grid and off the exponential prior tilt; 20 seeds):
  post-gap tracking error |ℓ̂−ℓ| 0.330 (identity) / 0.328 (widen-only) →
  **0.067 (anchor, −80%)**; |t̂−t| 0.235/0.228 → 0.111; stale-mastered gate
  calls (priority-1 FG analog) **71 / 11 → 0**; overhead ≈ 9 probes/returning
  session (adaptive 4–16); personal stability recovered S ≈ 9.7 d median
  from a 7 d prior against the power-law truth. `data_gap_anchor.npz`.
- **F62 (EXTSET dynamics anchoring — the first EMPIRICALLY-GROUNDED
  dynamics priors; OQ8=yes unlocked W3).** 1,158 (user × domain) penalized-
  ML fits on the real task2 reads (193 users ≥60 reads, LOO signals per
  F40, consensus y\* per OQ9, non-consensus reads fb=0 via the MA-3
  semantics, Pool(42); `data_extset_dynamics.npz`):
  **α_t median 0.097 IQR [0.046, 0.186]** — HALF the D7 default 0.200;
  **α_σ median 0.047 IQR [0.012, 0.087]** (default 0.060; τ_σ ≈ 21 reads);
  **ℓ_∞ median +0.127 IQR [+0.007, +0.202]** — real feedback-practice
  headroom ≈ +0.43 over the novice center (−0.3, F43), BUT the median
  fitted ceiling sits BELOW every v15 cut (0.155–0.354): for the median
  EXTSET-like novice, contest-style practice alone does not reach
  certification bars; upper-quartile users clear the lower cuts. Convergent
  validity: corr(mean fitted ℓ_∞, user gold-accuracy) = +0.595 (n=193).
  Caveats (recorded): between-read gaps ignored ⇒ α's deflated by any real
  forgetting (conservative D14 direction); one-vs-rest frame competition
  (F43/F49); engine λ assumed (benign per F63). **Action for the next
  iteration: adopt the anchored priors (α_t ≈ 0.10, α_σ ≈ 0.05) as the
  D7/D14 defaults in trainer studies — note the current default α_t = 0.2
  OVER-estimates reality ×2, which is exactly F17's anti-conservative
  direction.** Code-hygiene note: NM excursions to extreme ℓ produce benign
  overflow warnings in the exact kernel (probabilities clamp at the lapse
  floor/ceiling); a defensive ℓ-clip is a cosmetic follow-up.
- **F63 (M17 capstone — zoo graduation OC under the NEW default stack, v15
  cuts, 30 seeds × 400 trials, `data_m17_capstone.npz`).** Point filter:
  declares EVERY member 30/30 (the F28/F29 disease is cut-version-independent;
  static_below FG 30/30, careless FG-perf 30/30, anti FG 30/30). New stack
  (exact kernel + v15-anchored σ∞-mixture + probes/e-gate + derived band):
  **careless 0/30 (e-gate 29/30), anti 0/30 (e-gate 26/30), static_below
  11/30** (vs point 30/30; residual GREW vs the v14-conditioned M15 run
  because the wider v15 ceiling-to-cut margin halves the prior mass below
  cut — P(ceiling<cut) 0.22→0.12 — slowing evidence resolution; end-of-run
  trainability median 0.03 shows the mixture does resolve it, AFTER those
  11 premature declarations); genuine learners kept: wellspec 30/30 (FG
  4 vs point 4+1), asym_fitted 30/30 FG 3/0 (λ-robustness confirmed in
  closed loop), t3link 23/30, powerlaw 19/30 (censoring at budget 400 ≈ 10
  sessions — partly a budget artifact under the open-ended protocol).
  COST (priority-2): declaration lateness ≈ 72–92 vs point 18–43. Verdict
  under the D28 hierarchy: adversarial FG mass 79→11 latent (30→11 perf)
  for ~+60–80 trials of lateness — the right trade; certification-grade
  static_below discrimination remains D18 re-cert's job.
  **λ sensitivity (M17-6):** standalone calibration — coverage under the
  EXTSET-fitted lapse (λ_fa .0024/λ_miss .0095, filter assuming .025) is
  indistinguishable from well-specified (cov_t/cov_ℓ @90: 0.921/0.919 vs
  0.923/0.918); with the closed-loop asym_fitted OC (30/30, FG≈wellspec),
  the locked λ is CALIBRATION-BENIGN for training; D24's constant refresh
  stays a port nicety.
- **F59 (methodological, pilot-design input) — one-step prequential
  evidence is nearly FLAT in the dynamics parameters** (~0.005 nats/trial
  even for gross σ_∞ error; the filter's state-tracking self-corrects each
  step, so one-step prediction barely improves with correct dynamics).
  Consequences baked into the M16 designs: (i) per-learner fits need the
  TRANSIENT (biased sub-skill starts; post-convergence trials dilute rather
  than help); (ii) evidence comparisons need multi-seed CRN averaging
  (resample-branch noise ≈ the per-trial signal); (iii) do NOT power pilot
  dynamics analyses on ΔELPD — power them on decision-level outcomes
  (declaration lateness/FG, trainability classification). Pinned by
  test_audit_fixes check 20.

---

## 2B. Empirical-Bayes priors grounded in real artifacts (added M1.5; supersedes placeholder D7 defaults)

From `cert_config_general.yaml` (v14) + `Sigma_l_fitted_k7_general.npy`. Engine
coords (skill ℓ; σ=exp(−ℓ)). Per task domain1..7:

| task | ℓ* (cut) | σ*=exp(−ℓ*) (mastery) | expert_ℓ_mean | **σ_∞≈exp(−expert_ℓ)** (skill ceiling, LT1) | nonexpert_ℓ | Σ_l diag (cross-rater Var ℓ) |
|---|---|---|---|---|---|---|
| domain1 | 0.325 | 0.722 | 0.410 | 0.664 | −0.061 | 0.50 |
| domain2 | 0.256 | 0.774 | 0.629 | 0.533 | −0.091 | 0.84 |
| domain3 | 0.534 | 0.586 | 0.765 | 0.465 | −0.167 | 1.43 |
| domain4 | 0.330 | 0.719 | 0.643 | 0.526 | −0.150 | 1.70 |
| domain5 | 0.479 | 0.619 | 0.681 | 0.506 | −0.014 | 1.81 |
| domain6 | 0.486 | 0.615 | 0.740 | 0.477 | +0.024 | 2.11 |
| domain7 | 0.442 | 0.643 | 0.714 | 0.490 | −0.053 | 1.63 |

Load-bearing facts:
- **σ_∞ prior (skill floor / LT1 target) = exp(−expert_ℓ_mean)** per domain (0.46–0.66).
  Replaces the plan's hand-set σ_∞ (D10).
- **Every domain has expert_ℓ_mean > ℓ\*** ⇒ the expert ceiling is *past* the mastery
  bar (σ_∞ < σ\*): training toward the empirical ceiling guarantees a PASS. LT1 is
  achievable and data-grounded, not aspirational.
- **Initial-skill prior**: cross-rater ℓ spread Var = Σ_l diag (0.5–2.1); a fresh
  learner's σ_0 prior centers near non-expert (ℓ≈0, σ≈1) with that variance.
- **Var_prior(ℓ_k)=1** (unit-diag Corr_l) — the AD6 R-gate denominator, confirmed
  from the real npy.
- Bank signal scale: s_mean ≈ [−2.8, 3.6], s_sd medians 0.6–0.98 per task (s_sd is
  material — thread through reweight + selection, R5).

**v15 update (2026-06-11):** new ℓ* from the curated/credentialed data set, recorded
as `ell_star_unified_v15` in `cert_config_general.yaml`. All 7 cuts moved DOWN vs the
v14 values in the table above (domain1 0.325→0.238, domain2 0.256→0.155,
domain3 0.534→0.306, domain4 0.330→0.257, domain5 0.479→0.321, domain6 0.486→0.354,
domain7 0.442→0.304; deltas −22%…−43%), so σ*=exp(−ℓ*) rises to 0.70–0.86. ℓ* given
at 3 dp; per-task CIs/J/panel counts/expert means/source sha256 still pending — the
σ_∞ and Σ_l columns above are still v14-derived. If expert means hold, every
ceiling-to-cut gap WIDENS (domain1's OQ7-critical gap 0.085→0.172, easing the D16
mean-skill-branch pressure) and σ_∞<σ* (LT1) holds by larger margins; mean cut drops
0.407→0.276. **Validation received same day (ablation, synthetic raters, live SMC) —
the lower cuts are RECALIBRATION, not leniency:** v15 release = credentialed ℓ* +
corr_t + N=1200; isolated ℓ* lever = +17.4pp PASS@+0.6 (~80% of full-v15's +21.4pp)
while CUTTING both errors (false-PASS 1.14%→0.43%, false-FAIL 1.14%→0.57% at N600).
Full v15: PASS@+0.3/+0.6/+1.0 = 48/77/87% (62/88/94% at per_task=100 live bank),
false-PASS 0.29% / false-FAIL 0.43%; +18.4pp over shipped under realistic examinee
bias (θ≈−1.7; the earlier "7× collapse" was a sign-error artifact). corr_t is
resolution-neutral but trims false-PASS; N=1200 gives nominal SBC coverage at every
level (N=600 mildly under-covers, 0.932@95; N=2400 adds nothing) ⇒ N=1200 is the
calibration sweet spot, **staged post-pilot, live stays N=600** (bit-identical
instrument for in-flight tiered-pilot cohort). Remaining external validation:
coverage harness was K=6 Mode-A (K=7+corr_t run pending); real-pilot-response replay
is the reviewer-#1 circularity-breaker; pilot θ decides EB-centering upside.
**Consumers NOT switched:** `bank_adapter.ELL_STAR/SIGMA_STAR` and
`policy_k7_general` default block remain v14 pending the switch decision;
M10-pinned numbers (e.g. "mean cut 0.37", benchmark cells on domain3's 0.534 cut)
are v14-conditioned and would need re-pinning.

## 3A. Long-term objectives (user directives, 2026-06-10 — Phase-3 scope, hooks now)

These are standing product objectives stated by the user. They are NOT implemented
in Phase 2, but Phase-2 prototypes must not paint us into a corner: the hooks in
Decision D9 are mandatory from Step 1 onward.

- **LT1 — Drive *long-term* bias → 0 and *long-term* skill → maximum across all
  domains.** The plan's (σ_k, t_k) are within-session states; the product target is
  the *consolidated trait* that persists across days. Proposed methodological shape
  (to be rigorously constructed in Phase 3): a **two-timescale state-space model** —
  a slow consolidated trait (σ̄, t̄) per task, and a fast within-session state that
  starts each session near the consolidated trait and is partially consolidated at
  session end (session outcome updates the slow state). "Long-term skill maximum"
  is bounded by the learner's intrinsic floor σ_∞ (plan Eq. sigmaschedule), so the
  operational target is σ̄ → σ_∞ and |t̄| → 0, with mastery still gated by the cert
  cut-scores σ*_k = exp(−ℓ*_k) (F14). The cross-session learner ledger (sequence of
  TrainingSeeds/posteriors over calendar time) is the data structure that makes
  (σ̄, t̄) estimable.
  **Now grounded (M1.5):** σ_∞ per domain = exp(−expert_ℓ_mean) (§2B); since
  σ_∞ < σ\* in every domain, "long-term skill → maximum" has a concrete, achievable
  target (the expert ceiling) that clears the certification bar.
- **LT2 — Ebbinghaus forgetting curve and its counteraction.** Between sessions,
  skill (and possibly bias calibration) decays toward baseline. Standard form:
  retrievability `R(Δt) = exp(−Δt/S)` (or power law), with **stability S growing
  under successful spaced retrieval** — this is the principled replacement for the
  plan's SM-2 heuristic (plan §8, retention mode). Mathematical implementation
  path: (a) a **between-session decay kernel** `T_gap(θ' | θ, Δt)` applied to the
  belief when a new session opens (log σ relaxes back toward its pre-training
  baseline at rate set by S; process-noise widening with gap length); (b) a
  per-task/sub-category **memory-stability state** in φ, updated by retrieval
  outcomes; (c) review scheduling = present a retention trial when predicted R
  drops to a threshold (this *derives* the spacing schedule instead of hardcoding
  SM-2 intervals). Benchmarks already measure post-gap retention (plan Step 7);
  LT2 upgrades the gap simulation from "process-noise drift" to a real decay model
  once fitted.
- **Interaction LT1×LT2:** consolidation and forgetting are the two halves of the
  same slow dynamics — one model, two signs. Fitting both needs multi-session pilot
  data with absolute timestamps (hence D9).

## 3C. EXTSET real-response audit (M11.2, 2026-06-11 — user-requested feasibility study)

**Goal of the request:** replace the *simulated* training algorithm with real
EXTSET response data. **Verdict: partially feasible — EXTSET can replace the
learner RESPONSE MODEL and anchor dynamics priors, but CANNOT replace the
feedback-driven training BENCHMARK (no such real data exists).** Awaiting user
scope decision (see status log M11.2) before implementation.

**What EXTSET data we HAVE** (M11.2: in `the main repo/`,
the live multi-K7 repo — NOT in scratch. **UPDATE M13 2026-06-12: the scrubbed
release is NOW IN SCRATCH — current inventory and verified structure in §3E;
this section kept for the feasibility reasoning + M11.2c scope decision.**):
- `data/labels/external/extset_goldpanel_raw/extset_expert_labels.xlsx`
  — **gold panel**, 5000 pattern-class cases × 4 experts (expert_1/expert_2/expert_3/expert_4), complete,
  cross-sectional `pattern_class`. Ingested as `extset_expert` (20,000 rows).
- `…/extset_novice_labels.csv` — **the real response data**: 125,865 reads ×
  699 users, schema `Case ID, User ID, Read ID, Labeling State, Qscore, User Label,
  Origin, Response Submitted At`. **HAS a time axis** (timestamp + Read ID) and an
  in-platform competence proxy (Qscore). 100% crosswalk to bank seg_ids via
  `external/extset_2025/case_id_to_seg_id.csv` ⇒ s_mean/s_sd recoverable per read.
- demographics/tiers (`extset_users.csv`: 280 novice / 59 expert / 47 borderline /
  740 unknown), survey, Qscore (corr 0.873 w/ accuracy — a convergent-validity check
  for ℓ̂, per AUDIT §5).

**Structural facts that BOUND feasibility:**
1. **No feedback loop.** EXTSET was an unsupervised labeling CONTEST. The trainer's
   defining mechanism (veridical feedback + rationale driving the R–W criterion/skill
   update) was never in the loop. Qscore is a "rolling/windowed competence score, NOT
   a learning curve" and is non-monotone within user (AUDIT §5).
2. **No adaptive placement.** Users saw contest cases in contest order, not items
   placed at the learner's current 85%-difficulty point. The trainer's whole value
   (noise/dynamics-aware placement, F25/D15) has no counterpart to validate against.
3. **Multiclass, not binary.** Reads are 7-class pattern-class labels; the engine/trainer model
   binary target-present detection. Same one-vs-rest reduction as eval (D11) needed.
4. **Short sequences.** Median 22 reads/user (τ_σ≈33–50 ⇒ too short to see σ-learning);
   only 145 users ≥100 reads, 78 ≥200 — the longitudinal-capable slice is small.
5. **Weak, confounded learning signal.** Among the 145 ≥100-read users, 2nd-half −
   1st-half accuracy = **+0.018 mean / +0.022 median, 62% improving** — a real but
   small drift, confounded with case-order/fatigue/selection (no feedback to attribute
   it to). Enough to ANCHOR dynamics-prior magnitudes, NOT to validate trainer control.

**Three-tier feasibility (the fork the user must choose):**
- ✅ **(A) Replace the learner RESPONSE MODEL** (`learner_sim.respond`'s probit link)
  with an empirically-fitted/validated link from the 125,865 real reads (s recoverable,
  gold available). Replaces simulated *response generation* with real response behavior;
  benchmarks then run against an empirical learner. High value, fully feasible.
- ⚠️ **(B) Fit descriptive LEARNING DYNAMICS** on the ≥100-read subset to ground the
  Phase-3 dynamics priors (τ_σ, α_t, σ_∞ magnitudes) — feasible but NOT feedback-driven
  validation; it is unsupervised drift (caveat 1/5). Resolves part of F22/§3A, honestly
  labeled.
- ❌ **(C) Replace the training-OUTCOME benchmark** with real feedback-driven trainer
  trajectories — INFEASIBLE here. No human went through the trainer; the only real
  recorded sessions (`replay_mbw.py`, the 3 scratch `*_trials` CSVs) are EVAL/cert
  (no feedback). Requires a live pilot of the actual trainer.

**Adjacent real data noted:** `results/*_trials.csv` (recorded cert sessions with
`reaction_time_ms, answer_changes, n_interactions, select_ms` + per-trial posterior
fields) — these are the real EVAL response sequences `replay_mbw.py` replays; relevant
to the eval side / RT channel (Phase-3 F-items), not the trainer.

**SHARPENED CONCLUSION (M11.2b) — "can EXTSET replace the simulated runs to PROPERLY
TEST the learning-algorithm methodology?" → NO for the methodology as a whole; the
barrier is STRUCTURAL (two independent reasons), not sample size:**
1. **Off-policy / no overlap.** The trainer is a CLOSED-LOOP adaptive selector — every
   trial it argmaxes the NEXT item over the pool from the current belief (trainer_policy
   .select, benchmark_trainer._select, tier3_select). Testing that policy requires the
   learner's response to the item the POLICY chose, at that trajectory point. EXTSET
   users answered a FIXED contest set in contest order; the response to an
   adaptively-chosen item at an arbitrary belief state is a counterfactual the log does
   not contain. Restricting the trainer's choice-set to the ~22 items a user happened to
   see degenerates the policy into a fixed replay — that tests the FILTER's measurement,
   not the trainer's control.
2. **No feedback ⇒ no learning to track.** The methodology's core claim is that veridical
   feedback drives the R–W (σ,t) update and the trainer accelerates it. EXTSET delivered
   no per-item feedback, so no feedback-driven state evolution occurred; there is no real
   learning trajectory to validate the filter's tracking against, and trials-to-mastery
   (the benchmark headline) is undefined.
**What EXTSET CAN properly test (open-loop COMPONENTS, real value):** (a) the OBSERVATION
MODEL — does p=λ+(1−2λ)Φ((s−t)/σ) predict real reads given bank s? (b) FILTER MEASUREMENT
— replay a user's real (s,y) sequence through the filter (à la replay_mbw, eval-style),
check ℓ̂ vs Qscore/gold accuracy (convergent validity). Both are measurement checks, not
control-loop tests.
**The honest reframe:** the achievable upgrade is **empirically-grounded simulation** —
fit the generative model (response link + dynamics-prior magnitudes + skill/bias start
distribution) on EXTSET, then keep simulating the closed loop. That removes "the
simulator's response model is postulated" for the open-loop pieces and makes the synthetic
runs real-data-anchored, but the LOOP stays simulated. Model-based off-policy evaluation
cannot escape this: it requires the very fitted model you'd be grounding, so it returns to
simulation by construction. **Definitively testing the methodology end-to-end requires a
feedback-driven trainer PILOT** (real raters, adaptive placement, veridical feedback) — new
data collection, not minable from EXTSET or any existing file.

**M11.2c — user pushback (decision, 2026-06-11): the EXTSET routes are a SIDESTEP of the
methodology; rejected. Keep the learning algorithm AS-IS.** The user asked directly whether
the proposed EXTSET routes amount to swapping the real algorithm for one that merely runs
on EXTSET. Honest answer = yes, with precision:
- Route A (response-model fit) is NOT a rival learning algorithm — it is a STATIC
  measurement calibration sitting UNDER the trainer (the filter's likelihood). It grounds
  the simulator but tests none of the methodology's control claims (adaptive placement,
  feedback-driven skill/bias movement, trials-to-mastery). Sidesteps AROUND the algorithm.
- Route B (fit dynamics on EXTSET's ≥100-read drift) IS the substitution the user feared:
  it fits an UNSUPERVISED, no-feedback, non-adaptive drift model — a genuinely DIFFERENT and
  weaker learner than the feedback-driven adaptive trainer. Calling its fit "validation"
  would be bending the ALGORITHM to fit the available DATA, i.e. testing a surrogate.
⇒ **Decision: the feedback-driven closed-loop trainer stands unchanged; its validation
remains simulation-based** (now v15-instrumented + tier-3-extended, M11). EXTSET is reserved
ONLY for honestly-labeled component grounding / convergent validity IF desired later — never
as a methodology test. A real pilot is the only path to definitive end-to-end validation.

## 3D. Pre-publication / pre-release verification strategy (M11.3, 2026-06-11)

**Question:** how to verify the closed-loop feedback-driven trainer before publication
or release — is public/live testing the only way? **Answer: no.** Verification is a
LADDER; public release is only the last rung and should be confirmatory by then. Most
verification is pre-human; the irreducible external-validity gap needs real feedback-driven
data but that means a CONTROLLED PILOT, NOT public release.

**The ladder (rungs 1–4 need no new humans; we already hold most of the machinery):**
1. **Internal correctness** — code implements the math; filter recovers a known posterior
   (test_step3 <0.06 vs fine grid); identifiability hard gate (test_step4 |corr|≤0.17).
   DONE.
2. **Self-consistency (well-specified sim)** — under the assumed model the trainer
   graduates learners / beats baselines (M7/M10/M11). NECESSARY BUT CIRCULAR: the simulator
   IS the filter's assumed model, so this cannot catch mis-specification. Never the
   headline claim on its own.
3. **Robustness to mis-specification (HIGHEST-VALUE pre-pilot work).** Break rung-2
   circularity: run the trainer against simulated learners that DIFFER from its assumed
   model across every axis reality might differ — rule form (soft/hard ✓ F25/D14), rate
   misspec (2× ✓), **dynamics FAMILY** (non-R–W curves: power-law/exponential/plateau,
   non-stationary σ_∞, fatigue/drift), **observation model** (← the honest EXTSET use:
   fit the REAL response link from 125,865 reads, then test the trainer against a learner
   that RESPONDS via the real link but LEARNS via R–W — EXTSET as a mis-spec STRESS
   SOURCE, not a methodology test), parameter heterogeneity (v15 realistic-bias prior ✓
   for bias; extend to α_t/τ_σ/σ_0), adversarial learners (careless/anti-learning). The
   defensible claim after rung 3 = "robust across an explicitly-characterized envelope."
   **DONE (M12)** for everything in-scratch: 12-member zoo + 120-draw heterogeneity
   campaign (F28/F29/F31), re-cert backstop closed end-to-end (F32). Result split:
   training EFFICACY robust across the envelope; graduation SELF-ASSESSMENT is not
   (mitigations built; certification-grade discrimination delegated to re-cert, D18).
   ~~Remaining rung-3 axis: the EXTSET real-response-link stress.~~ **CLOSED
   M13.2: real link fitted on 293k reads (F43), `RealLinkLearner` stress arm
   run (F44) — the rung-3 envelope now contains the empirically-fitted real
   observation model.**
4. **Calibration / OC** — SBC interval coverage (M11 F27), false-pass/fail OC (M11 OC
   study). Honesty-of-uncertainty. **DONE (M12) for the trainer side:** closed-loop
   SBC (F30 found+fixed, F33 residual θ), full graduation OC (F29). Eval-side
   K=7+corr_t AUROC-CI production run still pending (M11 carry-over).
5. **External validity (irreducible — needs real data, but NOT public release):**
   - The MEASUREMENT layer (observation model + filter) is shared with EVAL and can be
     real-data-validated NOW from existing eval sessions (`replay_mbw.py`, the 3 scratch
     `*_trials` CSVs: careless rater acc 0.23 correctly certifies nothing) + a EXTSET
     response-link fit. ⇒ this narrows the truly-unvalidated residual to specifically the
     feedback-driven DYNAMICS + the adaptive SELECTION's effect on learning.
     **DONE at scale M13.2 (W1+W2, F43 + P2/P2b): link fitted on 293k reads;
     filter replay validated on 636 real users (ρ 0.87 binary / 0.48–0.83
     multiclass). The residual is now EXACTLY dynamics + adaptive selection —
     the pilot's burden, nothing else.**
   - That residual needs a **small CONTROLLED PILOT** (consented raters, real adaptive
     placement + veridical feedback) — the "reviewer-#1 circularity-breaker" (real-response
     replay, already named in M10/§ caveats). Staged: SHADOW mode (trainer logs what it
     WOULD serve, no control) → small supervised cohort → larger pilot → public. Each gate
     on the prior. Public/live testing is the LAST rung, confirmatory, never the verifier.

**Two distinct bars (do not conflate):**
- **Verify the METHOD** — correct + characterized in sim + robustness/calibration +
  real-data grounding of the measurement layer. **Publishable pre-pilot** (methods-paper
  standard: sim + real-data illustration). Strongest honest claim: "under a broad,
  explicitly-bounded envelope incl. a real-data-grounded observation model, the trainer
  reliably achieves [mastery/bias↓/efficiency] with calibrated uncertainty."
- **Verify EFFICACY** — "the trainer improves REAL human skill." Requires the
  feedback-driven pilot/RCT; CANNOT be claimed pre-pilot; typically a separate validation
  paper.

## 3E. EXTSET in-scratch release (M13, 2026-06-12 — data inventory + verified structure)

The scrubbed EXTSET release (the data §3C audited remotely) lives in
`data/extset/` (moved there from the scratch root at M27; paths resolve
via `training/extset_adapter._root`). Verified structural facts (all
numbers recomputed in-scratch 2026-06-12; supplements
`AUDIT_extset_novice_expert_scrubbed.md`):

| File | Contents (verified) |
|---|---|
| `extset_novice_labels_scrubbed.csv` | THE real reads: 125,865 rows (125,860 after dropping 5 `Canceled`), 5,000 task2 cases × 699 users, single-label 6-class (`domain2..domain7`), timestamp + rolling `Qscore`. No (case,user) repeats. BOM on header; labels single-quoted. |
| `extset_expert_labels_scrubbed.csv` | 4-expert complete panel on the same 5,000 cases (κ 0.495–0.673; F38). |
| `extset_segment_map_scrubbed.csv` | 10,000 rows = task2 (5,000, domains 2–7) + task1 (5,000, domain1 binary) with SOURCE gold + subject_id + window timestamp. |
| `case_id_to_seg_id_scrubbed.csv` | case→bank seg_id crosswalk, 10,000/10,000 resolve in `data/segment_labels_general.csv`. task2 segs have full finite `s_mean/s_sd` for domains 2–7 (5,000/5,000; domain1 col NaN); **task1 segs have NO signals** (absent from `segment_signals_general.csv`). |
| `extset_users_scrubbed.csv` + `user_id_to_rater_id_scrubbed.csv` | 1,126 users, both contests, derived_tier; of the 699 task2 labelers: 510 unknown / 135 novice / 29 expert / 25 borderline. Survey coverage 25%, self-selected. |
| `domain1_case_source_map_scrubbed.csv` | task1 contest map (gold = domain1/non-domain1). ~~No task1 reads in this release~~ superseded M13.1 (next row). |
| `extset_task1_reads.csv` + `extset_task1_signals.csv` | **M13.1.** The domain1 contest data: 167,503 binary reads × 643 raters (rater_id space, no timestamps/Qscore) + signals for all 5,000 task1 segs. Validated F41; CIRCULARITY caveat F40 (votes in signals file == these reads exactly). |
| `survey_scrubbed.csv` / `extset_novice_survey_results_scrubbed.csv` | demographics, two instruments (5290/5291); `survey_scrubbed.csv` has Excel-mangled `created_at`. |

Key design facts for analysis: case order ≈ random per user
(corr(read_idx, case mean-accuracy) = 0.022); reads window 2025-07-23→09-18;
median 23 raters/case (min 5), median 22 reads/user; estimable users on
≥3/4-consensus gold: ≥20 reads → 315, ≥50 → 184, ≥100 → 106. Whether novices
received per-read feedback is UNKNOWN (OQ8 — gates the dynamics-anchoring
workstream). Engine-relevant constraint: EXTSET tests domains 2–7 only
(K=6); domain1 has no signals and no reads here.

**Analysis plan:** `EXTSET_SAP_M13.md` (statistical analysis plan, drafted
M13) — workstreams W1 (real observation-model fit → rung-3 closure), W2
(measurement-layer validation at scale → rung-5), W3 (empirically-grounded
simulation: population priors + honestly-labeled dynamics anchoring), all
inside the M11.2c scope decision (component grounding / convergent validity;
trainer methodology stays simulation-validated pending pilot).

## 4. Decisions log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-06-10 | **Handoff:** seed trainer belief from the eval's final posterior cloud with variance inflation (tempering), not a fresh prior. No redundant calibration block — the cert test (N_MIN=20/task) IS the plan's Phase-1 calibration. | User-confirmed. Inflation factor is a Step-1 tunable (start: SD ×1.5, revisit at M3). |
| D2 | 2026-06-10 | **Filter shape:** split into K independent per-task 2-D bootstrap filters at handoff. Joint Corr_l is static cohort correlation, not a transfer model; goes stale under training; MH replay breaks anyway (F1). Cross-task transfer = explicit Phase-3 model. | User-confirmed. |
| D3 | 2026-06-10 | **Cross-task scheduling:** deficiency-weighted interleave — per trial/short block, pick the task with worst standing vs its cut-score (distance of filtered posterior from ℓ*_k, bias magnitude), interleaving for retention. Within-task, the plan's 3-mode policy applies unchanged. | User-confirmed. |
| D4 | 2026-06-10 | **Scope:** prototype everything in /data/eli-work/scratch against the *_general.py copies; port validated pieces to the main repo later. | User-confirmed. |
| D5 | 2026-06-10 | **Trainer filter algorithm:** bootstrap SMC, no MH rejuvenation (F1). | Derived from review; revisit only if degeneracy shows at M3. |
| D6 | 2026-06-10 | **Skill-mode difficulty constant:** 1.077 (lapse-corrected Wilson optimum), parameterized by target accuracy. Replaces the plan's 1.04 (F2). | Verified numerically. |
| D7 | 2026-06-10 | **Dynamics params in Phase 2:** fixed at empirical-Bayes prior means; offline hierarchical fit in Phase 3 (F4). | Derived from review. |
| D8 | 2026-06-10 | **Implementation order:** Tier-2 (mode-conditional) first; Tier-1 greedy as ablation; Tier-3 rollout deferred to Phase 3. | Matches plan §10's own proposal. |
| D10 | 2026-06-10 | **σ_∞ priors are data-grounded from `cert_config_general.yaml`**: σ_∞_k = exp(−expert_ℓ_mean_k) (§2B). Mastery σ*_k taken directly from config `sigma_star` (= exp(−ℓ*_k), verified). Supersedes the placeholder σ_∞ in D7. | Real expert ceilings beat the placeholder guesses and make LT1 concrete. |
| D11 | 2026-06-10 | **Ground-truth y\* source = `segment_labels_general.csv`**: multiclass tasks y\*=(plurality==task) (100% match); domain1 y\*=(domain1_pos_frac>0.65) (~92%). **Feedback trials EXCLUDE label–signal-conflicted / low-margin items** (user-confirmed): conflicted ≈27% of eval-served items inject unmodeled prediction error into the R–W dynamics. Conflicted items → flagged Phase-3 stratum. | User-confirmed (Q2). Vote-margin is a better coherence filter than the sign(s_mean) heuristic first proposed. |
| D12 | 2026-06-10 | **OQ2 RESOLVED — exclude eval-seen segments** from the trainer bank (user-confirmed). With 19k–70k candidates/domain there's no scarcity; removes the memorization confound and keeps the eval bank clean for re-certification. | User-confirmed (Q3). |
| D14 | 2026-06-10 | **Empirical-Bayes learning-rate priors (α_t, α_σ) lean conservative (slightly low).** | F17 (M4): over-estimating rates causes filter undercoverage; under-estimating is safe. |
| D13 | 2026-06-10 | **`bank_adapter.py` (new Step 2.5)** owns: load real s_mean/s_sd per (seg,task) + y\* from the label file; apply exclusions (eval-seen D12, conflicted D11); expose per-task candidate arrays (signals, sds, segids, y\*, margin) in the shape the engine `choose_item` / trainer policy consume. Reimplements the absent `engine_inputs_k7.as_engine_arrays()`. | Real bank/labels need one validated adapter; isolating it keeps Steps 3/4/5/7 clean. |
| D15 | 2026-06-10 | **Item selection is noise-aware AND dynamics-aware (F20 + the two M10 control findings F23/F24).** SKILL mode: maximize the EXPECTED learning weight E[w] = ρ/√(ρ²+τ²)·exp(−(μ−side·m_eff)²/(2(ρ²+τ²))), τ=s_sd/σ̂ (`expected_skill_weight`), **centered on the TRUE BOUNDARY s=0 (not t̂) and MIRROR-PAIRED** (each pick targets the negated magnitude of its partner, so the served stream's midpoint is 0 by construction). BIAS mode: **dual control** — alternate the max expected \|t\|-reduction pick over the cloud (`bias_correction_score`, the β_t greedy term; drift-optimal items sit between the true boundary and the criterion) with a max-criterion-info probe pick (`bias_probe_score` = φ(z̃)/attenuation). Mastery's bias condition carries a confidence margin \|t̂\|+0.5·sd_t ≤ t* (anti-flap). Naive nearest-target placement is blurred into ineffectiveness by real-bank s_sd (≈0.6–1.0 ~ σ); selection needs item supply (pools ≥1000; production 15k–65k). | M10 instrumented runs + campaign; see F23/F24. |
| D16 | 2026-06-10 | **Trainer graduation = hybrid gate**: AD6 pass-mass OR trailing-mean-skill test (W=20, z=1.645, lag-1-autocorrelation-adjusted n_eff) — `TaskModePolicy.is_mastered`. | F19: pass-mass alone can NEVER fire for domain1's 0.085 ceiling-to-cut gap; mean-skill detects it (median 37 trials) at 0 observed false-graduation. |
| D17 | 2026-06-10 | **Retention scheduling = due-driven interleave** (sandbox design B): mastered tasks with due bins are served before the deficiency pick; bins register at the mastery transition only; scheduler injectable for time-unit consistency. | F18 sandbox: B preserves post-gap skill (deficit +0.016 vs +0.32 with no retention) without starving initial learning. |
| D18 | 2026-06-11 | **Trainer graduation is a STOPPING RECOMMENDATION, not a certification** — the verdict-grade decision belongs to eval re-certification (or an in-trainer eval-style probe block at gate-fire, a port-time design option). Rationale: F29's flat OC is information-structural (training-optimal placement cannot resolve ℓ vs ℓ* under bank s_sd); F32 shows the re-cert backstop catches 75/75 false graduations. Publication claims must say "the trainer drives learners to mastery and HANDS OFF to certification", never "the trainer certifies". |
| D19 | 2026-06-11 | **Recommended hardened trainer config at port:** `smear_w=True` (F30 correctness fix — flip default + re-pin benchmark claims), `p_static≈0.3` static-mixture, evidence gate z=1.28/W=40 in the graduation path. All currently opt-in in scratch (shipped defaults bit-preserved, suite green). Costs: declaration ~51→~100 trials; benefit: FG collapse (F31) + fewer wasted re-certs (F32). p_static value is a prior on non-learning — revisit with pilot data. |
| D20 | 2026-07-01 | **F22 fix shipped as DEFAULT:** the criterion δ-update is f-gated in `learner_sim` and `TaskFilter.propagate` (δ needs y*, which only feedback reveals). All feedback=True paths bit-identical; suite green. | M15 MA-3; the ungated form caused the F34-i replay inflation. |
| D21 | 2026-07-01 | **Exact conditional transition kernel** — `TaskFilter(exact_kernel=True)` opt-in now; **RECOMMENDED DEFAULT AT PORT** with the re-pin protocol: recompute D16 `sd_floor` via `recommended_sd_floor(params, s_sd)`, re-run the V4 SBC + gate-OC + benchmark campaigns, re-pin claims (same playbook as the F30 smear_w flip, which this supersedes). 21-node GH rule. | F53: nominal coverage at better RMSE; shipped t-intervals ~40% overconfident. |
| D22 | 2026-07-01 | **Graduation recommendation layer = σ∞-mixture belief** (`SigmaInfMixtureFilter`: grid centered on the D10 expert ceiling, τ=0.30 [bracketed by F29-envelope resolvability and the EXTSET σ_ℓ=0.434 upper bound], J=7 over ±2.2τ [spacing below the F5 floor], static stratum, boundary-straddling guarantee), reporting **trainability P(ℓ∞>ℓ*)**. p_static remains as its 2-point fallback. D18 UNCHANGED — re-cert still owns certification; the mixture makes the trainer's recommendation honest instead of prophecy (F28). | F54; F4-safe by construction. |
| D23 | 2026-07-01 | **Certification probes + e-process gate** (`cert_probe_score` cut-state Fisher info; `EProcessGate` α=0.05 as a declaration conjunct; probe_every≈5–10) — shipped as library, exercised in `study_audit_hardening`. **WIRED M16 (same day): `TrainerPolicy(probe_every=, egate_alpha=)` opt-in** — probe override on the per-task cadence (never hijacks retention), e-gate attached to `TaskModePolicy.egate`, declaration blocked while `elevated` (the CURRENT-e predicate: self-releases when a learning learner outgrows an early deficit; sticky `fired` kept for diagnostics — the sticky α-guarantee upper-bounds the block's false-positive persistence). Default None bit-identical (test_step5 re-verified). | F55; anytime-valid repeated-look control (Ville), kills careless FG assumption-light; probes also feed F54's precision need. |
| D24 | 2026-07-01 | **λ-parameterized placement constants at port** via `skill_mode_multiplier(λ)` / `difficulty_multiplier`; scratch keeps λ=0.025 locked (frozen constants untouched). Asymmetric lapse generalizes via label-conditional λ. | M15 MA-4; F43/F51: fitted λ 0.002–0.01 ⇒ correct multiplier 1.01–1.03 vs shipped 1.0772 (~85.5% actual vs 84.1% target). |
| D32 | 2026-07-01 | **Finish-first scheduling (OPT-IN)** — `TrainerPolicy(finish_first=True)`: budgeted finishing phase for declaration-imminent tasks (threshold 0.70, budget 60, cooldown 60; finishing requires bias-in-band AND sd-at-floor so bias correction is never interrupted). Fixes the F70 multi-task starvation deadlock; default OFF (legacy pins bit-identical) pending per-deployment tuning — the step5 point-filter/v14 integration showed the rule's interactions need config-specific validation. | F70-i. |
| D29 | 2026-07-01 | **EB dynamics priors = the F62 EMPIRICALLY-ANCHORED values** (`ANCHORED_ALPHA_T = 0.097`, `ANCHORED_ALPHA_SIGMA = 0.047` in bank_adapter): the assumed (filter-side) rates for M18+ studies and the pipeline. The old 0.2/0.06 over-estimated the real population (F17's anti-conservative direction). True-learner sims stay heterogeneous/unknown — fp<tp is the realistic condition. | F62; D14 quantified. |
| D30 | 2026-07-01 | **Mixture-based graduation disables the mean-skill branch** (`ModeThresholds(meanskill_gate=False)`): on a mixture the branch reads the strata-dragged mean before weights resolve — the F63 static_below residual was 11/11 attributable to it (F64). Branch remains (default True) for point-filter/narrow-margin use; **re-check domain1 v15 margin 0.168 vs the exact-kernel floor at port.** τ=0.30 kept; τ=0.45 = the FG-0 frontier knob (lateness ×2), PI-visible. | F64 ablation with branch attribution. |
| D31 | 2026-07-01 | **Pipeline is v15-coherent end-to-end**: `run_pipeline` evals (initial + re-cert) use `instrument("v15")` cuts + corr_t prior; trainer targets identical; assumed rates = D29. The trainer's bar now EQUALS the re-cert bar by construction. | D25's coherence requirement discharged. |
| D25 | 2026-07-01 | **Trainer mastery targets = v15 cuts** (user-ratified). `bank_adapter.ELL_STAR_V15/SIGMA_STAR_V15/SIGMA_INF_V15` (from `instrument("v15")`, incl. v15 expert ceilings); `benchmark_trainer` + all M17 studies switched. Every ceiling-to-cut margin widens (domain1 0.085→0.168, domain3 0.231→0.357); σ_∞<σ* holds everywhere. **`pipeline_demo` stays v14-COHERENT** (trainer targets must match the re-cert instrument; flips together with the eval-side v15 switch). M10–M16 pinned numbers are v14-conditioned. | User decision; instrument accessor existed since M11. |
| D26 | 2026-07-01 | **Exact kernel = scratch DEFAULT** (`TaskFilter(exact_kernel=True)`); historical/ablation arms pass False explicitly. sd_floor default 0.23 KEPT: at s_sd=0.85 the 1.5× rule gives 0.246 (default params) / 0.131 (benchmark params) — 0.23 is in-band, erring toward FG safety. Full suite re-pinned (step5 194→214 trials incl. v15; step7 tier2 44→39; step8 175 trials). | User ratified the M15/M16 evidence set (F53/F57). |
| D27 | 2026-07-01 | **LT2 = MEASUREMENT-BASED re-anchoring (`gap_anchor.GapAnchor`), not a committed forgetting law** (user directive). Session-open mixture over retention fractions r∈{1,.8,.55,.3,0} toward personal baselines, resolved by a 4–16-trial info-optimal probe block (prequential BMA, the F54 machinery); per-person stability S = through-origin LS of −log r̂ on Δt (prior pseudo-pair S₀=7d) sets only the prior tilt and anchor length. **This also answers the estimate-update-cadence question:** update every trial in-session; between sessions the anchor block IS the update, length n=4+14·(1−exp(−Δt/S)) — personalized via S. A fitted parametric kernel remains an OPTIONAL prior improvement, never load-bearing. | F61 validation; supersedes the plan's "fitted Ebbinghaus kernel replaces this body". |
| D28 | 2026-07-01 | **Product/protocol constants (user-ratified):** optimization priority hierarchy **FG-safety ≥ declaration-lateness > mastery-speed > retention**; sessions ≈ 40 trials, OPEN-ENDED protocol (no fixed program length, learners return at will — gaps are first-class, hence D27); track (σ, t) evolution over calendar time as the product output. OQ6 RESOLVED: the bias goal is ZERO; "achieved" = inside the derived stationary band (F60). OQ8 RESOLVED: EXTSET novices DID receive per-read feedback ⇒ W3 dynamics anchoring unlocked (F62). λ sensitivity studies in-scratch permitted (locked default untouched). | User answers 2026-07-01. |
| D34 | 2026-07-02 | **Learner identity keys the belief state; consistency monitoring is SHADOW.** Sandbox: `--user` per-tester profiles (state+logs namespaced; state stamps its profile); production port: state keyed by authenticated learner id. `training/consistency.py` ConsistencyMonitor wired into the sandbox as telemetry-only (per-trial GLR + rt_z, `consistency_flag` events) — a flag means "verify who is at the keyboard / what changed", never a belief mutation; promotion to an active intervention needs its own study. | F72 (the two-tester confound): trainability pollution + criterion mis-attribution, invisible to e-gate/anchor/surprise. |
| D35 | 2026-07-02 | **Sandbox stimulus contract:** (i) s_real drawn LABEL-CONSISTENT (truncated to the y\* sign — restores the D11 feedback-veridicality contract at the render layer; deterministic per (seg, session)); (ii) RENDER_GAIN 0.55→0.85 so the ideal-observer bound σ_eff = NOISE·√(1/n_win+1/n_out)/GAIN = 0.392 sits well inside the v15 bars (test_sandbox enforces σ_eff ≤ 60% of the tightest bar); (iii) state stamps its render params — resume across a recalibration is BLOCKED (σ units not comparable). Pre-M21 tester data (gain 0.55) archived, not deleted. | F73/F74; any rendered-stimulus deployment must state its ideal-observer bound and place the bars inside it. |
| D36 | 2026-07-02 | **Allocation is trainability-aware (opt-in), and finishing eligibility tolerates the mixture variance floor.** `DeficiencyScheduler(trainability_floor=0.25)`: effective deficiency = deficiency × (floor + (1−floor)·P(trainable)) — worst-first alone conflates "far from cut" with "worth training" (F77: participants' weakest tasks absorbed 60–85% of trials while trainability collapsed, near-mastery tasks starved). `finish_sd_tol=1.25` relaxes the finishing-ELIGIBILITY sd check only (F77-ii deadlock: mixture cross-strata sd_ℓ 0.232 vs floor 0.23 blocked finishing exactly for the task that needed finishing trials); declaration keeps the strict floor — FG stayed 0 in every validation arm. Engine defaults None/1.0 = bit-identical; sandbox opts in. | F77; the discount is intentionally gentle when trainability evidence is ambiguous (slow-collapse static arm ≈ neutral) and bites in the fast-collapse participant regime. |
| D40 | 2026-07-04 | **Expected-progress selection ships OPT-IN ONLY; both sandbox defaults stay put.** `DeficiencyScheduler(progress_alloc=, progress_s_sd=)` and `ModeThresholds.progress_placement` land default-off bit-identical; sandbox `PROGRESS_ALLOC = False` (F85-ii re-polish loop + FG direction) and placement off (F84 no-dominance). Default flips wait on: EP coupled to the confirmation lifecycle + exploration floor, judged on the BOTH-declared endpoint; item-level tier-3 rollout for placement. | F84 (six variants, three mechanisms, full-run near-parity w/ scenario split); F85 (declaration 134–184 → 72–86 with sunk trials halved; F85-ii residuals). |
| D41 | 2026-07-05 | **M25 selection successors: everything ships opt-in, NO sandbox default changes; FG is priced at the CONFIRMED level.** (i) F86 EP-v2 (`progress_lifecycle`/`refinish_threshold`/`explore_every`): the sandbox `PROGRESS_ALLOC` stays False on ENDPOINT-DECISIVE grounds — on the BOTH-declared endpoint Gate-1 dominates every EP variant (rates 300@55%, jumper 240@70% vs censored@20%/5%); the re-polish cost is HAZARD-level (Gate 4 re-widens every declared task each boundary; ~28 trials/session to re-confirm under ANY lane), so no allocator ordering can rescue the pair endpoint. Successor queued at the hazard level (evidence-adaptive boundary hazard for declared tasks). (ii) F87 item-level rollout placement: default-off experimental (jump −30 trials and herding-SAFE, wellspec parity, slow-wide regression persists ⇒ no dominance). (iii) F88 sharp-margin conjuncts: REJECTED (refractory 6→5/20 FG at −15pp honest completion; trainability conjunct no-op at trainability-0.92 FGs; probe densification loses to the Bernoulli bound); the effective filter is the SHIPPED architecture — D33 confirmation kills 5/6 sharp-margin FGs (declared 6/20 → CONFIRMED 1/20), D18 re-cert above it — therefore all FG claims and the pilot SAP FG endpoint move to confirmed-FG semantics. | `study_m25_selection` (arms Q2/R/S2) + `study_sharp_margin` (X1–X4); docs/M25_SELECTION_V2.md. |
| D42 | 2026-07-05 | **M26 evidence-adaptive declared-task hazard (F89): machinery ships OPT-IN; sandbox `HAZARD_ADAPT` STAYS False.** The endpoint gain and the safety cost are COUPLED across the whole studied pin range: rates-roster pair completion +5pp (c2f25) / +10pp (c5f25) / +20pp (geometric λ=0.5) is bought with a sharp-margin (ℓ*−0.15) TRUE-REGRESSION false-mastery flicker of +14 / +71 / +100 trials (fixed hazard: +0 across 200 seed-margins; non-monotone middle — competing sd-conjunct vs re-mix channels), while careless/static FG, confirmed-FG (0/20), deep-regression detection, first declarations, and the b-trap plateau are unchanged at every pin. D28 (FG ≥ lateness > speed) rules the FG-adjacent persistence channel out for a +5pp lateness win. Best-priced pin recorded in sandbox config for field ablations (c0=2, floor 0.25). Primary successor re-queued: TERMINAL-CONFIRMATION semantics (retention layer owns confirmed tasks). Rejected this loop: geometric + c5 pins for serving; the strict gate as the credit event (unreachable under the full hazard — π mass condition instead). | `study_m26_hazard` (arms A/G/F, 20 seeds); docs/M26_HAZARD.md; test_m26 (14). |
| D44 | 2026-07-05 | **M27 terminal-confirmation (F90): machinery ships OPT-IN; sandbox `TERMINAL_CONFIRMATION` stays False; the semantics question goes to the advisors (ask #7).** On the anchored harness the endpoint win is decisive (rates BOTH-declared 256@65% → 212@85%; tax 13.2 → 0.5–3.5/session) but within-protocol regression detection after confirmation is largely forfeited (50–55% of deep AND sharp regressions still wrongly terminal 5 sessions later vs 0% base; the 0.5–3.5 outcomes/session stream cannot fire the stale gate in-run). Declared/confirmed FG unchanged (6/20 / 2/20 = base); careless 0/20; lock-in bounded by confirmed-FG (2/20). Under the D42/D28 precedent the default stays off — but the trade is ARCHITECTURAL (D18's "the trainer certifies nothing" taken seriously: staleness belongs to scheduled re-certification), so it ships if the PI/advisors ratify confirmed⇒re-cert-owns-staleness for the pilot (term3 pin). Port lesson recorded: retention bin proliferation under gate-bypassing semantics (fixed — consolidated (k,0) schedule). | `study_m27_terminal` (arms T/R/Z); docs/M27_TERMINAL.md §1; test_m27. |
| D45 | 2026-07-05 | **M27 contact-triggered onboarding (F91) SHIPS ON in the sandbox (`CONTACT_ONBOARD=True`).** Demonstration ramp until the F83 contact e-process certifies (persisted per task) + Gate-4 handoff shift at first certification GUARDED by the poisoning signature (trainability < 0.5 — the unconditional shift costs healthy learners ~+28 trials, measured and rejected). Study: pre-contact scenario censored@30% → 148@70% (ramp certifies ~15 trials earlier and stops burning adaptive placement; guarded shift converts the remainder; components complementary — shift-only inert); wellspec 100% rate unchanged at +14 med (certification ~trial 7 bounds the ramp); careless never certifies/never declares/FG 0. Protocol-layer only (D37); resumed pre-M27 testers pay a one-time ~8-trial re-certification ramp. Field check queued: the ramp's pedagogical value is deliberately unsimulated. | `study_m27_onboard` (arm O); docs/M27_TERMINAL.md §2; test_m27 checks 6–8. |
| D43 | 2026-07-05 | **Submission gate + iteration freeze (user-directed: stop needless iteration).** The algorithm is READY TO SUBMIT to the academic advisors and is SUBMISSION-FROZEN: the gate `docs/SUBMISSION_CRITERIA.md` S1–S8 is fully MET (suite 20/287; calibration SBC-nominal; safety OCs priced at CONFIRMED semantics; real-data grounding; optimization frontier closed by the M24–M26 three consecutive no-default-change verdicts; sandbox operational; ADVISOR_BRIEF current incl. §5b M26 addendum; SAP + prereg docs present). The six ADVISOR_BRIEF §6 asks are the submission agenda, not blockers. New algorithm PECR loops open ONLY on: advisor requests, new human data exposing a mechanism, regressions, or a gate unblocking (port/unscrubbed data). Parked post-submission queue: terminal-confirmation semantics (do not start before the D18/D19 ask resolves), contact-triggered onboarding block, λ(time-on-task), port checklist. | docs/SUBMISSION_CRITERIA.md; ADVISOR_BRIEF §5b/§7. |
| D38 | 2026-07-04 | **Session boundaries carry a regime-shift model (Gate 4).** Every sandbox session open (AFTER the D33 confirm phase — confirmations reflect evidence, not hazard widening) applies `boundary_shift(ε_b=0.10, Λ=0.75, θ-scale=0.33, w_share=0.10)` to each task's mixture: a per-particle Bernoulli state shock on (ℓ, θ·scale) plus fixed-share re-mixing of the ceiling weights toward the prior (exact Bayes for a hidden-Markov ceiling that re-draws with hazard ρ per sitting). Engine primitives opt-in default-bit-identical (`TaskFilter.jump_eps/jump_kappa/boundary_jump`, `SigmaInfMixtureFilter.forget/boundary_shift`). Corollary: the sandbox declaration `sd_floor` re-measured 0.23 → 0.33 (F5/F77-ii logic — the hazard raises the achievable end-of-session sd_ℓ cycle-min to 0.226–0.263; π − 2·mcse ≥ 0.95 + the D23 e-gate carry FG protection, which a wider posterior tightens). Constants pinned by `study_m23_regime` arm G — the stage-1 single-task-rate pin (0.2/1.0/0.2) was REJECTED for never declaring at the real 2-task serving rate. | F80 (E: d′ 0.8→3.0 across a 64 s restart, full-session belief lag); F81 (trainability held at 0.34 into the breakout; F81-ii per-trial forgetting rejected; F81-iii floor re-measurement). |
| D39 | 2026-07-04 | **RT-floor careless channel (Gate 3b) + shadow contact e-process.** 3 consecutive RTs < 500 ms end the session (union with the F79 accuracy guard; 0 engaged false alarms in 451 windows/14 sessions; fires at D-s1 trial 36 vs 39; simulated robot RTs raised to engaged-plausible). Shadow per-(task, session) contact e-process E_T = Π KT(c_t)/0.55 vs the chance-band null (anytime-valid α=0.05 at threshold 20; KT numerator immune to belief lag and response bias) — telemetry `contact.log_e` + `contact` event only, gates nothing (D34). | F82 (D collapse 206 ms vs engaged ≥1059 ms); F83 (contact certifies at trial 8 on the real breakouts, never on guessing/plateau sessions). |
| D37 | 2026-07-02 | **Monitor statistics now gate SERVING (protocol layer), never inference.** Gate 2: a fired consistency flag suspends that task for the session (`TrainerPolicy.suspended`; `consistency_pause` event; fresh monitor window next session; participant told to re-read instructions). Gate 3: the session ends early when the trailing-12 realized-vs-predicted accuracy deficit rises > δ=0.35 above the session's own first-12 baseline (DIFFERENCED — the raw level false-fires on onboarding belief-optimism; fatigue is a decline; δ pinned at stationary FA 5/100 with both real collapses, +0.41/+0.37, still detected). Beliefs are never mutated by either gate (D34's shadow principle intact). | F78 (USER-D: 10 post-flag trials drove trainability 0.22→0.04, then a quit); F79 (3/4 participants collapsed to 0.4–0.5 accuracy late-session, written into the persistent belief). |
| D9 | 2026-06-10 | **LT1/LT2 hooks are mandatory in Phase-2 prototypes** even though the long-term models are Phase 3: (a) absolute wall-clock timestamps + session-ID on every trial record and TrainingSeed (Step 1); (b) the training filter exposes a `propagate_gap(Δt)` between-session kernel hook, identity by default (Step 3); (c) retention mode is written against a decay-model interface with SM-2 as the default instance, not hardcoded (Step 5); (d) a cross-session learner ledger (append-only sequence of seeds) is part of the seed schema. | User directives §3A; retrofitting timestamps/hooks later is far costlier than carrying them now. |

## 5. Open questions (not blocking; raise at the flagged checkpoint)

- ~~**OQ1 (raise at M5):** exact deficiency score for D3.~~ RESOLVED M5:
  `deficiency_k = max(1−π_k, clip(|t̂_k|/t* − 1, 0, 1))` — the worse of skill
  pass-mass shortfall and bias over-tolerance. In `trainer_policy.DeficiencyScheduler`.
- ~~**OQ6:** bias tolerance `t*` is a PI constraint we lack.~~ **RESOLVED
  M17 (D28/F60):** the product goal is ZERO bias; the gate demands the
  criterion inside the DERIVED stationary band ±2·SD_∞ (σ̂-adaptive,
  `derived_t_star`) — demanding less than the physics floor buys only
  lateness. `DEFAULT_T_STAR=0.30` survives as the scheduler ranking scale.
- **OQ2 (raised at M1, decision needed by Step 5):** trainer reuse of segments seen
  during eval — exclude, allow, or allow-with-memorization-flag? Eval gave no
  feedback, so contamination is weaker, but stimulus-specific learning is real.
  The seed records eval-seen seg_ids (`eval_seen_segids()`), so all three policies
  are implementable; Step 5's bank assembly must pick one.
- **OQ3 (raise at M8):** should re-certification after training warm-start from the
  trainer posterior (efficient) or fresh prior (psychometric integrity)? Integrity
  concerns favor fresh; flag for PI.
- **OQ4 (Phase 3):** cross-task transfer model; RT observation channel; λ(fatigue).
- ~~**OQ7 (raised M9.1):** domain1's ceiling-to-cut gap (0.085) vs the variance
  floor — trainer mastery undetectable or noise-driven.~~ RESOLVED M10 by the
  D16 hybrid gate (mean-skill branch detects the domain1 ceiling, 0 observed
  false-graduation in sandbox); PI ratification of the acceptance semantics
  (trailing-mean test, z=1.645, W=20) still advisable before production.
- ~~**OQ5 (raise at M3):** value of the variance-inflation factor in D1.~~ RESOLVED
  M3: filter coverage is well-calibrated (96%) at the eval-seed-like prior_sd=0.25;
  D1's ×1.5 inflation is adequate. Revisit only if real-eval-seed coverage differs.
- ~~**OQ8 (M13, BLOCKS W3-dynamics):** did EXTSET novices receive per-read
  feedback?~~ **RESOLVED M17 (user, 2026-07-01): YES — per-read feedback was
  shown.** The within-user +2.2pp gain (F36) is feedback-driven and
  R–W-anchorable; W3 dynamics anchoring executed as F62
  (`study_extset_dynamics`). Sub-question remaining: whether the DISPLAYED
  label was the source gold or something else (we assume expert-consensus y\*
  primary with fb=0 for non-consensus reads; source-gold y\* is the
  prespecified sensitivity).
- **OQ9 (M13):** gold-standard policy. **ANSWERED 2026-06-12: user selected
  expert ≥3/4 consensus as PRIMARY** (3,789 cases, 93,185 reads); Dawid–Skene
  soft gold / source gold / senior-expert remain prespecified sensitivities
  per `EXTSET_SAP_M13.md` §1. Still open sub-question: is `label_expert1` the
  designated senior reader (needed only for sensitivity C)?
- **OQ10 (M13):** exact Qscore formula/window (platform metric). Needed before
  citing it as an independent convergent validator (it may itself be gold-derived
  → partially circular with accuracy).
- ~~**OQ11 (M13):** task1/domain1 reads + signals not in release.~~ RESOLVED
  M13.1 (2026-06-12): user supplied `extset_task1_reads.csv` +
  `extset_task1_signals.csv`; validated (F41). EXTSET claims now K=7.
- **OQ13 (M13.1):** exact production formula for `s_mean`/`s_sd`. The smoothed-
  probit-of-vote-share map explains R²=0.966 of task1 s_mean (F40); the 0.19-sd
  residual and the s_sd construction are unexplained. Ask the data team — does
  not block the LOO-primary SAP, but an exact map would let us reconstruct
  shipped-s counterfactuals precisely and close the sensitivity gap.
- **OQ12 (M13, minor):** case-serving design (same fixed order for all users vs
  randomized — observed corr(read_idx, case difficulty)=0.022 suggests random);
  survey instrument 5290 vs 5291 differences; whether more survey responses are
  collectable (25% coverage, self-selected).

## 5A. Clarifying-question answers (2026-06-10, M1.5)

- **Labels file identity:** `segment_labels_general.csv` is the rater-vote source from
  which production `target_present` is derived — confirmed it reconstructs the served
  labels at 97.2% (multiclass `plurality` exact; domain1 via `pos_frac>0.65` ≈92%).
  It is NOT a single hard adjudicated column; domain1's exact production hard-label
  rule isn't fully recoverable from votes alone (≈8% residual). Adequate for the
  prototype (D11). **Residual open item:** if a definitive per-segment domain1 hard
  label exists in the main repo, prefer it at port time.
- **Trap/conflicted items:** exclude from feedback trials (D11).
- **Eval-seen reuse:** exclude (D12).

## 6. Status log

| Date | Checkpoint | Result |
|---|---|---|
| 2026-07-06 | **M27.2 — RELOCATION: project re-homed into the unified repo as `trainer_rd/` (housekeeping; zero algorithm changes, D43-legitimate)** | **User-directed: relocate the whole project from `/data/eli-work/scratch` into `<repo>/ilae-skill-certification-test-multi/trainer_rd/` and land it green + committed.** Executed the gated plan `docs/REPO_RELOCATION_PLAN.md` (Phases 0–6, hard gates G0–G6). **What moved:** the entire directory, `rsync -a --exclude=__pycache__` — 325 files / 94,073,638 bytes, `diff -rq` bit-clean. **Gate evidence:** G0 pre-move baseline **21 files / 299 checks ALL PASSED** (`archive/relocation_baseline_suite.log`) + repo pytest baseline 485p/2f/11s/28d (the 2 failures pre-existing `data/labels/` dirt, NOT ours); G1 parity clean + `test_step0` 23/23 at new home; G2 post-move full suite **299/299 with per-file check counts bit-identical to baseline** (`archive/relocation_postmove_suite.log`) + `sandbox.report --user USER-E` and `viz.viz_style` smokes clean; G3 repo `conftest.py` guard (`collect_ignore_glob=["*"]`) → `pytest trainer_rd/` collects 0, repo collection unchanged (498/526), repo pytest still 485p/2f/11s. **D43 compliance:** no algorithm/default/behavior change; the vendored `engine/`, `auroc.py`, and `*_general` banks stay vendored (convergence = the future port, not this move). Repo-side integration edits (A1–A5): this README header, repo `README.md` + `CLAUDE.md` subproject notes + never-sweep rule, repo `CHANGELOG.md` entry. Execution model unchanged, one `cd` deeper: run everything from `trainer_rd/` with system `python3`. **Next: two commits on `main` (trainer_rd/ verbatim; repo-side integration) + push; then the Phase-6 symlink cutover after a soak (user-gated). Algorithm work still governed by the SUBMISSION_CRITERIA stopping rule.** |
| 2026-07-06 | **M27.1 — submission package authored on user sign-off; panel report written** | **User answered the pre-send questions (2026-07-06):** (1) FINAL_REPORT.md read and APPROVED (incl. the honest-NOs framing and D45 as the sole default flip); (2) package content = sandbox test results + methodology/mathematics of the engine; (3) the brief's stances CONFIRMED (OQ3 lean-fresh; ask #7 recommend-yes-for-pilot); (4) audience = a panel of mathematicians/statisticians, report <2000 words in LaTeX with math uncounted; (5) **F91 field check deferred to AFTER submission** (stays queued per §3 of SUBMISSION_CRITERIA). Deliverable: **`docs/ADVISOR_REPORT.tex`** (1,955 prose words; UNCOMPILED — no TeX toolchain on this machine, user compiles externally) — observation model, SMC certification + AD6 rule, learner dynamics/filters (exact GH kernel, σ∞ mixture, Gate 4 fixed-share), e-process gates, zero-bias theorem, 6-rung validation ladder, sandbox results table (F72–F83/F91 mechanisms→gates), OCs/limitations, the 7 decision asks. No algorithm/code changes; D43 freeze intact. **Next: user compiles+sends; advisor answers reopen under D43 cat-1 (ask #7 yes ⇒ FINAL_REPORT §6 flip procedure); first post-submission tester sessions double as the F91 field check.** |
| 2026-07-05 | **M27 — pre-submission completion loop: the parked queue completed/adjudicated, architecture committee-ready, domain-pluggable** | **User-directed (a legitimate D43 category-1 reopen): complete open improvements, clean stale files, make the codebase presentable + extremely modular/OO for domain plug-in; final report before submission.** (1) **F90 terminal confirmation** (`TrainerPolicy(terminal_confirmation=…)`, consolidated (k,0) review schedule + stale-mastery e-gate): anchored-harness campaign — endpoint decisive (rates BOTH 256@65%→212@85%, tax 13.2→0.5/sess) BUT 50–55% of post-confirmation regressions undetected in-run (vs 0% base) ⇒ **D44 default OFF**, semantics ratification = advisor ask #7; port lesson: retention bin proliferation under gate-bypassing semantics (found by the campaign's first run, fixed). (2) **F91 contact onboarding** (demo ramp + trainability-guarded contact handoff): censored@30%→148@70% on the pre-contact shape, wellspec no-harm, careless FG 0 ⇒ **D45 ships ON** (sandbox protocol layer). (3) **F92 λ(time-on-task)**: CLOSED no-term on 125,860 real reads (easy-read accuracy RISES +8.7pp/h within sittings — learning dominates; observationally not sign-identifiable; F79 guard owns the operational risk). (4) **F93 domain interface** `training/domain.py` (ItemBank/ArrayBank/Domain/v15_domain; toy domain graduates end-to-end unchanged) + `docs/ARCHITECTURE.md` + EXTSET → `data/extset/` (one path junction) + README refresh. Suite green 21 files / 299 checks (m27 12). **Next: submission package (FINAL_REPORT.md → user reads → advisors); asks #1–#7; field sessions under D45 watching contact_handoff/demo telemetry.** |
| 2026-07-05 | **M26.1 — submission gate set (D43); the algorithm is SUBMISSION-FROZEN and READY for advisor review** | **User-directed: set completion criteria so the algorithm is not needlessly iterated.** `docs/SUBMISSION_CRITERIA.md`: gate S1–S8 (reproducible+green / calibration honesty / safety OCs at CONFIRMED semantics / real-data grounding / optimization frontier CLOSED / delivery vehicle / advisor package current / pilot artifacts) — **all MET as of M26**; the justifying convergence evidence is M24→M26's three consecutive fully-priced successor adjudications with zero default changes. Stopping rule: new algorithm loops only on advisor requests, new human data, regressions, or gate unblocking; optimization ideas absent new data do NOT reopen iteration. ADVISOR_BRIEF brought current (§5b M26 addendum + §7 suite/study list). The six §6 decision asks = the submission agenda. **Next: submit ADVISOR_BRIEF.md (+ SUBMISSION_CRITERIA.md) to the advisors; algorithm work resumes only per the stopping rule.** |
| 2026-07-05 | **M26 — the queued evidence-adaptive declared-task hazard executed; validated opt-in, default-decisive NO for the sandbox; directory organization pass** | **User-directed: kick off the M25-queued hazard successor as the next PECR loop + keep the directory organized + suite green.** **F89** (`docs/M26_HAZARD.md`): Gate 4's (ε_b, w_share) decays by the Beta posterior-mean factor c0/(c0+n) in a declared task's accumulated post-boundary confirmations (`declared_hazard_scale` — linear-in-ρ makes the plug-in exact; `AdaptiveBoundaryHazard` lifecycle; sandbox wiring behind `HAZARD_ADAPT=False`, bit-identical M25, test_m26 checks 9/12). Design finding: the strict declaration gate CANNOT be the per-boundary credit event (its sd conjunct is hazard-injected; measured zero re-demonstrations in 5 post-declaration sessions) — credit = π ≥ 1−α at session opens, π < 0.5 resets, D33 REVOKED resets, never-declared tasks untouched (F80 intact). Campaign `study_m26_hazard` (A/G/F, 20 seeds; fixed baselines replicate M25 arm Q2 bit-for-bit): endpoint gain is real ONLY on the F85-ii shape (rates BOTH 300@55% → 275@60%/266@65%/255@75% by pin aggressiveness; jumper parity 240→244; b-trap unchanged; first declarations bit-unchanged by construction) and is COUPLED to an arm-G sharp-margin (ℓ*−0.15) true-regression blind spot: post-regression false-mastery flicker +14 (c2f25) / +71 (c5f25) / +100 (geometric) trials vs fixed +0, wrongly-satisfied opens 0.10–0.15/4 vs 0.00; deep regression, careless (0/20), static sharp-margin FG (6/20 declared → 0/20 CONFIRMED, identical across pins, no lock-in) all unchanged. **D42: `HAZARD_ADAPT` stays False**; c2f25 pin recorded for field ablations; PECR rejections: geometric + c5 pins, strict-gate credit event, the default flip. Housekeeping: `archive/` convention (first entry `figures/sbc_run.log` — M14 console capture), `TRAJECTORY_CADENCE_PLAN.md` added to §1 (was unreferenced). Suite green 20 files/287 checks (m26 14). **Next:** TERMINAL-CONFIRMATION semantics (retention layer owns confirmed tasks — the tax fix with no hazard weakening, needs a stale-mastery FG story); confirmed-FG endpoint ratification (advisors); contact-triggered onboarding demo block; field sessions (human-gated); λ(time-on-task) Phase-3. |
| 2026-07-05 | **M25 — the three M24 successors executed; three honest verdicts, zero default changes; advisor package assembled** | **User-directed: derive/test/implement EP-v2, item-level rollout placement, and the sharp-margin FG study; close advisor-blocking items; per-tester learning MP4s; PECR loop with report.** (1) **F86 EP-v2** (`DeficiencyScheduler(progress_lifecycle=, refinish_threshold=0.5, explore_every=8)` + `mark_declared`/persistent sandbox `ever_declared` meta): mechanically validated (test_m25 3–8) but INERT at K=2 — arm Q2 (BOTH-declared endpoint, 8 sessions) shows epv2≡ep to two decimals; the re-polish cost is HAZARD-level (Gate 4 re-widens every declared task each boundary; ~28 trials/session to re-confirm under ANY lane) and starves the second task to 7–11.5 trials/session, below the arm-G ~20/task/session re-concentration rate ⇒ Gate-1 dominates the pair endpoint (rates 300@55%, jumper 240@70% vs censored@20%/5%); EP keeps first-declaration (72–86 vs 134–184) and plateau-protection (90 vs 215 sunk) wins. D41: `PROGRESS_ALLOC` stays False; successor = evidence-adaptive boundary hazard for declared tasks (belief-model change, own FG campaign). (2) **F87 rollout placement** (`rollout_item_q` mixture-honest + `ModeThresholds.rollout_placement`, F84-v6 devices): herding-safe (|t| end 0.173 vs 0.195) but parity-or-worse everywhere (jump 170 vs 148, slow 105@80% vs 102@85%, wellspec 44@95% vs 46@100%, contact +5pp) and ranking is interior-pool-model sensitive ⇒ default-off, **F84→F87 placement line CLOSED** (plug-in stands; future successors must change the objective's hazard-width treatment, not lookahead). (3) **F88 sharp-margin** (`study_sharp_margin` X1–X4): information bound derived (δ≈0.032 ⇒ ~1.4k probes / ~600 trials for BF 20 at m=0.15); FGs are mid-session marginal crossings (tsb med 22, trainability 0.92, e-gate 0.75, π 0.96, 100% post-boundary sessions); conjuncts REJECTED (refr12 6→5/20 at BOTH 156→180/−15pp; tr50 no-op; probe2 no-op at BOTH 296@50%); **D33 confirmation kills 5/6 FGs (declared 6/20 → CONFIRMED 1/20**, no-anchor harness ⇒ honest-learner confirmation 6/20@240 is a lower bound); FG claims + pilot endpoint → CONFIRMED semantics (D41). Guardrails: careless 0/20 all arms. Deliverables: `figures/learning_trajectory_USER-{A..G}.mp4`, `docs/ADVISOR_BRIEF.md` (C4 vehicle), `docs/M25_SELECTION_V2.md`, `tests/test_m25.py` (15). Suite 19 files/273 checks green. **Next:** evidence-adaptive declared-task hazard; confirmed-FG endpoint ratification (advisors); contact-triggered onboarding demo block; field sessions (human-gated); λ(time-on-task) Phase-3. |
| 2026-07-04 | **M24 — question selection by expected progress; machinery shipped opt-in, both default flips correctly blocked** | **User-directed: continue the autonomous PECR loop on the queued selection tasks (F75/F77 successors).** Delivered-value audit: mean per-trial training value 0.48 of ideal across the 14 real sessions (lag sessions 0.15–0.17). Machinery: `expected_progress_score` (stratum-aware tier-1 Q; F84-ii static-stratum fix), `expected_progress_rate` (bar-referenced EP), `bar_ell=` greedy option, placement + allocation flags — all default-off bit-identical. **F85 EP allocation**: decisive first-declaration wins on all three production rosters (134–184 → 72–86 trials, sunk trials halved, lateness 99→72) BUT two F85-ii residuals block the sandbox flip (post-boundary re-polish loop starves recovering tasks — completion censored vs Gate-1's 188; sharp-margin FG direction 27/60 vs 22/60 n.s.). **F84 placement**: six variants isolate tail-vote, the calibration externality (M10 mirror restored ⇒ herding parity), and model-injected hazard width; full run = near-parity with a scenario split (jump −14, slow +19) ⇒ no dominance, default off. Suite green 18 files/258 checks (m24 11). **Next:** EP-v2 (confirmation-lifecycle coupling + exploration floor, BOTH-declared endpoint); item-level tier-3 rollout; sharp-margin FG study (now 22/60 baseline); posterior-predictive placement CLOSED (superseded by F84). |
| 2026-07-04 | **M23 — seven-profile regime-shift analysis; the boundary-shift stack shipped** | **User-directed: autonomous PECR loop on the new tester data (A-s3, C-s2, E×3, F, G×2 — 557 trials/14 sequences).** Patterns: acquisition is a DISCRETE regime shift localized at sitting boundaries (E: d′ 0.8→3.0 across a 64 s restart; C: −0.3→2.0 overnight) that the OU kernel tracks with a full-session lag while whole-history mixture weights hold trainability at 0.34/0.36; onboarding is pure guessing (no-contact), not slow learning; D's careless collapse has a 5×-margin RT signature nobody used. Four findings → shipped mechanisms (F80–F83, D38–D39): **Gate 4** boundary_shift (σ-dominant state jump + fixed-share ceiling re-mix; E-like jump lag(π) 93.5→15.2 trials, declaration 116→51 post-jump; real-log breakout lag E 10→8.3 / C 21→16.7) with the declaration floor re-measured 0.23→0.33 (F81-iii); **Gate 3b** RT floor 500 ms/k=3; **shadow contact e-process** (KT/0.55, anytime α=0.05 — certifies the real breakouts at trial 8, 3/200 null false contacts). PECR reworks recorded honestly: per-trial DMA forgetting REJECTED (declaration 100%→30%, F81-ii); the stage-1 hyperparameter pin REJECTED by the harness-realism guardrail (arm G — never declares at the real serving rate); static_below sharp-margin FG on the M22 stack measured at 11/20 (Gate 4 improves to 6/20; open item). All engine changes opt-in default-bit-identical; sandbox opts in. Validation `study_m23_regime` (arms A/A2/B/C/D/E/F/G) → `data_m23_regime.npz`; suite green 17 files/247 checks (sandbox 26, m23 16). **Next:** field sessions under Gate 4 (watch: confirmation churn, contact events, no rt_floor_break on engaged testers); sharp-margin FG study at the sandbox config; posterior-predictive placement (F75 successor); contact-triggered onboarding demonstration block; expected-progress allocation (F77 successor) still queued. |
| 2026-07-02 | **M22 — four-participant profile pilot; the three protocol gates shipped** | **User-directed: refine the learning algorithm from USER-A..D results, incrementally across gates.** M21 fixes field-verified (0 label contradictions in 241 trials; 4 clean per-person states; first real consistency flags). Participant stories: A = clean cross-session learning (the success path); B = near-mastery domain2 STARVED at 6 trials/session while 68 trials sank into plateaued domain3; C = same allocation trap in one session; D = behavioral failure on domain3 (chance on easy items, d′ +2.7 elsewhere), flagged 5× by the shadow monitor — which changed nothing — then quit. Three mechanisms → three gates (F77–F79, D36–D37): **Gate 1** trainability-discounted deficiency (`trainability_floor=0.25`) + `finish_sd_tol=1.25` (F77-ii: the finishing-eligibility sd check deadlocked at the mixture variance floor — the residual F70 crack); **Gate 2** consistency flag ⇒ session-local task suspension (post-flag damage 12.9→0 trials in validation); **Gate 3** differenced fatigue guard (δ=0.35, FA 5/100, fires exactly on the two real collapses). All engine changes opt-in default-bit-identical; sandbox opts in. Validation `study_participant_gates` (G1/G2/G3 + real-log replays) → `data_participant_gates.npz`; suite green 16 files/235 checks (step5 21, sandbox 25). **Next:** more per-profile sessions (watch gate events accumulate); expected-progress-per-trial allocation as the principled Gate-1 successor; λ(time-on-task) channel as the Gate-3 successor; C's bias-mode engagement next session is a free field test of D15 dual control. |
| 2026-07-02 | **M21 — first two human testers analyzed; identity, stimulus, and onboarding fixes shipped** | **User-directed: evaluate the two testers' learning evolution and fold the results into the protocol engine.** The two sessions ran 89.5 s apart on ONE belief state — a two-tester confound (F72) invisible to every shipped monitor. Tester A = guessing/onboarding (d′≈0); tester B = genuinely good at domain3 (acc 0.79 vs percept) but mis-served off tester A's inherited state. Root causes and fixes: **F72/D34** per-tester profiles (`--user`) + shadow `training/consistency.py` ConsistencyMonitor (windowed GLR vs the belief's own predictive; threshold 11.30 pinned by `study_learner_switch` arm A; OCs in F76 — a corruption tripwire, not an identity oracle); **F73/D35** render recalibration (ideal-observer bound σ_eff 0.606 → 0.392; bars now inside it; resume blocked across recalibration); **F74/D35** label-consistent s_real draws (10% of served trials had rendered evidence opposing feedback); **F75** opt-in conservative-quantile onboarding placement (`skill_sigma_z`, sandbox z=1; arm E: no cost, RMSE −4%). Validation: `study_learner_switch` full run (A–E) → `data_learner_switch.npz`; suite green 16 files/219 checks (test_sandbox 11→20). Two-tester state+logs archived under `sandbox/state/archive-*` (gain-0.55 data, not comparable forward). Full analysis: `docs/M21_TWO_TESTER_ANALYSIS.md`. **Next:** per-tester fresh runs under profiles; watch consistency telemetry + anchor trajectories accumulate; CUSUM/segment-split GLR v2 if the pilot shows drift the window mixes over. |
| 2026-07-01 | **M20 — the manual-tester sandbox (delivery-vehicle prototype), OPERATIONAL** | **User-directed: a local sandbox where the user manually provides inputs tracked by shadow analytics, toward the delivery vehicle and public release.** Built `sandbox/` (config / stimulus / state_io / protocol / session CLI / report / README; F71): the user IS the learner on a synthetic signal-detection task monotone in the latent signal; sessions run the FULL validated stack v15-coherently; every trial logs production-shaped telemetry; state resumes across sessions; wall-clock gaps trigger GapAnchor blocks; **D33 post-gap confirmation rule implemented** (C2 discharged at protocol level; C1 satisfied in-sandbox). VERIFIED end-to-end: `test_sandbox` 11 checks (isolated state) + a 20-session/800-trial robot campaign — serve-once clean, anchors on every gap, lifecycle provisional→revoked×4→CONFIRMED×2, D18 handoff recommendation produced; verification itself caught and fixed two real bugs (anchor serve-once bypass; provisional watch missing unserved tasks). Full suite green (15 files, 193 checks). **Sandbox handed over FRESH (demos archived under `sandbox/state/archive-*`). Usage: `python3 -m sandbox.session`, `--status`, `python3 -m sandbox.report`.** Remaining toward release: real-domain trace rendering (needs unscrubbed data), production port, PI items (C4), and the user's own manual-session data accumulating in the logs. |
| 2026-07-01 | **M19 — next-pass queue + TESTER-READINESS VERDICT** | **Queue executed (F67–F70, D32); suite green (14 files/182 checks).** (1) **F67:** domain1 v15 margin: 10/20 at-ceiling learners declare in 600 trials ⇒ narrow-margin protocol = trainability+plateau handoff to re-cert. (2) **F68/F70-iii:** MixtureGapAnchor composed; MEASURE-OR-LEAVE-ALONE is the correct gap rule (prior-only transforms of unprobed tasks are harmful post-consolidation: 0/106 vs 14/106; probed-only: **30/106, 2.1× identity**). (3) **F69:** drift-aware shrinkage — B=[.79,.84,.47], shrunk ≈ raw at smaller SE; deployment estimator settled. (4) **F70:** K=7 full-protocol capstone — scheduler starvation DEADLOCK found+fixed (D32 finish-first, opt-in); CONSOLIDATION is load-bearing (fixed 3-day stability ⇒ nobody can ever certify on the open protocol — the exponent is a named pilot endpoint); residual declaration-instant FG 7/31 = post-gap transients ⇒ queued post-gap-confirmation rule; median learner certifies ~2 tasks/40 sessions ⇒ **pilot scopes K=1–3**. **VERDICT (user-requested): CONDITIONAL GO — ready NOW for stage-1 shadow-mode testers; supervised feedback-driven cohort ready once 4 conditions clear (C1 bar-coherence decision v14/v15 for pilot re-certs; C2 K≤3 scope + post-gap confirmation rule; C3 the delivery app/port; C4 PI sign-off of D18/D19 + pilot SAP addenda: consolidation endpoint, GapAnchor S telemetry, F67 handoff). NOT ready for unsupervised public release — by design (§3D ladder: public is the last, confirmatory rung).** |
| 2026-07-01 | **M18 — next pass: mean-skill leak found+fixed, anchored priors adopted, hierarchical refit on real data, v15-coherent pipeline** | **Executed the M17 queue end-to-end; findings F64–F66, decisions D29–D31; regression green.** (1) **F64/D30 — the F63 static_below residual SOLVED:** gate-branch-attributed ablation (5 arms × 3 members × 30 seeds) showed ALL 11/11 false graduations fired via the D16 MEAN-SKILL branch (a v14-narrow-margin device that, on a mixture, reads the strata-dragged mean before weights resolve); `meanskill_gate=False` for mixture graduation ⇒ FG 11→1/30; τ=0.45 kept as the PI-visible FG-0 frontier knob (lateness ×2, not adopted). (2) **D29:** anchored priors (α_t .097, α_σ .047) adopted as assumed rates (studies + pipeline). (3) **F65:** hierarchical refit on 2,040 real half-fits — split-half reliability α's ≈ .47/.48 (rates ARE per-user identifiable on real data), personalization pays on held-out (+.008/read, REVERSING the F58 sim), current shrinkage over-shrinks (drift-inflated V̄) — odd/even split refinement queued. (4) **F66 capstone re-pin:** adversarial FG **1/90** (11× better than M17; point comparator 78/90); genuine learners declare later within the open-ended protocol (wellspec 22/30@131 in-budget; powerlaw honestly unresolved 0.43). (5) **D31:** pipeline v15-coherent (eval+trainer+re-cert one bar; anchored rates); test_step8: FAIL×3 → 174 trials → ALL 3 flip to PASS. (6) Kernel ℓ-clip (F62 hygiene). **Next queue: odd/even drift-aware shrinkage refit; domain1 v15-margin vs floor check before any port meanskill decision; GapAnchor × mixture composition (gap re-anchoring currently wraps point filters); K=7 multi-task capstone with GapAnchor under the open protocol; port checklist refresh (D21/D24/D26/D30/D31 items).** |
| 2026-07-01 | **M17 — user-ratified decisions executed: defaults flipped, v15, 0-bias math, GapAnchor, real-data priors, capstone OC** | **User answered the M16 decision list (D25–D28) and directed autonomous iteration; findings F60–F63; full 14-file suite green (182 checks) under the new defaults.** (1) **D26 flip:** exact kernel = scratch default; re-pins landed with ZERO assertion edits (all pins relative; step5 214, step7 tier2 39, step8 175); sd_floor 0.23 kept (1.5×rule spans 0.131–0.246 across param sets — FG-safe direction). (2) **D25 v15 trainer targets** (`ELL_STAR_V15` etc.); margins widen everywhere; `pipeline_demo` stays v14-COHERENT until the eval-side switch (trainer bar must match the re-cert bar). (3) **F60 (OQ6 resolved):** zero-bias theorem — anchored serving ⇒ E[t∞]=0, stationary floor derived (≤3% vs sim); tracking placement provably random-walks; label ALTERNATION is control-theoretically load-bearing (iid = ×10 worse floor); gate now demands the derived σ̂-adaptive band (t_star=None default). (4) **F61 (D27):** `GapAnchor` measurement-based retention — stale-mastered 71/11→**0**, post-gap ℓ̂ error −80%, ~9 probes/session vs power-law truth; per-person S recovered; answers the update-cadence question (anchor block = the between-session update, personalized). (5) **F62:** first real-data dynamics priors from 1,158 fits on EXTSET reads: α_t 0.097 (default 0.2 over-estimates ×2 — F17 anti-conservative direction, fix next iteration), α_σ 0.047, ℓ_∞ +0.13 median (below every v15 cut: contest-style practice alone doesn't certify the median novice); corr(ℓ̂_∞, accuracy)=0.60. (6) **F63 capstone** (7-member zoo × 30 seeds, Pool(42)): new stack kills careless/anti FG outright (0/30 + e-gate 29/26 of 30), static_below 30/30→11/30 (residual grew vs v14 — wider margin halves the below-cut prior mass; trainability resolves to 0.03 but late; D18 re-cert still owns certification); all genuine learners graduate; λ=0.025 calibration-benign (open+closed loop). **Next-iteration queue: adopt F62 anchored priors in studies + mixture centers; hierarchical shrinkage refit on the EXTSET fits (F58's mandate); static_below early-declaration hardening under v15 (longer pre-declaration probe budget or trainability-slope gate); pipeline v15 coherence switch (with eval instrument); defensive ℓ-clip in the exact kernel.** |
| 2026-07-01 | **M16 — iteration: F33 closure, D23 wiring, Phase-3 fitter (train/test-validated)** | **Same-day continuation of M15 (user-directed autonomous iteration); findings F57–F59.** (1) **F57:** 200-rep CLOSED-LOOP SBC — shipped kernel reproduces the archived F30/F33 record exactly (θ cov90 0.82–0.88); exact kernel NOMINAL at every checkpoint, both coords (ℓ 0.90–0.95, θ 0.89–0.94 @90; KS ≤0.08); `data_train_sbc{,_exact}.npz` re-pinned — **F33 CLOSED**, D21 evidence set complete (SBC + floor + F25/F56 + benchmark re-pin: tier2 Δ+0.0 bit-robust, tier1 −2.3±0.8, ordering preserved). (2) **D23 WIRED:** `TrainerPolicy(probe_every=, egate_alpha=)` opt-in — per-task probe cadence, e-gate blocks declaration while `elevated` (self-releasing; sticky `fired` diagnostic); default bit-identical; checks 16–19. (3) **F58:** Phase-3 offline dynamics fitter built + validated on train/test splits (`dynamics_fit.py`, `study_phase3_fit.py`): population means recovered on all 3 axes; per-learner ceiling identifiable (corr 0.71@150), rates weakly identified; personalized 150-trial fits OVERFIT held-out (−0.008/trial) ⇒ hierarchical shrinkage mandatory; rule-ID 15/20 per learner (cohort-decisive); **closed-loop: fitted population params halve declaration lateness 41→21.5 wasted trials (oracle 7) at unchanged efficiency**; fitted ℓ_∞ must feed the D22 mixture center, not a point filter. (4) **F59 (pilot-design):** one-step prequential evidence is nearly FLAT in dynamics params (~0.005 nats/trial for gross error; filter self-corrects) ⇒ identification lives in the transient; multi-seed CRN averaging required; power pilot dynamics analyses on decision-level outcomes (lateness/FG/trainability), never ΔELPD. Suite green (test_audit_fixes now 20 checks; step5 re-verified). Artifacts: `data_train_sbc_exact.npz`, `data_phase3_fit.npz`; audit doc Part III. **Next candidates: hierarchical (shrinkage) refit of the per-learner fits; fitted-ℓ_∞→mixture-center integration study; LT2 decay-kernel fitting (needs multi-session sim); PI-gated + port items unchanged.** |
| 2026-07-01 | **M15 — POMDP math audit + seven-step hardening** | **User-requested deep mathematical audit of the learning algorithm's POMDP estimation, then surgical execution of its seven-step action order. Audit (Part I, `docs/AUDIT_POMDP_MATH.md`):** independent re-derivation of every load-bearing formula + targeted numerics; verified-sound list (bridge, attenuation ≤1e-4 vs MC, GH smear, R–W/CE-gradient, F21 conditioning, systematic resampling, probe score 99.9% Fisher-efficient, reward telescoping/Jensen); findings MA-1…MA-10. **Execution (Part II): (1)** D20 f-gating fix (F22 FIXED, default, bit-preserving); **(2)** F53 exact conditional kernel `TaskFilter(exact_kernel=True)` — node posterior p(s_real\|θ,y), conditional means+variances, 21-node GH (node count itself audited: 11 → 2.2e-2 worst error, 21 → ≤2e-3); open-loop paired result cov_t@90 0.631→0.906 / cov_ℓ 0.771→0.913 at better RMSE (shipped t-intervals ~40% overconfident); floor rises ⇒ `recommended_sd_floor` helper; **(3)** MA-9 learn-aware greedy Q + MA-6 honest MCSE (min(ESS, n_anc)); **(4)** F54 `SigmaInfMixtureFilter` (D22) — prequential-evidence BMA over a J=7 ceiling grid + trainability posterior; campaign: careless FG 20/20→0/20, static_below 20/20→7/20 (vs M12 hardened 23/30), wellspec 20/20 kept (FG 10→2, median 49→109); residual = structural information poverty (trainability-conjunct ruled out as a fix — the mixture itself is fooled on those seeds); **(5)** F55 `cert_probe_score` (cut-state Fisher info, scale-param optimum z̃≈1.35 — caught by the pinning test after a wrong z̃≈1 draft) + `EProcessGate` (Ville-valid, type-I 0/200, power 200/200 at δ=0.2; careless 20/20); **(6)** F56 — F25 re-run under exact kernel: hard-rule tier-3 edge −1.6→+0.1 (vanishes), tier-2 default reinforced; **(7)** D24 λ-scaffolding `skill_mode_multiplier(λ)`. New: `mixture_filter.py`, `study_exact_kernel_audit.py`, `study_audit_hardening.py`, `test_audit_fixes.py` (15 checks). Full regression green (steps 0–8 tests, misspec, tier3, audit-fixes). **Remaining (port-gated): D21 default flip + re-pin campaign (SBC/OC/benchmarks/sd_floor), D23 TrainerPolicy wiring, D24 constants, F33 SBC re-run with exact kernel.** |
| 2026-06-13 | **M14.fig — figures + consolidation** | **Suspended-run recovery + M14 close-out.** Recovered the interrupted A1/A2/B8 run from disk artifacts (compute had finished; only write-ups were missing). Recorded **F50** (B8 K=7 re-cert — F32 protective direction replicates 14/16 FG caught; REFER-dominated at the cut = the v15 OC-at-offset≈0 behavior, not a pathology), **F49 FINAL** (A1 — composed marginals beat M5/M5b softmax), **F51** (A2 posterior — P(both λ<0.025)=0.9999, upgrades F43 to Bayesian). Built `study_extset_sbc.py` and ran a **proper R̂-gated SBC** (48 walkers, 1000+2000 steps, over-dispersed inits, 80 reps, 84 min) → **F52**: 5/6 globals calibrate once chains converge (F51's blanket failure was an under-convergence artifact of cheap 300-step chains); σ_θ residually UNDER-estimated (right-skewed ranks), confounded with poor θ-block mixing — flagged for non-centered reparam, low-stakes (bias nuisance, not certified skill). Built **`make_m14_figures.py` → fig15–20** (PNG 600dpi + PDF + SVG) covering F45–F52. Corrected **F47 n=434→197** (matched study `MIN_READS=50` + artifact). M14 in-scratch analyses COMPLETE. Next (user-deferred): PI-gated decisions (D18/D19/D16, F26/F27, OQ6, v15 switch), the pilot (`PILOT_SAP_M14.md`, gated on OQ8), and the production port incl. the decisive K=7+corr_t coverage run. |
| 2026-06-12 | **M14 — Nature-gap program COMPLETE** | **Tier-A/B/C "Nature-gap" program** (user-ratified list). DONE: A3 spec-curve (F45, 42 branches all positive), A4 proper disattenuation (F46: P2b→1.00 capped, P2→0.648), A5 freeze manifest (`PREREG_FREEZE_M14.md`, SHA-256 + confirmatory/exploratory boundary), C9 shadow replay (F47: ×6.0 placement headroom, n=197), C10/C11 pilot SAP + empirical power (F48: 33/arm enrollment at α=0.005/0.9; `PILOT_SAP_M14.md`). DONE: A1 M5/M5b joint softmax fit (F49 FINAL — composed marginals beat the single-skill softmax even with an added per-user lapse dim; M5b lapse latent degenerate; `data_extset_m5.npz`); A2 Bayesian refit + proper SBC (F51 full-posterior P(both λ<0.025)=0.9999 upgrades F43 to Bayesian; F52 proper SBC — 5/6 globals calibrated once chains converge, so F51's blanket failure was an under-convergence artifact; only σ_θ residually under-estimated, a low-stakes bias nuisance; `data_extset_bayes.npz`, `data_extset_sbc.npz`); B8 K=7 re-cert replication (`study_recert_k7.py`, F50 — F32 protective direction replicates 14/16 FG caught, but K=7 re-cert is REFER-dominated/poor graduator, which is the instrument's OC-at-the-cut behavior, NOT a pathology — anchored to the v15 OC resolution curve; `data_recert_k7.npz`). New studies: extset_m5/bayes/sbc/speccurve/shadow + pilot_power + recert_k7. Figures: `make_m14_figures` fig15–20 (A1/A2/A3/A4/C9/C10). All in-scratch M14 analyses COMPLETE; remaining work is PI-gated + pilot + port (see 2026-06-13 row & header). |
| 2026-06-12 | **M13.3** | **SAP EXECUTION COMPLETE — all six prespecified endpoints PASS; EXTSET ladder done.** P1: task2 held-out ΔELPD/read vs engine M0 = +0.0006…+0.0137 (13/14 frames CI>0; heavy-tail M3 dominates LOO frames, class miss-lapse 0.06–0.34). P1b: task1 binary M2, **+0.0049 [+0.0037,+0.0062]**, fitted lapses BELOW engine's 0.025 (F43). P2: ρ=0.481 (n=315, p=1.3e-19; per-domain 0.67–0.83 Holm-sig). P2b: **ρ=0.868** (n=321, p=6e-99). P3: real link INSIDE the envelope, ≈wellspec on all metrics (F44). P4: cross-domain ρ=0.571 (n=136, p=4e-13; disattenuated 0.771; 7×7 manifold). W3 populations extracted per frame (F43). Code: `extset_adapter` (+17-check test), `study_extset_link/replay/xdomain/stress`, `RealLinkLearner`, `make_extset_figures` (fig9–12); stress campaign parallelized ×42. Deferred (explicit, prespecified secondaries): M5 softmax joint fit; second-half-per-user nonstationarity refit; DS soft-gold + senior-expert gold sensitivities; shipped-vs-LOO formal delta table (raw numbers in npz). Gated: W3 dynamics anchoring (OQ8 feedback answer pending). Full regression green (test_step0/2, misspec, tier3, extset). Artifacts: `data_extset_{link,replay,xdomain,stress}.npz`, fig9–fig12. |
| 2026-06-12 | **M13.2 (interim)** | **SAP execution underway** (`extset_adapter.py`+`test_extset.py` 17 checks green; W1 ladder full run in background). **W2 DONE (P2/P2b PASS):** task1 Spearman(ℓ̂, acc) = **0.868** (p=6e-99, n=321); task2 mean-ℓ̂ vs 6-class acc = **0.481** (p=1.3e-19, n=315); per-domain ρ 0.669–0.834, all Holm-sig; F34-pathology rate 0.7% (14/1890 cells). Qscore triangle resolved: raw ρ(ℓ̂,Q)=0.19 is a VOLUME confound (Q is rolling+engagement-loaded, ρ(n,Q)=0.82; ℓ̂ volume-orthogonal, ρ(n,ℓ̂)=−0.17); partial ρ(ℓ̂,Q\|n)=**0.591** — convergent validity holds (OQ10 reinforced). Survey-tier known-groups FAILS honestly: tier AUC 0.45 (n=11 vs 71) and W4 tier table shows derived_tier≈uninformative for task2 skill — report as a tier-label finding, not filter failure. **W4 DONE (P4 PASS):** Spearman(domain1 acc, domains2–7 acc) = **0.571** (p=4e-13, n=136), disattenuated **0.771** (split-half rel. 0.77/0.71); full 7×7 positive-manifold matrix (0.29–0.67) — first empirical Σ_l grounding. Data-quality note (F42 candidate): bank vote columns on task2 EXTSET segs diverge from the read release (miss 2,238 reads on 1,651 segs; hold ~14.5k extra historical votes) ⇒ adapter builds vote bases from reads+experts directly. Artifacts: `data_extset_replay.npz`, `data_extset_xdomain.npz`. |
| 2026-06-12 | **M13.1** | **task1/domain1 contest data landed and validated (F41); CRITICAL signal-circularity finding (F40) + LOO protocol adopted; SAP v2 (K=7).** `extset_task1_reads.csv` (167,503 binary reads × 643 raters, zero dupes, 100% key integrity, +6.4k reads vs older users-table snapshot) + `extset_task1_signals.csv` (5,000/5,000 finite, scale matches bank pool). F40: s_mean ≈ smoothed-probit of the crowd vote share EVERYWHERE (task1 R²=0.966 w/ votes == these exact reads; task2 corr 0.93–0.96 w/ votes = EXTSET reads+4 experts; bank domain1 pool 0.80) ⇒ all confirmatory fits use exact leave-one-user-out consensus signals (self-shift +0.026 toward own response otherwise); shipped-s demoted to sensitivity. Task1 headline: P(say\|s_loo) 0.026→0.852, per-user acc 0.763±0.104 (321 users ≥20), pooled AUROC 0.779 — the K=1 BINARY frame tests the engine likelihood exactly. NEW W4: 147 dual-contest users ≥20 reads both ⇒ empirical Σ_l cross-domain correlation grounding. Endpoints now P1/P1b/P2/P2b/P3/P4 (SAP v2 §6). OQ9 resolved (≥3/4 consensus); OQ11 resolved (data arrived); NEW OQ13 (exact s_mean/s_sd formula — residual 0.19 sd unexplained). Pending: OQ8 feedback confirmation (gates W3 dynamics only). |
| 2026-06-12 | **M13** | **EXTSET scrubbed release landed in scratch (10 files); full structural validation + EDA done; SAP drafted; awaiting OQ8–OQ12 answers before fits.** Verified: 125,860 valid reads × 699 users × 5,000 task2 cases, 100% case→seg_id crosswalk into the bank, full finite s_mean/s_sd for domains 2–7 (task1/domain1: no signals, no reads — K=6 only); 4-expert panel complete. New findings: F35 bank-vote leakage (bank votes on EXTSET segs = these reads + 4 experts; bank plurality circular as truth there), F36 within-user learning +2.2pp (pooled 0.27→0.58 curve is survivorship), F37 clean monotone psychometric link (P(say k\|s_k) 0.02→0.71 by decile; pooled AUROC 0.62–0.74; per-user acc 0.253±0.136) + Qscore corr 0.840 verified, F38 gold-panel noise (κ 0.50–0.67; ≥3/4 consensus 75.8%; plurality vs source gold 77%/92%). Analysis plan `EXTSET_SAP_M13.md`: W1 real link fit + zoo stress member (rung-3 closure), W2 filter replay validation n≈315 (rung-5 at scale), W3 empirically-grounded population/dynamics priors — all within M11.2c scope. Clarifying questions OQ8 (feedback?), OQ9 (gold policy), OQ10 (Qscore formula), OQ11 (domain1 reads), OQ12 (serving design) put to user 2026-06-12. |
| 2026-06-11 | **M12** | **Verification-ladder campaign (autonomous run; plan in `VERIFICATION_PLAN_M12.md`): rungs 3–4 COMPLETE in-scratch, rung-5 measurement grounding done; findings F28–F34; mitigations implemented opt-in; full suite green (13 files, ~145 checks incl. new test_misspec 20).** (1) Built the 12-member mis-specified learner zoo + dual graduation ground truths (latent cert-bar; pool-accuracy vs the designed acceptance envelope A_bar(σ*,t*)). (2) Rung-3 campaign (zoo × 3 policies × 30 seeds + 120 heterogeneity draws): training EFFICACY robust (true mastery 40–52 trials for every learning member) but graduation SELF-ASSESSMENT fails under misspec — F28 attractor disease, F29 flat OC (P(declare)=1.00 ∀Δ∈[−.2,+.3]; oracle arm = clean step fn, proving F19 was filter-specification-conditional). (3) Rung-4: closed-loop SBC found the well-specified graduation-window ℓ over-estimation (F30, fixed by GH-smeared skill-weight kernel — SBC nominal after fix; θ residual F33); full OC quantified. (4) Mitigations (F31): static-mixture filter + raw-evidence gate + smear_w, all opt-in (`TaskFilter(p_static=, smear_w=)`, study-level evidence gate); hardened stack kills careless/anti FG entirely, halves benign premature declarations, restores coverage; structural residual (static_below 23/30, sub-cut OC 0.68–0.97) is placement-information-poverty — not fixable by calibration. (5) Rung-3 closed END-TO-END: real-engine K=1 re-cert FAILs/REFERs 75/75 false graduates (F32); genuine graduates pass immediate re-cert only 2/15 ⇒ premature declaration also wastes evals. (6) Rung-5 in-scratch: all 3 real sessions replayed — rater ℓ̂ ordering matches accuracy exactly, production-ℓ̂ cross-val Spearman 0.80, careless certifies nothing; dynamics-on replay inflates ℓ̂ +0.26 on real no-feedback data (F34). (7) Decisions D18 (graduation = recommendation; certification = re-cert/probe block) + D19 (hardened port config). Artifacts: misspec_learners/study_misspec(+_mitig)/study_gate_oc/study_train_sbc/study_recert_backstop/study_real_replay/make_m12_figures + 8 npz caches + fig8. **Publication framing per §3D:** the honest claim is now "the trainer reliably drives skill to mastery across an explicitly-characterized misspecification envelope and hands off to a certification layer that catches its errors" — with F28–F31 as the characterization. Open: EXTSET link stress (outside scratch), K=7 re-cert replication, smear_w default flip + claim re-pinning at port, PI ratification of D18/D19. |
| 2026-06-11 | **M11.4** | **Repo scrub (user request): domain-code mapping + medical/proper-noun removal + external-pointer removal.** (1) Domain codes confirmed mapped to domain1..7 everywhere (text already anonymized; CSV `pattern_class_true`/`response_label`/`plurality`/`subclass` values already domain1..7). (2) Removed medical/anatomy/bio terminology (signal-acquisition column names → `layout`/`gain`/`band`/`filt`; clinical demographic headers/values neutralized; the methodology-paper system name → `ENGINE`) and proper-noun dataset/person/institution names (neutral codenames: external label sets → `extset`/`corpus_a`/`corpus_b`/`profiler_a`/`group_a`/`group_b`/`platform_a`/`crowd2025`/`rct_a`; people → `expert_1..4`/`source`/`the authors`; `participant_name` PII → `participant_n`). YAML task keys `combined_/sparcnet_domainN` → neutral `task_domainN` (`policy_general` key-builder updated in lockstep). (3) Removed pointers outside the directory: `parent.parent`/`sys._MEIPASS` repo resolution in `policy_general`/`policy_k7_general`/`session_controller_general` repointed to the in-directory self path; external repo references in docs replaced. Verification: all `test_step*`/`test_v15`/`test_tier3` green; `bank_adapter` loads the scrubbed CSVs with identical pool sizes (domain1 = 19,332) and the large numeric CSV columns round-tripped byte-identically (md5 unchanged); zero residual of target terms on full-repo sweep. Pre-scrub backup at `/tmp/scratch_prescrub_backup.tgz`. |
| 2026-06-11 | **M11.2** | **EXTSET real-response feasibility audit (user request).** Located the data in the live multi-K7 repo (NOT scratch): 4-expert gold panel (5000×4, cross-sectional) + **novice contest reads `extset_novice_labels.csv` — 125,865 reads × 699 users WITH a timestamp axis + Qscore, 100% crosswalkable to bank signals.** Full analysis in §3C. Verdict: **partially feasible** — real reads CAN replace the simulated learner's response model (A, feasible/high-value) and anchor Phase-3 dynamics priors (B, feasible but unsupervised-drift only: +1.8pp within-user accuracy drift over 145 ≥100-read users, confounded, no feedback); CANNOT replace the feedback-driven training benchmark (C, infeasible — no human ever ran the trainer; only EVAL sessions recorded). **BLOCKED on user scope decision (A vs A+B vs C-needs-pilot)** before building. No code written yet — audit only. **M11.2b sharpening:** asked directly whether EXTSET can replace simulated runs to PROPERLY TEST the methodology → NO end-to-end (structural: closed-loop off-policy item selection + no feedback ⇒ no learning trajectory). CAN test open-loop components (response model, filter measurement). Best achievable = empirically-grounded simulation; definitive methodology test needs a feedback-driven pilot. See §3C SHARPENED CONCLUSION. |
| 2026-06-11 | **M11** | **v15 staged instrument integrated + OC-replicated; Tier-3 ("method 3") MC rollout built + benchmarked. Full suite green (steps 0–8 + test_v15 21 + test_tier3 9 checks).** (1) **v15 staging:** `cert_config_general.yaml` v15 block completed to v14 parity from `v15_simulation_spec.txt` (16-digit ℓ\*, credentialed expert/non-expert means, J/CIs/panel n, sha256s, domain3 robust-trim) + new `eb_examinee_bias_prior_v15` block (μ_t; Σ_t from npy). `instrument_v15.py` = the versioned accessor (v14 frozen: Corr_l both blocks/600p; v15 staged: Corr_t t-block/1200p — N is PARTICLE count, stopping rule unchanged); `pipeline_demo._run_eval` gained opt-in instrument kwargs (defaults bit-preserve shipped; step-8 re-passed). bank_adapter constants stay v14. Spec confirms: Σ_l/Corr_l/bank/labels UNCHANGED; t\* targets DO NOT EXIST (OQ6 stays open). (2) **OC study** (264 K=7 sessions, 420-q regime, real bank, fig6): per-arm anchoring → v15 +6.6pp @+0.6 (78.6 vs 72.0), +4.8pp @+1.0 (99.4 vs 94.6), cuts-alone ablation FLAT; credentialed-bar anchoring → v14 49.4% vs v15 78.6% (+29pp), nearly reproducing the claim table ⇒ **F26: the claimed ℓ\*-lever is mostly bar recalibration; same-anchor instrument resolution gain is the smaller corr_t+1200p effect.** Realistic θ~N(μ_t,Σ_t) costs ~20pp both arms (spec caveat is first-order). (3) **SBC** (200 sessions): 95% ℓ-CI coverage 0.933@600p / 0.919@1200p — **F27: particle doubling does NOT fix posterior-ℓ coverage here** (spec's nominal-at-1200 was AUROC-CI, K=6 Mode-A; the production K=7+corr_t AUROC-CI run remains the decisive check). (4) **Tier-3** (`trainer_rollout.py`, plan §6.3): batched H=6/L=8 rollouts, π0=tier-2, CRN across candidates LOAD-BEARING (without it the argmax is a coin flip); 30-seed campaign (fig7, `data_benchmark_tier3.npz`): **F25** — beats tier2 only under response-conditional (hard-rule) dynamics (−2.9 [−5.2,−0.8] paired), small MC tax under soft; tier2 stays default. Artifacts: `study_v15_oc.py`, `study_v15_sbc.py`, `make_v15_figures.py` (fig6/fig7 PNG+PDF+SVG + 3 npz caches). Open → PI: anchoring convention behind the claim table (F26), coverage metric for the 1200p claim (F27), OQ6. |
| 2026-06-11 | **v15 ℓ\* received + validated** | User supplied curated/credentialed-data-set ℓ\* cuts (3 dp) and, same day, the ablation/robustness/coverage stats. Added `ell_star_unified_v15` block to `cert_config_general.yaml` with derived σ\*=exp(−ℓ\*) + validation summary; v14 untouched and still the live default everywhere. All cuts dropped 22–43% — established as recalibration, not leniency (ℓ\* lever alone: +17.4pp PASS@+0.6 AND both error rates cut; full v15 48/77/87% at false-PASS 0.29%/false-FAIL 0.43%, robust to realistic bias). N=1200 staged post-pilot; live stays N=600. Details in §2B "v15 update". Open: per-task CIs/J/panel metadata + sha256; K=7+corr_t coverage run; real-pilot replay; consumer-switch decision. |
| 2026-06-10 | **M10.1** | **Visual demonstration + performance figure suite** (user request; aesthetic per `beautiful_figure_example_general.py`, codified in `viz_style.py`). (1) `figures/learning_algorithm_demo.mp4` — 62 s, 1080p animated demo of the REAL pipeline (eval → Tier-2 training with LT2 gap → re-cert; seed0=23 run: FAIL/FAIL/REFER → 156 train trials → **PASS/PASS/REFER**), panels: belief cloud in (t,σ) with mastery zone, psychometric truth-vs-estimate with served item ±s_sd, skill ℓ̂±SD vs cuts, bias vs tolerance band, posterior SD, mode strip. Rendered via bundled imageio-ffmpeg (pip-installed; no system ffmpeg). (2) Five figures (PNG 300 dpi + PDF + SVG, data cached as npz in `figures/`): fig1 benchmark trials-to-mastery with paired bootstrap CIs (this regenerated 30-seed campaign is now the normative record — see corrected numbers in M10(3)); fig2 F17 coverage asymmetry + believed-vs-actual mechanism; fig3 F5 floor trajectories with Riccati line and both s_sd regimes; fig4 F23 attractor (criterion-centred vs boundary-centred vs mirror-paired — the visual proof that placement sets the bias); fig5 recorded sandbox outcomes (F18 designs, F19 gate detection). All scripts re-runnable (`make_figures.py`, `make_demo_video.py`). (3) Second MP4 added on user request: `figures/param_evolution_demo.mp4` — per-question skill/bias line-pair comparison across Tier-2/Tier-1/random on a shared axis (see `make_param_video.py` row in §1). |
| 2026-06-10 | **M10** | **Audit-fix round complete; full suite green (~93 checks).** (1) **Incident recap (M9.1):** `bank_adapter.py` was lost from disk and reconstructed from spec; user ratified the reconstruction; M10 added per-domain distribution validation (see §1 row). The M9.1 review also verified the three audit flags: F5 floor fix confirmed analytically (Riccati ℓ-SD 0.138/θ-SD 0.131 vs measured 0.154/0.132; multi-seed 0.152±0.002) and F17/D14's mechanism quantified (2× over-estimated rates → believed SD 0.118 vs actual error 0.191 — anti-conservative direction, confirming conservative priors). (2) **Fixes implemented + power-verified:** F21 observed-y hard-rule propagate (RMSE(t) −13.7%); systematic resampling (posterior-mean MC error −33%/−22% θ/ℓ); vectorized greedy (bit-identical, ≥5×); F20 production-faithful s_sd physics end-to-end incl. the engine's Phase-3.5 eval path; D15 noise-aware selection; D16 hybrid mastery gate; D17 due-driven retention; dead-code cleanup (pipeline_demo, trainer_policy). (3) **Benchmark claims RE-BASED under realistic physics** (30 seeds, paired bootstrap, pool 600, task domain3, final D15 design; raw data cached `figures/data_benchmark.npz`): tier2 best-or-tied in EVERY cell — soft 1× **45.8±8.7** [42.7,48.8] ≈ tier1 47.0 < measure_opt 49.1 ≪ staircase85 69.2 ≪ random 371 (24/30 censored); hard 1× 46.7±10.4; hard 2× 52.8±13.6 vs measure_opt 63.8. tier2 has the lowest variance of all policies and is rate-misspec-insensitive on the soft learner (×2: 45.9). Paired tier2−staircase −23.4 [−31.7,−15.7] (soft 1×) through −34.7 [−52.2,−17.7] (hard 2×), every CI excluding 0 — **the headline win is noise-aware, dynamics-aware placement vs naive placement**; the old "tier2 ≪ measure_opt" gap was an s_sd=0 artifact (criterion placement gets smeared INTO the productive band; tier2's edge over it returns under rate misspec). Pool-size sensitivity: tier2 74→49→47 at pools 300→1000→3000 (selection needs supply; pilots now use 1000). test_step7/test_step8 claims re-pinned accordingly (per-task point-estimate-clears-cut at re-cert was also an s_sd=0 artifact; replaced by per-task improvement + mean-clears-cut). Step-8 e2e under faithful physics (final run): FAIL/REFER ×3 → 152 train trials → re-cert flips domain3 to PASS, ℓ̂ improved on every task, mean ℓ̂ 0.71 vs mean cut 0.37. The fix round also surfaced the two control-theoretic findings **F23 (criterion = served-stream-midpoint attractor) and F24 (dual-control: corrective items blind the filter)** — see §3; test_step5 integration now graduates 3/3 in 194 trials with ZERO mode switches. (4) **Sandboxes** `sandbox/exp_retention.py`, `sandbox/exp_gating.py` ran the point-2/3 design questions (results in §3 F18/F19); deleted after recording per plan. Open: F22 (Phase 3); OQ3/OQ6 unchanged; PI ratification of D16 acceptance semantics. |
| 2026-06-10 | M0-pre | Review of plan + engine complete (findings §3). Decisions D1–D8 recorded. Constants verified (§2). `INTEGRATION_PLAN.md` written. No code yet. |
| 2026-06-10 | **M9** | Step 9 DONE — `PHASE3_AND_PORT.md` (Phase-3 spec + port checklist). **Phase-2 prototype pipeline COMPLETE: all 9 steps (0–8) done + Step-9 docs, ~90 smoke checks green, identifiability hard gate passed, seamless eval→train→re-cert demonstrated on the real 89k bank.** Remaining work is Phase-3 (offline fitting, LT1/LT2 models, RT channel) + porting to the main repo. Open for PI: OQ3 (warm-start re-cert), OQ6 (t*/mastery targets); still awaiting the two source memos. |
| 2026-06-10 | **M8** | Step 8 DONE — **the seamless eval→train→re-cert pipeline works end-to-end** (`pipeline_demo.py`+`test_step8.py` **7/7**, 57s). A mid-skill candidate FAILs all 3 trained tasks (true ℓ −0.19/0.08/−0.12 < cuts), trains 160 trials across sessions until true ℓ rises to 0.60/0.70/0.66, and **re-certifies PASS on all 3** (recovered ℓ̂ clears every cut). Exercises every prior module against the real bank+labels with eval-seen exclusion (D12), feedback-safe filtering (D11), per-task seed handoff (D1/D2), and the LT2 between-session gap hook (D9). Verdict-resolution is particle-count-sensitive (300 particles strand strong candidates at REFER_BORDERLINE; 500 resolve PASS — matches HOW_THE_TEST_WORKS §11). Phase-2 prototype COMPLETE. Next: Step 9 (Phase-3 spec + port checklist, docs only). |
| 2026-06-10 | **M7** | Step 7 DONE. `benchmark_trainer.py`+`test_step7.py` **5/5**. **Headline results (trials-to-mastery, single task, biased+sub-skill learner):** tier2 ≈ 37, tier1 ≈ 36, staircase85 ≈ 38, measure_opt ≈ 53, random ≈ 294. **Claims supported:** (1) tier2 ≪ measure_opt (~30% faster) — measurement-optimal placement IS the wrong objective for training (plan §6.4); (2) tier2 ≈ 8× faster than random; (3) under the HARD-rule learner tier2 stays stable (42±8) while staircase degrades to 51±29 (3.6× variance) — tier2 is more robust to model misspecification. **Honest caveat:** on a single well-specified task tier2's edge over staircase is modest (balanced 85%-point placement incidentally reduces bias too); tier2's real advantages are robustness + multi-task + vs-measurement. Full campaign 1m50s; pilot 18s. Next: Step 8 (e2e seamless demo). |
| 2026-06-10 | **M6** | Step 6 DONE. `trainer_greedy.py`+`test_step6.py` **6/6**. Tier-1 greedy comparator built; both F6 degeneracies witnessed as regressions and shown fixed by the constraints: F6b base-rate exploit (unconstrained label excursion 24 → constrained 1), F6a retention spam (ungated bonus picks the trivial easy item → eligibility-gating picks a productive one). Test-design note: F6a only manifests when the bonus attaches to the mastered-bin item gated by due-ness (not granted to all items). Next: Step 7 (benchmark campaign). |
| 2026-06-10 | **M5** | Step 5 DONE. `trainer_policy.py`+`test_step5.py` **15/15** (1.8s). Tier-2 mode policy + deficiency-weighted scheduler. Unit: mode gating, label-balance stays within ±1 over 300 bias trials (F6b), SM-2 spacing fires/grows/resets, scheduler picks worst task + consec cap. **Integration: simulated trainee graduated all 3 tasks in 159 trials, mode-switches/task=[2,0,4] (no thrashing).** OQ1 resolved (deficiency score); OQ6 raised (t* is a PI constraint, assuming 0.30). Next: Step 6 (Tier-1 greedy ablation). |
| 2026-06-10 | **M4** | Step 4 DONE — **identifiability HARD GATE PASSED** (6/6). `study_identifiability.py`+`test_step4.py`: σ/t error-confusion |corr| ≤ **0.17** (≤0.5 criterion), posterior coupling ≤0.13, correct-spec coverage **0.91**, RMSE <0.20, graceful under ±2× rate misspecification (no divergence). New finding F17 + decision D14 (conservative rate priors). The trainer can disentangle skill from bias while both move — the central modeling risk is cleared. Next: Step 5 (Tier-2 policy + scheduler). |
| 2026-06-10 | **M3** | Step 3 DONE. `training_filter.py`+`test_step3.py` **7/7**. Bootstrap filter (no MH, F1/D5) tracks a drifting learner at **96% credible coverage** (σ and t), RMSE σ=0.10/t=0.18; static filter matches a fine-grid posterior to <0.06 in all moments (Bayes step verified correct); **variance floor ℓ-SD=0.154** (F5 quantified). Real careless rater (acc 0.23) correctly certifies NO task. Fixed a `steady_state_sd` bug (was measuring unbounded diffusion). OQ5 (inflation factor): coverage is well-calibrated at the eval-seed-like prior_sd=0.25, so D1's ×1.5 inflation is adequate — no change. Full regression 65 checks green. Next: Step 4 (identifiability gate). |
| 2026-06-10 | **M2.5** | Steps 2 + 2.5 DONE. `learner_sim.py`+`test_step2.py` **8/8** (τ_σ fit 33.3/33.3 exact; \|t\|→0.029; hard≈soft z=1.4). `bank_adapter.py`+`test_step2_5.py` **9/9**: pool sizes exactly domain1 19,332 / others 69,806; label reconstruction **98.0%** vs the real session; σ_∞<σ\* holds all tasks. **Insight:** whole-bank conflicted fraction is only 12% but eval-*served* items were 27% — the eval selector preferentially picks near-threshold (ambiguous) items, so D11's feedback-safe filter matters most exactly where the trainer would otherwise want to probe. Next: Step 3 (training filter). |
| 2026-06-10 | **M1.5** | Real production artifacts ingested + analyzed (89k bank, fitted Σ, v14 cut-scores, label file, 3 real sessions, session controller). Revisions R1–R9 to plan; decisions D10–D13; §2B grounded priors added; F4/F10 updated; clarifying Qs answered (§5A). Plan got new **Step 2.5 (bank adapter)** and amendments to Steps 2/3/5/7/8. Key numerical results: label file → target_present 97.2%; σ_∞=exp(−expert_ℓ) per domain 0.46–0.66 all < σ\* (LT1 achievable); s_sd material (med 0.6–0.98). No new test code yet — implementation resumes at Step 2. |
| 2026-06-10 | **M0** | Step 0 DONE. `bridge_conventions.py` + vendored `auroc.py` + `test_step0.py` — **23/23 checks pass**, incl. engine end-to-end (29-question hier session, 4 rejuvenation events, mean MH accept 0.61). Exact skill-mode constant pinned: `SKILL_MODE_MULTIPLIER = 1.0772256` (earlier memo value 1.077015 came from rounded accuracy; corrected in §2). LT1/LT2 directives recorded (§3A); D9 hooks added; plan Steps 1/3/5/9 amended accordingly. Next: Step 1 (TrainingSeed). |
| 2026-06-10 | **M1** | Step 1 DONE. `training_seed.py` + `test_step1.py` — **18/18 checks pass**. Smoke ran a real 209-trial K=3 eval session (engine loop + AD6Policy) yielding the designed mixed verdicts [PASS, FAIL, REFER_BORDERLINE]; seed round-trips exactly (13 arrays, dtypes preserved); inflation verified SD×1.5 / mean-invariant per task; seg-id completeness/uniqueness; D9 epochs+ledger working. Seed schema v1. Notes: (a) AD6Policy exposes verdicts/diagnostics only via private attrs (`_verdicts`, `_last_diag`) — fine for prototyping, add public accessors at port time; (b) borderline task consumed its entire 120-item bank, as expected for ℓ near cut. OQ2 now needed by Step 5. Next: Step 2 (learner simulator). |

## 7. Tombstones (superseded info)

- (2026-06-12, M13.2 maintenance: the 2026-06-10 multiplier tombstone aged out
  per §0 rule 2 and was deleted.)
- superseded (2026-07-01, M15): F22 "OPEN, Phase 3" → FIXED as default (D20).
- superseded (2026-07-01, M15): F30's "smear_w default ON at port"
  recommendation → D21 exact kernel (smear_w is its unconditional ℓ-half).
- superseded (2026-07-01, M15): F25's tier-3 value boundary ("pays under
  hard-rule dynamics") → F56 (advantage vanishes under the exact kernel).
- superseded (2026-07-01, M15): F33's "SMC approximation error, document" →
  cause identified as the propagate kernel's dropped y↔s_real coupling
  (F53); fixable, re-run SBC at port.
