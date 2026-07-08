# Label-Schedule Randomization — PECR record

Status: **PECR round 3c — awaiting critique-4 exit ruling + owner
review.** Rounds: 1 (walk + debt queue; echo leak), 2 (EWMA equalizer;
bank-geometry leak + short-stream |t| effect), 3a (precision-blind bins;
FG failure, reverted), 3b (precision-aware paired bins; all label/|s|
gates pass), 3c (documentation + measurement completeness; s_sd channel
→ §6 owner item). Owner-ratified goals (2026-07-08):

* **Scope**: remove every deterministic *label-sequence tell* in the
  trainer's serving layer. Four devices are in scope:
  1. skill-mode strict side alternation (`trainer/policy.py:723`,
     `cortex_web/apps/web/trainer/policy.ts:300`);
  2. the mirror-pairing "same difficulty → flipped label" echo
     (`_last_skill_s` partner targeting);
  3. bias-mode label-balance forcing (`policy.py:555`, `policy.ts:277`)
     and the corrective/probe flip (`_bias_probe`);
  4. the cert-probe label flip (`policy.py:1184`, opt-in machinery,
     not live in prod).
  Plus one session-level tell found in recon: prod TS resets policy
  state every session, so **the first skill trial of every session is
  always the positive case** (`skillSide = 1` initial, seed hardcoded
  to 7 at `App.tsx:347`).
* **Frozen surface** (the "don't alter the POMDP" contract): belief
  filters, dynamics, reward/score functions, mode gates, mastery/
  e-gates, and the cross-task scheduler are untouched. Only the
  label-side *scheduling devices* that sit on top of them may change,
  and their three mathematical guarantees must be re-proven:
  (i) anchoring `E[t∞] = 0`; (ii) bounded label imbalance (F6b);
  (iii) served-stream midpoint pinned at the true boundary (M10).
* **Acceptance gates** (PECR exit conditions):
  * **G-A (adversary)**: the optimal history-informed guesser's
    next-label accuracy ≤ **0.55**, (a) in closed form under the
    stationary chain, and (b) measured on simulated sessions — per
    mode-stream and on the interleaved stream as the learner sees it,
    including an adversary given served-difficulty features (the
    mirror-echo side channel), and including the pattern the owner
    actually observed (lag-1 label prediction = 50±ε%).
  * **G-B (band)**: stationary criterion fluctuation band inflation vs
    strict alternation ≤ **15%** across the operating envelope
    (σ̂ × s_sd grid, python defaults α_t=0.15/q_t=0.05 AND prod TS
    α_t=0.097/q_t=0.05), by closed form AND direct recurrence
    simulation (re-creating the un-ported `test_audit_fixes` §10b
    check, schedule-general).
  * **G-C (OC non-regression)**: on matched seeds, legacy vs
    randomized: false-graduation not worse, declaration lateness and
    delivered training value (`delivered_value` / w_true) within
    noise, terminal |t_true| calibration unchanged, bias-mode
    convergence (t̂ overshoot / time-in-mode) unchanged. Harness:
    `trainer/oc.py::single_task_train_session` +
    `trainer/orchestrator.py::run_closed_loop`.
  * **G-D (code)**: new behavior behind a flag whose default is
    **bit-identical legacy** (existing drift-guard
    `tests/test_trainer_g1_policy.py` choice-sequence parity must stay
    green untouched, plus a dedicated flag-off equivalence test);
    python↔TS bit-parity of the schedule stream (shared splitmix64);
    suite green.
* **Delivery**: python reference first (this PECR loop), then TS port
  with golden-vector parity, then **pause for owner review** (deploy +
  SAP disposition are owner decisions; trainer is live under
  `docs/TRAINER_PILOT_SAP.md`).

---

## 1. Why the alternation exists (the constraint set)

The learner's criterion follows a soft Rescorla–Wagner recursion
(`trainer/sim.py:50`): t' = t + α_t(p̂(s, t) − y*) + ξ, ξ ~ N(0, q_t²).
Under boundary-anchored mirrored skill serving (items at s̄ = ±a·σ̂,
label = side), linearizing at t = 0 with restoring gain
κ = α_t(1−2λ)φ(a/σ_e)/σ_e gives

    t_{n+1} = (1−κ) t_n − α_t δ̄ σ_n + α_t ε_n + ξ_n

where σ_n ∈ {±1} is the served side, δ̄ = 1 − p̄(a) the mean
correct-side prediction error, ε_n the stimulus-noise fluctuation of
p̂. The stationary variance splits into a schedule-independent
diffusion term and a schedule term:

    Var∞ = (q_t² + α_t²·Var[p̂]) / (κ(2−κ))  +  α_t² δ̄² · V[σ]

    V[σ] = (1/2π) ∫ |H(ω)|² f_σ(ω) dω,   H(ω) = 1/(1 − (1−κ)e^{−iω})

**V[σ] is the low-frequency-weighted spectral mass of the side
process** — the AR(1) transfer function is a low-pass filter. This is
the whole design problem in one line:

* strict alternation puts all spectral mass at ω = π (maximally
  attenuated): V = 1/(2−κ)² — exactly the intra-pair oscillation term
  in `bias_stationary_sd` (`policy.py:270`);
* iid sides are spectrally flat: V = 1/(κ(2−κ)) — the docstring's
  "order-of-magnitude worse" diffusion injection;
* any anti-persistent schedule sits in between, and for a Markov side
  process with lag-1 autocorrelation φ: V(φ) = (1+βφ)/((1−β²)(1−βφ)),
  β = 1−κ, giving a continuous frontier.

The other two guarantees: anchoring needs E[σ_n] = 0 (symmetric
schedule); midpoint pinning needs the served stream's signed-magnitude
sum bounded (mirror pairing gives per-pair exact cancellation).

## 2. Round-1 frontier computation (exact, real constants)

Design family analyzed: imbalance walk D_n = D_{n−1} + σ_n on
{−b..b}, P(σ=+1|D=d) = clip(½ − γd/2b, 0, 1), forced at |D| = b
(γ=0 ⇒ big-stick/fair-interior; γ=1 ⇒ Ehrenfest). Exact stationary
distribution, exact V via the chain autocovariance, exact optimal
guesser G = Σ_d π(d)·max(p(d), 1−p(d)) (label history determines D, so
this IS the optimal history-informed guesser). Constants: α_t = 0.15,
q_t = 0.05 (`trainer/dynamics.py:22-25`); band checked over
σ̂ ∈ {0.6, 1, 1.6, 2.5, 4} × s_sd ∈ {0.3, 0.6, 1}.

Result (worst-regime band inflation vs strict alternation):

| design | G (optimal guesser) | worst inflation |
|---|---|---|
| b=2, fair | 0.625 | 2.0% |
| b=4, fair | 0.562 | 6.0% |
| b=5, fair | 0.550 | 7.7% |
| **b=6, fair** | **0.542** | **~8.5%** |
| iid (no cap) | 0.500 | 11.4% |

Two structural findings:

1. **The band gate does not bind.** Even iid sides inflate the band
   only ~11% at the shipped constants. The binding constraint is
   predictability vs imbalance control: for fair-interior capped walks
   G = ½ + 1/(4b) exactly, so the cap must be ≥ 6 to clear G ≤ 0.55
   with margin. Tilt (γ > 0) only trades the wrong way here (raises G
   faster than it lowers inflation, and inflation has slack).
2. **Run length needs its own cap.** An imbalance cap alone admits
   perceptually-noticeable same-label runs; a run cap r (force a flip
   after r consecutive same labels) is itself a forcing rule, so it
   costs G. Round-1 design choice: analyze the augmented chain
   (state = imbalance × signed run length) exactly and pick (b, r)
   jointly; working hypothesis **b = 6, r = 3**.

## 3. Shipped design (round 3b; §5b/§5c record the pivots)

**One walk primitive + static paired-bin serving, four consumers.**
Module `trainer/label_schedule.py`. Shipped schedule params (the §2 b=6
hard-cap working hypothesis was superseded — hard rules are free wins
for the optimal guesser): `ScheduleParams(cap=8, d0=3, gamma_d=0.04,
r0=3, gamma_r=0.04, q_min=0.25)` — fair coin with SOFT anti-imbalance
and anti-run tilts, hard-forced only at |D| = 8; G = 0.5485,
worst-envelope band inflation 9.8%, P(run ≥ 5) = 3.2%. Mirror device =
`paired_bin_choice` (§5c; the round-1 debt queue and round-2 EWMA
equalizer both leaked and are deleted). The bin-profile kernel uses a
width floor max(target, 0.3) (code: `label_schedule.py`).

* `ScheduleRng` — splitmix64 counter-based stream, keyed
  (seed, stream_id, counter), mirroring the splitmix64 already in
  `cortex_web/apps/web/engine/rng.ts` bit-for-bit → python↔TS golden
  parity of every schedule decision. Independent of `TrainerPolicy`'s
  numpy Generator (which stays untouched on legacy paths).
* `SideScheduler(ScheduleParams(...))` — fair coin with soft tilts
  (see the §3 header for the shipped params); hard-forced only at the
  |D| = cap backstop. Tracks SERVED labels of the trials it PROPOSED
  (`info["sched"]`-gated commits at `note_served`, so pool-exhaustion
  deviations — the `_argmax_score` empty-mask fallback — update the
  true state, while retention serving never feeds the walk).
  Closed-form analysis on the augmented finite chain:
  `guess_rate()`, `side_autocov()`, `v_factor(kappa)`.
* `stationary_sd_scheduled(...)` — schedule-general extension of
  `bias_stationary_sd` (which is *unchanged*; it remains the
  strict-alternation reference and the derived t* authority).

**Consumers** (all inside `TaskModePolicy`/`TrainerPolicy`, flag-gated):

1. *Skill side*: `side = lsched.propose()` replaces `_skill_side *= −1`.
2. *Mirror pairing → EWMA equalizer* (round 2; the round-1
   magnitude-debt queue leaked, §5b(1)): serving side σ targets the
   OPPOSITE side's served-magnitude EWMA, driving the two sides'
   difficulty distributions together — the bank-asymmetry DC offset
   (the herding channel) dies distributionally, and at equilibrium
   difficulty carries no side information. Midpoint pinning is
   MEASURED and gated (persistent served-stream |mean| ≤ 0.08).
3. *Bias mode*: `want` from the SAME per-task label walk (the learner
   sees one label stream per task — its predictability is what G-A
   bounds); corrective/probe flip from a cap-2 `FLIP_PARAMS` scheduler
   (replacing the strict `_bias_probe` flip) — dual-control preserved
   (probe fraction ½, cap-bounded gaps).
4. *Cert-probe want* (`_probe_flip`): rides the same per-task walk.

**Session start**: initial state D=0 + fair first draw kills the
"first trial is always positive" tell. Seed: per-session (prod:
derived from `trainingId`; sim: TrainerPolicy seed), schedule stream
only.

**Flag**: `TrainerPolicy(label_schedule=None)` default ⇒ code path
bit-identical to today (G-D). `"randomized"` ⇒ the above.

## 4. What the closed form does NOT cover (sim-carried burden)

* Bias-mode stationary behavior (placement near criterion, not
  mirror-anchored) — G-C sims carry it (t̂ convergence, overshoot,
  mode dwell).
* Short-stream / interleaved-stream effects (per-task streams are
  short and mode-switched; stationary G is conservative) — G-A(b)
  measures empirically.
* The difficulty-echo side channel — G-A(b) adversary with |s|
  features.
* Pool-exhaustion label deviations — unit + integration tests.

## 5. PECR protocol

* **P**: this document (updated per round).
* **E**: implement + closed forms + validation battery
  (adversary oracle, recurrence-vs-formula band check, OC campaign on
  matched seeds, unit/drift tests).
* **C**: fresh-context adversarial critique agent attacks the
  derivation, the code, and the results (residual predictability
  channels, invariant breakage, edge cases, parity risks). Findings
  logged below with verdicts.
* **R**: revisions; loop to E until all gates pass and a critique
  round returns no confirmed findings.

## 5b. Round 2 (post-critique revision)

Critique round 1 (fresh-context adversarial agent) verified the closed
forms (G to 7 decimals, V-factor vs brute double-sum to 4e-15, TS BigInt
splitmix64 parity bit-exact incl. negative seeds), confirmed G-A
(history-only), G-B, G-D-python, and the death of the owner-observed
exploits (anti-repeat, session-start tell, same-difficulty echo at lag 1)
— and found three structural defects, all accepted and revised here:

1. **Mirror-debt echo (exact-difficulty adversary 0.568 > 0.55).** Root
   cause is general: any HISTORY-DEPENDENT per-item magnitude coupling
   (the round-1 debt queue, and equally a feedback controller on the
   signed-magnitude sum) leaks the current item's label through its
   perceived difficulty. The revision separates the two jobs the legacy
   mirror device conflated: (a) killing the bank-asymmetry DC offset
   (precise positives ≈ +0.96 vs precise negatives ≈ −0.55 herd the
   criterion to the stream midpoint — the true M10 pathology) needs only
   DISTRIBUTIONAL symmetry, not per-pair matching; (b) per-pair exact
   cancellation was never load-bearing beyond (a). Round-2 device:
   **per-side served-magnitude EWMAs (M+, M−); serving side σ targets
   m = (1−g)·a₀ + g·M₋σ** — the equalizer drives M+ → M−, so at
   equilibrium difficulty carries no side information (the leak is
   self-annihilating: its magnitude is proportional to the residual
   asymmetry it exists to remove), there is no queue, no evictions, no
   per-item echo. The M24 window mask uses the same equalized target.
2. **Walk corruption + overstated invariant.** Retention/maintenance
   trials (all-positive review labels) flooded the per-task walk and the
   |D| ≤ cap claim held only for the clamped state, not the true served
   imbalance. Revision: the walk commits ONLY trials it proposed (skill
   non-probe, bias, cert-probe/maintenance wants); retention is scoped
   OUT of the label-schedule surface explicitly — review items sit
   ≈ 3σ̂ easy where the label is readable off the signal (perception,
   not protocol leak), and legacy shares the pattern. Invariant restated
   and tested on walk-proposed trials.
3. **Commit-site symmetry + gates.** All schedule commits (walk, flip,
   equalizer) move to `record()`/`note_served` behind the trial's
   `info` (the propose/commit contract for callers: a `step()` without
   `record()` must be abandoned, not resumed); the battery gains the
   exact-difficulty adversary (mode-blind variant included), an
   analytic pin `guess_rate() = 0.5485 ± 0.001`, binomial-CI empirical
   gates, served-stream DC/cumsum midpoint gates, and a true-imbalance
   test. G-C harness wiring (`oc.py`, `run_closed_loop`) gains a pinned
   `label_schedule=None` passthrough; campaign results land as a repo
   artifact.

Promise restatements (critique findings 5, 7, 8, 12): lag-1
anti-repeat is inherently ≈ 0.531 (anti-persistent), bounded by G —
not 50/50; §3's shipped params are the soft-tilt cap-8 family (the §2
b=6 hard-cap hypothesis was superseded mid-round-1 — hard rules are
free wins for the guesser); G-B is envelope-scoped (α_t 0.097–0.15,
q_t 0.05) — α_t = 0.35/σ̂ = 0.6 style corners run ~27% inflation,
direction still protective; prod MUST derive `schedule_seed` per
session (a fixed seed replays the label sequence across sessions).

## 5c. Round 3 (post-critique-2 revision)

Critique round 2 verified the round-2 numbers exactly, confirmed the
plumbing (full path trace: no desync/double/missing commits), G-A
history channel, G-B, G-C single-task, G-D — and found the headline
flaw: **the equalizer's self-annihilation holds only where the bank
cooperates.** On the real K=7 bank the served skill magnitudes stay
side-split (+0.597 ± 0.251 vs −0.506 ± 0.283) and the correct
exact-difficulty adversaries reach 0.61 (own-side-EWMA) / 0.66
(held-out logistic); on a synthetic bank with the documented M10
geometry (+0.96 / −0.55 precision clusters) the leak is 0.97. The
equalizer equalizes TARGETS; the served distribution is bank-limited.
(For scale: legacy on the same real bank is 1.00-predictable — round 2
was a large improvement, but not the stated gate.)

Round-3 device — **paired-bin serving** (static, history-free):
skill-mode candidates are binned by |s| (width 0.2); only bins where
BOTH labels have feedback-safe items are eligible; the bin is chosen
SIDE-BLIND by magnitude-profile × precision-availability (round 3b:
the precision factor 1/(1 + max_side(min in-bin-side s_sd)²) is the
F20 lesson at the bin level — round 3a's precision-blind variant
served the bank's s_sd-blurred overlap band and inflated sharp-margin
FG 0.60 → 0.80, ledger §3a); the walk's side is then served by
within-bin E[w] argmax (precision-aware, side-legitimate inside a
side-blind bin). Served magnitudes match across sides to within bin
width BY CONSTRUCTION with zero history dependence — there is no
dynamic channel to attack, the static residual is bounded by the
within-bin side asymmetry (≤ bin width), and the bank-asymmetry DC
offset dies at the source. `MirrorEqualizer` is DELETED (its
clip-bound fixpoints and EWMA warmup transient — the plausible cause
of the closed-loop short-stream |t_true| effect, measured REAL at
+0.025, p = 0.042, n = 16 pairs — go with it; the effect vanished
under the static device). Empty paired-bin set ⇒ legacy unconstrained
argmax fallback (counted; 0 in all measured runs).

Battery upgrades (critique-2 findings 2/4/5/6): the G-A difficulty
gate runs on an ASYMMETRIC M10-geometry bank fixture (the environment
that killed round 2) with the own-EWMA and cluster adversaries; the
empirical gate is anchored at the analytic G + binomial CI on ≥4k
pooled trials (catches the round-1 0.568 and round-2 0.61 leaks);
the retention test forces actual retention serving (non-vacuous);
midpoint pins are scoped as synthetic regression pins with the
deployment claim carried by G-C terminal |t_true|; the closed-loop
disposition is re-made on ≥16 pairs against the round-3 build.

## 6. Open owner items (dispositions due at the delivery review)

1. **Perceived-ambiguity channel (s_sd)** — critique-3 F1. Bank-content
   limitation: on the real training pool, feedback-safe positives are
   precise (task-2 mean s_sd 0.276) and negatives noisy (0.698) on every
   task, so "looks controversial → answer no" is learnable from feedback.
   Measured on real-bank served streams
   (`results/label_schedule_oc/realbank_adversaries.json`, 8 matched
   seeds): legacy sd-cluster **0.935** (with anti-repeat 1.000 — the
   live trainer leaks BOTH channels today), randomized **0.998** (labels
   0.520/0.530 — closed; the ambiguity channel is saturated in both
   builds; the within-bin per-side precision argmax nudges it further
   into saturation). No serving policy can close it — precise negatives are
   near-absent in the pool (0–3 per measured task) — and s_sd-matching-by-blurring is the
   proven round-3a FG mechanism. Owner options were: (a) bank curation
   (derive/admit precise negative exemplars per task); (b) revisit the
   feedback-safe margin rule that induces the skew; (c) (|s|, s_sd)-
   paired serving restricted to both-sides-precise bins (pool-coverage
   and FG re-validation required); (d) accept as a shared-with-legacy
   limitation, monitored via the SAP.
   **OWNER DISPOSITION (ratified 2026-07-08): (d) + (a) queued** —
   accept + monitor via the SAP admin training-monitor, deploy proceeds
   (killing the 1.000-predictable label sequence), and a content
   workstream is QUEUED for bank curation of precise negative exemplars
   (the only real fix). Recorded as an owner-signed accepted deviation.
2. **G-B envelope** — the ≤15% band-inflation guarantee is scoped to
   α_t ∈ [0.097, 0.15], q_t ≈ 0.05, σ̂ ∈ [0.6, 4], s_sd ∈ [0.3, 1];
   out-of-envelope corners (e.g., α_t 0.2–0.35, σ̂ 0.6) run 16–27%
   inflation, direction still protective (lateness, never FG).
3. **M24 `progress_placement` × randomized schedule** — combinable,
   plumbing-verified, but unstudied (the M24/M25 sandbox studies predate
   the schedule); needs its own study before any production use of that
   opt-in path under the randomized flag.
4. **Prod schedule seed** — **RESOLVED (owner-directed, 2026-07-08)**:
   `App.tsx` now passes `labelSchedule: 'randomized'` with
   `scheduleSeed = seedFromString(trainingId)` (FNV-1a 64 over the
   server-issued session id — unique per session, and the served label
   sequence is reproducible from the session record for audit). The
   filter seed (7) is unchanged — the belief engine is untouched.
   Tests: FNV known vectors + distinctness/determinism in
   `labelSchedule.test.ts`; typecheck + prod build green.

## 7. Ledger

| Round | Phase | Outcome |
|---|---|---|
| 1 | P | This proposal. Frontier computed exactly; band gate shown slack (binding constraint is G vs imbalance control). Design pivot during frontier work: HARD run caps are too expensive (a hard force-flip after 3 same labels alone costs G ≈ 0.59 — every deterministic rule is a free win for the guesser); all interior pressure made SOFT (tilts, floored at q_min), hard forcing only at the |D| = 8 backstop. Shipped params (cap 8, d0 3, γ_d 0.04, r0 3, γ_r 0.04, q_min 0.25): G = 0.5485, worst-envelope inflation 9.8%. Second pivot: ONE label walk per task (not per mode) — the learner sees one per-task label stream, so skill sides, bias wants, and cert-probe wants all consult the same walk, making G-A exactly the predictability of what the learner observes. |
| 1 | E | `trainer/label_schedule.py` (primitive + exact chain analysis + schedule-general band formula) + flag-gated wiring in `trainer/policy.py` + `tests/test_trainer_label_schedule.py` (13 tests). Results: **G-A** closed form 0.5485; measured on policy streams (synthetic bank, pooled n=1139): optimal oracle 0.553, anti-repeat 0.541 — vs LEGACY anti-repeat 0.97–1.00 (the owner-observed exploit, quantified). **G-B** formula vs direct recurrence: 0.4–0.9% error (tolerance 20%); measured inflation 1.5–4.8%; alternation-limit override reproduces `bias_stationary_sd` bit-exactly. **G-D** legacy golden `test_trainer_g1_policy` green untouched; all fast trainer tests 76/76; slow G2 OC + closed-loop 5/5. G-C campaign + critique round in flight. |
| 1 | E | **G-C campaign 1** (real K=7 bank, `SimBankAdapter`, mixture stack, matched seeds, weak task 2): trainable soft learner (l_true 0.0/0.006, 2×10 seeds, budget 240) — delivered value 0.614 → 0.65 (**+6%, better**), terminal \|t_true\| 0.20 → 0.13 (**better**), median lateness 96 → 100 (+4%), declared-in-budget 10/10 → 9/10; static-below-cut FG scenario (l_true 0.15, 15 seeds) — **false graduation 0.40 → 0.33 (better)**. Verdict: no regression; FG and calibration improve; small lateness cost in the predicted protective direction. Sharp-margin slice (l_true 0.25 = 0.056 below cut, soft + static arms) added after review of scenario coverage — in flight. |
| 1 | E/R | **G-C sharp-margin slice** (l_true 0.25 = 0.056 below the task-2 cut): static-learner FG **0.60 → 0.60 (unchanged)** with false declarations LATER (median 172 → 188, protective); soft near-cut learner — dv 0.577 → 0.620 (better), terminal \|t_true\| 0.207 → 0.138 (better), but median lateness 97 → 122; paired analysis: mean +33 trials, t = 3.28, 8/9 seeds positive — a REAL effect, outside "within noise". **Diagnosis** (5-seed conjunct telemetry): the bias-band conjunct blocked 0/259 post-legacy-declaration trials (the alternation-derived t\* is NOT the cause); the block is entirely the skill-evidence gates, with served placement quality equal-or-better under randomized. **Round-1 R disposition: ACCEPT the near-cut lateness** under the repo's documented FG ≥ lateness priority (M17/M25): the extra ~25–30 trials at the margin are the price of an exploit-proof evidence stream, the gates' autocorrelation-aware demands are behaving correctly (not being fooled), and legacy's speed advantage is partly fictitious in production — it assumes a learner who does NOT exploit the 97–100%-predictable label sequence, and a marginal learner exploiting it corrupts declaration evidence in the FG direction. Owner reviews this tradeoff at the delivery pause. |
| 2 | E | Equalizer build: battery 17/17 + legacy golden green. Adversaries on policy streams (pooled n=1176): oracle 0.5527, **exact-difficulty adversary 0.5383** (round-1 debt queue: 0.568 — the echo channel is closed; difficulty features now HURT the adversary vs pure history, the self-annihilation property measured), anti-repeat 0.5372. Midpoint: persistent served-stream \|mean\| 0.035 (legacy 0.018; round-1 failure 0.151) — gated at ≤ 0.08; windowed(40) excursions 0.327 (mean-zero transients, priced into the stationary band; tripwire 0.45; physical gate = G-C terminal \|t_true\|). Walk commits proposal-gated: walk state == sched-trial imbalance exactly, clamp events 0. G-C campaign rerun (equalizer) + closed-loop matched pairs + slow-G2 regression in flight. |
| 2 | E | **G-C campaign, equalizer build** (matched seeds, real K=7 bank): FG 0.40 → 0.40 (moderate margin) and **0.60 → 0.47 (sharpest margin — better)**; delivered value +6–13% everywhere; terminal \|t_true\| 0.197–0.207 → 0.105–0.118 (**~45% better criterion calibration** in single-task streams); declared-in-budget 10/10 in all soft arms. **Cost: median lateness +25–39%** at the margin (96→120, 97→135) — larger than the round-1 ledger's +4–26%; accepted under the documented FG ≥ lateness priority, flagged prominently for owner review. Slow-G2 regression 5/5 after the `oc.py`/`run_closed_loop` passthrough (pinned defaults). Closed-loop 3-seed probe showed equal-or-better TRUE learning but noisier final exam verdicts (6→3, 7→3, 5→7 passes) with residual \|t_true\| ≈ 0.13–0.16 — hypothesis: short (~25-trial) per-task streams leave transient walk imbalance un-annealed at exam time; 8-seed per-task-telemetry batch running to resolve signal vs verdict noise. |
| 2 | E/R | **Closed-loop 8-seed batch** (`results/label_schedule_oc/closed_loop_pair8.json`): true learning IDENTICAL (mean final ℓ_true 0.835 legacy vs 0.837 randomized); final-exam verdict delta −1.38 ± 2.50 (paired t = −1.55, n.s.) inside instrument noise — the harness exam (110 q across 7 tasks, 100 particles) fails ℓ_true ≫ cut tasks in BOTH arms (legacy seed 17: 1/7 passes at ℓ̄ 0.85); residual short-stream \|t_true\| +0.024 ± 0.019 (n.s.), small vs the t\* band and vs the equalizer's −0.09 improvement in full-length streams. **Disposition: round 2 ACCEPTED**; the ready mitigation if live SAP monitoring ever shows short-session bias drift is a warmup profile (tighter anti-imbalance tilt for a task's first ~20 walk commits) — pre-analyzed here, deliberately NOT shipped on a non-significant signal (it would trade real early-stream predictability for an unproven cost). |
| 2 | E | Environment note: the tracked repo suite under `.venv` (3.11) shows 13 pre-existing failures traceable to the working tree's pre-session state (`pyproject.toml`/`requirements.txt` deleted before this work — packaging/CLI-wiring tests) plus 2 pre-existing `soft_smear` fixture-vs-interpreter mismatches in the untracked trainer suite; none are reachable from this change (zero tracked files modified). Gates for THIS change: trainer battery 17/17, legacy golden drift guard, slow G2 5/5 — all green under the trainer suite's de-facto interpreter. |
| 3 | E | Paired-bin build: battery 19/19 + goldens green (30 total with filter/sim guards). Adversaries, pooled n=2500/bank, analytic-anchored gate 0.5649: symmetric bank — oracle 0.5452, own-EWMA 0.4656, cluster 0.4784, anti-repeat 0.5336; **asymmetric M10-geometry bank (round-2's kill zone: 0.61–0.97) — oracle 0.5412, own-EWMA 0.4936, cluster 0.5104** — the difficulty channel is at CHANCE; the static device has no dynamic channel to attack. Midpoint (correct statistic — length-weighted pooled + long-stream means; the round-2-era per-short-stream max conflated SE≈0.09 sampling noise with the DC channel): long-stream (≥200 trials) \|mean\| ≤ 0.015 on BOTH banks (legacy 0.011–0.023; round-1 failure 0.151); within-bin want-side pick anchored to the bin's pooled median magnitude (side-blind, static). Zero paired-bin fallbacks in all runs. Non-vacuous retention test (forced retention serving; walk state == sched-imbalance exactly, clamped 0). G-C reruns + 16-pair closed loop in flight. |
| 3a | E/R | **G-C FG FAILURE, caught and reverted.** The round-3a within-bin pick anchored to the pooled median DISCARDED the F20 precision preference — serving migrated to the bank's s_sd-blurred overlap band, whose attenuated accuracy flatters a just-below-cut learner: sharp-margin FG 0.60 → **0.80**, false declarations 50 trials earlier. Lesson recorded: the mirror device's symmetry constraint and F20's precision constraint are BOTH load-bearing; round 3b satisfies both (precision-aware side-blind bin choice — a bin is only as discriminating as its less-precise side — plus within-bin E[w]). |
| 3b | E | Battery (17 tests) + 2 policy goldens green. Adversaries (pooled n=2500/bank, gate 0.5649): symmetric — oracle 0.5424, own-EWMA 0.4760, cluster 0.4995; asymmetric — oracle 0.5424, **own-EWMA 0.5164, cluster 0.5138** (chance; the leak stays closed with precision-aware bins). **G-C**: dv +25–38%; median lateness 96–97 → **69–70 (−28% — the round-2 cost flipped to a gain)**; terminal \|t_true\| 0.093–0.131 vs legacy 0.197–0.207; FG 0.40→0.47 and 0.60→0.67 (±1 seed per n=15 arm; pooled 15/30→17/30 n.s.) with earlier false declarations (172→127) — the pattern matches a time-rescaled declaration process at a fixed 240-trial window (evidence flows ~25% faster, so more of the same FG hazard is realized in-window); budget-equalized legacy check run to test exactly this. **Closed loop (16 pairs): all null** — pass −0.31 (t=−0.49), \|t_true\| +0.007 (t=+0.47, the round-2 +0.0246/p=0.042 effect GONE with the equalizer), ℓ_true +0.013 (t=+0.89). Verdict-deficit forensics (round-3 16-pair data): deficits ANTI-correlate with learning (r=−0.36) and failing tasks sit median +0.57 above cut — shared-bank exam-depletion artifact, absent in prod (disjoint test/train pools per D-INT). |
| 3b | E | **Time-rescaling check**: legacy sharp_static at the evidence-equivalent budget 325 (= 240·172/127; unit-consistent — static arms serve the weak task near-exclusively) shows FG **0.73** vs randomized 0.67 at 240; rescaling randomized declaration times by 172/127 reproduces legacy's 9/15 at the 240 horizon. Reading (critique-3-corrected): the sharp-margin FG hazard tracks evidence exposure (M25: information-bound), and per unit of evidence randomized FG is **indistinguishable from legacy — n = 15 cannot rank the arms**; the in-window +1-seed readings are consistent with the fixed-window artifact of ~25%-faster evidence flow, which also buys the −28% true-declaration lateness and +25–38% dv at UNCHANGED declared-time true skill (paired decl_ltrue deltas −0.010 to −0.017, t ≤ 0.97 — the "borrowed accuracy" confound refuted in-sim). Artifact: `results/label_schedule_oc/fg_budget_check.py`. The architectural FG filters (D33 confirmation lifecycle, D18 re-cert backstop) are untouched. |
| 3c | R | Critique round 3: NO-EXIT on a NEW finding class — the |s| channels are closed everywhere incl. the real bank (own-EWMA 0.522, cluster 0.500, oracle 0.530 real-bank measured), but **item precision s_sd is label-correlated in the BANK itself** (feedback-safe positives mean s_sd 0.276 vs negatives 0.698 on task 2; direction holds on 6/7 tasks, task 0 reversed, label-correlated on all 7), so a perceived-ambiguity guesser ("looks controversial → answer no") reads 0.940 on round-3b's served stream — **and 0.914 on the legacy trainer live today**; the channel survives 0.2-SD perceptual blurring (0.77–0.80) and its ~0.85–0.91 floor is set by bank content, not serving (precise negatives do not exist to serve). Round-3c disposition (same discipline as the warmup-profile call): **no device change** — s_sd-matched serving either blurs precision (the PROVEN round-3a FG mechanism) or collapses the pool; instead the channel is now MEASURED and PINNED (battery: isolating precision-skewed fixture + sd-cluster adversary, gated at legacy + 0.06; campaign artifact `realbank_adversaries.*` with both arms + legacy baseline), all overclaims corrected (§5c/docstring re-scoped to the \|s\| family; per-evidence FG claim softened), and the channel is an **OPEN OWNER ITEM** (§6). Sim-blindness caveat recorded: the G-C simulator responds to s_mean only, so cue-exploitation-inflated accuracy is invisible to every sim arm — equally for legacy. |
| 1 | E | Accepted deviation (recorded, owner-visible): `derived_t_star`/`bias_stationary_sd` remain alternation-derived while randomized serving's true band is ≤ ~10% wider — the mode/mastery gates therefore demand slightly tighter bias than the schedule's fluctuation floor. Direction is protective (possible slight lateness, no FG channel); G-C measures the cost. Out-of-envelope note: a harder-learning learner (α_t = 0.2, q_t = 0.04, σ̂ = 0.6) shows 16.5% closed-form inflation — outside the declared envelope; flagged for the owner. |
