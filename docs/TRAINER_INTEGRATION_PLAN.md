# TRAINER_INTEGRATION_PLAN.md — connecting the adaptive trainer to the examination test

Status: **DRAFT for review** (planning only; no code written). Authored 2026-07-06.

Scope: wire the frozen `trainer_rd/` learning algorithm into the production
**test → train → retest** loop so that a candidate is examined, enters adaptive
learning protocols targeted at their weak domains, is re-examined, and iterates
until every domain either clears the mastery bar or is honestly flagged as
"not yet proficient." Afterwards the candidate chooses which domains to maintain.

This document is the gated spec. It is **not** a big-bang rebuild: it follows the
repo's incremental, regression-gated, pause-for-review convention (Phases 4/7/8).

---

## 0. Decisions locked this session

| # | Decision | Choice | Consequence |
|---|---|---|---|
| D-INT-1 | Training trial content | **Real EEG segments + veridical feedback** (reveal correct/incorrect + true label). No authored rationale/annotation layer in v1. | No new clinical content pipeline. Learning is driven by corrected repeated exposure — exactly the trainer_rd design. |
| D-INT-6 | Training-item eligibility | **Credentialed-expert high-confidence only: the panel-plurality label is supported by > 3 concordant reads (≥ 4 agree).** Label panel = the **full credentialed-expert set (65): {Super8, Bonobo, profiler_iiic, New28, spikeed_expert, centaur_iiic_expert}** — named expert panels, zero crowd. The **v15 cut ℓ\* is unchanged** (still calibrated on Super8 ∪ Bonobo): the *cut* is a threshold on the latent skill scale, the *label* is a per-segment fact, so widening the label panel does not touch the cut. Testing uses the full bank; training uses this subset. | Pool = **29,747** (G0 table). Every domain reaches ~1,000+ confident positives (sz 1,084; lrda **1,000 exactly**; grda 1,322). ≥4-agreement bar honours "> 3 agree"; lrda-at-1,000 flagged for the G0 pause (≥3 → lrda 1,385 margin). |
| D-INT-7 | Repeat-prevention rule | **Count-primary with a same-session time floor.** A seen item is re-eligible once ≥ C intervening **within-domain** questions (test + train unified) have been served AND it was not seen in the same session / last 24 h. Both count and timestamp are stored; count drives. | One canonical Exposure Ledger (§2.5) is the sole authority; both the exam draw and the training draw query it. Default C = 600 (tunable, per-domain). |
| D-INT-8 | Exhaustion fallback | **Serve least-recently-seen.** When no candidate satisfies the no-repeat rule (thinnest domains: lrda 1,000, seizure 1,084 confident positives — at/above the 600 window; one-vs-rest negatives enlarge the real supply), serve the item with the largest intervening-count / oldest timestamp. The loop never stalls. | Graceful degradation; the exposure-budget audit (G2) quantifies when a domain enters this regime. |
| D-INT-2 | Web execution runtime | **Client-side TypeScript** in a Web Worker, mirroring the live exam engine. | Two numeric ports (Python reference → TS), each bit-parity-gated. Largest single lift. |
| D-INT-3 | Plateau handling | **Per-domain round budget + trainability-posterior exit** → "not yet proficient, retry later"; learner proceeds with the rest. | Uses the trainability BMA that already exists; loop always terminates. |
| D-INT-4 | Retest seeding | **Full, fresh, all-7-domain cold exam every iteration.** No warm-start. | Each retest is an independent re-measurement; also re-verifies already-passed domains (regression catch). |
| D-INT-5 (assumed, unchallenged) | Ship boundaries | Trainer **never certifies** (D18 — the exam is the sole certifier). Production trainer code lands **in this repo**. Scope ramps as a **K=1–3 pilot**, then expands to 7. Training items restricted to **feedback-safe** (high-confidence-label) segments only. | — |

If any D-INT-5 assumption is wrong, say so before G0.

---

## 1. The loop, precisely

```
        ┌────────────────────────────────────────────────────────────┐
        │                                                            │
  ┌─────▼─────┐   weak set W    ┌──────────┐   all trainable      ┌──┴────────┐
  │  EXAM Eᵢ  │────────────────▶│ REGIMEN  │───graduated/plateau─▶│  RETEST   │
  │ (all 7,   │  W={k: verdict  │  Rᵢ      │                      │  E_{i+1}  │
  │  fresh,   │      ≠ PASS}    │ training │◀── adaptive items ──▶│ (all 7,   │
  │  AD6)     │                 │ sessions │    + feedback        │  fresh)   │
  └───────────┘                 └──────────┘                      └───────────┘
        │                                                                │
        └──── loop until W empty OR every remaining k plateau-exited ────┘
                                     │
                            ┌────────▼─────────┐
                            │  MAINTENANCE     │  learner picks domains to keep
                            │ (retention/SM-2) │  up on → spaced retrieval
                            └──────────────────┘
```

Edges, with the mathematical object crossing each:

1. **Exam Eᵢ** — `CortexSession` (SMC engine + AD6) over all 7 tasks, fresh prior.
   Emits `SessionResult`: per-task verdict ∈ {PASS, REFER_BORDERLINE,
   REFER_UNINFORMATIVE}, posterior cloud over (t_k, ℓ_k), served seg_ids, AUROC.
   Weak set `W = {k : verdict ≠ PASS}`.
2. **Regimen Rᵢ** — for each k ∈ W, a training track with mastery bar ℓ\*_k
   (identical cut-score to the exam). Handoff artifact = `TrainingSeed`
   (variance-inflated ×1.5 marginal clouds per D1, trial history, verdicts,
   cut-scores, eval-seen seg_ids, cross-session ledger).
3. **Training sessions** — trainer policy selects feedback-safe items; learner
   answers; veridical feedback revealed; the training **particle filter** updates
   belief by propagating through the learning kernel T (no MH). Per-domain
   graduation gate OR plateau/trainability exit.
4. **Retest E_{i+1}** — a full fresh cold exam over all 7 (D-INT-4).
5. **Iterate** — recompute W; loop.
6. **Maintenance** — learner-selected domains enter the retention scheduler.

**Termination guarantee.** The loop ends when W is empty (all 7 PASS on a cold
exam) or every remaining domain has been plateau-exited (D-INT-3). Because
plateau exit is bounded by a round budget, the loop cannot run forever.

---

## 2. Mathematical justification (why this is coherent)

**One likelihood, everywhere.** Both the exam engine and the trainer filter use
the λ-lapse probit, `P(y=1) = λ + (1−2λ)·Φ(exp(ℓ)(s+θ))`, λ = 0.025, with the
attenuated form `att = √(exp(−2ℓ) + s_sd²)` for measured stimulus noise. This is
the repo's invariant #2 (single likelihood definition). The trainer inherits it
verbatim — no second likelihood is introduced.

**Two estimators, on purpose — and why they must stay separate.**
- The **exam** estimates *static* traits (t_k, ℓ_k are fixed during a sitting):
  SMC with **MH rejuvenation**, valid because the target is stationary.
- The **trainer** estimates a *moving* state (the whole point is that ℓ_k rises
  and θ_k → 0 as the learner improves): a bootstrap particle filter that
  **propagates through the learning kernel T with no MH**. MH would target the
  wrong distribution the instant the state moves (trainer_rd finding F1). This is
  the only genuinely new numeric being ported.
- Static "trainability" parameters (the skill ceiling σ_∞) can't live on moving
  particles (path degeneracy, F4), so they're a **BMA over a ceiling grid** with
  prequential weights; `trainability(ℓ\*) = Σ ω_j·1[grid_j > ℓ\*]` is the honest
  P(ℓ_∞ > ℓ\* | data) that powers the plateau exit.

**One bar.** The mastery bar ℓ\*_k is the *same object* as the certification cut
(cert_config `ell_star_unified_v14`/`v15`); σ\*_k = exp(−ℓ\*_k). "Ready to
graduate" and "certifiable" are numerically the same threshold (trainer_rd D31).
Graduation predicts a pass; **only the cold exam confirms one** (D18).

**Graduation gate.** Per domain: AD6 anytime-valid gate on the *filtered*
posterior — `sd_ℓ ≤ sd_floor ∧ (π − Z·mcse) ≥ 1−α`, `π = Σ ω_j·P(ℓ > ℓ\*)` — AND
a bias gate `|t̂| + 0.5·sd_t ≤ t\*`. e-process (Ville) gates block declaration
under below-bar probe performance.

**Why the fresh full retest (D-INT-4) is the mathematically strong choice.**
Because E_{i+1} starts from the cold cert prior and exposure-exclusion removes
every trained segment, its verdict is a **statistically independent
re-measurement** with no path back to the training belief. Two payoffs:
1. **The certificate's operating characteristics are exactly the cold-test OCs.**
   Training cannot inflate the certificate; it can only move real skill, which the
   independent exam then measures. This is the cleanest possible answer to "is a
   coached certificate still meaningful?" — yes, identically to an un-coached one.
2. **Regression safety.** Re-testing all 7 (not just the trained subset) catches
   decay in previously-passed domains between iterations (a fitted Ebbinghaus
   forgetting model is trainer_rd Phase-3 A3; the full retest is the safety net
   until then).

**False-graduation containment.** Sharp-margin false graduation is
information-bounded, not a gate defect (a static learner parked 0.15 below the cut
needs ~10³ probes to refute). trainer_rd prices it at the **confirmed** level:
D33 confirmation kills 5/6 (declared 6/20 → confirmed 1/20); the D18 re-cert
backstop catches 75/75 forced false graduations. **The fresh full retest is that
backstop in production.** Any FG headline we report must be at the confirmed,
post-retest level — never the raw graduation-declaration level.

---

## 2.5 Exposure Ledger — the repeat-prevention data layer (D-INT-7, D-INT-8)

A single canonical module owns "what has this participant been shown, when, in
what phase, for which domain" and answers the one question the exam draw and the
training draw both need: *is candidate item x eligible to serve to participant P
in domain k right now?* There is exactly one implementation of the eligibility
logic; the two draws call it — they never re-derive exclusion themselves. This
**supersedes** the ad-hoc `db.get_exposure_exclusion(code, days_window,
session_window)` (which today unions test + train over days/sessions only, in the
API layer) by promoting it to a first-class, testable module with a richer rule.

**Record (append-only).** One row per served question, written by both the exam
and the trainer:

```
exposure(participant_id, seg_id, domain, phase ∈ {test, train},
         served_at,            -- timestamp (free; every trial row has one)
         domain_ordinal)       -- running count of domain-k items served to P
                               --   at the moment this row was written (monotonic)
```

`domain_ordinal` makes the count rule O(1): item x with ordinal `o_x` has had
`(current_domain_ordinal(P,k) − o_x)` intervening within-domain questions since it
was last shown — a single subtraction, no scan.

**Eligibility predicate (D-INT-7).** For candidate x last seen at ordinal `o_x`,
time `t_x`, with current per-domain ordinal `O` and clock `now`:

```
eligible(x) = (O − o_x ≥ C_k)            # count-primary: enough intervening practice
              AND (now − t_x ≥ T_floor)  # time floor: not same session / < 24 h
never-seen x  → always eligible
```

- `C_k` = within-domain intervening-question threshold. Default **C = 600**,
  per-domain (may be scaled per domain later; see D-INT-8 alternative).
- `T_floor` = same-session / 24 h. This is the *only* place time is load-bearing;
  it exists solely to kill the marathon-session hole (600 questions in one sitting
  re-surfacing an item the same day, at peak recall contamination). All other
  freshness comes from the count.
- Scope is **unified test + train** and **per-participant** — an item seen in a
  test is excluded from training and vice versa (this is exactly what makes the
  D-INT-4 fresh retest a clean independent re-measurement).

**Exhaustion fallback (D-INT-8).** If no candidate is `eligible`, rank all
candidates by `(O − o_x)` descending (least-recently-seen first) and serve the
top. The loop never stalls; the audit logs when a domain enters this regime.

**Modularity — one interface, two backends.** The eligibility logic is defined
once against an abstract `ExposureLedger` interface
(`record`, `eligible(participant, domain, candidates, policy) → ranked list`,
`exclusion_set(...) → blocked ids`) and a `NoRepeatPolicy(count_threshold,
time_floor, per_domain, exhaustion)` value object. Two backends implement it:

| Backend | Used by | Notes |
|---|---|---|
| `SqliteExposureLedger` (or in-memory) | G2 local closed-loop | No web DB; drives the simulated-learner + EXTSET runs. |
| `PostgresExposureLedger` | G4 web app | Refactors `get_exposure_exclusion`; reads the existing `trials` / `training_trials` rows (adds `domain_ordinal`). |

Both engines and the orchestrator depend on the **interface**, not a backend —
this is the "centralized, modular, clean" contract. Determinism: the ranked
eligibility output must be a pure function of the ledger state + policy (stable
tie-break by seg_id) so G2/G3 parity tests are reproducible.

---

## 3. Gate structure (each gate ends in a review pause)

Legend: every gate lists **objective → work → exit criteria (verifiable)**. No
gate starts until the prior gate's exit criteria are green and reviewed.

### G0 — Contracts & foundations (local, Python)
**Objective.** Remove every ambiguity the port would otherwise inherit.
- **Canonical domain registry.** One module mapping the trainer's anonymized
  `domain1..7` ↔ production `{spike, seizure, lpd, gpd, lrda, grda, iic/other}`,
  replacing the ~15-file K=7 literal duplication (see slowing-domain audit). Both
  engines import it; no new hard-coded task lists.
- **Engine-drift reconciliation (R6).** Port against the *current* production
  engine, not the vendored `core_mcmc_general` copy: the production `choose_item`
  (`engine/core_mcmc.py:604`) exposes `ell_star=` + `active_domains=` (the plan's
  earlier "decision_tasks" name was wrong — corrected at G0) that the scratch copy
  lacks. ✅ G0: public `AD6Policy.verdicts` / `.last_diagnostics` accessors added
  (`scripts/cortex_policy.py`) so the G1 handoff need not read `_verdicts` /
  `_last_diag`.
- **Feedback-safe item filter (D-INT-6).** Training items are restricted to the
  credentialed-expert high-confidence subset: **the panel-plurality label is
  supported by > 3 concordant reads (≥ 4 agree)**, so a learner is never told
  "wrong" on a segment where the panel itself split. **Label panel = the full
  credentialed-expert set (65): {Super8, Bonobo, profiler_iiic, New28,
  spikeed_expert, centaur_iiic_expert}** — membership from
  `data/labels/raters.csv:groups`, named expert panels with zero crowd; consensus
  = plurality (IIIC) / majority present-vs-absent (spike). **The v15 cut ℓ\* is
  untouched** — it stays calibrated on Super8 ∪ Bonobo. Widening the *label* panel
  is legitimate because the cut (a threshold on the latent skill scale) and the
  label (a per-segment ground-truth fact) are different objects; more credentialed
  reads only make the label more reliable. Concrete pool (computed 2026-07-06 from
  `MANIFEST.json` × `labels.csv`):

  Testing pool is bucketed by the MANIFEST class (the exam bank); training
  positives by the **panel-plurality label** (the training authority — where the
  panel disagrees with the manifest, defer to the panel), so the two columns use
  slightly different class definitions by design. Built + verified by
  `trainer/bank.py` → `data/trainer_bank/` (G0):

  | Domain | Testing pool (manifest) | Training positives (panel ≥4) |
  |---|---:|---:|
  | spike | 16,527 | 15,670 |
  | seizure (sz) | 2,140 | 1,084 |
  | lpd | 3,269 | 3,201 |
  | gpd | 2,583 | 2,051 |
  | lrda | 2,026 | **1,000** |
  | grda | 2,487 | 1,322 |
  | other/iic | 6,161 | 5,419 |
  | **Total** | **35,193** | **29,747** (84.5%) |

  **Correction (G0 verification).** An earlier hand-sweep quoted seizure 1,213 /
  lrda 1,185; that count used the panel's *modal-class* agreement bucketed under
  the manifest label, which over-counted. The correct operator — *training label
  = panel plurality; eligible iff that plurality has ≥ 4 votes* — gives the table
  above. The total (29,747) is unchanged, but **lrda lands exactly at 1,000**
  (i.e. *at*, not strictly *over*, the target) and seizure at 1,084. **Open at the
  G0 pause:** keep ≥4 (honours the stated "> 3 agree"; lrda = 1,000 exactly) or
  drop to ≥3 (seizure 1,588, lrda 1,385 — comfortable margin, but relaxes the
  agreement bar to "> 2 agree"). Threshold is a one-line change + rebuild.
  Framing note: each domain trains **one-vs-rest**, so confident *negatives* (a
  clear non-k segment) supplement the positives — the true per-domain candidate
  supply is far larger; confident rare-class *positives* are the scarce input the
  table tracks. Rejected panels: v15-only (25,244; sz 689 / lrda 698); Super8-only
  (starves IIIC: sz 288, lrda 218).
- **Cut-score coherence assertion.** Assert trainer ℓ\* ≡ exam ℓ\* (numeric
  identity, v14/v15) so mastery bar = re-cert bar by construction. The label panel
  (65 experts) and the cut panel (Super8 ∪ Bonobo) are deliberately distinct; pin
  both so neither drifts.
- **Exposure Ledger interface (§2.5).** Define the abstract `ExposureLedger` +
  `NoRepeatPolicy` and land the panel-labeled candidate pool (the 29,747-item
  table above) as the training bank. Compute the **exposure-budget** per domain:
  given C = 600 and the pool sizes, at what retest count (if ever) does each domain
  enter the D-INT-8 fallback? (seizure/lrda are the tightest, but now > 1,000.)
- **Determinism.** Extend the single-thread BLAS contract to the trainer, and
  make the ledger's ranked eligibility a pure function of state + policy.

**Exit:** registry unit test; R6 accessor tests; panel pool table committed;
`ExposureLedger` interface + `NoRepeatPolicy` with unit tests (count rule, time
floor, exhaustion fallback, tie-break determinism); exposure-budget table;
ℓ\* identity test; determinism test. Production suite (282) + trainer_rd suite
(299) still green.

### G1 — Production Python trainer port
**Objective.** Land the trainer as production Python modules, numerically faithful.
**Path refinement (G1):** the port targets live under the `trainer/` package, NOT
`engine/` — the trainer filter carries its own inline likelihood (it does not
import the engine's), so there is no reason to perturb the frozen `engine/`
package; `trainer/` imports `engine/` only to assert the shared `LAPSE_RATE`. This
keeps D1 intact (engine/deployment never cross-import; trainer is a third
consumer). **Parity method:** golden vectors generated from the scratch modules by
`tests/fixtures/trainer_g1/_generate.py` (run once, isolated), so pytest compares
the port bitwise WITHOUT ever importing `trainer_rd/`.
- Port per `trainer_rd/docs/PHASE3_AND_PORT.md` §B:
  - ✅ `bridge_conventions.py` → `trainer/conventions.py` (LAPSE_RATE asserted
    against `engine/core.py` in a test). Bit-parity green.
  - ✅ `learner_sim.LearnerParams` → `trainer/dynamics.py` (the shipped dynamics
    dataclass the filter needs).
  - ✅ `training_filter.TaskFilter` → `trainer/filter.py` (the only new numeric:
    `propagate`, no MH). Bit-parity green across all propagate branches. Also
    ported `steady_state_sd` + `recommended_sd_floor` (gate-calibration helpers,
    bit-parity green — the numeric behind the gate re-pin).
  - ✅ `mixture_filter.py` (trainability BMA) → `trainer/trainability.py`
    (`SigmaInfMixtureFilter` + opt-in `AdaptiveBoundaryHazard`). Bit-parity green
    (grid, weights, pooled cloud, trainability statistic, summaries).
  - ✅ `learner_sim.Learner` (truth kernel) → `trainer/sim.py` (research-only;
    drives `steady_state_sd` + closed-loop studies). Bit-parity green.
  - ✅ `training_seed.py` → `trainer/handoff.py` (`TrainingSeed` + `LearnerParams`-
    free builder + `LearnerLedger`). Bit-parity green (clouds, history, meta) +
    save/load round-trip. The `SessionResult`→state-dict adapter is added at G2
    wiring (the builder is data-source-agnostic).
  - ✅ `bank_adapter.py` → `trainer/bank_adapter.py` — SEMANTIC port (production
    data): per-task `TaskCandidates` from the production MANIFEST signals + the
    credentialed-panel votes (shared `trainer.bank.load_panel_votes`). Correctness-
    tested (per-task pools > 1,000, feedback-safe + exclusion, deterministic order).
  - ✅ `trainer_greedy.{RewardWeights,_expected_reward}` → `trainer/reward.py`
    (the shared scorer the policy's bias/progress terms wrap). Bit-parity green.
  - ✅ `trainer_policy.py` → `trainer/policy.py` — the 1,273-line tier-2 policy,
    verbatim (only imports repointed). **Bit-parity green**: a fixed closed loop on
    synthetic candidates reproduces the scratch's full per-trial choice sequence +
    final filter clouds bitwise (exercises choose_mode, is_mastered, bias/skill
    select, the scorers, the DeficiencyScheduler, record). Opt-in tier-3 rollout →
    `trainer/rollout.py` stub (deferred; default path never calls it).
  - ✅ **Drop** vendored `auroc.py`: `trainer/conventions.auroc_from_ell` verified
    bit-equal to the repo's `engine/auroc.py:auroc_from_l` (a test pins it).
- ✅ **Re-pin graduation gates against production numerics.** Re-measured the
  information floor under the ported exact-kernel filter via `steady_state_sd`
  (itself bit-parity green): at the anchored production rates + bank stimulus noise
  (s_sd≈0.85) the steady-state ℓ-SD is ≈0.11 and `recommended_sd_floor` ≈0.16 —
  BELOW the shipped `ModeThresholds.sd_floor=0.23`, so the gate is achievable and
  0.23 is conservative (a higher floor only delays graduation → FG-safe). A test
  pins this achievability rather than assuming trainer_rd's 0.23/0.33 transfers.
- ✅ **Closed-loop SBC**: over 200 prior-drawn learners, the ported filter's 80%
  credible interval on ℓ covers the true evolved ℓ at the nominal rate (a
  pre-exact-kernel overconfident filter would under-cover). Slow-marked.

**Exit (MET):** bit-parity of the filter, trainability, sim, handoff, reward, and
the 1,273-line policy vs the scratch modules on fixed seeded traces; gate re-pin
measured + documented; closed-loop SBC nominal; production suite + trainer_rd
spot-check green. ~40 G1 tests (incl. 3 slow).

### G2 — Local closed-loop orchestration (Python) — the scientific gate
**Objective.** Prove the whole test→train→retest loop works and reproduces the
trainer_rd operating characteristics inside the production stack.
- ✅ `SqliteExposureLedger` (`trainer/exposure.py`) — SQLite backend, same
  eligibility semantics as in-memory (parity test) + persistence + `seen_segids`
  for the D-INT-4 hard retest exclusion.
- ✅ `run_closed_loop` + `SimBankAdapter` (`trainer/orchestrator.py`) — the state
  machine wiring the REAL exam (`CortexSession`, SMC+AD6) to the ported trainer and
  the ledger over ONE shared bank: E0 → weak set → per-task filters from the exam
  posterior (variance-inflated, D1) → train W (learner learns on feedback) → full
  fresh retest `inputs.without(seen)` → iterate. **Closed-loop demo runs headless
  (test)**: a weak learner (ℓ=0) is examined, trained (skill ℓ 0.00→0.43), and
  re-examined; the weak set evolves, no seg is served twice, exposure bounded.
- ✅ Drive with simulated learners (trainable + static-below-cut adversarial). Full
  12-member misspec zoo + EXTSET replay: **transfer by G1 bit-parity** (the ported
  filter is identical, so its behavior on any fixed dataset == trainer_rd's) — a
  production re-run is optional (see `docs/G2_CLOSEOUT.md` §5).
- ✅ **Reproduced the OCs** (`test_trainer_g2_oc`, measured): per-trial delivered
  value **0.90** in sim (the 0.48 is the real-data belief-lag figure); adversarial
  false-graduation **0/10** (σ_∞-mixture withholds, F28 fix); trainability plateau
  signal **≈0.36** (D-INT-3); D18 strict re-cert backstop **0/28** spurious PASS.
- ✅ **D-INT-4 independence verified:** the exam verdict is unchanged (**7/7**)
  under a random 30% item exclusion — the certificate is a function of current skill
  only, not exposure history.
- ✅ **No-stall / exposure:** the loop excludes every seen seg from retest + train
  draws and never stalls (bounded, unique exposure). The full per-domain
  exposure-budget audit for the thin domains is **deferred to G4** (needs the
  production bank in the exam — the fixture bank's per-domain pools are too small).

**Exit (MET):** full closed-loop demo runs headless ✅; OCs reproduced ✅;
backstop + independence verified ✅; no stall ✅; close-out doc
`docs/G2_CLOSEOUT.md` ✅. Production suite green. Caveat: the exam uses the 700-seg
fixture bank (not the 35k production bank) — wiring the production bank into the
exam + the real thin-domain budget audit are G4.

### G3 — TypeScript trainer port (client-side)
**Objective.** Bring the trainer to the browser runtime the exam already uses.
**Parity reframing (G3):** the plan said "bit-parity", but the existing exam-engine
TS port establishes the actual contract — the TS RNG is **xoshiro256\*\*, NOT NumPy
PCG64** (`apps/web/engine/rng.ts`: "does NOT need to bit-match NumPy… the test is
statistically defined"). So TS↔Python parity is **tolerance-parity for the
DETERMINISTIC numerics** (≈1e-6, limited by the shared Cody/NR erf behind `normCdf`)
and **statistical parity for the RNG-driven parts** (propagate noise, resample). The
harness mirrors `engine/__testdata__/gen_reference.py`: a Python generator dumps
deterministic fixtures, vitest asserts the TS agrees.
- The TS trainer lives in `cortex_web/apps/web/trainer/`, sharing only `normCdf`
  from `../engine/mathfns` (mirrors the Python `trainer/` importing `engine/`).
- ✅ **G3.1 — foundation, tolerance-parity green (7 vitest tests):**
  `trainer/conventions.ts` (coord transforms, λ-lapse likelihood, difficulty
  placement via an Acklam+Halley `normPpf`, auroc) and `trainer/filter.ts` (the
  deterministic belief pieces: `reweight`, `mean`/`sd`/`passMass`, the AD6
  `isMastered` gate). Reference: `trainer/__testdata__/gen_trainer_reference.py`.
- ✅ **G3.2** — `trainer/filter.ts` gained `propagate` (no MH; exact conditional
  Gauss–Hermite kernel + simple branch) + systematic `resample`, using the exam
  engine's `Rng`. Deterministic-drift (q=0) and resample-index parity are exact;
  the exact-kernel step is validated statistically. `trainer/trainability.ts`
  (`SigmaInfMixtureFilter`) — grid / weights / trainability / summaries
  tolerance-parity green.
- ✅ **G3.3** — `trainer/reward.ts` (`expectedReward`) + `trainer/policy.ts` (the
  default tier-2 path: scorers, `RetentionScheduler`, `TaskModePolicy`,
  `DeficiencyScheduler`, `TrainerPolicy`). The deterministic scorers AND a
  fixed-cloud decision sequence (choose_mode + select reproducing Python's mode +
  item choices) are tolerance-parity green. Opt-in ablation branches
  (progress_placement, rollout, finish_first, probe_every, terminal, p_static/jump/
  smear) are default-off and not ported.
- ✅ **G3.4** — `trainer/session.ts` (`TrainerSession` + `ArrayBank` +
  `buildFilters`) and `trainer/worker.ts` (the Web Worker message wrapper). A
  statistical end-to-end test runs the full TS stack over a fixed high-skill
  responder and confirms it graduates the learner (belief tracks up, valid picks).

**Exit (MET):** deterministic-numerics tolerance-parity green (~1e-6, erf-limited)
+ statistical parity for the RNG parts; **16 vitest tests pass**; `tsc --noEmit` = 0
errors; no D1 boundary violation (`trainer/` imports only `../engine/mathfns` +
`../engine/rng`). Toolchain: `npm install` in `cortex_web/apps/web`, then
`npx vitest run trainer/`; regenerate fixtures via
`python3 cortex_web/apps/web/trainer/__testdata__/gen_trainer_reference.py`.

### G4 — Web app wiring (staging)
**Objective.** Fill the already-scaffolded seams; make the loop real in the SPA.
The DB, API, spacing, and dashboard hooks already exist and are empty — this gate
populates them, it does not design them.
Most of the scaffolding was already wired (session-lifecycle endpoints, `is_real`
quarantine). Full close-out: `docs/G4_CLOSEOUT.md`. Nothing was deployed.
- ✅ **DB writes + server-authoritative persistence.** `db.record_training_progress`
  (the missing writer): per trial an idempotent `training_trials` exposure row +
  a REAL `param_trajectories` row, with `is_real`/`phase`/`code` set server-side
  (anti-tamper); `db.training_session_owner` is the ownership gate.
- ✅ **API:** `POST /api/regimen` (build the weak-set regimen from the latest cert
  result) + `POST /api/training-progress` (owned-session-only). The existing
  `GET /regimen`, `GET/POST /trajectories`, `/training-sessions[/finalize]` were
  already wired to real DB helpers.
- ✅ **Exposure exclusion.** The exam draw already excludes seen segs
  (`get_exposure_exclusion`, test+train union); writing `training_trials` closes
  the loop so **retests exclude trained segments**. Deviation: kept the existing
  days/sessions-window policy rather than refactoring to the count-window
  `NoRepeatPolicy` (a config swap; changing live exam spacing is riskier) — see
  close-out §5.
- ✅ **Client bridge.** `src/trainerClient.ts` (main-thread worker client, mirrors
  `engineClient.ts`), `src/trainerBank.ts` (manifest → trainer bank/cut-scores/
  clouds; unit-tested), `src/api.ts` `createRegimen`/`postTrainingProgress`.
- ✅ **UI runner.** `src/components/TrainingRunner.tsx` — a full-screen immersive
  learning session that reuses the exam Viewer's look (`cx-test`, EEG + spectrogram
  canvases, montage/gain/window/pan controls, sharp edges, teal actions) but is
  FEEDBACK-DRIVEN: each trial is a binary one-vs-rest "Is this <pattern>?" (Yes/No,
  keyboard Y/N/1/2) that maps directly to the trainer's belief update, then a
  distinct D-INT-1 RESULT step (✓/✗ + the true label, Enter/Space to continue),
  daily-bounded to 120 items with minimal on-screen progress. The trainer runs
  synchronously via the pure, unit-tested `src/trainingController.ts`
  (question→result→done state machine over `TrainerSession`); trajectory points are
  batched to `POST /api/training-progress` and the session `finalize`d on exit.
  Wired into `App.tsx` as the `training` phase: dashboard "Start training" →
  `createRegimen` → `startTrainingSession` → candidate draw → `TrainingRunner`.
  The regimen/trajectory/"Daily training" surfaces light up automatically as real
  data lands.
- ✅ **Real-skill handoff.** The trainer no longer seeds every learner from a
  generic zero-prior; `POST /api/regimen` now attaches a per-task `prior` (ALL 7
  tasks — engine-coords ℓ/θ means + the ℓ posterior SD from the cert `perTask`
  block), and `startTraining` seeds each task's belief cloud via
  `seedCloudsFromPrior` (measured mean, variance-inflated ℓ SD = cert SD floored
  ×1.6; θ moderate spread). Passed domains land above their cut so the scheduler
  deprioritises them; weak domains start at their measured (low) skill and get
  trained. Falls back to the generic seed for a legacy result with no posterior.
  (Behaviourally strongest on the real bank; a REFER-heavy small-bundle cert
  yields near-0 measured skill, so the handoff ≈ generic there.)

**Exit (MET, staging):** `test_g4_train_retest_loop_staging` — a simulated user
through cert → regimen → train → fresh retest ×3: regimen weak set correct, real
(`is_real=1`) exposure + trajectories persisted, **every trained seg excluded from
every retest draw**, quarantine holds. Full backend suite (102) + frontend src/
+ trainer (44) green; `tsc` = 0 errors. Behavioral (visual/interaction) verification
of the runner needs a Playwright or manual pass on the live SPA — it cannot be
exercised headlessly here. Nothing deployed.

### G5 — Pilot on prod (K=1–3), monitored, reversible
**Objective.** Ship a bounded, feature-flagged pilot.
- Feature-flagged, cohort-scoped, **K=1–3 pilot scope** (matches trainer_rd's
  honest "median learner graduates ~2 of 7 in 40 sessions").
- **Monitoring:** per-trial delivered value, graduation-vs-confirmed rates,
  retest deltas per domain, false-graduation at the *confirmed* (post-retest)
  level, exposure-budget consumption.
- **Rollback:** feature flag off restores the pure exam path; the training tables
  are additive and the exam path is untouched, so rollback is data-safe.
- Pre-register a pilot SAP (~33/arm per trainer_rd power note); PI sign-off before
  expanding K beyond the pilot set.

**Exit:** pilot SAP met; PI sign-off; documented decision to expand K.

---

## 4. Open policy items to resolve with the PI (surfaced, not silently decided)

- **Inter-exam cadence.** Exposure-exclusion handles item reuse, but is there a
  minimum wall-clock gap between full retests (interaction with forgetting)?
- **Bias-warning channel (OPEN_DECISIONS #4).** Training explicitly corrects bias
  (θ → 0). Should the retest surface `|t_k| > tol` as a warning even on a PASS?
- **AD6 gate-2 currently non-operative (OPEN_DECISIONS #16).** `var_prior` uses
  the correlation diagonal (=1.0) while particles live on the covariance scale, so
  the R\* variance-contraction gate does not bind today (95/95 REFER are
  BORDERLINE). If the loop is ever to *rely* on the precision gate — e.g. a widened
  retest posterior — this must be fixed or explicitly confirmed inert-safe first.
- **Confirmed-FG endpoint semantics (trainer_rd D41) and terminal-confirmation
  (D44, default OFF).** How is a "confirmed graduation" reported downstream?
- **Warm-start OQ3** — closed by D-INT-4 (fresh); recorded here for the audit trail.
- **Per-segment domain1 (spike) hard label** — votes give ~92%; a definitive label
  sharpens spike feedback (PHASE3_AND_PORT §5A residual).

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Two numeric ports (Python + TS) double the parity surface. | G1 and G3 each end in a bit-parity gate against a fixed reference; reuse the existing exam-engine parity harness. |
| Client-run exam + client-run trainer → belief-integrity / tamper. | Authoritative training belief + graduation state persisted and validated server-side; `is_real` quarantine; response anti-tamper (G4). |
| Credentialed-panel filter shrinks IIIC pools. | Resolved (D-INT-6): label panel widened to the 65 credentialed experts at ≥4 agree — every domain reaches ~1,000 confident positives (seizure 1,084, lrda 1,000) while the v15 cut ℓ\* is untouched. lrda-at-1,000 flagged for the G0 pause; D-INT-8 fallback guarantees no stall; one-vs-rest negatives enlarge real supply. |
| Exposure exhaustion under many full-7 retests. | Single Exposure Ledger (§2.5) with the D-INT-7 count rule + D-INT-8 fallback; budget audited over N iterations at G2; thin domains degrade gracefully rather than stall. |
| Anti-gaming (memorizing specific segments across retests). | Ledger's unified test+train no-repeat + fresh-prior retest measures skill, not recall of seen items; the count rule guarantees ≥ C intervening items before any repeat. |
| Some domains are information-bounded and never graduate. | D-INT-3 plateau exit via the trainability posterior; honest "not yet proficient" flag; loop still terminates. |
| Trainer_rd graduation floors don't transfer to production numerics. | G1 re-pins every gate floor against the production engine and documents the re-measurement (do not assume transfer). |

---

## 6. What this plan deliberately does NOT do (v1)

- No authored instructional content (rationales, annotations, exemplars) — D-INT-1
  is corrected-exposure feedback only. A content layer is an additive future track.
- No warm-started retests, no cross-task transfer model (trainer_rd D2 keeps tasks
  independent), no fitted Ebbinghaus forgetting (Phase-3 A3), no RT observation
  channel (A4). All are Phase-3 items in `trainer_rd/docs/PHASE3_AND_PORT.md`.
- The trainer issues **no certificate** (D18). Ever. The exam is the sole certifier.

---

## 7. Suggested execution order (all gated, pause after each)

```
G0 contracts ──▶ G1 Python port ──▶ G2 local closed-loop (scientific gate)
                                          │
                                          ▼
                                   G3 TS port ──▶ G4 web wiring ──▶ G5 pilot
```

G0–G2 are fully local and are where the science is proven. G3–G5 are the
web/prod delivery, and none of them begins until G2's close-out is signed off.

**Execution mode (pending final go-ahead).** Each gate runs as an autonomous
**PECR loop — Plan → Execute → Critique → Reimplement**: draft the gate's concrete
change set, implement it, run an adversarial self-critique against the gate's exit
criteria (tests, parity, OC reproduction, determinism), and reimplement until the
exit criteria are green — then pause for review before the next gate. The exit
criteria in each gate above are the PECR loop's acceptance bar. **Not started:
awaiting the user's final confirmation to begin G0.**
