# Integration plan: the cleaned learning engine in a mixed binary / n-way certification pipeline

Version 1.0 (2026-07-18). Audience: the integration team. Terminology is
deliberately generic: the seven certification domains are `domain1..domain7`
(the integration repo's own anonymized codes); `domain1` is the **binary
detection family** (present/absent judgment on a segment); `domain2..domain7`
are the **n-way identification family** (one pick among six signal classes
on a segment). Code identifiers and file paths are quoted verbatim where
needed to locate things; their meaning is described generically.

Everything below is grounded in a read-only survey of the integration repo
(trainer packages, data schemas, calibration configs, response data) and of
the learning-engine parent repo's decision log (cited as Dnn).

---

## 0. Executive summary — answers to the three questions

**Q1 (integration target): Standalone parallel algorithm — your pick — built
against the production trainer's EXISTING strategy seams.** The production
`trainer/` package already exposes everything an A/B needs as interfaces:
`run_closed_loop(policy_factory=...)`, the `ExposureLedger` ABC (shared
ledger = the natural join point), the bank-adapter `candidates(...)`
interface, filter classes with a common `step(s, y, y_star, s_sd, feedback)`
API, and an existing arm-comparison harness (`trainer/oc.py`). So option 3
and option 1 are not rivals: build the standalone adapter (option 3) so that
it plugs into those seams (option 1's promotion path) from day one. Nothing
in `trainer/` or the frozen `trainer_rd/` is edited; the adapter is a new
module alongside `learning-engine-cleaned/`. Option 2 is ruled out by the
integration repo's own rules (`trainer_rd/` is submission-frozen).

**Q2 (binary vs n-way): Native multiclass for the n-way family — your pick
is right — PLUS native binary for the detection family, with certification
semantics left untouched.** Three corrections to the option-2 box:

- `domain1` should use the validated **binary probit** family (the learning
  engine's original model), NOT multiclass with M=2. An M=2 softmax is
  over-parameterized (only two of its four state combinations are
  identified) and silently changes the lapse convention (λ/2-mixture vs the
  shared (1−2λ) design). The integration repo's own trainer already
  implements exactly this binary family with the same λ = 0.025 and the same
  soft-prediction-error dynamics — the two systems share a mathematical
  ancestor, which makes this the low-risk half of the work.
- `domain2..domain7` become ONE joint multiclass task (the D30 model:
  lapse-mixed softmax over per-class z, per-class criterion and skill,
  cross-domain transfer). This is not a re-model on faith: the multiclass
  model was **fitted and validated on this project's own external cohort**
  (see §2), where it recovered per-class learning curves, rejected
  response-driven update rules against a calibrated null, and supported the
  difficulty-gate law. The current pipeline binarizes the six-way pick
  ("did the candidate assert class k?"), which discards the identity of the
  wrong pick — about 1.6 bits per trial — and cannot represent the
  compensating per-class bias structure the multiclass fit shows is real.
  The full pick is already captured by the live system (`response_label` in
  the session trial logs), so this is a modeling upgrade, not a data change.
- **Verdict and cut-score semantics do NOT change.** The box's cost line
  ("redefines per-domain verdict & cut-score semantics; exam contract & bank
  schema change") is avoidable and should be avoided: certification stays
  with the existing exam instrument and its per-domain cuts (the D-INT-5
  boundary: the trainer never certifies). The multiclass belief is bridged
  to the per-domain cut scale by a **derived reduction map** (§3.4): for
  each domain, the belief's implied one-vs-rest discrimination on the bank's
  case-mix → the instrument's AUROC(ℓ) bijection → the ℓ scale the cuts are
  defined on. Derived quantity, no new constants, no contract break.

**Q3 (population artifact): the recommendation ("simulate/derive first")
needs a major amendment — a genuine real-data artifact for the n-way family
already exists.** The learning-engine parent repo's entire real-data program
(hierarchical population fit, learning-rate/floor estimates, update-rule
adjudication, gate support) was fitted on a de-identified copy of the
integration project's own external validation cohort (§2). So the honest
provenance ladder is: (a) n-way dynamics = refit of that already-validated
hierarchical model on the integration repo's own evidence axes
(calibration-stage, JAX-isolated, frozen artifact — the repo's existing
`core_mcmc` pattern); (b) state priors and cross-domain correlation = real
and already fitted in the integration repo (`Sigma_l_fitted_k7.npy`, expert
panel levels); (c) binary-family dynamics = the one place derivation from
the truth model is currently unavoidable (the binary contest data has no
response ordering, so no dynamics are fittable) — labeled as such, with a
caveat that the trainer's anchored rates come from one-shot per-user fits, a
method the parent repo measured to inflate rates; (d) a pre-registered refit
on longitudinal trainer data once the trainer runs live (the integration
repo's own Phase-3 intent), plus a readiness-recalibration holdout before
any certification-adjacent use. No quantity is invented; every field carries
a provenance label and a refit trigger.

---

## 1. Ground truth: what the integration pipeline is (generic)

- **Seven domains, two task families.** `domain1`: binary present/absent
  detection. `domain2..domain7`: six-way identification (one pick among six
  classes, with a catch-all class). Segments are candidates for exactly one
  family (disjoint source banks).
- **Per-domain latent state.** Every domain is scored as a one-dimensional
  SDT problem: skill ℓ = ln(1/σ) and criterion/offset, with the response
  model P = λ + (1−2λ)Φ(e^ℓ(s + ... − t)), λ = 0.025 — the SAME blessed
  lapse and the same latent parameterization as the learning engine
  (contract mapping: their ℓ = −u, their offset = −t; there is a known
  criterion sign-flip between the two coordinate systems, already handled by
  `trainer/conventions.py`; any adapter must route through it).
- **Evidence axes.** Each (segment, domain) carries a probit-scale evidence
  value `s_mean` with uncertainty `s_sd`, estimated from tiered annotator
  votes by a hierarchical SDT calibration — the same construct as the
  learning engine's crowd-anchored axes, estimated more completely
  (uncertainty included; the exam engine attenuates evidence by
  z /= sqrt(1 + (e^ℓ s_sd)²)).
- **Certification.** An adaptive exam resolves each domain PASS/FAIL by
  posterior pass-mass against per-domain cuts ℓ*_k (config block
  `ell_star_unified_v14` live; `v15` staged as the trainer's mastery
  target), with resolution gates (N_MIN, information ratio R*, α, Z) — all
  field-supplied design inputs in the sense of the learning engine's
  constant inventory.
- **The production trainer (`trainer/`)** runs a closed loop: exam → weak
  domains → per-domain SMC filters seeded from the exam posterior (×1.5
  variance inflation) → training with feedback on a curated high-confidence
  bank → fresh re-exam. Placement uses a fixed-width Gaussian difficulty
  weight centered at the ~85% point (mode multiplier ≈ 1.077, width
  ρ = 0.5); learning rates are fixed literals (α_t = 0.097, α_σ = 0.047)
  taken from one-shot per-user fits on the external cohort; the exposure
  ledger prevents item repeats across exam and training (count threshold +
  time floor); all six n-way domains are trained as independent one-vs-rest
  binary tasks.
- **Constraints.** `trainer_rd/` is frozen (read-only reference). JAX/NumPyro
  are banned from the runtime and live only in the calibration pipeline,
  which serializes frozen artifacts the runtime reads through a single path
  registry — the exact pattern this plan reuses. Determinism gates (single-
  thread BLAS, drift-guard tests) apply to anything runtime-adjacent; the
  learning engine's session code is numpy/scipy-only and satisfies this; its
  fitting code is JAX and must stay calibration-side.

## 2. The dataset identity and what it buys

The learning-engine parent repo's "real pilot" (699 participants, ~125.9k
responses on the 6-way task; ~167.5k on the binary task; 5,000 items each;
de-identified, classes recoded to generic signals with the same catch-all
collapse) is a reorganized copy of the integration project's **external
validation contest data**. Consequences:

1. The multiclass population fit (per-class learning rates 0.004–0.044,
   skill floors, one-factor cross-class transfer, lapse ≈ 0.010) is a
   REAL-DATA artifact for `domain2..domain7`, estimated on this project's
   own population and task family — not a simulation-derived stand-in.
2. The update-rule adjudication (soft prediction-error learning; response-
   driven and score-chasing rules rejected against a flexibility-calibrated
   null) was performed on this cohort, and it directly warns about the
   trainer's anchored rates: one-shot non-hierarchical fits systematically
   inflate learning rates (the measured flexibility bias), and the anchored
   α_σ = 0.047 is 2–5× the hierarchical per-class estimates. The A/B (§4)
   will adjudicate this on the trainer's own terms.
3. The difficulty-gate result (the ~85%-point placement law supported at
   +331 nats over difficulty-blind on this cohort) supports the trainer's
   existing placement center — and replaces its arbitrary width ρ = 0.5 with
   the parameter-free derived gate (§3.5).
4. The two axis constructions (parent repo: EB-shrunken leave-one-out crowd
   rates; integration repo: hierarchical-SDT vote posteriors `s_mean/s_sd`)
   are estimators of the same construct on the same votes. For integration,
   the repo's own axes are canonical; the dynamics artifact is REFIT on them
   (§5) so state coordinates match the production bank exactly.

## 3. Mathematical reconciliation (the core of the plan)

### 3.1 One learner state, two observation links

State per learner: criteria t (7,), log-skills u (7,), floors u_inf (7,) —
unchanged from the cleaned engine. What changes is the observation model,
which becomes per-family:

- `domain1` (binary link): P(yes) = λ + (1−2λ)Φ((s − t_1)/exp(u_1)) — the
  engine's original validated family (D1–D27), identical in form to the
  integration repo's own likelihood.
- `domain2..domain7` (n-way link): P(pick = m) = λ/6 + (1−λ)softmax(z)_m,
  z_m = (s_m − t_m)/exp(u_m) over the six per-class evidence values — the
  D30 model, validated on this cohort.

Dynamics are the SAME law in both families (this is what makes the mixed
model coherent): soft prediction error on the criterion, gated relaxation of
log-skill toward the floor, gate evaluated at each domain's own |z|. A
binary trial updates only its domain's state; an n-way trial updates the six
identification domains (each through its own z, per D30). Cross-domain
coupling enters through the population prior (one-factor transfer /
correlation matrix), not through trial-level cross-updates.

### 3.2 Why not M=2 softmax for the binary family

A two-channel softmax has four state parameters per domain of which only two
combinations enter the likelihood (the paired differences), and its lapse
mixture (λ/2 per channel) differs from the shared (1−2λ) design convention.
The binary probit family is exactly identified, matches the integration
repo's own model, and needs zero adaptation of cut-score semantics for
`domain1`.

### 3.3 Why native multiclass for the identification family

- **Information:** the full six-way pick carries log2(6) ≈ 2.58 bits; the
  one-vs-rest binarization keeps ≤ 1. Belief convergence and placement
  quality scale with per-trial information.
- **Structure the binarization cannot represent:** the multiclass fit on
  this cohort shows compensating per-class criteria (which wrong class
  absorbs errors) and cross-class transfer — both invisible to six
  independent binary filters, and both load-bearing for placement (serving
  the class whose margin-gain is largest requires knowing the confusion
  structure).
- **Training economics:** one n-way trial provides evidence about six
  domains simultaneously; the current per-domain scheduler pays one trial
  per domain per unit of evidence.
- **No data change:** the live exam already records the full pick; the
  trainer's per-trial record needs one additive field (the pick's class
  index) next to the existing binary response.

### 3.4 The verdict bridge: a derived reduction map (no new constants)

Certification stays with the existing instrument: per-domain cuts ℓ*_k with
the existing resolution gates. The multiclass belief is mapped onto that
scale per particle:

    R_k(state) = the one-vs-rest discrimination for domain k implied by the
    full response model on the certification case-mix: compute
    P(assert k | segment) for k-segments and non-k-segments across the bank
    (their binarized reduction applied to OUR predictive), form the implied
    AUROC, and invert the instrument's AUROC(ℓ) bijection to get ℓ_k.

Properties: fully derived (a deterministic functional of the belief and the
published case-mix — no fitted or tuned quantity); exact for `domain1`
(where the link IS binary, R is the identity); and it reproduces the
instrument's semantics by construction, because it applies the instrument's
own reduction to the richer model. Per-domain readiness then uses the
integration repo's own gates (pass-mass π_k = P(ℓ_k > ℓ*_k) with their
α/Z/mcse discipline) or, equivalently, the engine's double-η rule on the
reduced scale — both are constant-free given the field-supplied cuts.

Honesty discipline (pre-registered, the Signal-B doctrine): the reduction
map is a model-derived bridge, so before any certification-adjacent use, the
first ~50 trainees flagged ready by the new stack take the real exam and the
map's implied readiness is recalibrated against realized verdicts
(isotonic, population-estimated).

### 3.5 Domain extensibility as a first-class requirement (user directive)

The pipeline must accept ADDED domains — new binary detection tasks and/or
new n-way identification tasks (including additional independent n-way
groups) — without engine code changes. This hardens the mixed-link design
into a **registry-driven** one:

- **The domain registry is the single declarative input.** A domain spec
  is data: `(code, link)` where `link` is either `binary` or membership in
  a named n-way GROUP (a group = the set of classes competing in one
  softmax pick). The engine supports any number of binary domains and any
  number of disjoint n-way groups simultaneously; the current pipeline
  (one binary domain + one 6-way group) is just one registry value. The
  integration repo already has the seed of this (`trainer/domains.py`
  with its per-domain `family` field); the registry generalizes `family`
  from an enum of two hardcoded cases to (link type, group id).
- **State, artifact, bank, and exam are domain-indexed, never
  shape-hardcoded.** The belief state is (t, u, u_inf) over the
  registry's domains; items declare their domain (binary) or group + gold
  class (n-way); the exam case-mix, the reduction map, the readiness
  gates, and the placement loop all iterate the registry. No module may
  assume K=7, one group, or a fixed group size.
- **Cold-start doctrine for a new domain (no new constants).** A freshly
  added domain has no fitted rates or floors. The hierarchical population
  structure already answers this: the artifact carries the POPULATION
  HYPERPRIOR blocks (means and spreads of rates, floors, state priors,
  and the transfer factor), not just per-domain point values, so a new
  domain is instantiated by partial pooling — it enters with the
  population-level distribution and earns domain-specific values as data
  accumulate (exactly how the hierarchical fit treats a sparsely-observed
  domain). Its cut-score remains a field-supplied input, as for every
  domain. This makes "add a domain" a data/config operation with a
  defined uncertainty, not a modeling project.
- **Fit-side extensibility.** The population model already plates over
  domains; the mixed-link likelihood dispatches per trial on the
  registry. Adding a domain to the fit = adding columns/rows to the
  cohort arrays plus one registry entry; incremental refits reuse the
  hyperprior as the prior for the new domain.
- **Acceptance gate (added to P1):** construct a synthetic registry with
  one EXTRA binary domain and one EXTRA n-way group (e.g., a 3-way) that
  the shipped pipeline has never seen; generate a synthetic cohort, fit,
  build the artifact, run engine sessions, and produce readiness
  verdicts end to end **with zero engine code changes** — registry and
  data only. This gate, not documentation, is what certifies
  extensibility.

### 3.6 Constants audit (the parsimony ledger)

| quantity | today (incumbent trainer) | under this plan |
|---|---|---|
| lapse λ | 0.025 shared | unchanged (already shared by both systems) |
| placement law | Gaussian bump, center 1.077, width ρ = 0.5 (both fixed) | derived Wilson gate (parameter-free prior mean), learnable simplex gate where designed variation exists (D23/D42 machinery ships in the package) |
| learning rates | fixed literals α_t = 0.097, α_σ = 0.047 (one-shot fits) | hierarchical population estimates per class (real-data artifact, §5); rate dispersion carried per-particle |
| skill floors | per-domain expert-level points + mixture spread τ = 0.45 | population floor posterior with per-particle floors (same medicine, estimated not set); expert-panel levels remain the anchor |
| process noise q_t, q_σ | fixed 0.05 / 0.02 | carried as calibration inputs, bounded by the parent repo's identifiability result (below-resolution noise is provably invisible; do not tune it) |
| cuts ℓ*_k, resolution gates (N_MIN, R*, α, Z) | field-supplied | unchanged, authoritative (field inputs, not tuning constants) |
| exposure ledger (C, time floor) | operational design inputs | unchanged, shared by both arms |
| evidence uncertainty s_sd | closed-form attenuation in the exam engine | adopted into the learning engine's observation model (optional input, closed-form, no constant) |
| verdict bridge | n/a | derived reduction map (§3.4), zero constants; isotonic recalibration is population-estimated |

Net: two fixed constants are RETIRED into estimation (rates, gate width),
none added.

## 4. Integration architecture (Q1 in detail)

New module (suggested name `learning_engine_adapter/`, sited next to
`learning-engine-cleaned/` in the integration repo; both stay untracked
until the team commits them):

1. **Coordinate + naming adapter.** One thin translation layer between the
   exam engine's session output (`t_traj/l_traj/w_traj` final clouds,
   `task_codes`, `served_seg_ids`) and the learning package's handoff schema
   (`response_offset/log_sensitivity/weights/dimension_names`), routing the
   criterion sign-flip through the SAME convention module the production
   trainer uses (`trainer/conventions.py`). This is pure renaming + sign
   discipline; shapes already match, λ already matches.
2. **Bank adapter.** Consume the production bank through the existing
   `candidates(...)` interface (evidence vector per segment for the n-way
   family, scalar for the binary family, `s_sd`, gold, feedback-safe and
   margin filters) — no schema change; one additive field in the trial
   record for the full n-way pick.
3. **Belief/filter stack.** The mixed-link belief (§3.1) presented behind
   the SAME per-call API the incumbent filters use
   (`step(s, y, y_star, s_sd=..., feedback=...)` per trial, plus the
   reduction map for the `pass_mass`/`is_mastered` equivalents), so the
   orchestrator and A/B harness treat both stacks uniformly. The
   incumbent's `trainability` statistic (probability the learner's skill
   CEILING clears the cut — its plateau/honesty signal) is supported
   natively: the engine's per-particle floors give
   P(floor clears ℓ*_k) directly, where the incumbent approximates it
   with a mixture over a ceiling grid. Note the belief upgrade this
   carries: the incumbent's per-domain filters are independent by
   construction (its truth model assumes no cross-domain transfer),
   while the population fit on this project's own external cohort found
   real one-factor transfer — the joint belief lets strong domains
   inform weak ones at seeding and during training.
4. **Policy object.** The engine's margin-gain placement and readiness,
   exposed with the incumbent policy's `step()/record()` surface, sharing
   the ExposureLedger and cut-score loader.
5. **Runtime discipline.** Session-time code is numpy/scipy only (already
   true of the cleaned engine) and runs under the repo's single-thread BLAS
   determinism contract; JAX fitting lives exclusively in the calibration
   stage (§5), which serializes frozen artifacts + a provenance manifest —
   the repo's existing pattern for fitted quantities.
6. **A/B by construction.** Arm A = incumbent `TrainerPolicy` + per-domain
   filters; arm B = the adapter stack. Same bank, same shared ledger, same
   cuts, same exam. First on simulated learners through the existing
   arm-comparison harness (`trainer/oc.py` pattern and the closed-loop
   driver's `policy_factory` seam), then live behind participant-level
   assignment. Primary endpoints: questions-to-verdict-flip on weak domains,
   verdict honesty at re-exam, exposure consumption; the parent repo's
   dry-run lesson applies — verify the improvement window (bar inside the
   trainable range) before powering the comparison.

`trainer_rd/` is never touched. `trainer/` is not edited in phase 1–3; if
the A/B promotes arm B, promotion = registering the adapter's factory at the
orchestrator's existing seams (the option-1 endgame, then a team decision).

## 5. The population artifact: provenance ladder (Q3 in detail)

| artifact field | source | provenance label | refit trigger |
|---|---|---|---|
| n-way rates α_t,m, α_σ,m + dispersion | hierarchical refit of the validated multiclass model on the external cohort, using the integration repo's own `s_mean` axes (calibration-stage JAX, frozen `.npz` + manifest) | REAL (population, cross-learner) | longitudinal trainer data (their Phase 3) |
| n-way floors + dispersion | same refit; anchored against expert-panel skill levels | REAL | same |
| gate shape | Wilson prior mean; learnable simplex only where designed placement variation exists (the trainer's own serving creates some; banded arms would create more) | DERIVED prior + estimable | accumulation of designed variation |
| state priors (t, u) + cross-domain correlation | the integration repo's fitted covariance (`Sigma_l_fitted_k7.npy`) and cohort tier fits; both real | REAL (cross-sectional) | scheduled recalibration |
| binary-family rates/floors | truth-model constants (the anchored values), CAVEAT: one-shot-fit method measured to inflate rates; treat as upper bounds | DERIVED (assumed) | first longitudinal binary-family training data |
| process noise q | integration repo's values, carried not tuned | DESIGN INPUT | RT-channel instrumentation |
| lapse λ | 0.025 shared | DESIGN (blessed) | never |
| population hyperpriors (rate/floor/state-prior means + spreads, transfer) | carried in the artifact alongside per-domain values | REAL (from the same fits) | every refit; they are the cold-start prior for ADDED domains (§3.5) |
| cuts ℓ*, resolution gates | certification config | FIELD INPUT | the field's own recalibration process |
| readiness-bridge calibration | isotonic on first ~50 flagged trainees' real exam outcomes | REAL (pre-registered holdout) | per cohort |

This is option 1's speed with option 2's substance where it already exists,
and it makes every remaining assumption legible and expiring.

## 6. Phased workplan with acceptance gates

- **P0 — axis and contract verification (days).** Reproduce the mapping
  between the two evidence-axis constructions on the shared cohort
  (correlation, calibration curve); round-trip the coordinate adapter
  against a captured exam session cloud (gate: reconstruction ≤ 0.005, the
  parent repo's handoff standard); confirm full-pick availability in trainer
  serving.
- **P1 — mixed-link engine in the cleaned package (week).** Add the domain
  REGISTRY (§3.5) and the binary link to the belief/fit code (additive;
  default registry = one n-way group reproduces current behavior); the
  reduction map; `s_sd` attenuation; artifact schema extended with the
  population hyperprior blocks. Gates: parameter recovery on synthetic
  mixed cohorts (binary + n-way, planted truths); reduction map exactness
  on binary domains; the EXTENSIBILITY gate of §3.5 (an unseen extra
  binary domain + an unseen 3-way group run end to end via registry/data
  only); all existing package checks green.
- **P2 — artifact refit on canonical axes (days of compute,
  calibration-side).** Hierarchical multiclass fit on the external cohort
  with the integration repo's `s_mean` axes; binary-family artifact derived
  from truth-model constants with provenance labels. Gates: sampler honesty
  (R-hat/divergences/tree), state coordinates verified against the
  production bank, manifest complete.
- **P3 — adapter + simulated A/B (week).** Build the adapter surfaces (§4);
  run both arms on simulated learners spanning the incumbent's truth model
  AND the refit artifact's population (cross-truth robustness, both
  directions). Gates: arm B ≥ arm A on questions-to-verdict-flip at equal
  verdict honesty under BOTH truth models; exposure parity; determinism
  suite green.
- **P4 — live shadow (team decision).** Arm B runs read-only alongside live
  training sessions (consumes the stream, recommends, never serves),
  logging placement divergence and readiness deltas. Gate: readiness-bridge
  recalibration curve built from realized exam outcomes.
- **P5 — live A/B and promotion (team decision).** Participant-level
  assignment; pre-registered endpoints; promotion through the orchestrator
  seams if arm B wins.

## 7. Risks and open questions for the integration team

1. **Axis mapping quality (P0).** If the two evidence constructions diverge
   materially on shared segments, the refit must use the repo's axes end to
   end (planned anyway) and the parent-repo validation numbers become
   directional rather than quantitative.
2. **Criterion semantics across links.** Detection criterion (binary) and
   identification handicap (n-way) are different decision parameters; the
   plan keeps them separate per domain and lets the population fit couple
   them. Do not share t across links by assumption.
3. **Serving UI for n-way training.** Native multiclass training wants the
   six-way pick with true-class feedback in the TRAINING flow (the exam UI
   already has the control; the training flow currently asks one-vs-rest
   questions). If the training UI cannot present the full pick soon, the
   engine can still consume binarized training responses (with the known
   information loss) while the exam's full picks feed the belief — state
   this as a temporary degradation, not a design.
4. **Frozen-cut interplay.** The trainer targets the staged conservative
   cuts while the exam judges on the live ones; the reduction map must be
   evaluated against BOTH blocks in reporting so the A/B can't be gamed by
   the gap.
5. **Rate discrepancy adjudication.** The anchored rates vs the hierarchical
   estimates differ by 2–5×; P3's cross-truth design settles which matters
   operationally, but the team should know the direction of the known
   estimator bias in advance.
6. **Exposure-ledger interaction with joint training.** One n-way trial
   informs six domains; the ledger counts per (segment, domain) — confirm
   the intended accounting so arm B doesn't get spurious exposure credit.

## 8. What was deliberately left out

- Any edit to `trainer_rd/` (frozen) or to `trainer/` internals.
- Any re-derivation of cuts, resolution gates, or exam policy.
- Any new tuned constant. Two existing ones are retired into estimation;
  everything new is derived, field-supplied, or population-estimated with a
  pre-registered recalibration.
