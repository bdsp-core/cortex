# Adoption roadmap: the cleaned learning engine as the production trainer

Version 1.0 (2026-07-19). Decision context: the team has chosen the
cleaned learning engine as the forward approach — one generative model,
one belief, every decision (placement, pacing, stopping, readiness)
derived from it, every number derived / population-estimated /
field-supplied — over incremental patching of the incumbent trainer.

Current status (2026-07-21): the server-side engine is the sole deployed
trainer. The phase markers below preserve the adoption decision and validation
record; they are not a second deployment checklist. Runtime ownership and
focused verification commands are maintained in `../README.md` and
`../../../docs/LIVE_TRAINER.md`.
The 2026-07-22 host overlay is native n-way practice mode with Thompson domain
allocation and a 0.34 exposure-share cap; it is documented in
`../../docs/PRODUCTION_BASELINE.md`. Those serving controls do not refit the
artifact or alter the certification engine.
This document details what must be accomplished to (1) keep that
cohesion intact, (2) measurably improve on the incumbent, (3) be
acceptable to the current testing/learning implementation, and (4) stay
cheap to extend when domains are added. Status markers: ✅ done and
verified (P0–P3), 🔧 build, 🧪 run/validate, 👥 needs the team.

Evidence base: P0 (axes r = 0.875–0.954 on ~5k shared segments/class;
contract round-trip ≤ 0.0025), P1 (registry-driven mixed-link engine;
extensibility gate passed with unseen domains), P2 (real-data artifact
on the host's own axes, R̂ 1.05, 0 divergences), P3 (adapter drove the
host's real closed loop; fixture-scale A/B surfaced the three gaps in
§2).

---

## 0. The one-paragraph structure

The engine replaces the incumbent's per-domain filter stack + mode
policies + deficiency scheduler + mixture-trainability + retention
heuristics with a single joint belief over all registry domains and
four derived decisions: SEED from the exam posterior (contract adapter,
sign-flip in one place), PLACE by expected margin gain at the learned
gate peak, STOP-OR-CONTINUE by attainability (floor posterior) and
readiness (pass-mass on the field's cuts through the derived reduction
map), and RE-EXAM when readiness says the learner would pass — not on a
fixed round cadence. Certification stays entirely with the existing
exam and its cuts (the trainer never certifies).

## 1. What is already in place (do not rebuild)

- ✅ Registry-driven mixed-link engine (`learning_engine/registry.py`,
  `mixedengine.py`): any number of binary domains + disjoint n-way
  groups; binary probit and group softmax under ONE learning law;
  per-trial link dispatch in the fit (`dmask`/`is_bin`); nothing
  hardcodes a domain count or group size. Extensibility proven by the
  unseen-domain gate, not asserted.
- ✅ The derived reduction map (projection onto the certification
  instrument's model family; exactly −u on binary domains) and the
  AUROC(ℓ) bijection — verdict/cut semantics preserved with zero new
  constants.
- ✅ The real-data population artifact on the host's own evidence axes
  (`artifacts/nway_dynamics_v1.json`) with the hyperprior block for
  cold-starting added domains, provenance manifest and caveats.
- ✅ The contract adapter (`adapter/le_adapter.py`): JAX-free,
  incumbent-parity surfaces, host design inputs passed as arguments;
  verified against the live exam engine end to end.
- ✅ Runtime split: session code numpy/scipy-only; fitting JAX-isolated
  calibration-side, frozen-artifact handoff (the host's own pattern).

## 2. What must be accomplished — the algorithm (the P3 lessons)

P3's fixture-scale A/B failed its economy gate and exposed exactly what
"effectively improving on the incumbent" requires. These three items
are the core build; each replaces an incumbent heuristic with a derived
counterpart, keeping the constants ledger clean.

### 2.1 🔧 Attainability before placement (the honest-fraction doctrine)

Measured: under the fitted (real-rate) truth, NEITHER trainer's PASS
verdicts were honest at the tested budget (incumbent 0.29) — training
cannot beat exam noise when the cut is unattainable for the learner.
And the adapter's cold futility gate (α on a wide floor posterior)
barely throttled, while the incumbent's ceiling-mixture did real work.

Build: per-domain attainability = P(floor clears the cut) from the
belief's per-particle floors, with the floor PRIOR anchored where the
host's data actually is — the expert-panel skill levels per domain
(field-supplied anchors) with population-fitted dispersion (the P2
artifact for the n-way family; the binary domain from the same panel
until §2.4 data exists). A domain is served only while attainable at
the declared risk and unmastered; an unattainable domain is REPORTED as
such (the engine says "this bar is not reachable for this learner"
instead of spending exposure). No new constant: the risk level is the
field's α; the anchors are the field's panels.

Validate: on the production bank, the attainability rule must (a)
terminate every session without bank exhaustion, and (b) reproduce or
beat the incumbent's exposure consumption at equal-or-better verdict
honesty — the P3 gates, rerun at scale (§4 V1).

### 2.2 🔧 Readiness-triggered re-exams (the engine's core economy win)

The incumbent re-exams on a fixed round cadence (train 150, re-exam,
repeat), burning fresh exam items each round (the fresh-retest rule).
The engine's native structure is: train until the belief's pass-mass
(through the reduction map, on the field's cuts, with the field's
resolution gates) says the learner would PASS the weak domains — then
re-exam once. Fewer exams, same certification authority. This is the
validated questions-to-honest-certification value proposition,
transplanted.

Build: the session loop in the adapter (train → readiness → single
re-exam → verdicts), with the plateau signals logged every question
(never controlling), and the readiness bridge recalibration holdout
pre-registered (first ~50 flagged learners' real verdicts fit the
isotonic correction before any certification-adjacent claim).

### 2.3 🔧 Exposure economy as a first-class constraint

Measured: the finite bank is a real budget (the fixture exhausted; the
production bank is 35k items but per-domain feedback-safe pools are
much smaller, and the ledger adds count/time thresholds).

Build: the serving loop terminates on the triad (all domains mastered ∨
unattainable ∨ no eligible candidates), consumes the host's
`ExposureLedger.eligible(...)` contract directly (count threshold, time
floor, least-recently-seen fallback are HOST design inputs), and the
placement objective already maximizes value per item (margin gain).
Add the accounting decision the team must make: 👥 how a native n-way
trial charges the ledger (one segment shown, six domains informed —
recommend charging the segment once, per its display, since exposure is
about item familiarity, not information).

### 2.4 🔧 Binary-family dynamics from real data (retiring the last
derived block)

The binary domain's rates/floors are truth-model-derived (labeled in
the manifest). Once the trainer runs live, its own ledgers (the host's
TrainingSeed already carries per-trial wall-clock, feedback flags, seg
ids) are exactly the longitudinal feedback data the hierarchical fit
needs. Pre-register: refit the full mixed-link population model on the
first cohort of real training ledgers; the binary block moves from
DERIVED to REAL; per the standing rule, model-class changes (e.g.
between-session forgetting terms) only enter by beating the shipped
model held-out by the pre-set margin.

### 2.5 🔧/👥 Native n-way training serving (UI-gated upgrade)

The exam already captures the full pick; the training flow binarizes.
Until the training UI presents the full pick with true-class feedback,
the engine trains on binarized responses (built and working) — a known
information loss (~1.6 bits/trial), not a design change. When the UI
lands: flip the bank adapter to emit n-way items; the belief, fit,
reduction map, and readiness need NO changes (P1's extensibility gate
covered n-way groups from day one).

### 2.6 🔧 Replay-based seeding from the raw test sequence (team input)

The trainer receives, per completed test: the per-domain skill/bias
posterior AND the exact question sequence (segment ids, asked domains,
full responses). The posterior is already consumed (the contract
seeding). The sequence is worth more than exposure exclusion: the
testing engine's posterior was computed from BINARIZED responses,
while the log records the full n-way pick — replaying the raw sequence
through the engine's own mixed-link observation model (reweight-only
`observe`; correct because certification tests give no feedback and
the learning law is feedback-gated) recovers the confusion structure
the summary discarded, ~1.6 bits/trial on n-way items, at zero extra
questions. Rule that keeps it honest: NEVER combine replay with cloud
seeding — the same evidence would enter twice. Seed from the
population prior and replay (preferred; any future model improvement
automatically re-extracts more from the same logs), or seed from the
cloud (fallback when raw logs are unavailable). Validate: replay-seeded
beliefs must match or beat cloud-seeded beliefs on state fidelity and
downstream training economy (V1 includes both seeding modes).

## 3. What must be accomplished — acceptability to the host repo

1. 🔧 **Determinism + drift guards, host-style.** Seeded bitwise
   reproducibility test for `MixedBelief`/`MixedSessionEngine`/adapter
   under the single-thread BLAS regime; drift-guard tests pinning
   default behavior (their contributor convention: every new kwarg
   ships with a guard). The adapter is already numpy-only, so the BLAS
   contract holds by construction — the tests make it enforceable.
2. 🔧 **Production data adapters.** Replace the harness's sim-bank
   shim with their real `BankAdapter`/`TaskCandidates` (feedback-safe
   + margin filters are host design inputs), the `ExposureLedger`
   backends, and the TrainingSeed handoff (both the live SessionResult
   path and the serialized artifact path).
3. 🔧 **Case-mix standardization for the reduction map.** The
   projection needs a declared evaluation case-mix per n-way domain;
   recommend the certification bank's per-domain candidate pools
   (already curated and versioned) so the reduced ℓ is stable and
   auditable. One config entry, no constant.
4. 🔧 **Packaging + CI.** The package as an installable module in
   their repo; requirements extended (+h5py, +pyyaml for host
   loaders); the P1 gates (recovery, reduction exactness, unseen-domain
   end-to-end, demo) ported into their `tests/` as permanent
   regressions — the extensibility gate especially, so "easy to add
   domains" stays true under maintenance.
5. 👥 **Team decisions to schedule:** ledger accounting for n-way
   trials (§2.3); which cut block the trainer targets (they stage a
   conservative training block vs the live exam block — keep both
   evaluated, as their current trainer does); the recalibration
   holdout size for the readiness bridge; retirement plan for the
   incumbent trainer package after V1–V3 pass.

## 4. The validation sequence (what to RUN, in order)

- 🧪 **V1 — production-scale A/B rerun** (completes P3 honestly):
  production bank, n ≥ 40 learners per cell, both truth models,
  attainability-first + readiness-triggered rounds. Gates (pre-set,
  same spirit as P3): engine total questions ≤ incumbent at
  equal-or-better PASS honesty under BOTH truths; zero bank
  exhaustions; unattainable domains reported not served.
- 🧪 **V2 — attainability audit**: on learners sampled from the P2
  artifact, the predicted honest-fraction per domain vs realized —
  the D36 audit transplanted to their cuts; this is also the evidence
  the team needs for the standing question of whether the current
  all-domain PASS requirement is reachable by their population.
- 🧪 **V3 — determinism/drift suite green** in their repo (§3.1).
- 🧪 **V4 — live shadow** (their P4): the engine consumes live
  sessions read-only, logs placement divergence, readiness deltas,
  and the plateau signals; the readiness-bridge isotonic fits on
  realized verdicts.
- 🧪 **V5 — staged rollout** (their P5): participant-level assignment,
  pre-registered endpoints, incumbent retired when V5 confirms V1.

## 5. The add-a-domain runbook (the future-proofing contract)

Adding a domain (binary or a class in a new/existing n-way group)
must remain a data/config operation. The runbook, all steps already
supported and gate-tested:

1. Registry: add `(code, "binary")` or `(code, <group>)` — one line of
   config.
2. Bank: provide items carrying the new domain's evidence column(s)
   (the host's existing vote-calibration pipeline produces these).
3. Cut: the field supplies ℓ* for the new domain (its standard-setting
   process, unchanged).
4. Artifact: nothing required on day one — the engine cold-starts the
   domain from the hyperprior block (partial pooling), and the
   attainability/readiness machinery works immediately at honestly
   wider uncertainty.
5. First refit: when the new domain has training data, the scheduled
   calibration refit gives it fitted rates/floors; the artifact
   manifest records the provenance transition.
6. Regression: the CI extensibility test (an unseen binary + an unseen
   3-way group, end to end) guarantees steps 1–5 keep working.

## 6. The constants ledger after adoption (the cohesion check)

Retired with the incumbent: fixed gate width, anchored literal rates,
fixed mode thresholds/deficiency heuristics/retention bins as decision
rules (their roles are absorbed by derived quantities). Remaining
numbers, by category — DESIGN: λ = 0.025 (shared, blessed). FIELD:
cuts ℓ*, resolution gates (N_MIN, R*, α, Z), exam spec, exposure
thresholds (C, time floor), expert-panel floor anchors.
POPULATION-ESTIMATED: rates, floors + dispersion, state priors,
transfer, gate shape (Wilson prior, learnable where designed variation
exists), signal thresholds/calibrations (isotonic, holdout-fitted).
DERIVED: the reduction map, placement targets, readiness statistics,
attainability. Anything proposed outside these four categories is a
red flag in review — that is the discipline that keeps this approach
from becoming the next knot.
