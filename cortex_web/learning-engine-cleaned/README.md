# CORTEX learning engine

This package is the runtime-authoritative learning model used by the deployed
server trainer. It is vendored inside `cortex_web/` so a web release contains
the model, adapter, frozen population artifact, tests, and methodological
documentation as one unit.

The browser does not run this model and has no local trainer fallback. It sends
authenticated start/record requests; `cortex_web/services/api/engine_trainer.py`
owns the deployed session state and uses this package for belief updates and
item selection.

## Deployed path

1. `learning_engine.registry.Registry` defines one binary domain plus the
   native n-way group from the served manifest.
2. `learning_engine.mixedengine.MixedBelief` maintains the joint particle
   belief over learner criterion, skill/noise, rates, and floors.
3. `adapter.le_adapter.LETrainerPolicy` derives pass mass, trainability,
   mastery, allocation, and gate-targeted placement from that belief.
4. `services/api/engine_trainer.py` adds certification replay, candidate-bank
   construction, native n-way serving, retention, interleaving/exposure guards,
   persistence recovery, and the HTTP-facing session manager.
5. `apps/web/trainer/serverSession.ts` adapts the API responses to the browser
   training controller.

The frozen runtime artifact is `artifacts/nway_dynamics_v1_1.json`. Session-time
imports are NumPy/SciPy only; JAX/NumPyro are calibration-only lazy imports.

## Package map

```text
learning_engine/
  learnmodel.py     learned feedback-gate utilities
  multimodel.py     calibration-side hierarchical population model
  msengine.py       original multi-signal model and research utilities
  mixedengine.py    deployed mixed binary/native-n-way particle belief
  registry.py       domain/link registry and certification projection
adapter/
  le_adapter.py     deployed selection and belief-update policy
artifacts/
  nway_dynamics_v1_1.json  frozen deployed population fit
docs/
  handoff_contract.md      current certification/training API contract
  learning_engine_derivation.md  mathematical derivation
  V1_V2_VALIDATION_REPORT.md     fixed-budget validation evidence
  ADOPTION_ROADMAP.md            historical adoption decision record
  CORTEX_WEB_PHASE0_NOTES.md     historical rollout notes
tests/
  test_package_smoke.py          package/import/behavior smoke checks
demo_end_to_end.py               synthetic research demonstration
```

`msengine.py` and the synthetic demo remain useful for reproducing the method's
research lineage. They are not alternate production session managers.

## Runtime model

The artifact carries population estimates for learning rates, process noise,
the learned feedback gate, initial state, rate/floor dispersion, transfer, and
the conditional skill-floor model. The registry makes domain count and link
type data-driven. The leading task uses a binary probit link; the remaining
tasks share a native categorical softmax observation.

At session start, the host builds a fresh expanded belief from the frozen
artifact and replay-seeds it from the participant's latest finalized
certification stream. Replay is observation-only because certification gives
no feedback. The cloud is then contracted to the deployed particle count.
If no certification result exists, the population prior is retained.

During training, feedback advances the learned dynamics. The policy targets
items near the fitted learning gate, reports per-domain attainability, and
derives mastery from pass mass plus the host-supplied uncertainty/risk gates.
The host supplies certification targets from the same served bundle manifest;
the trainer cannot certify or alter those targets.

For the exact task/pick encoding, HTTP fields, single-writer ledger rule, and
restart behavior, see `docs/handoff_contract.md`.

## Quick verification

From this directory with Python 3.11 and the repository's web API environment:

```bash
python -m pytest -q tests/test_package_smoke.py
```

The deployed integration checks run from the API and web workspaces:

```bash
cd ../services/api
python -m pytest -q ../../learning-engine-cleaned/tests/test_package_smoke.py \
  test_engine_trainer.py

cd ../../apps/web
npx vitest run trainer/serverSession.test.ts \
  src/trainingController.test.ts src/trainingReveal.test.ts
```

The root `trainer-policy/` package is a reviewer-oriented Python mirror. Its
tests compare source content, frozen artifacts, deterministic decisions, and
session behavior with this production package.

## Calibration and data boundary

The deployed service never fits a population model. Calibration functions in
`multimodel.py` require the optional JAX/NumPyro stack and governed response
data; they emit a reviewed artifact that must be frozen before deployment. The
real pilot rows and identity crosswalk are not distributed. The included
synthetic demo illustrates fit-to-session mechanics but is not production
qualification evidence.

## Non-negotiable invariants

- Certification replay and a supplied posterior are mutually exclusive.
- Task order and cuts come from the served manifest.
- The server may select only media present in the submitted session pool.
- Native n-way gold is the segment's true class, not necessarily the domain
  currently prioritized for learning.
- The API decision service is not a second response-ledger writer.
- Trainer changes do not modify the certification likelihood, selector,
  particle profile, PrecisionPolicy, stopping behavior, or verdict logic.

See `docs/INTEGRATION_PLAN_MIXED_TASKS.md` for current runtime ownership and
`docs/V1_V2_VALIDATION_REPORT.md` for the stated validation claims and limits.
