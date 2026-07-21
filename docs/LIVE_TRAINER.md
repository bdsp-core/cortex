# Live adaptive trainer

The production adaptive trainer is the server-side learning engine vendored at
`cortex_web/learning-engine-cleaned/`. It supersedes the retired root-level
prototype and Python port.

## Runtime ownership

- Model and session engine: `cortex_web/learning-engine-cleaned/learning_engine/`
- Frozen population artifact:
  `cortex_web/learning-engine-cleaned/artifacts/nway_dynamics_v1_1.json`
- Host adapter: `cortex_web/learning-engine-cleaned/adapter/le_adapter.py`
- API decision service: `cortex_web/services/api/engine_trainer.py`
- API routes: `cortex_web/services/api/routers/training_engine.py`
- Browser adapter: `cortex_web/apps/web/trainer/serverSession.ts`
- Browser orchestration: `cortex_web/apps/web/src/trainingSetup.ts` and
  `cortex_web/apps/web/src/trainingController.ts`

The browser does not run a fallback learning model. It requests the next item
from the authenticated server trainer, submits the response, and renders the
returned posterior snapshot. The server loads the frozen artifact, reconstructs
the learner belief from the certification and training ledgers, and supports
native n-way training.

## Reviewer entry points

Start with:

1. `cortex_web/learning-engine-cleaned/README.md`
2. `cortex_web/learning-engine-cleaned/docs/learning_engine_derivation.md`
3. `cortex_web/learning-engine-cleaned/docs/handoff_contract.md`
4. `cortex_web/learning-engine-cleaned/docs/V1_V2_VALIDATION_REPORT.md`
5. `cortex_web/services/api/engine_trainer.py`

The package is self-contained for methodological review. Historical trainer
prototypes, ports, parity fixtures, and their method-specific tests are not part
of the live implementation.

## Root-level Python representation

`trainer-policy/` presents the same deployed algorithm under the unified
import name `trainer_policy`. It contains byte-exact copies of the core model,
the content-identical frozen `nway_dynamics_v1_1` artifact, and import/path-only rewrites of the
policy and deployed session service. Its parity suite compares deterministic
choices, particle states, and deployed session behavior against the production
sources. The web paths above remain runtime-authoritative; the root package is
the modular reviewer surface, not a second deployed entry point.

## Focused verification

```bash
cd cortex_web/services/api
../../../.venv/bin/python -m pytest -q \
  ../../learning-engine-cleaned/tests/test_package_smoke.py \
  test_engine_trainer.py

cd ../../apps/web
npx vitest run trainer/serverSession.test.ts \
  src/trainingController.test.ts src/trainingReveal.test.ts

cd ../../../trainer-policy
../.venv/bin/python -m pytest -q
```
