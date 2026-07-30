# Live adaptive trainer

The production adaptive trainer is the server-side learning engine vendored at
`cortex_web/learning-engine-cleaned/`. It supersedes the retired root-level
prototype and Python port.

The verified non-secret production settings and release identity are recorded
in [`../cortex_web/docs/PRODUCTION_BASELINE.md`](../cortex_web/docs/PRODUCTION_BASELINE.md).
At the 2026-07-22 snapshot, training and the server engine are available to all
accounts, native n-way serving is on, practice-mode `ALPHA=0` is in effect, and
the allocator is Thompson with a 0.34 per-domain exposure-share cap. These are
serving controls, not certification-policy inputs.

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
- Shared waveform display pipeline:
  `cortex_web/apps/web/src/features/eeg/useEegDisplay.ts`

The browser does not run a fallback learning model. It requests the next item
from the authenticated server trainer, submits the response, and renders the
returned posterior snapshot. The server loads the frozen artifact, reconstructs
the learner belief from the certification and training ledgers, and supports
native n-way training. Training, spike-exam, and IIIC-exam waveforms share one
tested montage/filter pipeline; display filtering never enters the trainer or
certification belief state.

Known telemetry boundary: the browser's persisted trajectory point is built
from the snapshot visible at answer time and trails the server's post-answer
belief by one update. The next question is still selected from the updated
server state; only the displayed/history trajectory lags.

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
../../.venv/bin/python -m pytest -q \
  ../../learning-engine-cleaned/tests/test_package_smoke.py \
  test_engine_trainer.py

cd ../../apps/web
npx vitest run trainer/serverSession.test.ts \
  src/trainingController.test.ts src/trainingReveal.test.ts

cd ../../../trainer-policy
../.venv/bin/python -m pytest -q
```
