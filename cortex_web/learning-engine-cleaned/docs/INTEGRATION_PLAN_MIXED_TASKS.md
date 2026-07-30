# Live CORTEX integration

This package is the learning model used by the live server-side trainer. The
older repository-root trainer prototype and its Python port were retired after
the server-driven path became the sole trainer on 2026-07-17.

The verified 2026-07-22 host posture uses native n-way serving, practice-mode
`ALPHA=0`, Thompson domain allocation, and a 0.34 exposure-share cap. These are
host serving settings, not frozen artifact fields; see
[`../../docs/PRODUCTION_BASELINE.md`](../../docs/PRODUCTION_BASELINE.md).

The active request path is:

1. `cortex_web/apps/web/src/trainingSetup.ts` constructs a
   `ServerTrainerSession`.
2. `cortex_web/apps/web/trainer/serverSession.ts` calls the authenticated
   training-engine API.
3. `cortex_web/services/api/routers/training_engine.py` owns the HTTP boundary.
4. `cortex_web/services/api/engine_trainer.py` loads this package and the frozen
   `artifacts/nway_dynamics_v1_1.json` population artifact.

The live service imports `MixedBelief`, `Registry`, and `LETrainerPolicy` from
this package. It seeds each sitting by replaying the participant's latest
certification stream, serves native n-way items where configured, updates the
belief after feedback, and persists responses through the API ledger.

No repository-root trainer package is a runtime or fallback dependency. The
focused integration checks are:

```bash
cd cortex_web/services/api
../../.venv/bin/python -m pytest -q \
  ../../learning-engine-cleaned/tests/test_package_smoke.py \
  test_engine_trainer.py

cd ../../apps/web
npx vitest run trainer/serverSession.test.ts \
  src/trainingController.test.ts src/trainingReveal.test.ts
```

For the model, artifact schema, operating rules, and standalone demonstration,
see the package [`README.md`](../README.md),
[`learning_engine_derivation.md`](learning_engine_derivation.md), and
[`handoff_contract.md`](handoff_contract.md).
