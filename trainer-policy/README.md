# CORTEX trainer policy

This directory is the root-level, Python-only reviewer representation of the
trainer currently deployed through `cortex_web`. The importable package uses
the unified Python name `trainer_policy`; the filesystem directory keeps the
requested publication-facing title `trainer-policy/`.

Production ownership does not move here. The live API continues to load:

- `cortex_web/learning-engine-cleaned/learning_engine/` for the learned
  population and particle-belief algorithms;
- `cortex_web/learning-engine-cleaned/adapter/le_adapter.py` for item selection,
  mastery, trainability, and response updates;
- `cortex_web/services/api/engine_trainer.py` for certification replay,
  native n-way serving, retention, interleaving, and restart reconstruction;
- `cortex_web/learning-engine-cleaned/artifacts/nway_dynamics_v1_1.json` for the
  frozen population fit.

The files here are an auditable equivalent with package-local imports and a
content-identical package-local copy of that frozen artifact. They do not import production code
at runtime. The equivalence tests compare this representation against the live
sources and fail when either side drifts.

## Package map

```text
trainer_policy/
  learnmodel.py       learned feedback gate and learner-state utilities
  multimodel.py       calibration-side hierarchical population model (lazy JAX)
  msengine.py         multi-signal particle belief and session engine
  mixedengine.py      mixed binary/n-way belief and session engine
  registry.py         domain/link registry and skill transformations
  policy.py           reusable LETrainerPolicy selection/update policy
  deployed.py         deployed serving policy and EngineSession/EngineManager
  api.py              unified TrainerPolicy/TrainerSession/TrainerManager names
  artifacts/          frozen nway_dynamics_v1_1 population artifact
docs/                 derivation, handoff contract, and validation report
tests/                source-drift and deterministic behavioral equivalence gates
```

The session-time import path is NumPy/SciPy only. JAX and NumPyro are imported
only when a calibration function from `multimodel.py` is requested.

## Use

From this directory:

```bash
python -m pip install -e .
python -m pytest -q
```

Public policy imports use one naming convention:

```python
from trainer_policy import MixedBelief, Registry
from trainer_policy.api import TrainerManager, TrainerPolicy, TrainerSession
```

`EngineSession` deliberately accepts the web host's database and bank
interfaces rather than owning storage. It is therefore deterministic decision
code, not a second database writer or a second deployed service.

## Synchronization rule

The `cortex_web` sources remain authoritative. Any production algorithm change
must be copied here in the same change and must update `PROVENANCE.json` only
when an intentional import/artifact-location rewrite changes. Run the parity
suite before review or release.
