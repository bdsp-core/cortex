# Python integration reference for `cortex_web`

This directory is the Python integration reference and authorized improvement
boundary for the current TypeScript CORTEX testing engine. Concrete stopping
policies are independently testable sibling packages in `../ad6-policy/` and
`../precision-policy/`, sharing the contract in `../termination-policy/`. No
file in `cortex_web` is changed by this Python phase.

The production termination backstop is a **60-question per-domain cap**. A
domain that remains unresolved after 60 of its own questions is removed from
selection and finalized as `REFER`; the old global 500-question sweep is not
the production stopping policy.

The approved cut-independent design is implemented as the opt-in
`PrecisionPolicy`. `scripts/cortex_policy.py` preserves the historical import
surface, and `scripts/session_controller.py::build_cortex_session` connects the
standalone packages to the local engine. AD6 remains the default and rollback
policy. See `PRECISION_POLICY_IMPLEMENTATION.md` and `../POLICY_LAYOUT.md` for
the exact contract.

The earlier `c=1.40`, five-item profile used interval half-width and is
explicitly superseded as evidence. Stopping uses the stricter point-centred
radius `max(mean-L, U-mean)`. A verified 840-reader served-bank frontier and
100-reader CRN-paired composed sweep selected the opt-in K=7 development
profile `c=[1.50,1.30,1.35,1.30,1.30,1.30,1.25]` with three items per signal
band. Its independently qualified MCSE guard, protocol, owner frontier, and
decision report live under `calibration/precision_frontier/` and
`calibration/PRECISION_FRONTIER_PROTOCOL.md`. A later 270-session factorial
stopping-time screen found material extreme-skill tail undercoverage but no
false certifications. That limitation is accepted and disclosed. Development
is closed: no calibration repair, tuning, or new stopping family is part of
the frozen profile. AD6 remains the default.

## Included runtime mapping

| TypeScript engine area | Python reference source |
|---|---|
| `mathfns.ts`, `likelihood.ts`, `prior.ts`, `particles.ts`, `choose_item.ts` | `engine/core_mcmc.py` |
| `auroc.ts` | `engine/auroc.py` |
| `policy.ts` | `../ad6-policy/`, `../precision-policy/`, and K=7 adapter in `scripts/cortex_policy_k7.py` |
| `session.ts`, `advance.ts`, `worker.ts` | `scripts/session_controller.py` |
| `types.ts`/manifest engine inputs | `scripts/cortex_engine_inputs_k7.py` |
| Browser-bundle construction | `cortex_web/apps/web/scripts/prepare_web_bundle.py` |

The corresponding frozen inputs are also local:

- `Sigma_l_fitted_k7.npy` — K=7 `Corr_l` and `Corr_t`.
- `calibration/cert_config*.yaml` — v13/v14 and v15 cut scores.
- `data/labels/segment_signals.csv` — per-segment signal estimates.
- `data/eeg_bank.h5` — the 700-segment desktop/reference bank.
- `cortex_web/.../v1.6-k7-35k/manifest.json` — the exact current web engine
  manifest. EEG and spectrogram display blobs are deliberately not duplicated;
  the particle engine requires only the manifest.

Python full sessions now use the served v15 numerical profile: 1,200 particles,
`Corr_l` for skill, and the distinct `Corr_t` for bias. The earlier reference
used 600 particles and incorrectly reused `Corr_l` for the bias block.

`PROVENANCE.json` pins unchanged copies to their source SHA-256. For an
explicitly authorized reference improvement, it records both the original
source hash and the current reference hash so provenance is retained without
claiming that the improved file remains byte-identical to its source.

## What “parity” means

`parity/cortex_web_parity.test.ts` runs the actual modules under
`../cortex_web/apps/web/engine` against reference output generated only from
this isolated Python tree. It covers:

- v1.6 release inputs (`Corr_l`, `Corr_t`, v15 cuts, task order, particle count,
  and per-domain cap);
- hierarchical prior density with separate v15 bias correlation;
- ten sequential likelihood updates with signal uncertainty;
- A-optimal item selection;
- ESS, posterior means, AD6 diagnostics/verdicts; and
- posterior AUROC means and intervals.

Discrete outputs—selected task, segment, and verdict—must match exactly.
Continuous outputs use the existing port tolerance of `1e-6`; aggregate ESS
uses `1e-5`. The browser implements its documented ~`1e-7` normal-CDF
approximation, while Python uses SciPy, so bit-for-bit floating-point identity
is neither expected nor claimed.

Full randomly initialized sessions are also not byte-identical: Python uses
NumPy's generator and the browser uses xoshiro256**. The deterministic fixed-
cloud trajectory removes that irrelevant RNG difference and tests the shared
engine computation directly. Existing browser session tests separately cover
deterministic browser replay, resampling, rejuvenation, and termination.

## Run the complete gate

From this directory:

```bash
./run_tests.sh
```

The gate:

1. verifies all isolated hashes, preserved source hashes, and that originals
   remain unchanged;
2. tests the standalone contract, AD6, and PrecisionPolicy packages, then
   imports and builds the integrated Python K=7 stack;
3. regenerates both Python reference artifacts and requires exact structure and
   discrete values, with platform-level floating-point drift below `1e-12`;
4. runs the extended Python↔TypeScript parity test; and
5. runs the existing `cortex_web` math, pipeline, replay, termination, and
   v15 drift tests.

It also exercises fail-closed transactional updates, reversible precision
completion, the hard 60/domain budget, floor-aware selection, band reservation,
raw-response telemetry, and the additive policy interface. No `cortex_web`
file is changed by these tests.

## Deliberate exclusions

The 28 GB production HDF5 bank, 106 GB source spectrogram bank, raw EEG source,
calibration fitters, simulation studies, desktop GUI, storage, and research
variants are upstream or downstream of the TypeScript particle engine. They
are not needed to execute or validate the port and are therefore not copied.
The current 35k manifest is retained so the served engine configuration remains
auditable without duplicating the 11 GB browser display bundle.

Make changes here only with explicit authorization. Keep `cortex_web` unchanged,
update provenance and tests for every authorized reference change, regenerate
parity evidence, and transfer changes to TypeScript only after explicit approval.
