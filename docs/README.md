# Documentation map

The code, frozen artifacts, and executable tests are the source of truth. This
index separates current reviewer entry points from dated scientific records so
historical plans are not mistaken for production instructions.

## Current reviewer entry points

- [`../README.md`](../README.md) — repository purpose, package map, and setup.
- [`METHODS.md`](METHODS.md) — historical Python mathematical derivation; use
  the web documentation below for deployed runtime ownership.
- [`../cortex_web/README.md`](../cortex_web/README.md) — canonical deployed
  application, qualification commands, and runtime controls.
- [`../cortex_web/docs/PRODUCTION_BASELINE.md`](../cortex_web/docs/PRODUCTION_BASELINE.md)
  — verified live release, configuration assumptions, telemetry semantics,
  API surface, and rollback anchors.
- [`../cortex_web/docs/README.md`](../cortex_web/docs/README.md) — current web
  architecture, operations, and product/data documentation.
- [`LIVE_TRAINER.md`](LIVE_TRAINER.md) — deployed server trainer and the
  reviewer-facing `trainer-policy` representation.
- [`../POLICY_LAYOUT.md`](../POLICY_LAYOUT.md) — Python stopping-policy and
  trainer-policy ownership boundaries.
- [`../data/DATA_PROVENANCE.md`](../data/DATA_PROVENANCE.md) — source and
  artifact provenance.
- [`INVARIANT_AUDIT.md`](INVARIANT_AUDIT.md) — frozen scientific invariants.
- [`OPEN_DECISIONS.md`](OPEN_DECISIONS.md) — unresolved scientific and
  publication decisions.

## Reference and validation records

- [`EEG_SEGMENT_BANK_REPORT.md`](EEG_SEGMENT_BANK_REPORT.md) — reconciled bank
  inventory, selection rules, and release caveats.
- [`LABELED_EEG_RATER_CENSUS.md`](LABELED_EEG_RATER_CENSUS.md) — canonical and
  source-question rater-count census with its governed-data boundary.
- [`DATA_UNIFICATION_ANALYSIS.md`](DATA_UNIFICATION_ANALYSIS.md) — historical
  analysis of a joint calibration proposal that was not promoted.
- [`DEPLOYMENT_INTEGRATION.md`](DEPLOYMENT_INTEGRATION.md) — historical Python
  deployment reference, not the live web runtime.
- [`NATURE_MEDICINE_AUDIT.md`](NATURE_MEDICINE_AUDIT.md)
- [`PHASE7_CLOSEOUT.md`](PHASE7_CLOSEOUT.md) and its supporting Phase-7
  validation reports.
- [`PHASE9_CLOSEOUT.md`](PHASE9_CLOSEOUT.md) — K=7 rebuild record; sections
  describing the retired native desktop application are historical only.

## Historical records

The following files preserve dated evidence or decisions. Their commands,
test counts, version numbers, and “next” sections describe the repository at
the stated date and are not current release instructions:

- `MERGE_SOURCE_MANIFEST.md`
- `PHASE7_PAPER1_FIGURES.md`
- `PHASE7_REPLAY_DESIGN.md`
- `PHASE7_REPLAY_HEADLINE.md`
- `PHASE7_TIER2_OC.md`
- `PHASE7_VALIDATION_K7.md`
- `SIM_BANK_QUALITY_SWEEP.md`
- `SIM_V1_2_0_REPORT.md`
- `SIM_V1_3_5_CONSEC_CAP.md`
- `AD6_RESOLUTION.md`

Obsolete pre-execution plans, native-desktop integration instructions,
assistant-specific context, and internal reviewer prompts were removed. Their
history remains available through Git and does not belong in the reviewer
navigation surface.
