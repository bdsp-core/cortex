# Unified label tables

These tables are the canonical research corpus for rater × segment × label
observations across spike/IED and IIIC sources. They are research inputs, not
web-serving dependencies. The complete derivation and source history is in
[`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md).

## Current table inventory

Counts below describe the files committed for reviewer release:

| File | Rows | Granularity |
|---|---:|---|
| `raters.csv` | 5,306 | canonical/source-scoped rater identities |
| `segments.csv` | 95,327 | source segments and retrieval metadata |
| `labels.csv` | 2,115,793 | `(seg_id, rater_id, label_type, value, source_dataset)` |
| `segment_labels.csv` | 95,327 | per-segment aggregate labels |
| `segment_signals.csv` | 89,138 | fitted per-segment signal estimates |
| `datasets.csv` | 5 | registered source datasets |

Use the files themselves and their release hashes as authority; do not encode
these counts into algorithms.

## Foreign keys and identity rules

- `labels.seg_id` → `segments.seg_id`
- `labels.rater_id` → `raters.rater_id`
- every source record retains `source_dataset`
- source-native person IDs must be reconciled through an explicit governed
  crosswalk, never through coincident numeric values or fuzzy names
- unresolved source identities remain distinct and source-scoped

`raters.csv` contains potentially identifying clinician information. It and
any name/alias crosswalk require an explicit publication review. Integer IDs
are not anonymous when released with a joinable identity table.

## Registered sources

`datasets.csv` is the machine-readable registry. The current corpus includes:

- `sn1_combined_v2`
- `sparcnet50K`
- `pd_rda_profiler`
- `iiic_crowdsourcing:kong2025`
- `centaur_iiic_expert`

`labels.csv` also contains `centaur_2025_iiic` and `centaur_2025_ied` source
namespaces, but `datasets.csv` does not yet contain their registry rows. The
Centaur case-ID crosswalk and crowd reads are present; the registry mismatch and
the 5,000 IED segments missing from `segment_signals.csv` are explicit release
reconciliation exceptions, not silently upgraded counts.

## Label taxonomy

| `label_type` | Value form |
|---|---|
| `spike` | binary/string-normalized spike decision |
| `pattern_class` | seizure, LPD, GPD, LRDA, GRDA, Other/native source classes |
| `frequency_hz` | numeric string |
| `laterality` | source categorical value |
| `spatial_channels` | JSON list |
| `spatial_extent` | numeric/categorical source value |
| `discharge_times` | JSON event times |
| `wave_times` | JSON event times |

Model predictions, when deliberately included, must use a versioned
`model:<name>` rater identity and a distinct label type so calibration code
cannot silently treat them as human annotations.

## EEG boundary

The tables contain retrieval metadata, not a public license to distribute the
underlying EEG. Raw HDF5/MAT banks remain external and access-controlled. Some
source rows do not have a resolved waveform location. A missing URI must remain
missing; do not infer paths or treat a source filename as proof that two rows
represent the same physical window.

## Known reconciliation limits

- Some Centaur user identities cannot be linked safely to existing canonical
  people; they remain distinct source-scoped identities.
- Cross-dataset physical-window deduplication is incomplete. Matching segment
  identifiers or filenames are insufficient by themselves.
- IED waveform recovery is incomplete even though IED reads and fitted signal
  estimates are present.
- Source-specific native class vocabularies are preserved in provenance even
  where the deployed six-way IIIC mapping collapses classes into `other`.

## Rebuild policy

The original one-off corpus builders are not all part of the shareable tree.
Do not claim the tables can be regenerated from a missing script. Any future
rebuild must run from governed upstream snapshots, preserve source-native IDs,
emit a new provenance/hash manifest, and pass row-count, foreign-key,
crosswalk, and duplicate/conflict validation before replacement.
