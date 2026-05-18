# Unified label tables

A single canonical schema for all rater × segment × label-type observations
across the spike (Centaur / SpikeEd / bonobo / SN1) and IIIC
(sparcnet50K / pd-rda-profiler) corpora.

Built by `scripts/build_unified_labels.py`. Re-run that script to rebuild
from the upstream sources.

## The five unified files

| File | Rows | Size | Granularity |
|---|---:|---:|---|
| **`raters.csv`** | 2,682 | 242 KB | one row per canonical rater |
| **`segments.csv`** | 84,555 | 18 MB | one row per unique EEG segment, with absolute S3 URI to the source EEG file |
| **`labels.csv`** | 1,303,201 | 45 MB | long-form: `(seg_id, rater_id, label_type, value, source_dataset)` |
| **`segment_labels.csv`** | 84,555 | 21 MB | wide-form per-segment aggregates (spike vote counts, IIIC plurality, etc.) |
| **`datasets.csv`** | 3 | 1 KB | registry of source datasets |

## Foreign keys

- `labels.seg_id` → `segments.seg_id`
- `labels.rater_id` → `raters.rater_id`
- `segments.s3_uri` → absolute S3 path to the source EEG file (provisional — see "S3 paths" below)

## Source datasets

See `datasets.csv` for details. The three sources are:

| dataset_id | label type(s) | segments | raters |
|---|---|---:|---:|
| `sn1_combined_v2` | spike (binary) | 20,521 | 2,574 |
| `sparcnet50K`     | pattern_class (sz / lpd / gpd / lrda / grda / other) | 50,478 | 124 |
| `pd_rda_profiler` | pattern_class, frequency_hz, laterality, spatial_channels, spatial_extent, discharge_times, wave_times | 13,556 | 9 |

Within `sn1_combined_v2`, segments are further sub-categorized in the
`source_dataset` column as `sn1`, `bonobo_only`, or `fabio_spikeed`
(SpikeEd RCT educational arm).

## Label type taxonomy

| `label_type` | Source | Value format |
|---|---|---|
| `spike` | sn1_combined_v2 | `"0"` or `"1"` (binary spike/no-spike) |
| `pattern_class` | sparcnet50K, pd_rda_profiler | string: `seizure`, `lpd`, `gpd`, `lrda`, `grda`, `other` |
| `frequency_hz` | pd_rda_profiler | numeric (string-encoded) |
| `laterality` | pd_rda_profiler | string: `lateralized` / `generalized` / etc. |
| `spatial_channels` | pd_rda_profiler | JSON list of channel names |
| `spatial_extent` | pd_rda_profiler | numeric / categorical |
| `discharge_times` | pd_rda_profiler | JSON list of seconds (per-event timestamps) |
| `wave_times` | pd_rda_profiler | JSON list of seconds |

## Rater alias resolution

The `raters.csv` table consolidates names observed across all sources via
**`raters_aliases_draft.yaml`** (v5; reviewed and edited). 87 multi-source
canonicals were merged; 2,580+ single-source crowd raters kept as-is.
Specials: `IIIC_crowd`, `corrected`, `pending` are flagged with
`is_person=False` (these are profiler-only placeholders, not people).

Still-unresolved first-name-only entries: `Anil`, `GT`, `Sean` (no full
names available in any source).

## S3 paths to source EEG

`segments.s3_uri` points to the absolute S3 location of each segment's
source EEG. Three regimes, depending on `source_dataset`:

| Source | `s3_uri` | `s3_uri_bipolar` | `s3_h5_index` |
|---|---|---|---|
| `sn1_combined_v2:*` (all spike segments) | `s3://bdsp-opendata-restricted/spike-test/v2/SN1_combined_v2.h5` | — | integer; slice `/eeg/signals[s3_h5_index]` |
| `sparcnet50K` (IIIC raw) | `s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/{SUBTYPE}/segments_raw/{BIDS_filename}.mat` | — | — |
| `pd_rda_profiler` (IIIC raw + bipolar) | (same morgoth1 path as sparcnet50K) | `s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/<mat_file>` | — |

Where `{SUBTYPE}` ∈ {`LPD`, `GPD`, `LRDA`, `GRDA`, `SEIZURE`, `IIIC`}
(IIIC = "other"/catch-all). For sparcnet50K the subfolder is derived
from the segment's vote-argmax across the 6 IIIC classes; segments with
zero votes (n=7,531) have null `s3_uri`. The `BIPD` subtype uses
`morgoth2` instead of `morgoth1` with no `/segments_raw/` subdir.

The two-version situation for pd_rda_profiler reflects that GROND
created bipolar-montage pre-processed copies at `iiic-freq3/data/eeg/`
for its 9,857 curated segments. Both raw and bipolar paths are
populated when available; use whichever you prefer.

Credentials for both buckets: register at [bdsp.io](https://bdsp.io).

Sources verified from:
- spike: `ideal-test/README.md` + `combined_spike/README.md`
- IIIC raw paths: `pd-rda-profiler/code/data_management/*.py`
  (download_s3_segments.py, harvest_iiic_*.py)
- IIIC bipolar path: `pd-rda-profiler/README.md` "Data Access" section

## Provenance / legacy files (kept alongside the unified ones)

The original pd-rda-profiler files are kept since some carry richer per-segment
information not captured in the unified long-form table:

| File | Why kept |
|---|---|
| `annotations.csv` | per-(segment, rater) annotation rows including frequency, no_pd flag, skipped flag |
| `channel_involvement.json` | per-segment per-channel involvement |
| `channel_pseudolabels.json` | CNN-generated per-channel pseudo-labels |
| `discharge_times.json` | richer JSON structure of per-event timing |
| `rda_wave_labels.json` | per-wave (peak/trough) timing |

These are *redundant* with `labels.csv` for the columns it covers, but contain
extra structure for the JSON-valued label types.

## Known caveats

1. **`sparcnet50K` segments have null `s3_uri`.** Per-segment file location
   not documented in either upstream repo. The CSV gives us labels but the
   underlying EEG would need to be located in a separate SPaRCNet-specific
   BDSP bucket (likely accessible to people with SPaRCNet credentials).
2. **IIIC crowd data is missing.** The Centaur-style IIIC crowd labels are
   reportedly held by Tianyu and not yet integrated.
3. **Cross-dataset segment de-duplication not done.** If the same underlying
   EEG recording was annotated for *both* spike (in `sn1_combined_v2`) and
   IIIC patterns (in `pd_rda_profiler` or `sparcnet50K`), it appears as
   *two separate seg_ids* — one per source dataset, since the source keys
   are different (`file_key` vs `mat_file` vs BIDS `file`). For most
   analyses this is fine: you query labels by `seg_id` and the source
   dataset is independent. Building a cross-walk would let you ask
   "give me all labels (spike + IIIC) for the same physical recording" —
   but that's not needed for per-task work.
4. **Sparcnet vote encoding** verified empirically as
   `0=other, 1=seizure, 2=lpd, 3=gpd, 4=lrda, 5=grda`.
