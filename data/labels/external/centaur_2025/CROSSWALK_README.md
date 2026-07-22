# Centaur case and rater crosswalk provenance

This directory contains the publishable mapping products used to ingest the
2025 Centaur/DiagnosUs IED and IIIC contests. Private Box locations, raw export
filenames, survey identities, and operator-specific paths are intentionally not
documented here.

## Case mapping

Centaur reads are keyed by a source `Case ID`/event image. Source task lookup
tables map the normalized event image to its original image and source/gold
label. `case_id_to_seg_id.csv` then maps both contests into the unified segment
namespace:

- task 5290 IED: unified segment IDs 89201–94200;
- task 5291 IIIC: unified segment IDs 84555–89200.

`case_id_to_seg_id.csv` contains 10,000 mappings plus its header.
`task1_case_source_map.csv`, `task2_case_source_map.csv`, and
`centaur_segment_map.csv` retain the source-side provenance needed to audit the
join.

## Rater mapping

Centaur user IDs belong to a source-specific namespace. They are joined to the
unified corpus only through the governed local `user_id_to_rater_id.csv`
crosswalk, which is not distributed in a fresh source clone;
numeric coincidence with another dataset is never accepted as identity
evidence. Missing or ambiguous person-level mappings remain distinct
source-scoped identities.

The crosswalk is a governed identity artifact. Review it separately before a
public release even when it contains only numeric identifiers.

## Ingested results

The current unified `labels.csv` contains:

| Source dataset | Reads |
|---|---:|
| `centaur_2025_ied` | 167,503 |
| `centaur_2025_iiic` | 128,872 |
| `centaur_iiic_expert` | 20,000 |

The IED vocabulary is normalized according to the source registry, with `ied`
used as the positive spike decision and non-IED source classes retained in
provenance. The IIIC expert panel retains its native source labels before the
deployed mapping to six reporting classes.

## Validation requirements

A rebuild must fail on duplicate case mappings, missing source cases,
conflicting segment assignments, ambiguous rater mappings, foreign-key
failures, or an unexpected read count. Generated replacements require a new
hash/provenance record; never hand-edit an identifier to make a join pass.
