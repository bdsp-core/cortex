# Labeled EEG segment and rater-count census

**Census date:** 2026-07-16

## Answer

There are **104,070 labeled dataset-question memberships** across the six identifiable
EEG question pools, with **2,255,284 primary scoring events**. This is the appropriate
count when the question is, "How many labeled questions exist across all datasets?"

There are **94,138 canonical labeled `seg_id`s**. This smaller number is the
repository's cross-source registry count. It is not an exhaustive count of source
questions because the canonical ingest:

- collapsed 8,905 retained Kong problem IDs onto 7,817 canonical IDs;
- omitted another 1,799 raw Kong problem IDs; and
- reused canonical IDs when Centaur or Kong questions matched SPaRCNet intervals.

Neither number is a proven **unique physical EEG-window total**. Dataset memberships
overlap, and the outstanding raw Kong crosswalk does not justify either adding or
deduplicating all of them as physical recordings.

## Source-question count and raters per question

The Centaur four-expert panel is combined with the same 5,000 Centaur-IIIC questions.
The anonymous 30-expert workbook is not added because its 70,205 cells are exact
SPaRCNet value-vector equivalents. The Centaur novice snapshot is an alternate version
of the same response keys, not a new question pool.

| EEG question pool | Labeled questions | Primary scoring events | Raters/question, median (IQR) | Mean | Range |
|---|---:|---:|---:|---:|---:|
| SN1 combined v2 | 19,332 | 964,206 | 17 (8–111.25) | 49.88 | 1–213 |
| Centaur 2025 IED | 5,000 | 167,503 | 30 (27–36) | 33.50 | 18–115 |
| SPaRCNet | 50,478 | 290,268 | 3 (2–5) | 5.75 | 1–32 |
| pd-rda-profiler | 13,556 | 27,110 | 2 (2–2) | 2.00 | 1–2 |
| Kong 2025 raw pre-filter | 10,704 | 657,325 | 43 (33–56) | 48.09 | 1–3,079 |
| Centaur 2025 IIIC crowd + four experts | 5,000 | 148,872 | 28 (26–30) | 29.77 | 9–93 |
| **All dataset-question memberships** | **104,070** | **2,255,284** | **4 (2–25)** | **20.30** | **1–3,079** |

The seven raw Kong problems with exceptionally large rater counts are the same
high-volume problems that lack a Results identity and were excluded from the
historical `test_df4` ingest. They remain in the exhaustive source inventory, flagged
as unresolved, rather than being silently discarded.

## Quick rater-count distribution

This table answers how many of the 104,070 dataset-specific questions were scored by
one, two, three, or more distinct raters. For raw Kong, a rater is an upstream
`user_id`; elsewhere it is a canonical `rater_id` within that source question.

| Distinct raters on the question | Questions | Share |
|---:|---:|---:|
| 1 | 8,926 | 8.58% |
| 2 | 19,384 | 18.63% |
| 3 | 23,010 | 22.11% |
| 4 | 1,780 | 1.71% |
| 5–7 | 2,337 | 2.25% |
| 8–9 | 2,253 | 2.16% |
| 10–19 | 15,294 | 14.70% |
| 20–49 | 20,775 | 19.96% |
| 50–99 | 5,362 | 5.15% |
| 100+ | 4,949 | 4.76% |
| **Total** | **104,070** | **100%** |

## Canonical-ID view

When all labels attached to the same canonical ID are combined, the census has 94,138
rows and 2,094,176 canonical primary response rows. A canonical ID has a median of
three distinct primary raters (IQR 2–24; mean 20.80; range 1–601).

| Distinct raters on the canonical ID | Canonical IDs | Share |
|---:|---:|---:|
| 1 | 8,432 | 8.96% |
| 2 | 19,217 | 20.41% |
| 3 | 21,256 | 22.58% |
| 4 | 1,601 | 1.70% |
| 5–7 | 1,598 | 1.70% |
| 8–9 | 2,253 | 2.39% |
| 10–19 | 12,473 | 13.25% |
| 20–49 | 16,468 | 17.49% |
| 50–99 | 5,709 | 6.06% |
| 100+ | 5,131 | 5.45% |
| **Total** | **94,138** | **100%** |

The canonical count can have more raters than a source-specific row because one
canonical ID may combine SPaRCNet, Kong, Centaur crowd, and expert-panel labels.

## Evidence and reproducibility boundary

The census was produced during a private clean-room annotation audit. Its detailed
item-level exports contained governed source identifiers and are intentionally not
distributed in this repository. The numerical results above are retained as a frozen
audit report, not as a promise that those private exports can be regenerated from a
public clone.

The repository-visible inputs are the canonical tables in `data/labels/`. Production
membership and per-payload hashes are recorded by the governed local
`data/production_bank/MANIFEST.json` when that artifact is present. The production
builder is `scripts/build_cortex_bank_k7_production.py`.

Reproducing the exhaustive 104,070-question source view additionally requires the
governed raw source exports and approved identity crosswalk. Joins must use
`(source_namespace, source_rater_id)`; missing, ambiguous, or conflicting mappings
must stop the build. The frozen assertions are 94,138 canonical IDs, 104,070 source
memberships, 2,094,176 canonical primary rows, 2,255,284 exhaustive source response
rows, 10,704 raw Kong problems, and 35,193 production IDs.
