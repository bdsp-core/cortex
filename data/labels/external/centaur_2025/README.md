# Centaur 2025 IIIC + IED contest data

Snapshot date: 2025-09-23. Two parallel Centaur Labs / DiagnosUs contests:

| Task ID | Pattern | Cases | Readers | Reads |
|---|---|---:|---:|---:|
| 5290 | Brief sharp events (spike / IED) | 5,000 | 603 | 161,065 |
| 5291 | Harmful brain activity (IIIC) | 5,000 | 699 | 125,865 |

(Reads = the 2025-09-23 snapshot counts. After joining to source and
ingesting, the unified store holds 167,503 IED and 128,872 IIIC reads
from 643 / 736 distinct raters — the source-join recovered reads beyond
the snapshot. The 603/699 "Readers" are the survey-flagged participants in
`centaur_users.csv`; the larger ingested rater counts include readers not
back-propagated into that table — a known reconciliation gap.)

Both share Centaur's internal `Case ID` namespace; the two case pools are
disjoint (5290: 26255616–26260615; 5291: 26392792–26397791).

275 readers participated in both contests; total 1,050 unique Centaur users.

## What's here

- **`centaur_users.csv`** — one row per Centaur user_id (1,050 rows).
  Survey-derived tier (`expert` / `borderline` / `novice` / `unknown`)
  plus participation flags and read counts per contest.
- **`case_id_to_seg_id.csv`** — the Case-ID → unified seg_id crosswalk
  for **both** contests (10,000 rows; IED segs 89201–94200, IIIC segs
  84555–89200). This is the crosswalk that was previously missing.
- **`task1_case_source_map.csv` / `task2_case_source_map.csv`** — per-case
  Centaur Case ID → source image + gold/source label, for IED (task1) and
  IIIC (task2) respectively.
- **`centaur_segment_map.csv`** — both tasks' cases joined to subject_id,
  gold_label, and (where resolved) BDSP S3 retrieval pointers.
- **`centaur_ied_case_table.csv`** — consolidated per-IED-case table:
  gold label + crowd-read vote distribution + signal-availability flag.
  Built by `scripts/build_centaur_ied_case_table.py` (self-checking).

The actual reads are ingested into the unified label store, keyed by the
crosswalk above:

| `data/labels/labels.csv` `source_dataset` | rows | what |
|---|---:|---|
| `centaur_2025_ied`     | 167,503 | IED 5290 per-reader reads (`label_type=spike`) |
| `centaur_2025_iiic`    | 128,872 | IIIC 5291 per-reader reads |
| `centaur_iiic_expert`  |  20,000 | 4-expert IIIC IRR panel (Stage 1b) |

IED reader vocabulary is `{ied, vertex wave, other, posts, wickets, bets}`;
it maps to the gold vocabulary as `ied → spike`, `other → non-spike`, and
identity for the rest. (See the consolidated table builder for the join.)

## What's NOT here (and where it lives)

- **Raw IED EEG signals** — the 5,000 IED segments (seg_id 89201–94200)
  carry only `file_key` / `pat_key` in `segments.csv`; every waveform
  pointer (`s3_uri`, `h5_local_path`, `spike_h5_idx`, `profiler_mat_file`)
  is empty. Provenance splits into the **Bonobo/SpikeNet** spike & non-spike
  pool (`retrieval_status=bonobo_pending`, ~1,329 cases) and **BDSP**
  subjects (`subject_missing` on S3, ~3,618 cases; only ~49 ever resolved
  an `s3_uri`). Recovering the signals needs the Bonobo dump plus a BDSP
  subject-retrieval pass — that is the remaining gap, not the reads.
  (IIIC signals are largely resolved; see `centaur_segment_map.csv`.)

## Survey tier breakdown (177 IIIC + 207 IED surveyed; 755 not surveyed)

|             | expert | borderline | novice | unknown |
|---|---:|---:|---:|---:|
| did IIIC    |     26 |         23 |    127 |     523 |
| did IED     |     24 |         25 |    158 |     396 |
| any contest |     44 |         36 |    215 |     755 |

`derived_tier` mapping rule (from the 4-field Centaur survey):

- **expert** = self-reports "Specialized in adult/pediatric epileptology" OR role is "Fellow - epilepsy or clinical neurophysiology"
- **borderline** = role is "Resident" or "Practicing physician (non-neurologist)", not specialized in epileptology
- **novice** = role is "Medical student" / "Nurse" / "EEG technologist" / "Other", not specialized
- **unknown** = no survey response

The unsurveyed 755 are mostly low-volume drop-in readers; a reasonable
default if needed is `unknown → novice` for the lower tail.

## Stage map

| Stage | What | Status |
|---|---|---|
| 1a | Survey responses → `centaur_users.csv` | ✅ done |
| 1b | IRR study labels (4-expert IIIC xlsx) | ✅ done — `centaur_iiic_expert`, 20,000 rows |
| 2  | IIIC 5291 reads → unified labels | ✅ done — `centaur_2025_iiic`, 128,872 rows |
| 3  | IED 5290 reads → unified labels | ✅ done — `centaur_2025_ied`, 167,503 rows |
| 4  | TianyuCentaur (superseded by IED 5290) | skipped permanently |

The case-ID crosswalk that blocked Stages 1b/2/3 has since arrived
(`case_id_to_seg_id.csv`) and all three streams are ingested into
`data/labels/labels.csv`. The remaining open item is IED *signal*
retrieval (see "What's NOT here").

## Reconciliation with existing raters.csv

Centaur `User ID` is an independent source namespace. Reads are joined through
the governed local `user_id_to_rater_id.csv` mapping (not distributed in a
fresh source clone), never by matching numeric IDs or fuzzy names. A mapped
Centaur identity may reuse a canonical person only
when the governed crosswalk supplies that decision; otherwise it remains a
distinct source-scoped rater. Missing, ambiguous, duplicate, or conflicting
mappings are validation failures rather than candidates for automatic repair.

See `CROSSWALK_README.md` for the shareable case/rater mapping contract. Treat
both the survey table and person crosswalk as governed identity artifacts
before public distribution.
