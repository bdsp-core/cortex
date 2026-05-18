# Centaur 2025 IIIC + IED contest data

Snapshot date: 2025-09-23. Two parallel Centaur Labs / DiagnosUs contests:

| Task ID | Pattern | Cases | Readers | Reads |
|---|---|---:|---:|---:|
| 5290 | Brief sharp events (spike / IED) | 5,000 | 603 | 161,065 |
| 5291 | Harmful brain activity (IIIC) | 5,000 | 699 | 125,865 |

Both share Centaur's internal `Case ID` namespace; the two case pools are
disjoint (5290: 26255616–26260615; 5291: 26392792–26397791).

275 readers participated in both contests; total 1,050 unique Centaur users.

## What's here

- **`centaur_users.csv`** — one row per Centaur user_id (1,050 rows).
  Survey-derived tier (`expert` / `borderline` / `novice` / `unknown`)
  plus participation flags and read counts per contest.

## What's NOT here (and where it lives)

- **Raw per-read labels** (`Reads for ... .csv` files, ~290k rows total)
  are the proprietary Centaur dump and remain in
  `~/Downloads/Fw_ [External] Re_ new labels_/`. To re-build this folder:
  unzip the two `Centaur_reads_for_task_*.zip` archives into
  `/tmp/centaur_iiic/` and `/tmp/centaur_ied/`, then run
  `python3 scripts/ingest_centaur_surveys.py`.

- **Case-ID → BDSP seg_id crosswalk** for both contests is MISSING. The
  reads carry only Centaur internal IDs (e.g. `task2_event4958.png` for
  IIIC, `task1_event3480.png` for IED), with no published mapping to
  the BDSP segments in our unified labels. Until that crosswalk arrives
  (need to request from Centaur / Tianyu / WanYee), the reads cannot be
  merged into `data/labels/labels.csv`.

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
| 1b | IRR study labels (4-expert IIIC xlsx) | pending; same case-id crosswalk problem as Stage 2 |
| 2  | IIIC 5291 reads → unified labels | ⏸️ blocked on case-id crosswalk |
| 3  | IED 5290 reads → unified labels | ⏸️ blocked on case-id crosswalk |
| 4  | TianyuCentaur (superseded by IED 5290) | skipped permanently |

## Reconciliation with existing raters.csv

**Not yet merged.** Centaur `User ID` is an 8-digit integer in Centaur's
own namespace, distinct from our unified `rater_id`. Numerical overlap
between the two namespaces is coincidental (3/1,050 in a quick check).

If a Centaur user is in fact already represented in our unified pool as
a Crowd rater (likely for the long-running platform users), the merge
needs an explicit attribute-based reconciliation (e.g. on Centaur
internal username or first/last name from the survey), which we don't
currently have access to.

The safe interim policy: leave the Centaur user table external; when
the crosswalk arrives and reads are ingested, also bring in any newly
discovered Centaur users as fresh `rater_id` entries with
`groups: ["Centaur"]` and a flag pointing back to the Centaur ID.
