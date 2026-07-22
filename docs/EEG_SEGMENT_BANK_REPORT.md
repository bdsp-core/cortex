# Labeled EEG segment inventory and production-bank derivation

**As audited:** 2026-07-16<br>
**Bank of record:** `data/production_bank/MANIFEST.json` and
`data/production_bank/eeg_bank_production.h5`<br>
**Canonical annotation source:** `data/labels/`

## Executive summary

There are three different quantities in this repository that must not be called the
same "segment count":

1. The canonical annotation registry has **95,327 registered `seg_id`s**, of which
   **94,138 have at least one label**.
2. The canonical label table has **2,115,793 stored rows**: 2,094,176 primary
   spike/pattern decisions plus 21,617 profiler auxiliary field-value rows.
3. The deployed K=7 EEG bank has **35,193 unique waveform items**: 16,527 spike
   items and 18,666 IIIC items.

The 35,193 total was not a target sample size and it is not the exhaustive labeled
corpus. It is the arithmetic result of the production builder's actual filters:

```text
Spike:  frozen SN1 case table, n_raters >= 8                       16,527

IIIC:   calibrated pattern segments, n_raters >= 8                20,503
        minus absent from eeg_bank_spec.h5                           -743
        minus spec_source == kong_precomputed                      -1,094
        minus other spec source / nonstandard waveform shape            0
                                                                    ------
        retained IIIC                                              18,666

Production bank = 16,527 + 18,666                                 35,193
```

This calculation was reproduced from the current inputs. Its 35,193 `seg_id`s are
identical to the manifest membership, with 35,193 unique IDs and 35,193 unique HDF5
group paths.

The production artifact was built on 2026-05-31. It predates the later clean-room
annotation audit and the 2026-06-12 Centaur-IED signal extension. It therefore should
be described as a **derived, filtered bank of record**, not as "all labeled EEG".

## 1. Counting units

| Unit | Count | Interpretation |
|---|---:|---|
| Registered canonical segment | 95,327 | Row in `segments.csv`; includes 1,189 IDs with no label |
| Canonical labeled segment ID | 94,138 | Unique `seg_id` in `labels.csv` |
| Canonical primary classification row | 2,094,176 | 1,131,709 spike plus 962,467 pattern-class rows |
| Canonical auxiliary profiler row | 21,617 | Frequency, laterality, spatial, discharge, and wave fields |
| Canonical stored label row | 2,115,793 | All primary and auxiliary rows, including repeats |
| Exact-distinct canonical long row | 2,017,635 | Distinct five-column label records; not a deduplication policy |
| Canonical `(segment,rater,label_type,source)` key | 1,982,446 | Another row-level unit, not a segment count |
| Raw pre-filter Kong response event | 657,325 | Unique upstream response IDs |
| Raw Kong problem ID | 10,704 | Source questions; cross-dataset physical identity is unresolved |
| Production waveform item | 35,193 | Unique manifest/HDF5 item after production filters |

The 94,138 labeled IDs divide into **24,332 spike IDs** and **69,806 IIIC pattern
IDs**, with no canonical `seg_id` overlap between those two families. That does not
prove that no underlying physical EEG window occurs in both families: the physical
cross-dataset union has not been established.

## 2. Canonical labeled corpus by dataset and stream

"Segment memberships" below are unique canonical IDs within a source stream. They do
not sum to 94,138 because several IIIC streams label the same canonical IDs.

| Source stream | Source questions / registered rows | Canonical segment memberships | Primary rows | Raters | Native choices | Membership in final bank |
|---|---:|---:|---:|---:|---|---:|
| SN1 combined v2 | 20,521 | 19,332 | 964,206 | 2,551 | binary `0`, `1` | 16,527 |
| Centaur 2025 IED | 5,000 | 5,000 | 167,503 | 643 | IED, vertex wave, other, POSTS, wickets, BETS | 0 |
| SPaRCNet | 50,478 | 50,478 | 290,268 | 124 | seizure, LPD, GPD, LRDA, GRDA, other | 14,731 |
| pd-rda-profiler pattern labels | registry says 13,557; canonical has 13,556 | 13,556 | 27,110 | 2 | seizure, LPD, GPD, LRDA, GRDA, other, two BIPD rows | 0 |
| Kong retained crowd | 8,905 problems / 8,902 filenames | 7,817 | 487,683 | 1,529 | seizure, LPD, GPD, LRDA, GRDA, other | 6,689 |
| Kong retained named experts | Same retained question stream | 5,272 | 8,534 | 8 | same six classes | 4,488 |
| Centaur 2025 IIIC | 5,000 | 5,000 | 128,872 | 736 | seizure, LPD, GPD, LRDA, GRDA, BIPD, BIRDS | 4,289 |
| Centaur four-expert panel | Same 5,000 Centaur IDs | 5,000 | 20,000 | 4 | native BIPD/BIRDS are stored canonically as `other` | 4,289 |

The final-bank column is a **source-membership count**, so it also overlaps. For
example, 14,731 retained items have SPaRCNet labels, but only 7,833 are attributed to
SPaRCNet in the manifest because many also have Kong or Centaur labels.

### 2.1 Why source memberships overlap

The 69,806 canonical IIIC IDs can be understood from the registry provenance as:

```text
SPaRCNet registry IDs                                      50,478
pd-rda-profiler registry IDs                              +13,556
new Kong registry IDs                                      +1,126
new Centaur-IIIC registry IDs                              +4,646
                                                          -------
unique canonical IIIC IDs                                  69,806
```

This does not mean Kong had only 1,126 questions. Its 7,817 canonical memberships
largely reuse existing SPaRCNet IDs. Likewise, Centaur has 5,000 cases, of which 354
reuse SPaRCNet IDs and 4,646 are new registry IDs. The four-expert panel labels the
same Centaur cases and adds no segment IDs.

The final production IIIC source-set combinations are:

| Label-source set on a retained item | Items |
|---|---:|
| SPaRCNet only | 7,833 |
| Kong crowd + Kong expert + SPaRCNet | 4,399 |
| Centaur crowd + Centaur expert panel | 3,935 |
| Kong crowd + SPaRCNet | 2,145 |
| Centaur crowd + expert panel + SPaRCNet | 209 |
| Centaur crowd + expert panel + Kong crowd + Kong expert + SPaRCNet | 89 |
| Centaur crowd + expert panel + Kong crowd + SPaRCNet | 56 |
| **Unique retained IIIC items** | **18,666** |

### 2.2 Native primary-label counts by source

These are stored response rows, not consensus labels.

| Source | Stored choice counts |
|---|---|
| SN1 combined | `0` 341,147; `1` 623,059 |
| Centaur IED | IED 41,125; vertex wave 34,935; other 28,409; POSTS 26,082; wickets 19,949; BETS 17,003 |
| SPaRCNet | seizure 63,763; LPD 51,226; GPD 28,305; LRDA 19,137; GRDA 25,001; other 102,836 |
| pd-rda-profiler | seizure 644; LPD 8,632; GPD 6,900; LRDA 3,944; GRDA 6,980; other 8; BIPD 2 |
| Kong crowd retained | seizure 94,150; LPD 69,348; GPD 83,960; LRDA 84,989; GRDA 88,172; other 67,064 |
| Kong named experts retained | seizure 1,505; LPD 1,245; GPD 1,275; LRDA 1,290; GRDA 1,329; other 1,890 |
| Centaur 2025 IIIC | seizure 25,209; LPD 17,092; GPD 19,488; LRDA 17,615; GRDA 18,957; BIPD 12,114; BIRDS 18,397 |
| Centaur four-expert panel, canonical storage | seizure 2,918; LPD 4,403; GPD 4,280; LRDA 3,081; GRDA 3,466; other 1,852 |

The raw four-expert workbook shows that the last 1,852 `other` rows are a reversible
canonical collapse of 702 native BIPD and 1,150 native BIRDS decisions.

### 2.3 Aggregate native-domain inventory

"Positive segment IDs" means IDs with at least one selected vote for the native
choice; it is not a plurality or gold-label count. For BIPD/BIRDS/other, the
native-restored column reverses the four-expert panel's canonical collapse.

| Domain | Canonical stored selected rows | Native-restored selected rows | Positive segment IDs | Omitted raw Kong rows | Known selected rows after adding omitted Kong |
|---|---:|---:|---:|---:|---:|
| Spike (`1` or IED) | 664,184 | 664,184 | 20,449 | 0 | 664,184 |
| Seizure | 188,189 | 188,189 | 31,067 | 28,567 | 216,756 |
| LPD | 151,946 | 151,946 | 27,474 | 24,866 | 176,812 |
| GPD | 144,208 | 144,208 | 20,136 | 26,489 | 170,697 |
| LRDA | 130,056 | 130,056 | 18,748 | 29,307 | 159,363 |
| GRDA | 143,905 | 143,905 | 21,171 | 29,133 | 173,038 |
| Other | 173,650 | 171,798 | 25,556 | 22,746 | 194,544 |
| BIPD | 12,116 | 12,818 | 3,832 | 0 | 12,818 |
| BIRDS | 18,397 | 19,547 | 4,143 | 0 | 19,547 |

The source protocols did not offer a common universal option set. In particular,
Centaur offered BIPD/BIRDS but not native `other`, whereas SPaRCNet, Kong, and the
profiler used `other` and did not share the Centaur taxonomy. These rows therefore
must not be interpreted as if every reader saw every one-vs-rest option.

### 2.4 Profiler auxiliary labels

These 21,617 rows are included in the 2,115,793 canonical total but are not primary
K=7 task decisions and do not affect production-bank membership.

| Label type | Rows | Segment IDs | Raters |
|---|---:|---:|---:|
| Discharge times | 3,475 | 2,594 | 4 |
| Frequency | 8,791 | 4,965 | 6 |
| Laterality | 4,327 | 3,499 | 6 |
| Spatial channels | 1,419 | 733 | 3 |
| Spatial extent | 3,090 | 1,053 | 4 |
| Wave times | 515 | 515 | 1 |
| **Total** | **21,617** | overlapping | — |

## 3. Raw-source material beyond the canonical table

The canonical 94,138 IDs are exact for `labels.csv`, but not exhaustive for every
source question now available.

### Kong pre-filter export

The raw export has 657,325 unique response events, 4,951 users, and 10,704 problems.
The historical `test_df4` ingest retained 496,217 events, 1,537 users, and 8,905
problems. Thus 161,108 response events and 1,799 problem IDs are outside the canonical
ingest. The omitted events raise the presently verified primary-response inventory
from 2,094,176 to at least **2,255,284 non-overlapping rows**, but they do not provide
a defensible additive segment count: their physical overlap with canonical datasets
is unresolved.

Even inside the retained Kong set, 8,902 filenames were mapped to 7,817 canonical
IDs. There are 446 IDs that absorb multiple filenames, up to 23 filenames per ID;
77,129 response rows are affected and 1,085 filename memberships disappear in the
many-to-one crosswalk. The clean-room audit therefore quarantines Kong segment-level
pluralities pending a rebuilt crosswalk.

### Anonymous 30-expert workbook

The workbook has 70,205 nonmissing cells over 7,778 filenames. Every one of its 30
columns has a full-coverage, zero-mismatch SPaRCNet value-vector equivalent. All cells
are already represented by canonical SPaRCNet data, so they are not added again.

### Centaur version discrepancy

The raw novice snapshot has 125,865 segment-rater rows. All crosswalk to canonical
keys, but 5,575 shared keys disagree with the canonical values, while the canonical
stream has 3,007 additional rows. Neither version has been declared authoritative.

## 4. Exact production-bank construction

### 4.1 Source artifacts and actual threshold

The production builder is `scripts/build_cortex_bank_k7_production.py`. The manifest,
not the builder's stale introductory prose, records the actual run parameters:

- full mode;
- K=7;
- **minimum 8 raters**, for both spike and IIIC;
- spike candidates from `data/labels/fits/spike/cases.csv`;
- K=7 latent segment signals from `data/labels/segment_signals.csv`;
- waveform payloads from `data/SN1_combined_v2.h5` and
  `data/eeg_bank_spec.h5`;
- IIIC `spec_source == "morgoth1_recompute"`;
- IIIC `data30s.shape == (20, 6000)`.

The builder's docstring and its comments describe the earlier >=5-reader design:
16,681 spike and 21,927 IIIC candidates, or 38,608 before waveform filters. That is
not the shipped bank. Re-running candidate selection at >=8 produces 16,527 spike and
20,503 IIIC candidates and reproduces every final manifest ID.

### 4.2 Spike funnel

The spike selector:

1. reads the frozen spike case table;
2. keeps cases with `n_raters >= 8`;
3. attaches K=7 `s_mean_spike` and `s_sd_spike` where available, with the older case
   signal as fallback;
4. joins the segment registry and accepts only the three SN1 sub-sources; and
5. writes every qualifying item in full mode.

| SN1 sub-source | Registered IDs | Labeled IDs | Production items |
|---|---:|---:|---:|
| `sn1` | 13,262 | 13,140 | 12,559 |
| `bonobo_only` | 3,966 | 3,966 | 3,966 |
| `fabio_spikeed` | 3,293 | 2,226 | 2 |
| **SN1 total** | **20,521** | **19,332** | **16,527** |

No additional waveform/provenance filter removes a selected spike case. Centaur IED
is not in the frozen spike case input and contributes zero production items.

All spike manifest items have `pattern_class="spike"`; that field is a family label,
not the true present/absent answer. Using the stored vote rate with a >=0.5 majority
rule, the bank contains 10,835 spike-present and 5,692 spike-absent items.

### 4.3 IIIC candidate labeling and >=8-reader filter

The IIIC selector:

1. sums the available tier-count columns in `segment_signals.csv` to obtain
   `n_raters`;
2. keeps `n_raters >= 8`;
3. reads every canonical `pattern_class` row for the remaining IDs;
4. maps BIPD and BIRDS to `other`;
5. counts the six resulting choices and assigns the maximum-count class; and
6. keeps seizure, LPD, GPD, LRDA, GRDA, and other.

The >=8 candidate pool is:

| Assigned class before waveform filtering | Candidates |
|---|---:|
| Seizure | 2,641 |
| LPD | 3,480 |
| GPD | 2,747 |
| LRDA | 2,217 |
| GRDA | 2,775 |
| Other/IIC | 6,643 |
| **Total** | **20,503** |

The source attribution produced at this step is not truly a "dominant source": the
code stores the lexicographically first source string on a multi-source item. It is
useful as a deterministic manifest field, but the overlapping source-membership table
in section 2 is the more accurate provenance account.

### 4.4 IIIC waveform/provenance filter

Every >=8 IIIC candidate must have a group in `eeg_bank_spec.h5`, must have
`spec_source="morgoth1_recompute"`, and must contain a 20-channel, 6,000-sample
30-second EEG payload. The exact disposition is:

| Disposition | Seizure | LPD | GPD | LRDA | GRDA | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kept | 2,140 | 3,269 | 2,583 | 2,026 | 2,487 | 6,161 | 18,666 |
| Kong-precomputed source | 192 | 178 | 140 | 180 | 273 | 131 | 1,094 |
| Missing from spec HDF5 | 309 | 33 | 24 | 11 | 15 | 351 | 743 |
| Other spec source | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bad waveform shape | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

By the builder's deterministic source attribution, the 743 missing items are 711
Centaur and 32 Kong, and all 1,094 `kong_precomputed` exclusions are Kong. Every
SPaRCNet-attributed >=8 candidate passes the waveform filter.

### 4.5 Final domain composition

Each final item belongs to exactly one task domain by manifest `pattern_class` and is
scored using that task's own `s_mean`/`s_sd` pair.

| Engine domain | Manifest class | Items | Share of bank |
|---|---|---:|---:|
| Spike | spike | 16,527 | 46.96% |
| Seizure (`sz`) | seizure | 2,140 | 6.08% |
| LPD | lpd | 3,269 | 9.29% |
| GPD | gpd | 2,583 | 7.34% |
| LRDA | lrda | 2,026 | 5.76% |
| GRDA | grda | 2,487 | 7.07% |
| Other/IIC (`iic`) | other | 6,161 | 17.51% |
| **Total** | — | **35,193** | **100%** |

The runtime bank loader independently asserts exactly 35,193 items and verifies that
every item has finite own-domain `s_mean` and `s_sd` values in
`segment_signals.csv`.

### 4.6 Manifest-attributed source composition

This table uses the builder's single, lexicographically selected source field. It is
not an additive account of all label provenance.

| Manifest source | Spike | Seizure | LPD | GPD | LRDA | GRDA | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SN1 `sn1` | 12,559 | 0 | 0 | 0 | 0 | 0 | 0 | 12,559 |
| SN1 `bonobo_only` | 3,966 | 0 | 0 | 0 | 0 | 0 | 0 | 3,966 |
| SN1 `fabio_spikeed` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| SPaRCNet | 0 | 388 | 1,610 | 525 | 286 | 421 | 4,603 | 7,833 |
| Kong crowd | 0 | 1,109 | 963 | 1,234 | 933 | 1,130 | 1,175 | 6,544 |
| Centaur 2025 IIIC | 0 | 643 | 696 | 824 | 807 | 936 | 383 | 4,289 |
| **Total** | **16,527** | **2,140** | **3,269** | **2,583** | **2,026** | **2,487** | **6,161** | **35,193** |

## 5. What the production filter did not do

The full builder keeps all items satisfying its reader-count and waveform rules. It
does **not**:

- balance the six IIIC class sizes;
- randomly subsample the full bank;
- require an expert-only consensus;
- require a minimum consensus proportion or margin;
- exclude duplicate stored label rows before calculating plurality;
- preserve BIPD and BIRDS as separate production domains;
- ingest the 161,108 raw Kong events omitted from `test_df4`;
- include Centaur IED or profiler-only items; or
- resolve physical duplicates across source datasets.

Random, difficulty-stratified sampling exists only for the smaller offline-fallback
mode, not for the 35,193-item full bank.

## 6. Material caveats for interpreting the 35,193

1. **Kong many-to-one crosswalk.** The production IIIC plurality reads canonical
   `labels.csv`, including Kong IDs affected by the verified filename-to-`seg_id`
   collapse. It also counts all stored rows, including repeats. The later clean-room
   audit explicitly quarantines retained Kong segment pluralities until that crosswalk
   is rebuilt.
2. **Forced ties.** After collapsing BIPD/BIRDS to other, 3,012 of all 69,806
   canonical IIIC IDs have a maximum-count tie. There are 731 ties in the >=8
   candidate pool and **707 ties in the final 18,666 IIIC bank**. The builder uses
   Python `max` on insertion-ordered counts and therefore assigns one class rather
   than marking the tie. The clean-room native-taxonomy audit instead preserves ties.
3. **Taxonomy loss.** BIPD and BIRDS are irreversibly represented as `other` in the
   production manifest unless the raw labels are consulted.
4. **Source attribution is lossy.** Multi-source items are assigned the
   lexicographically first source string, despite a code comment calling it dominant.
   Source-membership counts should be used for provenance analysis.
5. **The current annotation corpus is newer than the bank.** Centaur IED now has
   5,000 labeled segments and calibrated spike signals, but the production bank's
   frozen spike case input still contains only SN1-family cases.
6. **The 94,138 total is canonical, not physical or exhaustive.** The 1,799 omitted
   raw Kong problems cannot be simply added because cross-source physical identity is
   unresolved. Conversely, HDF5 waveform membership alone is not annotation evidence:
   `eeg_bank_spec.h5` has 132,291 groups, 63,231 of which are not registry IDs.
7. **Dataset registry metadata has one known mismatch.** It reports 13,557 profiler
   source segments, while both the segment registry and canonical pattern labels have
   13,556.

## 7. Reproducibility and evidence map

Repository-visible primary evidence:

- `data/labels/labels.csv`, `segments.csv`, `raters.csv`, and `datasets.csv` —
  canonical source tables.
- `scripts/build_cortex_bank_k7_production.py` — production selection and write logic.
- `data/production_bank/MANIFEST.json` — governed local record of the >=8 filter,
  drop counters, 35,193 membership rows, and per-payload hashes when the production
  artifact is present.
- `data/DATA_PROVENANCE.md` and `data/labels/README.md` — current ownership,
  identity, and release boundaries.

This report also incorporates a private clean-room audit of raw-source membership,
overlap, and HDF5 reconciliation. That audit workspace and its item-level exports were
removed from the shareable repository because they depended on governed source data.
The reconciliation reran the production builder's selectors at thresholds 5 and 8,
applied the HDF5 provenance and shape predicates read-only, and compared the calculated
ID set with the manifest ID set. Reproduction therefore requires the governed raw
inputs and approved identity crosswalk; the public repository alone is insufficient.
