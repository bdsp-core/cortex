# Centaur ICU EEG annotation search results

Source found in Box:

`box:Brandon - PHI/0_People/Chenxi Sun’s files/Home/File transfer/Centaur_2025April`

Primary files:

- `Reads for Brief sharp events classification in EEG 2025.10.07_22.37.41.174816.csv`
- `Reads for Harmful brain activity classification 2025.10.07_22.37.56.912550.csv`
- `task1/task1_labels_ided.xlsx`
- `task2/task2_labels_ided.xlsx`

Mapping method:

- Centaur read exports contain per-user labels keyed by `Origin`.
- `Origin` is normalized to the basename, for example `task2_images_v3/task2_event4958.png` becomes `task2_event4958.png`.
- The normalized origin is joined to `task*_labels_ided.xlsx` column `lut1`.
- The lookup sheets provide:
  - `lut1`: Centaur event image
  - `lut2`: original/source image identifier
  - `lut3`: source path used when constructing the task
  - `lut4`: source/gold label

Local outputs:

- `task1_brief_sharp_events_centaur_reads_joined_to_source.csv`
- `task1_brief_sharp_events_case_source_map.csv`
- `task2_harmful_brain_activity_centaur_reads_joined_to_source.csv`
- `task2_harmful_brain_activity_case_source_map.csv`
- `centaur_search_summary.json`

Join validation:

- Task 1: 167,507 Centaur reads, 5,000 cases, 644 users, 0 missing source mappings.
- Task 2: 128,872 Centaur reads, 5,000 cases, 736 users, 0 missing source mappings.
