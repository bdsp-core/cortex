# Rater alias decisions needed

The auto-clustering left the following uncertainties. Please confirm or correct.

## 1. Likely same-person, need confirmation

Heuristic match suggests these pairs are the same person. **Confirm or reject** each:

| Cluster A | Cluster B | Reason | Decision |
|---|---|---|---|
| `Aaron Struck` | `Aaron F. Struck` | middle initial added | merge? |
| `Gamal Osman` | `Osman Gamaleldin` | first/last reversed (Gamal Osman Gamaleldin) | merge? |
| `Ji Yeoun Yoo` | `Jiyeoun Yoo` | space vs no-space | merge? |
| `Hiba Arif` | `Hiba A. Haider` | same first name but different last names | likely DIFFERENT people |
| `Zubeda Karim` | `Zubeda Sheikh` | same first name | likely DIFFERENT people |

## 2. pd-rda-profiler initials → full names

These initials appear in pd-rda-profiler's `labels.csv` and `annotations.csv`.
Need to map each to a canonical full name:

| Initial | Likely full name? | Decision |
|---|---|---|
| `MW`        | M. Brandon Westover? | confirm/correct |
| `SZ`        | Sahar Zafar? | confirm/correct |
| `LB`        | ? | needs input |
| `PH`        | ? | needs input |
| `TZ`        | ? | needs input |
| `AS`        | Aaron Struck? Aline Stern? | needs input |
| `IIIC_crowd`| (collective label, not a person) | leave as is (group label, not a person) |
| `corrected` | (a 'rater' string in labels.csv — probably a label-correction marker) | leave as is or drop |
| `pending`   | (same — placeholder?) | leave as is or drop |

## 3. The "Sean" entry

Single-token name "Sean" appears in `combined_spike_meta`. Likely Sean Devlin or Sean Lipsy? Please confirm full name.

## 4. Other singletons to confirm

The following single-token names appear in ideal-test — they look like nicknames:

`alice` → Alice Lam (confident)
`aline` → Aline Herlopian (confident)
`doug` → Douglas Maus (confident)
`ioannis` → Ioannis Karakis (confident)
`jj` → Jin Jing (confident; the meta says so)
`jon` → Jonathan J. Halford (confident)
`marcus` → Marcus Ng (confident)
`mbw` → M. Brandon Westover (confident)
`syd` → Sydney S. Cash (confident)
