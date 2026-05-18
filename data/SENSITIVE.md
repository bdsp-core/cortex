# Sensitive-Data Inventory — ilae-skill-certification-test-multi-main

**F3.4 (2026-05-15).  INVENTORY ONLY.**  This document records every
PHI / re-identifiable file that the Paper-1 (Multi-AUROC) pipeline reads
or writes.  Per the 2026-05-15 scope decision, Phase 3 makes **no**
`.gitignore` changes and runs **no** anonymizer — the repository stays
private until journal acceptance, at which point
`scripts/anonymize_rater_data.py` (spec'd, not yet built) will be run.
This file is the authoritative checklist for that future step.

## IRB coverage

All clinician annotation data are covered by:

- **IRB 2016P000058** (BIDMC)
- **IRB 2013P001024** (MGH)
- Waiver of consent for retrospective annotation data.

The IRB covers *internal research use*.  It does **not** authorise public
deposition of re-identifiable clinician identities; hence the
hash-at-acceptance plan below.

## What counts as PHI here

The EEG segment data are de-identified at source.  The re-identification
risk in THIS repo is the **clinician rater identities** (board-certified
neurophysiologists named in panel rosters, fit tables, audit logs, and
crosswalks).  Aggregate numeric artifacts (covariances, thresholds,
coverage tables) are NOT sensitive.

## Inventory

Legend — Action at public release:
`HASH` = replace names with `rater_<sha256[:12]>` via the anonymizer;
`EXCLUDE` = gitignore + keep private (never in the public artifact);
`SAFE` = no names, publish as-is.

### This repo (`ilae-skill-certification-test-multi-main/`)

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `audit_logs/*.jsonl` (118 files, 596 KB) | **YES** | clinician name in **filename** (`Aaron_Struck_seed3.jsonl`) AND `rater_id` field | **EXCLUDE** (gitignore at release; regenerate hashed if needed) |
| `youden_ell_star.json` | **YES** | CV-panel rosters contain canonical clinician names | **HASH** (numeric ℓ*/J safe; names in roster lists must be hashed) |
| `data/labels/raters.csv` | **YES** | per-rater identity table | **HASH** |
| `data/labels/raters_aliases_draft.yaml` | **YES** | name → alias mapping | **EXCLUDE** (pure identity crosswalk) |
| `data/labels/datasets.csv` | review | dataset metadata; verify no embedded names before release | **REVIEW → SAFE/HASH** |
| `data/labels/annotations.csv` | review | case-level labels; rater column is integer-coded — verify | **REVIEW → SAFE** |
| `results/phase1_figures/expA_real_k6_v2.json` | **YES** | `rater_id` = canonical name (e.g. "M. Brandon Westover") from the patched Phase-1 v2 run | **HASH** |
| `results/phase1_figures/example_trajectory_v2.json` | **YES** | `rater_id` = "M. Brandon Westover" | **HASH** |
| `results/phase1_figures/fig_phase1_supp_speedup_heatmap.{pdf,png}` | **YES** | per-rater heatmap with clinician names on the y-axis | **HASH** (re-render from hashed JSON) |
| `results/phase2_validation/sparcnet_test_retest.csv` | review | `rater_id` is the integer SDT-fit code (no name) — confirm join artifacts carry no name | **REVIEW → SAFE** |
| `cert_config.yaml` | no | numeric constants + `mode_b_legacy` thresholds | **SAFE** |
| `Sigma_l_fitted.npy` | no | aggregate 6×6 covariance / correlation | **SAFE** |
| `results/phase2_validation/*.{json,md}` (SBC, coverage, sigma, lapse, gold-chain) | no | aggregate statistics only | **SAFE** |
| `results/phase1_figures/fig_phase1_main.{pdf,png}` | no | aggregate medians/IQR; panel (c) labelled "M. Brandon Westover" → re-caption to "Expert E1" at release | **REVIEW → re-caption** |

### Sibling repo read by the Paper-1 pipeline (`ilae-skill-certification-test-main/`)

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `data/prepared/cross_domain_rater_matrix.csv` | **YES** | `confirmed_canonical_name` / `confirmed_sparcnet_name` clinician names | **HASH** |
| `data/prepared/sparcnet_{sz,lpd,gpd,lrda,grda,iic}_sdt_fits.csv` | **YES** | `rater_name` column | **HASH** |
| `data/prepared/sparcnet_*.csv` (long format) | no | integer `rater_id` only — but it is the join key to names | **SAFE** (publishable once the name map is hashed) |
| `gold_standard_raters.yaml` | **YES** | named expert rosters per contrast | **EXCLUDE or HASH** |
| `data/h5_v2/SN1_combined_v2.h5` | **YES (PHI-bearing)** | already gitignored in test-main; spike-domain only | **EXCLUDE** (unchanged) |

### Project root

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `../name_crosswalk_audit_v3.0.csv` (+ v2, -reference) | **YES** | authoritative spike↔Bonobo↔SPARCNET name map | **EXCLUDE** (pure identity crosswalk; never public) |

## Corrections to the prior architecture audit

The architecture-audit agent stated `youden_ell_star.json` contained "only
numeric ℓ* — safe to publish".  **That is wrong**: it embeds the CV-panel
clinician rosters.  Likewise the Phase-1 result JSONs / heatmap generated
during this work carry canonical names.  All are flagged `HASH` above.

## Anonymizer spec (to run at acceptance — NOT this phase)

`scripts/anonymize_rater_data.py` (to be written) shall:

1. Build one deterministic map `canonical_name → "rater_" + sha256(name +
   SALT)[:12]` from `cross_domain_rater_matrix.csv` ∪
   `name_crosswalk_audit_v3.0.csv` (SALT stored only in the private repo).
2. Rewrite the `HASH`-tagged files in place into a public fork, replacing
   every name occurrence (CSV cells, JSON roster lists, YAML keys).
3. Re-render the `HASH`/re-caption figures from the hashed inputs.
4. Add the `EXCLUDE`-tagged paths to the public-fork `.gitignore`.
5. Emit a coverage report asserting zero residual matches of any known
   clinician surname in the public tree.

Acceptance check for the future step: `grep - rIl` for the known surname
list returns **only** this `SENSITIVE.md` (which intentionally lists
example names) in the public artifact.
