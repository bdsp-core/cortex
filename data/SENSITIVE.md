# Sensitive-Data Inventory — ilae-skill-certification (unified)

**Refreshed 2026-05-20 (Phase 8 sub-8.2).** Original F3.4 inventory
(2026-05-15) extended for Phase 5 (`engine_inputs/` vendor) and
Phase 7 (curated banks vendor + real-rater replay outputs).
**INVENTORY ONLY** — this document records every PHI / re-identifiable
file the unified pipeline reads or writes; no `.gitignore` changes,
no anonymizer run. The repo stays private until journal acceptance,
at which point `scripts/anonymize_rater_data.py` (spec'd in §"Anonymizer
spec" below, not yet built) will be run.

## IRB coverage

All clinician annotation data are covered by:

  - **IRB 2016P000058** (BIDMC)
  - **IRB 2013P001024** (MGH)
  - Waiver of consent for retrospective annotation data.

The IRB covers *internal research use*. It does **not** authorise
public deposition of re-identifiable clinician identities; hence the
hash-at-acceptance plan below.

## What counts as PHI here

The EEG segment data are de-identified at source. The re-identification
risk in THIS repo is the **clinician rater identities** (board-certified
neurophysiologists named in panel rosters, fit tables, audit logs, and
crosswalks). Aggregate numeric artifacts (covariances, thresholds,
coverage tables, integer-keyed bank signals) are NOT sensitive **so
long as** the name → rater_id crosswalk is not co-published.

## D9: External EEG source

Per merge plan D9, `SN1_combined_v2.h5` (the spike-domain raw EEG
file) **stays external**. It is **never vendored** into this repo.

**Retrieval path** (documented per merge plan §"Phase 8" sub-3):

  * Canonical location: sibling repo
    `ilae-skill-certification-test-main/data/h5_v2/SN1_combined_v2.h5`
    (gitignored in that repo; managed via direct copy or a controlled
    object-store mirror).
  * Re-acquisition (BIDMC/MGH internal users): the file is part of the
    BDSP HDF5 collection; request access via the IRB-listed PI
    (M. B. Westover, MGH).
  * Public users (post-acceptance): not provided. The published
    artifact uses only the derived case bank signals
    (`data/curated_banks/*.json` — 168 KB of s_probit values, NOT raw
    EEG) and the integer-keyed rater data tables. No PHI-bearing EEG
    leaves the internal repo.

The unified repo's `data/curated_banks/` (Phase-7.4-A vendor) holds
derived per-segment signal banks (`s_probit`, `frac_yes`, integer
`case_keys`); it does NOT carry EEG. md5-verified against
`docs/_manifests/reference_consulted.md5`.

## Inventory

Legend — Action at public release:
`HASH` = replace names with `rater_<sha256[:12]>` via the anonymizer;
`EXCLUDE` = gitignore + keep private (never in the public artifact);
`SAFE` = no names / aggregate-only, publish as-is.

### Data layer (`data/`)

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `data/labels/raters.csv` | **YES** | per-rater identity table — `confirmed_canonical_name`, `confirmed_sparcnet_name`, `expertise_level` columns | **HASH** |
| `data/labels/raters_aliases_draft.yaml` | **YES** | name → alias mapping | **EXCLUDE** (pure identity crosswalk) |
| `data/labels/labels.csv` | review | rater column is **integer-coded** `rater_id` (PHI-clean alone; join key to `raters.csv`) | **SAFE** once `raters.csv` is hashed |
| `data/labels/datasets.csv` | review | dataset metadata; verify no embedded names | **REVIEW → SAFE/HASH** |
| `data/labels/segments.csv` | no | integer `seg_id` + metadata, no rater info | **SAFE** |
| `data/labels/fits*/` (six SDT-fit variants, deployment_prior, hier) | review | sometimes carries `rater_id` integer + `rater_name` columns; per-fit basis | **REVIEW → HASH where named** |
| `data/labels/external/centaur_2025/_raw_source/centaur_iiic_*.{csv,xlsx}` | review | raw Centaur releases (provenance only, never re-ingested) | **EXCLUDE** (provenance vault, never public) |
| `data/labels/external/centaur_2025/AUDIT_centaur_iiic_novice_expert.md` | no | audit notes; verify no embedded names | **REVIEW → SAFE** |
| `data/deployment_prior/Sigma.csv` | no | aggregate K=7 14×14 prior covariance | **SAFE** |
| `data/deployment_prior/ell_thresholds.csv` | no | 7 per-task ℓ* + numeric metadata | **SAFE** |
| `data/deployment_prior/case_bank.csv` | review | per-seg-id signal bank; integer keys | **SAFE** |
| `data/deployment_prior/summary.json` | no | aggregate fit constants | **SAFE** |
| `data/deployment_prior/figures/fig{1..5}*.png` | no | synthetic candidate panel figures, no real names | **SAFE** |
| `data/deployment_prior/sim/candidates.csv` + `trajectories.npz` | no | synthetic Bernoulli-sim outputs | **SAFE** |
| `data/deployment_prior/sim_modea/results.npz` + `summary.json` | no | synthetic Mode-A panel outputs | **SAFE** |
| `data/deployment_prior/deployment_replay.csv` | no | per-candidate Bernoulli-sim outputs (mislabelled in plan; NOT real-rater data, see Phase 7.3 audit) | **SAFE** |
| `data/deployment_prior/_pi_baseline_frozen/*` | no | PI K=6 baseline snapshot, aggregate only | **SAFE** |
| `data/engine_inputs/cross_domain_rater_matrix.csv` | **YES** | Phase-5 vendored copy from sibling; `confirmed_canonical_name` + `confirmed_sparcnet_name` columns | **HASH** |
| `data/engine_inputs/sdt_fits.csv` | review | 14,214 rows; `rater_id` integer (PHI-clean alone) | **SAFE** once `raters.csv` is hashed |
| `data/engine_inputs/sdt_fits.spike.csv` | review | per-rater spike fits | **SAFE** (integer keys) |
| `data/engine_inputs/cross_domain_rater_matrix.q2locked.csv` | **YES** | locked variant of the matrix; same canonical-name columns | **HASH** |
| `data/engine_inputs/MANIFEST.json` / `README.md` | no | provenance docs | **SAFE** |
| `data/curated_banks/sparcnet_*.json` + `combined_spike.json` | no | Phase-7.4-A vendor; aggregate `s_probit`, `frac_yes`, integer `case_keys`, `n_annotations` only | **SAFE** |
| `data/replay/rater_replay_bank.csv.gz` (gitignored, regenerable) | review | 5.5M rows; `rater_id` integer + per-seg `(y, s_mean, s_sd)` | **SAFE** alone; PHI **only** when joined to `raters.csv` |
| `data/replay/rater_replay_bank.indexed.pkl` (gitignored) | review | pickled multi-index of above; same schema | **SAFE** alone |
| `data/replay/rater_replay_summary.csv` (gitignored) | **YES** | per-(rater, task) aggregates; **has `canonical_name` column** ("Aaron F. Struck", "M. Brandon Westover", …) AND `expertise_level` | **HASH** (or strip `canonical_name` at publication; numeric aggregates are safe) |
| `data/SENSITIVE.md` (this file) | no | inventory + example names | **SAFE** (intentional name examples; documents the hash plan) |
| `data/DATA_PROVENANCE.md` | no | sha256 chain + provenance prose | **SAFE** |

### Calibration layer (`calibration/`)

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `calibration/cert_config.yaml` | no | v13 numeric constants + per-task ℓ\* + provenance sha256s | **SAFE** |
| `calibration/youden_ell_star.json` | **YES** | CV-panel rosters contain canonical clinician names | **HASH** (numeric ℓ\*/J safe; names in roster lists must be hashed) |
| `calibration/CALIBRATION_PROVENANCE.md` | no | provenance prose | **SAFE** |

### Results + experiment outputs (`results/`)

All `results/` paths are gitignored as regenerable build artifacts.
Listed here for completeness so the anonymizer also rewrites local
copies before any deliberate publication.

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `results/phase1_figures/expA_real_k6_v2.json` | **YES** | `rater_id` = canonical name (e.g. "M. Brandon Westover") in the Phase-1 v2 output | **HASH** |
| `results/phase1_figures/example_trajectory_v2.json` | **YES** | `rater_id` = "M. Brandon Westover" | **HASH** |
| `results/phase1_figures/expB_kscaling_v2.json` | no | synthetic K-scaling (no real raters) | **SAFE** |
| `results/phase1_figures/fig_phase1_main.{pdf,png}` | no | aggregate medians/IQR + panel (c) labelled with rater name → re-caption "Expert E1" at release | **REVIEW → re-caption** |
| `results/phase1_figures/fig_phase1_supp_speedup_heatmap.{pdf,png}` | **YES** | per-rater heatmap with clinician names on the y-axis | **HASH** (re-render from hashed JSON) |
| `results/phase2_validation/tier2_oc_simstudy_rows.json` | no | synthetic ℓ-grid OC, no real raters | **SAFE** |
| `results/phase2_validation/oc_summary.json` | no | aggregate KM medians + bootstrap CIs over synthetic cells | **SAFE** |
| `results/phase2_validation/fig_oc_{surface,delta_censored,hier_gain}.{pdf,png}` | no | aggregate OC figures, no rater identities | **SAFE** |
| `results/phase2_validation/sparcnet_test_retest.{csv,md}` (if regenerated) | review | `rater_id` is the integer SDT-fit code; confirm no name join | **REVIEW → SAFE** |
| `results/phase2_validation/sbc_*.{json,md,png}` (if regenerated) | no | aggregate SBC ranks | **SAFE** |
| `results/phase2_validation/coverage_*.{json,md,csv}` (if regenerated) | no | aggregate coverage stats | **SAFE** |
| `results/phase2_validation/gold_chain_*.{json,md,png}` (if regenerated) | no | aggregate gold-chain moments | **SAFE** |
| `results/phase2_validation/{lapse,sigma}_sensitivity.*` (if regenerated) | no | aggregate sensitivity sweeps | **SAFE** |
| `results/replay/replay_per_task.csv` (29,646 rows) | review | `rater_id` integer + per-(rater, task) decision | **SAFE** alone; PHI only via `raters.csv` join |
| `results/replay/replay_per_candidate.csv` (42 rows) | review | `rater_id` integer + per-candidate roll-up | **SAFE** alone |
| `results/replay/replay_run_summary.json` | no | headline aggregates | **SAFE** |
| `results/replay_mode_a/replay_per_candidate.csv` (42 rows) | review | `rater_id` integer + Mode-A AUROC posterior summaries | **SAFE** alone |
| `results/replay_mode_a/replay_run_summary.json` | no | Mode-A headline aggregates | **SAFE** |

### Pipeline + audit artifacts

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `pipeline/name_crosswalk_audit_v3.0.csv` (if present) | **YES** | authoritative spike↔Bonobo↔SPARCNET name map | **EXCLUDE** (pure identity crosswalk; never public) |
| `pipeline/_calib_work/` (gitignored; Phase-3 staging) | review | reference-fitter intermediate artifacts; some carry `rater_name` | **EXCLUDE** (regenerable from `data/labels/` post-hash) |
| `pipeline/joint_calibration/*_posterior.npz` (gitignored) | no | bulky JAX posterior arrays; no identity columns | **SAFE** (gitignored anyway as bulky-and-regenerable) |
| `bridge/audit_logs/*.jsonl` (if present; Mode-B legacy) | **YES** | clinician name in filename AND `rater_id` field | **EXCLUDE** (gitignore at release; regenerate hashed if needed) |

### Project-root crosswalks

| Path | PHI? | Detail | Release action |
|---|---|---|---|
| `../name_crosswalk_audit_v3.0.csv` (parent of repo) | **YES** | authoritative spike↔Bonobo↔SPARCNET name map | **EXCLUDE** (pure identity crosswalk; never public) |
| `../UNIFIED_REPO_MERGE_PLAN.md` | no | merge plan + decisions D1–D9; no PHI | **SAFE** |

## Corrections + notes

  * `data/curated_banks/*.json` (Phase-7.4-A vendor) was previously
    flagged by the F3.4 inventory as "sibling-repo only"; vendored
    in-repo at sub-7.4-A. Confirmed SAFE (aggregate signal banks,
    no rater identifiers, md5-verified).
  * Phase-7 real-rater replay outputs (`data/replay/`,
    `results/replay/`, `results/replay_mode_a/`) all use **integer
    `rater_id`** as the join key. The only file in this group with a
    name column is `data/replay/rater_replay_summary.csv` (joined
    from `raters.csv` for human-readable audit logs).
  * `cross_domain_rater_matrix.csv` lives in BOTH `data/engine_inputs/`
    (Phase-5 vendored) and (historically) the sibling repo; the
    vendored copy is byte-identical (Phase-5 MANIFEST verifies sha256).
    Both must be HASHed at release.
  * The architecture-audit agent had previously stated
    `youden_ell_star.json` was "only numeric ℓ\*" — **incorrect**, it
    embeds CV-panel rosters; flagged HASH.

## Anonymizer spec (runs at acceptance — NOT this phase)

`scripts/anonymize_rater_data.py` (to be written) shall:

  1. Build one deterministic map `canonical_name → "rater_" +
     sha256(name + SALT)[:12]` from `data/labels/raters.csv` ∪
     `data/engine_inputs/cross_domain_rater_matrix.csv` ∪
     `../name_crosswalk_audit_v3.0.csv` (SALT stored only in the
     private repo).
  2. Rewrite the **HASH**-tagged files in place into a public fork,
     replacing every name occurrence (CSV cells, JSON roster lists,
     YAML keys). Includes the Phase-7 additions:
     `data/replay/rater_replay_summary.csv:canonical_name`,
     `data/engine_inputs/cross_domain_rater_matrix.csv:
     confirmed_canonical_name`, `confirmed_sparcnet_name`.
  3. Re-render the **HASH**-tagged figures from the hashed inputs
     (`fig_phase1_supp_speedup_heatmap.{pdf,png}`,
     `fig_phase1_main.{pdf,png}` panel (c)).
  4. Add the **EXCLUDE**-tagged paths to the public-fork `.gitignore`.
  5. Verify `data/curated_banks/*.json` and the integer-keyed result
     CSVs publish unchanged (no rewrite needed; these are already
     SAFE).
  6. Emit a coverage report asserting zero residual matches of any
     known clinician surname in the public tree.

Acceptance check for the future step: `grep -rIl` for the known
surname list returns **only** this `SENSITIVE.md` (which intentionally
lists example names for documentation) in the public artifact.

## Re-audit cadence

This document is re-audited at every phase boundary that touches
data/ or results/:

  * F3.4 (2026-05-15): initial inventory (Phases 0–3).
  * Phase 8 sub-8.2 (2026-05-20): added Phase-5 `engine_inputs/`
    vendor + Phase-7 (`curated_banks/`, `data/replay/`,
    `results/replay/`, `results/replay_mode_a/`, Phase-2 validation
    figure paths) + D9 retrieval-path doc.
  * Next: re-audit at journal acceptance, then run the anonymizer.
