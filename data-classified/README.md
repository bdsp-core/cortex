# data-classified

Participant-level data exports, tracked in git by owner decision (2026-08-19):
the repo is private and these files carry no names or emails, but they DO carry
re-identifying quasi-identifiers (sex, race/ethnicity, country, practice
setting, years reading EEG, exact UTC timestamps) on a small (~55-reader)
clinician cohort. Treat this folder as classified: never copy it into a public
repo, a paper supplement, or a share package (cortex-share strips it by design).

Contents:

- `live_expert_experienced_novice_data/` — real CORTEX session exports
  (`<uuid>_summary.csv` / `<uuid>_trials.csv`; 8 top-level pairs plus 43 under
  `sessions/`). Primary source for the live-cohort AUROC figures and the sim-4
  blinded replay study.
- `97d70997-…_summary.csv` / `_trials.csv` — a later re-export of one session
  (adds a hashed `site_id` column); formerly loose under `results/`.

Compatibility: the frozen sim-4 replay study
(`cortex-sim-policy/four-model-sim/sim-4-run/`, incl. its built `dist/`
bundles) reads `live_expert_experienced_novice_data/` at the repo root. That
path is now a committed symlink into this folder — do not remove it, and do
not edit the frozen study to "fix" the path.

EEG bank HDF5s (PHI-bearing raw EEG) remain governed by D9 in the root
.gitignore and are NOT in this folder: they must never be committed under any
circumstances, private repo or not (and GitHub's 100 MB file limit blocks them
regardless). They move between machines by rsync/S3 only.
