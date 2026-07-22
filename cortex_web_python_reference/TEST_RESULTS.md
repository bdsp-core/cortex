# Verification status — 2026-07-21

This file reports the current reproducible boundaries. Older calibration and
R&D result ledgers were removed from the shareable documentation; their history
remains available in Git and in governed local artifacts.

## Standalone Python policy packages

Run independently with Python 3.11:

| Package | Result |
|---|---:|
| `termination-policy` | 1 passed |
| `ad6-policy` | 2 passed |
| `precision-policy` | 9 passed |
| `trainer-policy` | 9 passed |

The trainer suite includes source/artifact drift checks and deterministic
behavioral equivalence against the deployed server trainer.

## Canonical web application

A clean worktree at `origin/main` commit `c5d6356` passed the normal web
quality gate after a locked `npm ci`:

| Gate | Result |
|---|---:|
| ESLint | passed, zero warnings |
| TypeScript typecheck | passed |
| Vitest | 52 files / 217 tests passed |
| Intentional Vitest skips | 3 files / 8 tests |
| FastAPI/Python pytest | 205 passed |
| Production build and precompression | passed |
| CSP verification | passed |
| High-severity npm audit | zero vulnerabilities |

Real-browser smoke and full-bank replay are separate release gates because
they require Chromium and governed external artifacts. Their contract and
latest recorded qualification are documented in
`../cortex_web/docs/WEB_WORKER_QUALIFICATION.md`.

## Root research/reference suite

The clean checked-in root suite produced:

- 365 passed;
- 122 skipped;
- 28 deselected by the default slow/nightly policy; and
- 4 failures caused by release-packaging assumptions, not failed numerical
  assertions.

Three failures require the intentionally untracked `data/eeg_bank.h5`. The
fourth is a stale skeleton assertion for the deliberately removed `slides/`
directory. Until those tests are classified as governed-external or updated,
the root suite is not a fresh-clone release gate.

## Local integration reference

The directory can host a broader integration/parity workspace, but several of
its bank, calibration, parity, and extended-test fixtures are governed local
artifacts rather than tracked repository files. Consequently:

- the standalone packages and canonical web gate above are fresh-clone
  reproducible;
- `run_tests.sh` and the repository-level `run_policy_tests.sh` require the
  explicitly staged local reference fixtures they name; and
- absence of those fixtures must not be reported as an algorithm failure.

When the broader gate is run internally, it must validate hashes and
independent file identity before testing fixed-cloud Python↔TypeScript parity.
Discrete selection/status outputs match exactly; continuous comparisons use
the documented tolerance for the browser normal-CDF approximation. Freely
sampled Python and browser sessions are not claimed byte-identical because
their RNG implementations intentionally differ.

## Accepted scientific boundary

The frozen PrecisionPolicy retains the disclosed extreme-skill interval
undercoverage limitation. No calibration repair, new stopping family, reduced
particle setting, or likelihood substitution is implied by this verification
summary.
