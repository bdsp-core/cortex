# Merge Source Manifest — immutable provenance record

**Created:** 2026-05-18 (Phase 0)
**Plan:** `../UNIFIED_REPO_MERGE_PLAN.md` · decisions D1–D9, §0/§0.1
**Why this file exists:** none of the source repos were git repositories, so
there are no commit SHAs. Per D8 (squash to a clean root commit), this
file-manifest + MD5 snapshot **is** the authoritative record of exactly what
state each source repo was in at merge time. Any later "where did X come
from / has X changed since merge?" question is answered by re-hashing against
the tables below.

## Source repos

| Role | Path (sibling of this repo) | Files* | Size | Manifest |
|------|------------------------------|--------|------|----------|
| **A — methodology / simulation engine** (MERGED IN) | `ilae-skill-certification-test-multi-main` | 441 | 430 MB | `_manifests/source_methodology.md5` |
| **B — data / clinical deployment** (MERGED IN) | `ilae-skill-certification-test-multi-main-PI-code` | 336 | 266 MB | `_manifests/source_pi_deployment.md5` |
| **C — reference** (CONSULTED ONLY, not merged) | `ilae-skill-certification-test-main` | 491 | 6.4 GB | `_manifests/reference_consulted.md5` |

\* excludes `__pycache__/`, `.pytest_cache/`, `.venv/`, `.git/`, `.DS_Store`.

### Manifest-of-manifests (tamper-evident root hashes)

These hash the manifest files themselves. If a manifest is regenerated and its
root hash differs, a source repo changed after the merge snapshot.

| Manifest | MD5 |
|----------|-----|
| `source_methodology.md5`  | `3c784adf1b2c23f05ac92761debda6ef` |
| `source_pi_deployment.md5` | `6c2e9c2ebd7ba19c92f5b8e8274a68c7` |
| `reference_consulted.md5` | `8197d98f2fddc040f89ea1b5081f38b7` |

## Role of each source in the unified repo (see plan §1, §4 disposition table)

- **A (methodology)** — canonical for the inference engine: hardened
  `core*.py` / `engine_mode_b.py` (log_ndtr, validated lapse mixture, fitted
  prior), `bridge/`, `tests/`, `diagnostics.py`, CV Youden calibration,
  `SENSITIVE.md`, raw Centaur source + AUDIT (provenance), `engine_inputs/`
  layout + `engine_paths.py`.
- **B (PI deployment)** — canonical for the **data corpus** (`data/labels/` is
  a proven strict superset of A, MD5-verified, 0 rows lost) and the clinical
  deployment runtime (`simulate_test.py` + shipping `Config`,
  `freeze_deployment_prior.py`, `fit_2pl_probit*`, `ingest_*`,
  `deployment_prior/`, the deployment slide decks) plus the methodological
  breadth modules (learn-r / beta-prior / K-scaling) to be ported onto A's
  hardened likelihood.
- **C (reference)** — **not merged.** Consulted to establish ground truth for
  ℓ / c (Rasch main effects → ×1/1.7 logit→probit → probit-lapse MLE λ=0.025 →
  Youden σ*=0.815 on TRAIN pool, experts 70/30). Used as the invariant
  checklist (plan Phase 6). Its data (incl. the PHI EEG `SN1_combined_v2.h5`,
  the 6.4 GB) stays external and is never vendored (D9).

## Environment at merge time

- Target Python: **3.11.9** (pyenv; `.python-version`). Active system Python
  was 3.14.4 — incompatible with the numpy 1.26.4 pin; a fresh `.venv` on
  3.11.9 was created with the union of A's and C's pinned requirements
  (conflict-free: every overlapping pin was byte-identical). See
  `../requirements.txt`.
- git: 2.39.5.

## Verification

Re-hash a source repo and diff against its manifest. **Write to a temp file
first — do not pipe `xargs md5` directly into `diff`**: `diff` may close the
pipe early on the first difference, sending SIGPIPE to `md5` and spuriously
reporting a failure even when the repo is unchanged (observed Phase 0).

```sh
TMP=$(mktemp)
find <source_repo> -type f ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \
  ! -path '*/.venv/*' ! -path '*/.git/*' ! -name '.DS_Store' -print0 \
  | sort -z | xargs -0 md5 -r | sed 's| <source_repo>/| |' > "$TMP"
diff "$TMP" docs/_manifests/<manifest>.md5 && echo "bit-identical"
rm -f "$TMP"
```

Empty diff ⇒ that source repo is bit-identical to its merge-time snapshot.
Equivalently, `md5 -q "$TMP"` must equal the manifest-of-manifests hash above.
