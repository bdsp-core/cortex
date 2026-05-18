# Engine inputs (self-contained)

The data the engine actually reads, in one place, in this repo.
Built + verified by `scripts/build_engine_inputs.py`; resolved in code
through `engine_paths.py`. Provenance, sha256 and proofs:
`MANIFEST.json`.

## Why this exists

Previously every consumer reached into the **sibling repo**
(`../ilae-skill-certification-test-main/data/prepared/`) via ~10
hardcoded relative paths, and `cross_domain_rater_matrix.csv` had no
committed producer. This directory removes the cross-repo dependency
and adds a self-verifying builder.

## Contents

| File | What | Provenance |
|---|---|---|
| `sdt_fits.csv` | **Source of truth.** Tidy long per-rater SDT fits, 219 rows = the 6 `sparcnet_{sz,lpd,gpd,lrda,grda,iic}_sdt_fits.csv` stacked with a `domain` column. Values verbatim (string-preserved → zero precision loss). | Round-trips **byte-identical** to all 6 originals. |
| `cross_domain_rater_matrix.csv` | 29-rater × (σ,θ,ℓ)×6 panel. **What the engine reads.** Byte-identical vendored copy of the former sibling file (sha256 unchanged ⇒ engine numeric behavior provably unchanged). | Also proven reproducible as `pivot(sdt_fits σ/θ by confirmed_sparcnet_name)` with `ℓ = −ln σ` (audit, max\|Δ\| ≈ 2e-16). |
| `MANIFEST.json` | source/output sha256, reproduction proof, frozen-Σ note. | — |

## Not here (by design — would add risk, not reduce fragility)

- **`Sigma_l_fitted.npy`** — stays at repo root. FROZEN 15-rater-era
  artifact (its `Corr_l` differs from the current 29-rater matrix by
  ~0.72; regenerating it would silently change the hierarchical prior
  and every result). Already in-repo (not a cross-repo problem) and
  referenced at root by `core_mcmc` + ~15 scripts; duplicating it
  would risk drift. Resolvable via `engine_paths.SIGMA_L`.
- **`cert_config.yaml`** — stays at repo root. Hand-maintained config,
  not derived data. Resolvable via `engine_paths.CERT_CONFIG`.

## Rebuilding

```
python scripts/build_engine_inputs.py
```

Reads the sibling `data/prepared/` originals, rebuilds `sdt_fits.csv`,
re-vendors the matrix, and **aborts** unless: the 6 per-domain blocks
round-trip byte-identical, the vendored matrix is sha256-identical to
the sibling original, and the matrix equals `pivot(sdt_fits)+ℓ=−lnσ`.
Sources are never modified.

## Regenerating the *upstream* sibling files

Out of scope here. Lineage: raw deID labels → `src/prep_*.py`,
`train_val_split_and_fit.py`, `prep_gold_banks.py` → per-domain
prepared CSVs → `src/fit_sdt_per_domain.py` → the 6
`sparcnet_*_sdt_fits.csv` consumed above.
