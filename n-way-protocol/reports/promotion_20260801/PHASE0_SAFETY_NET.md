# Phase 0 — Safety net (executed 2026-08-01)

Everything below existed and was verified BEFORE any promotion step ran.

## Git restore points (all confirmed present on origin via `git ls-remote`)

| tag | commit | holds |
|---|---|---|
| `restore/main-pre-draw-latent` | `0b310f2` | main exactly as it stood before the promotion |
| `archive/nway-f1-promotion-20260731` | `e806507` | the promotion branch at plan time |
| `archive/wip-precision-spike-20260731` | `ab0c704` | 54-file precision-spike/data WIP, never merged |
| `archive/v15-web-version` | `0009f97` | 2026-06-13 K=7 spike-first browser test, never merged |
| `archive/stash-20260730` | `04472d2` | stash@{0} (87 files, pre-existing WIP) |

`stash@{0}` was preserved by tagging the stash commit object directly
(a stash entry is already a commit carrying base+index parents), NOT via
`git stash branch`, which would have dropped the entry on success. The stash
list is untouched; `git stash list` still shows the entry.

## Production snapshot

- pg_dump: Box snapshot `snapshots/2026/08/2026-08-01T062011Z/db.sql.gz`,
  5,597,750 bytes (backup service run triggered 06:20:11Z, exit success;
  the script aborts before prune on any dump < 1 KiB).
- Release state at Phase 0:
  - current: `/opt/cortex/releases/20260725T231233Z-285f264a42ba`
  - previous: `/opt/cortex/releases/20260725T224452Z-a8dfb6d64ada` (populated —
    a one-step rollback target exists)
- Sessions carrying an n-way profile: 12 complete, 3 `in_progress_nway`,
  3 superseded (all `precision_v1`).

## Baseline suite counts (branch `nway-f1-promotion-20260731` @ `e806507`)

| suite | result |
|---|---|
| cortex_web engine vitest (`npx vitest run engine`) | 136 passed, 6 skipped |
| cortex_web full apps/web vitest | 278 passed, 6 skipped |
| cortex_web apps/web `tsc --noEmit` | clean (exit 0) |
| services/api pytest | 264 passed |
| n-way python (`python/tests/` + `draw_latent_rd/test_equivalence.py`) | 64 passed |
| n-way vitest (`npm run test:ts`) | 48 passed, 3 skipped |
| n-way `npm run typecheck` | KNOWN pre-existing failure at `src/precision_cli.ts:190` (3-arg `PrecisionPolicy.fromInputs` exists only on the WIP branch) |

## STOP conditions evaluated

- Any tag fails to push → all five tags confirmed on origin. PASS.
- pg_dump missing/empty → present, 5.6 MB. PASS.
- `release-state/previous` empty → populated. PASS.

## Data staged for Phase 1 (gitignored `.artifacts/`, never committed)

- `prod_nway_trials_20260801.csv` — 1,855 trials (1,483 categorical + 372
  binary/spike) from the 18 stamped sessions; columns restricted to
  session_id, trial_index, seg_id, task_k, pick, is_correct. No participant
  identifiers.
- `prod_manifest_engine_config.json` — served-manifest engine configuration
  (corrL/corrT 7×7 prior blocks, nParticles 1200, perDomainCap 60,
  bundle v1.6-k7-35k, engineProfile v15). Engine metadata only.
- Join check: all 1,301 distinct IIIC seg_ids in the real trials resolve in
  `.artifacts/categorical_bank_axes.csv` (0 missing); 13 sessions contain
  IIIC trials.
