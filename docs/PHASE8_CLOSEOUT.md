# Phase 8 — Close-out + v1.0.0-rc1 tag (sub-step 5)

Per `../UNIFIED_REPO_MERGE_PLAN.md` §"Phase 8":

> 1. Merge `README.md` / `CLAUDE.md`; correct stale 50/50 + δ
>    statements; document the two-engine architecture (D1) and the
>    single data+calibration layer.
> 2. `environment.yml` / `requirements.txt` = union, pinned; test
>    clean-env install.
> 3. `SENSITIVE.md` pass: confirm no PHI in shipped data; the EEG
>    source (`SN1_combined_v2.h5`) is **not** in this tree (lives in
>    sibling `-main` repo) — document the retrieval path, do not
>    vendor it.
> 4. Carry the open shipping decisions into an issue tracker /
>    `docs/OPEN_DECISIONS.md`.
> 5. Tag `v1.0.0-rc1`.
> 6. **Gate**: clean-env install + `pytest` + one end-to-end
>    `ilae-deploy` run + one `ilae-paper` Mode-A run, all green.

**Phase 8 SHIPPED — all six sub-steps closed.**

Close-out date: 2026-05-20.
Suite state at gate: **288 passed / 1 xfailed** in 9:28 (sub-8.4 gate).
Tag target: `eb1...` (Phase 8 sub-8.5 commit — this commit).

## 1. Phase-8 commit chain

| Sub-step | Commit | Deliverable |
|---|---|---|
| 8.1 | `6d836ca` | README.md + CLAUDE.md merge (Phase-6 invariant corrections baked in) |
| 8.2 | `07bc5d8` | `docs/OPEN_DECISIONS.md` + `data/SENSITIVE.md` refresh + D9 retrieval-path doc |
| 8.3 | `b968f9e` | Packaging — `pyproject.toml` version → 1.0.0rc1; `ilae-calibrate` rewired to `calibration.cli:main`; Phase-0 stub deleted |
| 8.4 | `1ce4341` | In-place gate (282→288 suite + ilae-deploy bit-reproducible + ilae-paper 6/6); BLAS-fix in cli.py + bridge for engine reproducibility contract |
| **8.5** | _this commit_ | **Close-out + tag `v1.0.0-rc1`** |

## 2. Per-sub-step evidence map

### Sub-step 1 — README + CLAUDE.md merge

Plan: "Merge README.md / CLAUDE.md; correct stale 50/50 + δ
statements; document the two-engine architecture (D1) and the
single data+calibration layer."

**Evidence**:
- `README.md` (193 lines, was 19): contributor-facing developer doc
  with quickstart, repo layout tree, two-engine table, hard
  invariants from Phase-6, reproducibility section, sensitive
  data + IRB pointer, "where to look for what" index.
- `CLAUDE.md` (162 lines, was 10): reviewer-grade context — the 9
  D-decisions (D1–D9), reference-truth invariants (5 hard pins + 4
  signed-off deviations), phase-by-phase index, "what NOT to do"
  rules, contributor conventions baked in from feedback memory.

**Phase-6 invariant corrections explicitly baked in** (the prior
CLAUDE.md placeholder flagged these as Phase-8 work):
- Expert split = **70/30** (corrected from the legacy 50/50 note).
- Expert membership = `data/labels/raters.csv:expertise_level`
  (corrected from the legacy `EXPERTS` Python set / `gold_
  standard_raters.yaml` notes, both of which named non-existent
  sources).
- `r_ℓ` K=7 = **0.36656350316581954** (corrected from the retired
  K=6 0.326).
- `LOGIT_TO_PROBIT = 1/1.7` + `λ = 0.025` pinned with code-line
  references.

✅ **Gate criterion SATISFIED**.

### Sub-step 2 — OPEN_DECISIONS + SENSITIVE refresh

Plan §"Phase 8" sub-3 + sub-4 combined.

**Evidence — `docs/OPEN_DECISIONS.md`** (new, 158 lines): decision
register for the 5 open shipping decisions:

| # | Decision | Status |
|---|---|---|
| 1 | Per-candidate roll-up policy | 🟢 **WORKING POLICY**: per-task certificates, no single roll-up (user scope 2026-05-20). Supported by sub-7.3-C empirical finding: 0/21 raters reach `all_pass=True` under either replay or Bernoulli at the conservative deployment stopping rule — full-7 conjunction is empirically degenerate |
| 2 | Real-rater replay vs Bernoulli | ✅ **RESOLVED** in Phase 7 (D6; replay is the v1.0 headline) |
| 3 | ℓ\* independent-panel reproducibility | 🟡 PARTIAL (Phase-3 ordinal D7 done; Centaur 4-expert panel is a candidate for full hold-out re-fit) |
| 4 | Bias-warning channel (`\|t_k\|>tol`) | 🔴 OPEN (no `t_k` warning channel in deployment runtime; Paper-2 / post-ship carry per Phase 6 §8) |
| 5 | Split-half reliability + external cohort | 🟡 PARTIAL (split-half ICC 6/6≥0.70 from Phase 2; external-cohort validation a v1.0-review carry) |

**Evidence — `data/SENSITIVE.md` refresh**: extended from F3.4
(2026-05-15; Phases 0–3) to cover Phase-5 (engine_inputs/ vendor)
+ Phase-7 (curated_banks/ vendor + real-rater replay outputs) + D9
retrieval-path doc. Title corrected ("test-multi-main" → "unified").
Phase-7 PHI audit pinned: `data/replay/rater_replay_summary.csv`
flagged HASH (has `canonical_name` column); integer-`rater_id`
replay outputs flagged SAFE alone; curated_banks JSONs confirmed
SAFE (md5-verified). Anonymizer spec extended with Phase-7 HASH
targets. Re-audit cadence section added.

**Evidence — D9 retrieval-path doc**: `data/SENSITIVE.md` §"D9:
External EEG source" documents the canonical sibling-repo location
of `SN1_combined_v2.h5`, BIDMC/MGH internal request path, and the
public-fork policy (derived `curated_banks/` signals ship; raw EEG
does not).

`tests/test_phase1_data.py:test_carry_forwards_present` updated:
dropped the stale Phase-1 byte-identity-to-MINE_SRC pin (the
Phase-8 refresh intentionally diverges); replaced with the
load-bearing IRB-coverage content invariant.

✅ **Gate criterion SATISFIED**.

### Sub-step 3 — Packaging

Plan §"Phase 8" sub-2 (env.yml/requirements union — already done at
Phase 0) + sub-5 (version bump to v1.0.0-rc1) + the ilae-calibrate
rewire (necessary because the Phase-0 stub was still pointed at
from pyproject.toml).

**Evidence**:
- `pyproject.toml` version `0.0.0.dev0` → **`1.0.0rc1`** (PEP 440
  normalized form of the merge plan's `v1.0.0-rc1` git tag).
- `pyproject.toml` `ilae-calibrate`: `calibration.run_youden_
  calibration:main` (Phase-0 stub) → **`calibration.cli:main`**
  (new wrapper).
- `calibration/cli.py` (NEW, 88 lines): thin `--help`-safe argparse
  wrapper around `pipeline.run_unified_calibration:main()`.
  Mirrors `deployment/cli.py` from Phase 4.7. Foot-gun fix: the
  Phase-3 orchestrator runs ~hours on any invocation including
  `--help`; the wrapper argparse-intercepts before any heavy import.
- `calibration/run_youden_calibration.py`: DELETED (orphaned
  Phase-0 stub; the real reference fitter lives at
  `pipeline/reference_calibration/run_youden_calibration.py`).
- `tests/test_phase0_skeleton.py`: ENTRY_POINTS map updated —
  `ilae-calibrate is_stub=True → False`; module path
  `calibration.run_youden_calibration` → `calibration.cli`.
- `tests/test_phase8_packaging.py` (NEW, 6 drift-guards at sub-8.3
  + 2 BLAS guards added at sub-8.4 = 8 total): version pin,
  entry-point target pin, deleted-stub absence, `--help`
  foot-gun guard, `--dry-run` plan output, deferred-import
  guarantee, two BLAS env-var source-inspection checks.

**Operational verification** (`pip install -e ".[test]"` refreshed
the `.venv/bin/ilae-calibrate` shim):
- `ilae-calibrate --help`  → 0.23 s (was: would start ~hours pipeline)
- `ilae-calibrate --dry-run` → 0.02 s step-by-step plan
- `ilae-calibrate all`     → invokes the real Phase-3 orchestrator

✅ **Gate criterion SATISFIED**.

### Sub-step 4 — In-place Phase-8 gate

Plan §"Phase 8" sub-6: "clean-env install + pytest + one end-to-end
ilae-deploy + one ilae-paper Mode-A, all green on a fresh checkout".

User scope (2026-05-20): in-place gate against this `.venv` (already
installed). Three verifications in sequence.

**Step 1 — full slow suite** (`pytest -q -m "" --override-ini ...`):
- **288 passed / 1 xfailed in 9:28** (no regressions vs sub-7.5's
  282; +6 sub-8.3 packaging drift-guards).
- 1 xfailed = `test_oc_borderline_pass_rate` (documented Mode-B
  termination issue; Mode-B is Paper-2 scope).

**Step 2 — `ilae-deploy all`** (end-to-end freeze → simulate → plot
on canonical paths):

The FIRST run uncovered a real reproducibility bug:
- `OPENBLAS_NUM_THREADS=` was unset → multi-thread BLAS active
  → **~1-ULP drift in `hat_*`/`sd_*` columns** of `sim/candidates.
  csv` vs the committed Phase-4.6-B baseline (400 rows changed at
  the 16th decimal digit).
- Diagnosis: the engine's bit-exact-reproducibility contract
  (`engine/core.py` + `docs/CLAUDE.md` §"Engine reproducibility
  contract") requires single-thread BLAS. `conftest.py:21-25`
  enforces this for pytest but the CLI did NOT.
- **Fix shipped in-line at sub-8.4**: BLAS env-var setter (5 env
  vars: OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_
  THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS) at the top of
  `deployment/cli.py` (and `bridge/run_multi_auroc_bridge.py` for
  `ilae-paper`). Set BEFORE any numpy-importing code; deferred
  imports throughout the CLIs keep numpy out of module load.
- Gated by 2 new drift-guard tests in
  `tests/test_phase8_packaging.py`.
- **Re-run after fix**: working-tree diff is **byte-clean** except
  for `sim/summary.json:wall_time_s` (intentional timing noise).

**Step 3 — `ilae-paper --max-raters 3 --method both --n-reps 1
--no-audit`**:
- **6 / 6 sessions in 1697 s (28:17 wall), zero errors**.
- hier: ~440 s/session, δ=0.05 stops at q=141 / 239 / 447.
- brute: ~125 s/session, δ=0.05 stops at q=943 / 701 / 449.
- hier-vs-brute n_q ratio @ δ=0.05: 6.7× / 2.9× / 1.0× across 3
  raters — directionally consistent with the Phase-1 v2
  paper-grade finding (hier dominates brute; weak-correlation
  regime can degenerate to parity).
- Output: `results/mode_a_auroc/mode_a_auroc_results.csv` (6
  rows; gitignored).

**Additional Phase-8 hygiene** (caught during sub-8.4):
- `data/eeg_bank.h5` (175 MB, PHI-bearing EEG, mtime 2026-05-20
  11:16) was untracked but NOT in `.gitignore`. From local
  `scripts/eeg_bank_viewer.py` exploration. **Defensive ignore
  added** to enforce D9 ("EEG stays external") without requiring
  operators to remember.
- `results/mode_a_auroc/` added to `.gitignore` (oversight from
  sub-8.3; matches the regenerable-build-artifact precedent of
  sub-7.3/sub-7.4).

✅ **Gate criterion SATISFIED** — all three in-place verifications
green; one real bug surfaced + fixed in flight.

### Sub-step 5 — Close-out + tag (this commit)

Plan §"Phase 8" sub-5: `Tag v1.0.0-rc1`.

**Evidence**:
- This document.
- CHANGELOG entry for sub-8.5 (CHANGELOG header updated to
  "Phases 0–8 COMPLETE — v1.0.0-rc1").
- Final full-suite gate re-confirmed at this commit (same
  288 / 1 xfailed as sub-8.4; doc-only changes).
- Git tag `v1.0.0-rc1` (annotated; message references the merge
  plan + Phase-8 gate evidence).

## 3. Phase-by-phase index → release candidate

| Phase | Closed | Headline deliverable |
|---|---|---|
| 0 | yes | Repo skeleton + pyproject.toml + entry-point stubs |
| 1 | yes | PI corpus adopted as D3 single source of truth; data provenance chain |
| 2 | yes | Hardened SMC + MCMC engine adopted byte-identical from methodology repo; PI variants ported onto hardened likelihood |
| 3 | yes | Reference-faithful Rasch + per-rater probit-lapse + CV-top-14 two-stage Youden; `cert_config` v13 |
| 3.5 | yes | Joint hierarchical s_j unification + engine s_sd propagation; SBC coverage 0.88→0.93 under UNCERT |
| 4 | yes | Deployment integration; K=7 re-freeze; one likelihood definition shared with Paper-1; `ilae-deploy` CLI |
| 5 | yes | `engine_inputs` provenance (D3) — sha256 chain proven |
| 6 | yes | Invariant audit (5 PASS + 4 signed-off deviations) |
| 7 | yes | Phase-2 K=7 validation + Tier-2 OC + D6 real-rater replay + Paper-1 figures + Phase-7 gate |
| **8** | **yes** | **Shippability — README/CLAUDE merge + OPEN_DECISIONS + packaging + gate + `v1.0.0-rc1`** |

## 4. v1.0.0-rc1 — what's shipping

**Engine + deployment**:
- Paper-1 SMC + MCMC engine (`engine/`) — Mode-A Multi-AUROC
  Precision Protocol; bit-exact under single-thread BLAS.
- Clinical-deployment runtime (`deployment/`) — Laplace + EKF;
  K=7, same likelihood definition as Paper-1 (`λ + (1−2λ)·Φ`,
  λ=0.025); shipping stopping-rule contract in
  `deployment/deployment_config.yaml`.

**Calibration**:
- `cert_config.yaml` v13 — K=7 production calibration (recomputed
  on PI superset corpus via CV-top-14 two-stage Youden).
- `data/deployment_prior/` — frozen K=7 Σ (14×14) + per-task ℓ\*
  + case_bank + 5 deployment figures.

**Tooling**:
- 3 console entry points: `ilae-deploy`, `ilae-paper`,
  `ilae-calibrate` (all real impls, all `--help`-safe).
- 288-test suite + 1 documented xfail.

**Documentation**:
- `README.md` (contributor-facing) + `CLAUDE.md` (reviewer-grade).
- `CHANGELOG.md` (forward-chronological Phase-0 → 8 trail).
- `docs/PHASE{7,8}_CLOSEOUT.md` (scientific gate sign-offs).
- `docs/OPEN_DECISIONS.md` (5 open shipping decisions register).
- `data/SENSITIVE.md` (PHI inventory + IRB + D9 retrieval).
- `docs/INVARIANT_AUDIT.md` (Phase-6 reference-truth checklist).
- `docs/MERGE_SOURCE_MANIFEST.md` (D8 provenance record).

**What does NOT ship** (intentional, per D9 / sensitive-data plan):
- Raw EEG (`SN1_combined_v2.h5`, ~hundreds of MB) — stays external.
- Audit logs with rater names in filenames — gitignored.
- Per-rater PHI tables (raters.csv, name crosswalk) — flagged HASH
  in `data/SENSITIVE.md`; anonymizer runs at journal acceptance.
- `pipeline/_calib_work/` (regenerable from labels.csv).

## 5. What's NOT in v1.0.0-rc1 (deferred carries)

All documented in `docs/OPEN_DECISIONS.md` or as paper-grade
follow-ons in `docs/PHASE7_CLOSEOUT.md`:

- **Open decisions for v1.0 review**: bias-warning channel
  (`|t_k|>tol`); external-cohort validation; ℓ\* full
  hold-out-panel reproduction.
- **Paper-grade Tier-2 OC**: K=7 only at `--n-reps 25` (~2-3 h) or
  full K∈{2,4,6,7,8} at `--n-reps 25` (~overnight). Re-launchable
  via `scripts/run_tier2_oc_simstudy.py`.
- **Paper-grade Mode-A Phase-1**: 27 raters × 3 methods × ≥25
  seeds (Phase-7.4-A already did 27 × 3 × 5 = 405 sessions).
  Re-launchable via `scripts/run_phase1_experiments_v2.py
  --paper-grade`.
- **Mode-A per_task replay headline**: 14,823 × 2 = 29,646
  sessions (~hours). Re-launchable via
  `pipeline/replay/run_replay.py`.
- **Phase-2 inference-validation figure regen**: K=6 → re-render
  if a Phase-9 reviewer asks. Sub-7.1 added the synthetic K=7
  engine-soundness pins via tests; figures are a viz layer over
  the validated K=6 numbers.
- **Anonymizer** (`scripts/anonymize_rater_data.py`): spec'd in
  `data/SENSITIVE.md`; runs at journal acceptance, NOT now.

## 6. Phase-8 gate satisfied — six criteria

Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 8":

  - ✅ **§1**: README.md + CLAUDE.md merged with Phase-6 corrections.
  - ✅ **§2**: environment.yml / requirements.txt union pinned
    (since Phase 0); `pip install -e ".[test]"` clean install
    confirmed at sub-8.3 + sub-8.4.
  - ✅ **§3**: SENSITIVE.md refreshed; D9 retrieval-path doc added;
    `data/eeg_bank.h5` defensive ignore enforces D9.
  - ✅ **§4**: docs/OPEN_DECISIONS.md created; 5 decisions
    registered.
  - ✅ **§5**: tag `v1.0.0-rc1` (this commit).
  - ✅ **§6**: pytest 288 / 1 xfailed; ilae-deploy bit-reproducible;
    ilae-paper 6/6.

## 7. Post-tag carries (Phase 9+ / paper acceptance)

Not blockers for v1.0.0-rc1:
- Run the anonymizer at journal acceptance.
- Resolve the open decisions in `docs/OPEN_DECISIONS.md` (per the
  v1.0-review meeting).
- External-cohort validation (deferred to paper revision if
  reviewers ask).
- Phase-2 figure regen (deferred to figures-only ticket if
  reviewers ask).
- Tag `v1.0.0` after the v1.0 review meeting signs off the open
  decisions (currently working policies are pinned; v1.0 final is
  the meeting-approved policy set).

## 8. The single xfailed test

Carried since Phase 0: `test_oc_borderline_pass_rate` (Mode-B
borderline OC). Documented in `CHANGELOG.md` §"CURRENT STATE":

> 1 xfail (documented Mode-B termination issue, addressed by the
> reframe).

Mode-B is **Paper-2 scope** (binary credentialing); Paper-1 is
Mode-A (Multi-AUROC). The Multi-AUROC reframe at Phase 1 made
Mode-A the production path, and Mode-B's unresolved borderline
case became a known carry. Not a v1.0.0-rc1 blocker.

---

**Tag `v1.0.0-rc1` ships at this commit.** Phase 8 → v1.0 review
handoff is clean.
