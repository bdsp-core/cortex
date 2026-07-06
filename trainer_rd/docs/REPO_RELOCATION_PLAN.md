# REPO_RELOCATION_PLAN.md — homing the trainer project into the unified repo

> **Task (2026-07-06, user-directed).** Relocate the whole learning-algorithm
> project at `/data/eli-work/scratch` to a subdirectory of the unified repo
> `/data/eli-work/repos/ilae-skill-certification-test-multi`, preserving ALL
> functionality (the 21-file / 299-check suite, the sandbox protocol, the
> studies, the viz pipeline) and leaving the repo's own 282-test pytest suite
> untouched.
>
> **This is a RELOCATION, not "the port."** In project vocabulary "the port"
> (`PHASE3_AND_PORT.md`) means integrating the trainer algorithm into the
> repo's production engines — that remains future, gated work. This plan only
> re-homes the directory. **D43 applies:** the algorithm is SUBMISSION-FROZEN;
> zero algorithm/code-behavior changes are permitted here. The relocation is a
> D43-legitimate activity (housekeeping / port-gate enabling, not a PECR loop).

---

## 1. Verified facts the architecture rests on (all checked 2026-07-06)

**Scratch side (`/data/eli-work/scratch`):**

| # | Fact | Evidence | Consequence |
|---|---|---|---|
| S1 | All code lives in six proper packages (`engine/ training/ sandbox/ studies/ tests/ viz/`, each with `__init__.py`); root holds only `README.md` + data/doc/config dirs. **No `__init__.py` at the scratch root** — the root is a working directory, not a package. | `find -maxdepth 2 -name __init__.py` | The directory can be dropped anywhere; nothing above it is import-visible. |
| S2 | All intra-project imports are absolute top-level (`from training.x import y`, `import engine.core_mcmc_general`). | grep over all packages | Modules resolve iff **cwd = project root** (`python3 -m` puts cwd on `sys.path`). Run-from-project-root discipline must survive the move — it does, it just becomes "cd into the subdir first". |
| S3 | Every data path is `__file__`-relative (`bank_adapter._HERE`, `extset_adapter._ROOT`, `instrument_v15._ROOT`, `policy_general._REPO`, `sandbox.config.ROOT`). **Zero cwd-relative data loads; zero absolute `/data/eli-work` strings in code.** | grep audits | Data loading survives relocation with no edits. Only `figures/` outputs and study npz caches are cwd-relative — satisfied by the same run-from-project-root discipline. |
| S4 | Suite = 21 script-style files / 299 checks, run `python3 -m tests.test_<x>` (NOT pytest). Canary re-verified today: `tests.test_step0` → "PASSED — 23 checks". | live run | Preservation gate = re-run the full suite from the new location, same interpreter. |
| S5 | Interpreter = system anaconda `python3` 3.12.4 with numpy 1.26.4 / scipy 1.13.1 / pandas 2.2.2 — **exactly the repo's pinned versions** (repo `.venv` is Python 3.11.15, used only for the repo's own pytest). | version checks | No dependency work needed; no new packages. The two suites keep their own interpreters. |
| S6 | Sizes: `data/` 51M (extset 27M, two `*_general.csv` 25M), `figures/` 35M (incl. 11M of MP4s), `sandbox/` 3.5M (per-tester logs/state, pseudonymized USER-A…G), everything else <1M. Total ≈ 92M. | `du` | Git payload decision needed (§4, decision R3). |
| S7 | Not a git repo today; no history to preserve — the history IS `docs/PROJECT_MEMORY.md`. | — | Plain `rsync` + first commit is lossless. |

**Repo side (`/data/eli-work/repos/ilae-skill-certification-test-multi`):**

| # | Fact | Evidence | Consequence |
|---|---|---|---|
| R1 | pytest collection is pinned: `testpaths = ["tests"]` in `pyproject.toml`; markers strict; slow/nightly excluded by default. | pyproject `[tool.pytest.ini_options]` | The repo's `pytest` will **never see** the subproject's script-style `tests/` by default. One belt-and-braces guard still recommended (§3, item A4). |
| R2 | Distribution packages are an **explicit list** (`engine`, `engine.variants`, `deployment`, `pipeline`, `calibration`, `bridge`). | pyproject `[tool.setuptools]` | The subproject can never leak into the wheel/install. No packaging edits needed. |
| R3 | Precedent exists: `methodology_rd/` and `discrimination_rd/` are self-contained R&D subprojects (own PLAN/REPORT/PROVENANCE/results/scripts) living beside the shipped packages. | `ls` | The trainer project joins as a third, much larger, sibling of this kind. |
| R4 | Exactly ONE tree-walking test: `tests/test_phase1_data.py:138` rglobs `<repo>/data/` only. | grep for walk/rglob | A new top-level subdir is invisible to the suite. Future repo-wide invariant scans must exclude it (documented, §3 item A5). |
| R5 | The working tree is **dirty with unrelated work** (web-app commits on `main`; uncommitted modifications incl. `engine/core_mcmc.py`, `data/labels/*`, `.gitignore`). | `git status` | Port commits must stage ONLY port paths; pre-existing dirt is snapshot-recorded at G0 and left alone. Repo-suite baseline must be taken on the dirty tree as-is (that's the honest baseline). |
| R6 | Name overlaps between the two projects (`engine/ tests/ data/ docs/ archive/ config/`) are **nested, not top-level** after the move — the only sharing mechanism would be `sys.path`, and neither root is on the other's path. Repo root itself is not importable (dashed dir name; the stray root `__init__.py` is a Phase-0 vestige). | S1/S2 + pyproject comment | No import collision is possible under run-from-project-root discipline. See §2 collision map. |

---

## 2. Target architecture

### 2.1 Placement

```
ilae-skill-certification-test-multi/
├── engine/            deployment/        pipeline/        ← shipped packages (untouched)
├── calibration/       bridge/            data/  docs/  tests/  …
├── methodology_rd/    discrimination_rd/                  ← existing R&D siblings
└── trainer_rd/                                            ← THE MOVE (decision R1, name)
    ├── README.md          (gains a "how this subproject runs" header)
    ├── conftest.py        (NEW, 3 lines: pytest collect-ignore guard — §3 A4)
    ├── engine/            ← vendored anonymized SMC engine copy (STAYS vendored, D43)
    ├── training/          sandbox/       studies/       tests/       viz/
    ├── config/            data/          docs/          figures/     reference/
    └── archive/
```

- **Recommended name: `trainer_rd/`** — matches the repo's `*_rd` R&D-subproject
  convention (R3). Alternatives if the user prefers: `adaptive_trainer/`,
  `learning_algorithm/`. Nothing in either codebase depends on the name.
- The subdirectory root gets **no `__init__.py`** — it is deliberately NOT a
  package and NOT importable from the repo root. It is a self-contained working
  directory, exactly as it is today.

### 2.2 Execution model (unchanged, one `cd` deeper)

| Today | After |
|---|---|
| `cd /data/eli-work/scratch` | `cd …/ilae-skill-certification-test-multi/trainer_rd` |
| `python3 -m tests.test_step0` | identical |
| `python3 -m studies.study_X --smoke` | identical |
| `python3 -m sandbox.session --user USER-E` | identical |
| `python3 -m viz.make_user_videos` | identical |

Interpreter stays system `python3` (anaconda 3.12.4). The repo's `.venv`
(3.11.15) remains the repo-suite interpreter. They never cross.

### 2.3 Collision map (why nothing shadows anything)

| Name | `trainer_rd/<name>` | repo `<name>` | Resolution |
|---|---|---|---|
| `engine` | vendored anonymized SMC copy (`core_mcmc_general.py`, `policy_general.py`, `instrument_v15.py`, vendored `auroc.py`) | production SMC+MCMC / Laplace engines | Only ever co-resolvable via `sys.path`; cwd discipline means exactly one is visible at a time. **The vendored copy STAYS** — D43 bit-identity outranks dedup. Converging them is the production port (future, `PHASE3_AND_PORT.md`). |
| `tests` | 21 script-style files, `python3 -m` | 282-test pytest suite | `testpaths` pin (R1) + `trainer_rd/conftest.py` guard (A4). |
| `data` | `*_general` anonymized derivatives + EXTSET + session fixtures | D3 label authority + provenance chain | trainer_rd data is self-contained (S3) and stays OUT of the repo's `data/DATA_PROVENANCE.md` chain — it is subproject-internal evidence, like `methodology_rd/data/`. The R4 rglob test can't see it. |
| `docs`, `archive`, `config` | M# records / move-log / v14 yaml copies | phase close-outs / repo archive / root yamls | Nested paths; no mechanism by which they interact. |

### 2.4 Explicitly OUT of scope (D43 freeze)

1. **No import rewrites**, no re-packaging, no renames inside the project.
2. **No dedup** of the vendored engine, `auroc.py`, `Sigma_l_fitted_k7_general.npy`,
   `cert_config_general.yaml`, or the `*_general.csv` banks against their repo
   ancestors. ("Main-repo auroc wins **at port time**" — port time is not now.)
3. **No algorithm or default changes** of any kind.
4. The production port itself (trainer → `deployment/`/`engine/` integration).

---

## 3. Repo-side integration edits (the complete list — 5 small items)

| # | File | Edit | Why |
|---|---|---|---|
| A1 | `trainer_rd/README.md` | Prepend: what this subproject is, run-from-this-directory rule, interpreter note, suite command, pointer to `docs/PROJECT_MEMORY.md` + `docs/ARCHITECTURE.md`, "not part of the shipped distribution", D43 freeze note. | First-contact orientation, matches `*_rd` convention. |
| A2 | repo `README.md` + `CLAUDE.md` | One row/paragraph each: `trainer_rd/` = adaptive-trainer R&D subproject (eval→training pipeline), self-contained, own suite + interpreter, frozen for advisor review; repo-wide invariant scans must exclude it. | Reviewer-grade transparency (repo convention). |
| A3 | repo `CHANGELOG.md` | Entry under the current unreleased section: relocation, provenance (`/data/eli-work/scratch`, 2026-07-06), suite evidence (299/299 post-move). | Repo convention. |
| A4 | `trainer_rd/conftest.py` (new) | `collect_ignore_glob = ["*"]` + a comment ("script-style suite; run `python3 -m tests.test_<x>` from this directory — pytest must not collect here"). Optionally also add `norecursedirs = trainer_rd` to pyproject as belt-and-braces. | Guards the one residual pytest hazard: an explicit `pytest trainer_rd/` invocation importing script-style tests (which execute on import). Travels with the directory. |
| A5 | `docs/OPEN_DECISIONS.md` or CLAUDE.md "what NOT to do" | One line: never sweep `trainer_rd/` into repo-wide tree-walk tests/scans; it is a frozen vendored subproject. | Future-proofs R4. |

No `pyproject.toml` packaging change, no `requirements.txt` change, no CI
change (none exists), no repo-test change.

---

## 4. Decisions to ratify at Gate 0 (defaults chosen; say "go" to accept all)

| # | Decision | Default (recommended) | Alternatives |
|---|---|---|---|
| R1 | Subdirectory name | `trainer_rd/` | `adaptive_trainer/`, `learning_algorithm/` |
| R2 | Git placement | Two commits directly on `main`, staging ONLY port paths (tree already carries unrelated dirt; gates run pre-commit) | feature branch `trainer-rd-port` if you want a merge point for review |
| R3 | What gets tracked | **Everything except `__pycache__`** (already gitignored): data 51M, figures 35M, sandbox logs/state 3.5M. Rationale: these ARE the validation/evidence record of a frozen, advisor-facing algorithm (study npz caches, per-tester telemetry behind F72–F91, publication figures); regeneration costs hours-to-impossible (human sessions). +92M is acceptable next to already-tracked 10M+ CSVs. | thin out MP4s (−11M) via `.gitignore`; or untrack `sandbox/logs*/state*` if the pseudonymized tester telemetry shouldn't enter git |
| R4 | `/data/eli-work/scratch` afterlife | After G6 soak: directory becomes a **symlink** → `…/trainer_rd` (muscle memory, mid-protocol testers, and old notes keep working) | plain pointer-README stub; or leave a full backup copy `scratch_pre_move_bak` for one soak period (default includes this as the intermediate state) |

---

## 5. The sequential/gated execution plan

Every phase ends in a **hard gate**; a red gate stops the line (fix or roll
back — rollback story in §6). Phases 1–4 are pure additions to the repo.

**Phase 0 — Ratify + baselines (no mutations)**
- 0.1 User ratifies R1–R4 (§4).
- 0.2 Scratch baseline: full suite from `/data/eli-work/scratch`, tee'd to
  `archive/relocation_baseline_suite.log` (+ dated row in `archive/README.md`).
  Record interpreter + numpy/scipy versions in the log header.
- 0.3 Repo baseline: `.venv/bin/python -m pytest -q` from repo root on the
  dirty tree; record result + `git status --short` snapshot (the
  pre-existing-dirt manifest) in the same log.
- **Gate G0:** scratch suite = 21 files / 299 checks ALL PASSED; repo suite
  result recorded (green, or failures attributed to pre-existing dirt);
  decisions recorded.
  - *Pre-run filed 2026-07-06:* scratch `tests.test_step0` 23/23 green.
  - *Repo baseline (`.venv/bin/python -m pytest -q`, 321 s):* **485 passed,
    2 failed, 11 skipped, 28 deselected.** Both failures are pre-existing
    dirt, NOT relocation-related: `test_phase1_data.py::test_canonical_table_sizes[datasets.csv-5]`
    (expects 5 rows, tree has 7) and `::test_carry_forwards_present` — both
    caused by the two uncommitted `data/labels/datasets.csv` rows
    (`centaur_2025_iiic`, `centaur_2025_ied`) from the in-progress data work.
    **This 485/2/11 result is the frozen G0 baseline.** The relocation touches
    nothing under `data/labels/`, so G3 and G5 must reproduce this EXACT
    tally — 485 passed / 2 failed (same two IDs) / 11 skipped. A different
    number is a regression the relocation caused.

**Phase 1 — Copy (non-destructive; scratch untouched)**
- 1.1 `rsync -a --exclude='__pycache__' /data/eli-work/scratch/ <repo>/trainer_rd/`
- 1.2 Parity audit: file counts + sizes match modulo the exclusion
  (`diff -rq` with `__pycache__` filtered).
- **Gate G1:** parity clean AND canary `cd trainer_rd && python3 -m
  tests.test_step0` → 23/23.

**Phase 2 — Full functional-preservation gate at the new home**
- 2.1 Full suite from `trainer_rd/`, same interpreter, tee'd to
  `trainer_rd/archive/relocation_postmove_suite.log`.
- 2.2 Diff the per-file "PASSED — N checks" lines against the Phase-0 log:
  must be identical (21 files, 299 checks, same counts per file).
- 2.3 Non-suite smokes: `python3 -m sandbox.report --user USER-E` (state-read
  path through the moved per-tester npz/json), and an import smoke of the viz
  stack (`python3 -c "import viz.viz_style"`). No figure regeneration needed.
- **Gate G2:** 299/299 with per-file parity; smokes clean.

**Phase 3 — Repo-side isolation hardening**
- 3.1 Record `pytest --collect-only -q | tail -1` count BEFORE edits.
- 3.2 Add `trainer_rd/conftest.py` guard (A4) (+ optional pyproject
  `norecursedirs`).
- 3.3 Re-run collect-only (count must be unchanged) + full repo pytest.
- **Gate G3:** repo suite bit-identical to the G0 baseline; collection count
  unchanged; `pytest trainer_rd/` now collects 0 items (guard proven).

**Phase 4 — Documentation + provenance stitching**
- 4.1 Edits A1, A2, A3, A5 (§3).
- 4.2 `trainer_rd/docs/PROJECT_MEMORY.md`: append §6 checkpoint row
  (relocation checkpoint: what moved, both gate logs, D43-compliance note —
  zero algorithm changes) per the §0 maintenance protocol.
- **Gate G4:** every path referenced by the new doc text exists; PROJECT_MEMORY
  row filed.

**Phase 5 — Commits (the only repo mutations besides §3/§4 edits)**
- 5.1 Commit A: `trainer_rd/` verbatim ("relocate adaptive-trainer R&D
  subproject from /data/eli-work/scratch; suite 299/299 post-move").
- 5.2 Commit B: integration edits (A2/A3/A5 + any pyproject line).
- 5.3 `git status --short` must equal the G0 pre-existing-dirt manifest
  exactly (nothing unrelated swept in).
- **Gate G5:** post-commit repo pytest green; `git show --stat` sanity; dirt
  manifest unchanged.

**Phase 6 — Cutover + afterlife (the only destructive step; user-confirmed)**
- 6.1 `mv /data/eli-work/scratch /data/eli-work/scratch_pre_move_bak` then
  `ln -s <repo>/trainer_rd /data/eli-work/scratch`.
- 6.2 Through-symlink canary: `cd /data/eli-work/scratch && python3 -m
  tests.test_step0` → 23/23 (old habits + any mid-protocol tester keep working).
- 6.3 Update Claude auto-memory pointers (note: per-project memory is keyed to
  the session start directory — future sessions should start in `trainer_rd/`
  or the repo root; `docs/PROJECT_MEMORY.md` remains the project's real memory
  and travels with the directory).
- 6.4 After a soak period (user's call): delete `scratch_pre_move_bak`.
- **Gate G6:** symlink canary green; memory pointers updated.

## 6. Rollback story

- Phases 1–4: `rm -rf <repo>/trainer_rd` + revert the ≤5 repo file edits
  (`git checkout -- <files>` for tracked ones). Scratch was never touched.
- Phase 5: `git revert` the two commits (or reset if unpushed).
- Phase 6: `rm` the symlink, `mv scratch_pre_move_bak` back. Nothing is
  deleted until the user closes the soak period.

## 7. Known residual risks (accepted, with mitigations)

| Risk | Mitigation |
|---|---|
| Someone runs `pytest` with cwd inside `trainer_rd/` — rootdir detection walks up to the repo pyproject and runs the REPO suite (surprising but harmless) | Documented in trainer_rd README (A1). |
| A future repo-wide invariant scan (tree-walk test, license/grep audit) sweeps `trainer_rd/` and trips on frozen vendored code | A5 convention line in CLAUDE.md / OPEN_DECISIONS. |
| Repo `.venv` (3.11) accidentally used for the trainer suite | A1 documents the interpreter; S5 showed the dependency pins match anyway, so even the failure mode is mild. |
| Mid-protocol human testers (`sandbox/state-USER-*`) hit the old path during cutover | Phase 6 symlink + scheduling the cutover between sessions. |
| +92M git payload | R3 decision point; alternatives listed. |
