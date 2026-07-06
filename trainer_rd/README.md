# trainer_rd — adaptive-trainer R&D subproject

> **What this is.** A self-contained R&D subproject: the eval→training→re-cert
> learning-algorithm prototype (SMC measurement of rater skill/bias → POMDP
> training protocol → re-certification). It was developed at
> `/data/eli-work/scratch` and relocated here **verbatim** on 2026-07-06
> (`docs/REPO_RELOCATION_PLAN.md`). It joins `methodology_rd/` and
> `discrimination_rd/` as an R&D sibling and is **not part of the shipped
> distribution** — it is absent from the `pyproject.toml` package list, and the
> repo's pytest suite (`testpaths = ["tests"]`) never collects it.
>
> **How it runs — from THIS directory, with the system interpreter.**
> ```sh
> cd trainer_rd
> python3 -m tests.test_step0                 # any test; full suite = 21 files / 299 checks
> python3 -m studies.study_misspec --smoke    # any study
> python3 -m sandbox.session --user NAME      # the manual-tester protocol
> ```
> Uses the **system `python3`** (anaconda 3.12.4: numpy 1.26.4 / scipy 1.13.1 /
> pandas 2.2.2 — the repo's exact pins). The repo's own `.venv` (Python 3.11)
> is for the repo pytest suite only; the two interpreters never cross. `pytest`
> does **not** run this suite — the tests are script-style and execute on
> import, so `conftest.py` here guards against accidental `pytest trainer_rd/`
> collection.
>
> **Frozen for advisor review (D43).** The algorithm is submission-frozen. The
> vendored `engine/` copy, `auroc.py`, and the `*_general` data banks stay
> vendored — converging them with the production engines is the future port
> (`docs/PHASE3_AND_PORT.md`), not this relocation. Do **not** sweep this
> directory into repo-wide invariant / tree-walk scans (see the repo `CLAUDE.md`
> "what NOT to do").
>
> **The real project memory** is `docs/PROJECT_MEMORY.md` (single source of
> truth: system map, findings F#, decisions D#, status log M#) — read it first;
> `docs/ARCHITECTURE.md` is the module map + math↔code correspondence, and
> `docs/ADVISOR_BRIEF.md` holds the results + open decision asks.

---

# Scratch — certification-engine + trainer prototype

New reviewer? Start with `docs/ARCHITECTURE.md` (module map, math↔code
correspondence, the domain plug-in interface), then
`docs/PROJECT_MEMORY.md` (the evidence trail) and
`docs/ADVISOR_BRIEF.md` (results + open decision asks).

## Layout

| Directory    | Contents |
|--------------|----------|
| `engine/`    | Certification engine: `core_mcmc_general` (particle engine), `auroc`, `policy_general` (AD6 stopping policy), `instrument_v15` (v15 staged instrument sim) |
| `training/`  | Trainer stack: `learner_sim` / `misspec_learners` (simulated raters), `bank_adapter` (real 89k bank), `bridge_conventions`, `training_seed`, `training_filter` (per-task particle filter), `trainer_greedy` / `trainer_policy` / `trainer_rollout` (Tier 1–3 selection), `benchmark_trainer`, `pipeline_demo` (end-to-end eval→train→eval) |
| `sandbox/`   | The manual-tester protocol (delivery-vehicle prototype): `python3 -m sandbox.session --user NAME` trains a human on the full validated stack with production-shaped telemetry; `sandbox.report` for analytics |
| `studies/`   | Experiment campaigns (`study_*.py`); most support `--smoke`, the v15 ones `--pilot`. Results land in `figures/data_*.npz` |
| `tests/`     | Script-style smoke tests (`test_step0` … `test_m27`, one file per shipped mechanism); the full suite must be green before a checkpoint closes |
| `viz/`       | `viz_style` (shared palette, `FIGDIR`) + `make_*` figure/video scripts |
| `reference/` | Port-target reference files (`session_controller_general`, `policy_k7_general`, `beautiful_figure_example_general`). These import modules of the production repo (`core_mcmc`, `policy`, `engine_inputs_k7`, …) and are **not runnable here** — kept verbatim for the port |
| `config/`    | `cert_config_general.yaml` (cut-scores), `deployment_yaml_general.yaml` |
| `data/`      | Bank CSVs (`segment_signals/labels_general.csv`), `Sigma_l_fitted_k7_general.npy`; `data/sessions/` holds the 3 real anonymized eval sessions; `data/extset/` is the scrubbed EXTSET real-response release (125,860 reads; inventory in `PROJECT_MEMORY.md` §3E) |
| `docs/`      | `ARCHITECTURE.md`, `PROJECT_MEMORY.md` (single source of truth), `ADVISOR_BRIEF.md`, the M# checkpoint analyses, plans and specs |
| `figures/`   | Generated outputs: campaign `.npz` caches, figures, videos |
| `archive/`   | Stale files moved aside (never deleted); each move logged in `archive/README.md` |

## Running

Everything runs **from this directory** with `python3 -m <pkg>.<module>`:

```bash
python3 -m tests.test_step0                # any test
python3 -m studies.study_misspec --smoke   # any study
python3 -m viz.make_figures                # figures from cached npz
```

Studies that write output use the CWD-relative `figures/` path, so run them
from this directory. Library code (config, bank CSVs, FIGDIR) resolves paths
relative to its own file location and works from anywhere.
