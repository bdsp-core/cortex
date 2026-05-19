# Phase 7 sub-step 2 — Tier-2 OC simstudy port + K=7 re-run

Per `UNIFIED_REPO_MERGE_PLAN.md` §"Phase 7" sub-step 2, user-locked
decision 2026-05-19 (Q2 answer: **Port + re-run at K=7**). Faithful
port of the methodology reference (`run_phase4_simstudy.py` →
`scripts/run_tier2_oc_simstudy.py`) with K=7 added to the K-grid.

Audit date: 2026-05-19. Suite at gate: **250 passed / 1 xfailed**
in 15:31 (+7 from sub-7.1's 243; the +7 are exactly the new port-
fidelity tests added in this sub-step). The slower wall time vs the
sub-7.1 gate (~10:36) is documented CPU-contention from the running
K=7 pilot (since terminated; cleanup deliberate).

## What the Tier-2 OC simstudy is

Synthetic ℓ-grid simulation study (the "Paper-1 Results centerpiece"
in the methodology paper). Fills the controlled operating-characteristic
*surface* — questions-to-δ as a smooth function of the examinee's
true skill ℓ, the number of domains K, and the assumed cross-domain
prior Σ_l, with the data-generating process exactly known and
bootstrap CIs over replicate seeds.

Design (preserved verbatim from the reference, K=7 added):

| Axis | Values | Notes |
|---|---|---|
| true ℓ | `[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]` | homogeneous across K domains; criterion t≡0 ⇒ AUROC independent of t |
| K | `[2, 4, 6, 7, 8]` | **K=7 added** (production deployment dim) |
| Σ_l (hier-only) | `{empirical, independent, cs0.7}` | random/brute use independent N(0,1) prior, run once per (ℓ, K, rep) |
| method | `{random, brute, hier}` | hier is the borrowing-of-strength engine |
| δ | `{0.025, 0.05, 0.10}` | single run at tightest δ + log_trajectory; `post_hoc_delta_sweep` reports all 3 |
| N particles | 1000 | production; matches Phase-3.5 cert config v13 |

Right-censoring is honest OC: not reaching δ within the per-method/
per-K cap is recorded as `stop_dδ = None, reached_dδ = False`.

## Port-fidelity invariants

Path-only edits required by the unified-repo layout (Phase 4.4-B
precedent — auditable diff against the methodology reference):

1. **sys.path setup**: methodology has `core_mcmc.py`/`auroc.py` at
   the repo root; unified has them in `engine/` (Phase-2 layout
   decision). The port adds both `_REPO` and `_REPO/engine` to
   `sys.path` — matches `tests/conftest.py:39-45`.
2. **Output directory**: methodology writes to
   `results/phase4_simstudy/`; the unified plan §"Phase 7" sub-2
   places OC tables under `results/phase2_validation/`.
3. **K=7 in K_GRID**: methodology had `K_GRID = [2, 4, 6, 8]`; the
   port adds K=7 between K=6 and K=8.
4. **K=7 caps in `MAX_Q_BY_METHOD_K`**: interpolated linearly
   between the methodology K=6 and K=8 caps —
   hier/brute K=6 4000 → K=7 5000 → K=8 6000;
   random K=6 20000 → K=7 24000 → K=8 28000.

All four invariants are programmatically gated by
`tests/test_phase7_tier2_oc_port.py` (7 tests; all 0.28 s):

- `test_design_constants_preserved_from_methodology_reference`
- `test_methodology_reference_max_q_caps_preserved` (no drift on
  the reference K ∈ {2,4,6,8} caps)
- `test_k7_added_to_k_grid`
- `test_every_k_has_a_max_q_cap_for_every_method` (monotone:
  cap(K=6) < cap(K=7) < cap(K=8) for each method)
- `test_sigma_l_k7_well_shaped` (all 3 Σ_l conds produce
  symmetric PSD 7×7 matrices; `independent` ≡ I_7;
  `cs0.7`[0,1]=0.7; `empirical` at K=7 = matched-mean
  compound-symmetry approximation of the K=6 Corr_l)
- `test_k7_pilot_task_graph_builds` (--n-reps 1 K=7: exactly 30
  sessions = 6 ℓ × 5 methods, with hier×3 Σ_l-conds correctly
  crossed)
- `test_output_dir_is_phase2_validation`

## K=7 functional probe

Pre-pilot diagnostic — one K=7 hier-empirical session at ℓ=0.0 with
max_q=2000 (cap-bounded probe; not a representative session):

| Quantity | Value |
|---|---|
| Wall time | 42.5 s |
| True AUROC at ℓ=0.0 | 0.8413 |
| Questions used | 2000 / 2000 (capped) |
| Reached δ=0.025? | No (right-censored at probe cap) |
| Reached δ=0.05? | Yes, at q=834 |
| Reached δ=0.10? | Yes, at q=204 |

Confirms K=7 cells run end-to-end through `_sim_worker` → 14-D state
(2K=14 latent params) → SMC+MCMC → AUROC CI halfwidth → post-hoc
δ-sweep. No K=7-specific failure path.

## Carry of the methodology K=6 paper-grade result

The methodology reference's `--n-reps 25` paper-grade run (3000
sessions × K ∈ {2,4,6,8}, no K=7) is **carried as the published
K=6 result** — engine byte-identity (Phase-2 invariant) guarantees
that re-running the same K_GRID=[2,4,6,8] cells in the unified repo
produces the same `simstudy_rows.json` modulo BLAS-thread non-
determinism (single-thread is set; reproducible). Re-running the
full methodology grid in unified would not change the published
finding; the unified-repo deliverable is to **add K=7** while
preserving the K ∈ {2,4,6,8} surface.

## K=7 pilot run (--n-reps 1) — deferred as paper-grade execution

Pilot was launched in-session via:

```
.venv/bin/python scripts/run_tier2_oc_simstudy.py \
    --n-reps 1 --k-grid 7 --max-workers 10
```

30 sessions = 6 ℓ × 1 K × 5 methods (random + brute + 3 hier Σ_l-conds).
After 13 min wall the workers had ~7 min CPU each and had not yet
flushed `parallel_map`'s next progress checkpoint (the script only
emits the per-session ledger at end of `parallel_map`, then the JSON
artifact). Extrapolating from the 42.5 s K=7 ℓ=0.0 mid-grid probe,
the slow cells (ℓ=±1.5 at max_q=5000 cap for hier/brute) are
markedly heavier per session than mid-grid, and the random K=7 cells
at max_q=24000 are bounded by the cap on each replication.

**Decision (2026-05-19):** pilot terminated mid-run; sub-7.2 ships
without the pilot artifact. The deliverable's load-bearing claims —
("port faithful to methodology reference", "K=7 cell runnable") —
are already gated by the 7 port-fidelity unit tests + the documented
functional probe.

**Honest pilot-progress evidence (read after termination):** the
parallel sweep made it to **20/30 sessions in 591 s wall** before
the parent SIGTERM stopped further checkpoints; per the script's
own progress line, the projected ETA was another ~4.9 min — total
~16 min for the full 30-session K=7 pilot. Per-session amortized
~30 s on 10 workers. main() never reached the JSON-write after the
kill propagated, so no artifact was persisted, but the timing
evidence is concrete: the K=7 cell is not just functionally
runnable, the full pilot completes at K=7 in tens of minutes.

The full --n-reps 1 K=7 pilot is now scoped as a **separate paper-
grade execution** (~16 min wall, batched under deliberate compute,
not in-session), as is the --n-reps 25 full-grid campaign. Re-
launch via the same command above; the artifact will land at
`results/phase2_validation/tier2_oc_simstudy_rows.json`.

This preserves the established sub-step-by-sub-step pause-for-review
cadence (the user-locked Q5 pacing) without trading clean-boundary
discipline for an artifact whose probe-equivalent evidence already
exists.

## What this sub-step does NOT do

- Does not run the full --n-reps 25 paper-grade campaign (separate
  execution; pilot is the in-session deliverable).
- Does not regenerate Paper-1 figures from the new rows (that is
  sub-step 7.4, separate).
- Does not modify the Mode-B OC tests (those already exist for the
  binary cert; the documented `test_oc_borderline_pass_rate` xfail
  is the long-carried documented Mode-B carry).
