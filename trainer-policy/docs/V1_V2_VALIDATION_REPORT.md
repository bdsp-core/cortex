# Trainer validation report: selection precision, improvement at fixed budget, modularity

Status: v2.0 — supersedes the earlier exam-verdict A/B framing in
full. Reason: the available response corpus comes from RANDOMLY
selected questions with binary right/wrong feedback — no usage data
under either training algorithm exists — so neither a "fitted truth"
nor the incumbent's assumed constants can referee a certification-
honesty contest, and passing bars inherited the same lineage. This
report validates the trainer claims directly, with no exam or cut
anywhere in the scoring. Terminology: domain1 = the binary
present/absent task, domain2..domain7 = the n-way identification
classes; code identifiers quoted verbatim only to locate things.

## 1. The three claims and their verdicts

1. **The engine selects questions more precisely.** **Proven,
   decisively.** Measured directly: the realized learning-gate weight
   of each served item against the simulated trainee's TRUE state.
   Engine 0.55–0.68 vs incumbent 0.38–0.47 — in every cell of a
   9-cell robustness grid. The engine's items land ~1.4–1.6× more
   on-gate.
2. **The engine improves skill and bias better per question.**
   **Skill: won or tied everywhere except the incumbent's own
   assumed world.** At a fixed 450-item budget the engine gains more
   skill on slow-learner cells (+0.029, +0.017 — the incumbent's
   fast-rate assumption under-trains slow learners), ties the mid-
   rate and every deviant cell (plateau, fatigue, random-walk), and
   trails only in the two cells whose truth equals the incumbent's
   own anchored rates (−0.071, −0.089) — the one hypothesis whose
   data lineage (one-shot fits) is known to inflate rates severalfold.
   **Bias: small uniform deficit** (−0.03..−0.06 over 450 items,
   against typical starting offsets of ~0.24), a deliberate
   robustness trade explained in §4.
3. **The methodology is modular.** **Proven.** An unseen binary
   domain plus an unseen 3-way group were added by registry/data
   only — no engine code changes — cold-started from the population
   hyperprior, served, and tracked end to end (also guarded
   permanently by the package's extensibility regression).

## 2. Design

- **Neutral truth grid, nobody's home field privileged**: simulated
  trainees at {slow 0.006, mid 0.020, fast 0.047} skill-learning
  rates × {low, high} state noise (the two legacy hypotheses appear
  only as unprivileged endpoints; expert-level skill ceilings), plus
  three deviant kinds — plateau (no headroom), fatigue (within-
  session degradation), random-walk (no learning). 24 paired
  trainees per cell.
- **Both trainers as deployed, blind to the truth**: the incumbent
  with its production filter constants and targets; the engine with
  its artifact-based belief. Fixed configs across all cells.
- **Shared pre-test, identical information**: one real testing-engine
  session per trainee seeds BOTH trainers with the same posterior
  cloud — differences are question selection, not initialization.
  Training draws from the real production bank (feedback-safe pools).
  For this comparison the engine runs in full-training mode (its
  stop-at-the-certification-bar economy is a separate feature,
  evidenced separately); both arms spend the full budget.
- **Scoring**: true skill (per-domain log-skill) and bias (criterion
  offset magnitude) trajectories, snapshotted at 150/300/450 served
  items. Gates per cell, paired, 2 SE.

## 3. Results (450-item budget; Δ = trainee's true state change)

| cell | Δskill incumbent | Δskill engine | Δbias engine−incumbent | gate wt inc. | gate wt eng. |
|---|---|---|---|---|---|
| slow, low noise | 0.094 | **0.123** | −0.040 | 0.43 | **0.63** |
| slow, high noise | 0.097 | **0.114** | −0.030 (ns) | 0.47 | **0.60** |
| mid, low noise | 0.251 | 0.241 (tie) | −0.063 | 0.42 | **0.61** |
| mid, high noise | 0.256 | 0.252 (tie) | −0.039 | 0.45 | **0.59** |
| fast, low noise | **0.425** | 0.354 | −0.061 | 0.43 | **0.59** |
| fast, high noise | **0.425** | 0.336 | −0.038 | 0.46 | **0.55** |
| plateau | 0.054 | 0.056 (tie) | −0.062 | 0.39 | **0.68** |
| fatigue | 0.248 | 0.257 (tie) | −0.059 | 0.40 | **0.65** |
| random-walk | −0.010 | −0.010 (tie) | −0.058 | 0.38 | **0.63** |

Formal gates: selection precision PASS 9/9; skill-at-budget PASS
7/9 (fails the two fast cells); bias-at-budget PASS 1/9; modularity
PASS.

## 4. Reading the two residuals — one shared root

**Fast-cell skill.** The engine places items better in those cells
too (0.55–0.59 vs 0.43–0.46) yet gains less, because allocation, not
placement, is the bottleneck: at 8×-faster-than-fitted learning, the
incumbent's filters — which assume exactly those rates — track each
domain's saturation and reallocate precisely, while the engine's
belief lags and keeps training domains that are already at ceiling.
(Adaptive placement partially masks the response signal that would
correct the lag.) Symmetrically, in the slow cells the incumbent's
fast-rate assumption declares progress that has not happened and
under-trains — there the engine wins. Neither trainer dominates the
grid; the engine dominates the region the only available real data
supports.

**Bias.** The engine's bias mode is evidence-gated: it fires only
when the belief is confident the criterion is actually offset, and
it anchors its items at the item pool's own label boundary. Both
guards exist because removing them was TESTED and measured to
backfire (an ungated bias mode chases the belief's own estimation
error and pushes real bias UP; a belief-anchored placement does the
same under tracking noise). The incumbent can afford an aggressive
bias mode because — under its own assumed dynamics — its criterion
tracking is trustworthy. The engine trades a small correction-rate
deficit (−0.03..−0.06 per 450 items) for never anti-correcting under
dynamics uncertainty.

**The shared root and its closure.** Both residuals are the same
missing dataset: trainee-population learning dynamics. Nobody has
them (the response corpus is random-selection, feedback-ambiguous,
cross-sectional for training purposes). The adoption roadmap
pre-registered the fix (§2.4): the trainer's own live ledgers are
the first longitudinal training data for this population; the
population refit replaces the borrowed dynamics, after which this
grid re-runs cheaply and both residuals are expected to close (the
engine already wins wherever its dynamics prior is nearer the truth
than the incumbent's).

## 5. Modularity (claim 3, detail)

Adding a domain is a data/config operation: one registry line, items
carrying the new domain's evidence columns, a cut from the field's
standard-setting when certification needs it. New domains cold-start
from the population artifact's hyperprior block (honestly wider
uncertainty, partial pooling) and tighten at the first scheduled
refit. Demonstrated in this validation (unseen binary + unseen 3-way
group, zero engine edits) and enforced by a permanent regression
test. The incumbent's per-domain filter stack has no analogous path —
each added domain replicates the full heuristic set.

## 6. Operational findings that stand regardless of framing

1. **Cut-block wiring**: the exam certifies at `ell_star_unified_v14`
   while the trainer-side loader defaults to `v15`; any readiness
   statistic aimed at v15 trains to a bar below the certifying
   instrument's. Fix the default or make the block explicit.
2. **Seeding**: replaying the raw test sequence (segment ids, asked
   domains, full picks, responses) through the engine's observation
   model beats consuming the summary cloud — better state fidelity
   and calibrated attainability estimates. The raw per-question log
   should be part of the standard test→trainer handoff. Never
   combine replay with cloud seeding (double counting).
3. **Exposure economy**: production feedback-safe pools (12–15k per
   domain) never bind — zero exhaustions across all sessions run.
4. **Determinism**: bitwise-reproducible engine and adapter sessions
   under the single-thread BLAS regime, re-verified after every
   change.
5. **Population attainability (context for cut policy, with the data
   caveat front and center)**: under a population resembling the
   random-selection validation cohort, the all-domain PASS
   requirement is reachable by at most ~3–6% (bottleneck domain3);
   under expert-panel-level ceilings every cut is reachable. Whether
   trainees resemble one or the other is exactly what live data will
   show; the cuts decision is the team's.

## 7. Requested decisions

1. **V4 live-shadow green-light** — the critical path: it produces
   the trainee ledgers for the §2.4 dynamics refit, which closes both
   measured residuals and enables the definitive rerun of this grid.
2. **Package sync** — the integration repo's engine copy predates the
   adapter (which now carries the validated selection stack); propose
   a drop plus the determinism suite in your CI.
3. **Raw-log handoff** — add the per-question test log to the
   test→trainer contract (§6.2).
4. **Cut policy discussion** — §6.5, at the team's discretion.

## 8. Provenance

Constants discipline maintained: DESIGN (λ=0.025) / FIELD (pools,
anchors, α, Z, exposure contract) / POPULATION-ESTIMATED (rates,
floors, dispersions, noise) / DERIVED (placement targets, value
objective, readiness/attainability statistics). Every negative
intermediate result is retained in the decision log (D50), including
the measured failure of an unthresholded bias/skill value comparison
— the reason the shipped bias mode is evidence-gated and boundary-
anchored. Artifacts: `sim/exp_v1b.py`, checkpoints `results/v1b.*`,
run logs 1–4; adapter `learning-engine-cleaned/adapter/le_adapter.py`.
