# learning-engine-cleaned

The learning protocol engine, packaged self-contained for sharing. It
sits between an adaptive testing system and a certification exam: it
ingests the tester's posterior over a learner's per-domain skill and
bias, chooses the training sequence that maximizes progress toward the
exam, and stops with a calibrated "ready" flag. Design stance: the only
fixed numbers are the design lapse (0.025) and the exam specification
the certifying field already owns; every other quantity (learning
rates, skill floors, the difficulty gate, state priors, dispersions,
signal thresholds) is estimated from a pilot population.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python demo_end_to_end.py
```

The demo needs no data and no services. It generates a synthetic
population, fits the hierarchical population model at smoke MCMC scale,
builds the deployable artifact, curates a certification exam, trains
fresh learners closed-loop with the production engine (readiness flags,
logged-only plateau signals), and exercises the testing-system handoff
with a mock result. A few minutes on a laptop-class CPU; it ends with a
`DEMO CHECKS:` line where every entry should read `ok`.

## Layout

    learning_engine/learnmodel.py   gate derivations and the learned-gate
                                    basis (configurable centers, rail
                                    check) — numpy/scipy only
    learning_engine/multimodel.py   M1: the multiclass hierarchical
                                    population model (NumPyro NUTS),
                                    exact masked likelihood, synthetic
                                    cohort generator
    learning_engine/msengine.py     M2: the production session engine —
                                    particle belief (per-particle floors
                                    and rates), exam-margin placement,
                                    double-eta readiness, score
                                    prediction/calibration, plateau
                                    forecasts, testing-system ports
    docs/handoff_contract.md        the exact contract with the adaptive
                                    testing package (both directions)
    docs/learning_engine_derivation.md  the derivation memo
    demo_end_to_end.py              runnable end-to-end demonstration

## Using it

```python
from learning_engine import (run_hier_multi, MSSessionEngine, MSBelief,
                             prior_blocks_from_population)

# M1: one hierarchical fit per repository -> population posterior
mcmc, post = run_hier_multi(cohort, form="soft", use_rt=True,
                            fatigue=True, dense_pop=True)

# artifact: the single deployable object (schema below)
art = build_artifact(post)          # see demo_end_to_end.build_artifact

# M2: train one learner against an exam
eng = MSSessionEngine(art, signals, bank, exam, eta=0.10, budget=300)
result = eng.run(learner_step,                  # item -> response index
                 testing_result=pre_test_dict,  # optional: seed belief
                 # ...or testing_trials=raw_stream (contract section 2a
                 # replay seeding — never together with testing_result)
                 on_question=logger)            # optional, passive
# result: ready_at, n_used, pass_prob, per-question log, final belief
```

- `cohort` is a dict of arrays (`s`, `ystar`, `y`, `mask`, `fpos`,
  optional `rt`, `display`, ...); see `make_multi_cohort` for the exact
  schema and a synthetic generator.
- `bank` is a list of `dict(item_id, s (M,), gold)`; `exam` is
  `dict(n_items, s, gold, pass_count)`.
- The belief can be seeded from the adaptive testing engine's posterior
  (`testing_result` / `seed_from_testing`) and the population fit
  supplies the tester's next-cohort prior
  (`prior_blocks_from_population`) — both directions of
  `docs/handoff_contract.md`.

## The artifact

The population fit is deployed as one dict:

    alpha_t, alpha_s        per-signal learning rates (criterion, skill)
    alpha_t_sd, alpha_s_sd  optional rate dispersion -> per-particle rates
    lam                     lapse
    q_t, q_s                process noise carried in the belief
    w_coef, w_centers       optional learned gate (simplex weights +
                            basis centers); absent -> Wilson gate
    state_prior             mu_t0, tau_t0, mu_u0, tau_u0, gamma
                            (one-factor transfer across signals)
    floor_prior             (mu, sd) per signal on the skill floor —
                            particles carry OWN floors (the honesty fix)
    floor_joint             optional (contract v1.1): per-signal
                            regression of the floor on starting skill
                            (slope, intercept, resid_sd) -> floors drawn
                            CONDITIONALLY on each particle's skill, at
                            construction and after posterior seeding
                            (repairs the severed state-floor joint,
                            the D49 attainability-optimism mechanism)

## Operating rules that ship with the algorithm

These are measured requirements, not conventions (decision-log IDs
refer to the parent repo's `DECISION_LOG.md`):

- **Exam curation.** The certification exam must satisfy
  floor-accuracy > start-accuracy under the fitted population geometry,
  with the pass bar inside the trainable window. A random item mix can
  make the fully-trained state score BELOW the start state and voids
  readiness semantics (D45). `demo_end_to_end.build_exam` implements
  the curation rule.
- **Exams are taken rested.** The fatigue-blind flag stays honest on a
  rested exam under planted fatigue (it costs questions, not honesty),
  but end-of-session fatigue breaks the guarantee (0.90 -> 0.62 at
  severe fatigue). Schedule pre/post/certification exams in fresh
  sessions (D45).
- **Readiness = the double-eta rule.** Flag when P(pass | state) >=
  1 - eta holds with posterior probability >= 1 - eta, with per-particle
  floors (D36). On a heterogeneous population only part of the cohort
  can honestly certify at a fixed bar — the engine says so rather than
  promising otherwise.
- **Plateau signals are logged-only.** `forecast_score` (deterministic)
  and `forecast_score_mc` (expected posterior path) are diagnostic
  signals; their honesty comes from pre-test seeding plus a
  population-calibrated threshold, and they must never stop a session
  (D45). Expected reweighting is a martingale under the belief's own
  model — no self-consistent forecast can promise the drift that comes
  from prior misconcentration.
- **The learned gate deploys RAW, jointly with its rates.** The simplex
  fixes the gate's scale and the rate carries magnitude; renormalizing
  a fitted gate is a deployment bug (D42). If a fitted gate rails at
  the basis edge (`gate_railed`), widen the basis (`gate_centers`) and
  refit; artifacts carry `w_centers` so deployment always matches the
  fit.
- **Sampler honesty.** No fit statistic is consumed without R-hat,
  divergence, and tree-saturation checks (`run_hier_multi` exposes
  them via the returned MCMC object).
- **Seeding is posterior XOR replay.** A session belief seeds from the
  testing posterior (`testing_result`/cloud) OR by replaying the raw
  pre-test stream through the engine's own observation model
  (`testing_trials`, contract §2a) — never both: they encode the same
  responses (double counting; enforced with a ValueError). Replay is
  preferred when the stream is available (D49 §2.6, D51); it runs on an
  expanded particle cloud contracted to session size (the depletion
  guard), and replayed items are never re-served. Replay is
  reweight-only: the test shows no feedback, so the learning dynamics
  do not advance.

## Not included (and where it lives)

- The real pilot dataset and fitted real-population artifacts (private;
  parent repo `learning-algorithm-real-data/`, `sim/results/`).
- The adaptive TESTING engine (separate package,
  `testing-algo-cleaned`); this package holds only the contract and the
  ports, which run standalone (the demo uses a mock result).
- The experiment/analysis scripts, the trial pipeline, and the decision
  log (parent repo `sim/exp_*.py`, `DECISION_LOG.md`).

## Provenance

Copied verbatim (import paths only rewritten) from the
`ideal-test-learning` repo at the D45 close-out state, 2026-07-16.
Validation history: model recovery D30, honesty D33/D36/D37, learned
gate D42, score calibration D43, pre-trial hardening D45. Updated
2026-07-16: contract v1.1 replay seeding + `floor_joint` (D51,
validated in the parent repo's `sim/exp_v0b.py`). Python 3.12,
CPU-only; versions pinned in `requirements.txt`.
