# ARCHITECTURE — module map, math↔code correspondence, and the domain interface

Audience: reviewers (mathematicians / computer scientists) reading the
codebase for the first time. Companion documents: `PROJECT_MEMORY.md`
(the evidence trail — findings F#, decisions D#, checkpoints M#),
`ADVISOR_BRIEF.md` (results summary), `learning_algorithm_plan.md` (the
POMDP design memo), `HOW_THE_TEST_WORKS.md` (the certification test).

## 1. Layers and dependency direction

```
                ┌────────────────────────────────────────────────┐
   validation   │  studies/   (pre-registered-style experiments) │
   & reporting  │  tests/     (script-style behavioral pins)     │
                │  viz/       (publication figures & films)      │
                └───────────────┬────────────────────────────────┘
                                │ imports
                ┌───────────────▼────────────────────────────────┐
   delivery     │  sandbox/   the manual-tester protocol:        │
   prototype    │  session CLI, stimulus render contract, gap    │
                │  anchors, D33 confirmation lifecycle, gates    │
                └───────────────┬────────────────────────────────┘
                                │ imports
                ┌───────────────▼────────────────────────────────┐
   trainer      │  training/  beliefs + policies + domain seam:  │
   (this        │  training_filter → mixture_filter (beliefs)    │
   project's    │  trainer_policy / trainer_greedy /             │
   core)        │  trainer_rollout (policy tiers 2/1/3)          │
                │  learner_sim (simulation ground truth)         │
                │  bank_adapter / domain (item supply)           │
                │  gap_anchor, consistency, dynamics_fit,        │
                │  training_seed, bridge_conventions             │
                └───────────────┬────────────────────────────────┘
                                │ imports
                ┌───────────────▼────────────────────────────────┐
   measurement  │  engine/    the certification engine (SMC      │
   (production  │  posterior over rater skill/bias, A-optimal    │
   home)        │  item selection, AD6 verdict policy,           │
                │  instrument_v15 versioned cut-scores/priors)   │
                └────────────────────────────────────────────────┘
```

Dependencies point strictly downward; nothing in `engine/` or
`training/` imports from `sandbox/`, `studies/`, or `viz/`. `config/`
and `data/` hold versioned instrument YAMLs and the datasets
(`data/extset/` is the scrubbed real-response release; see
`PROJECT_MEMORY.md` §3E). `reference/` is non-runnable production
reference code; `archive/` is the stale-file archive (never deleted).

## 2. Math ↔ code correspondence

| Mathematical object | Code | Notes |
|---|---|---|
| Probit-lapse SDT observation model P(y=1) = λ + (1−2λ)Φ(e^ℓ(s+θ)) | `engine/core_mcmc_general.py`, `training/bridge_conventions.py` | one model, two parameterizations; the bridge (σ = e^{−ℓ}, **t = −θ**) lives in exactly one module |
| SMC posterior over 2K static traits (reweight → ESS-triggered resample → MH rejuvenation) | `engine/core_mcmc_general.py` | the certification engine; MH-rejuvenation is valid for STATIC traits only |
| Anytime verdict rule (π_k / MCSE / R̂ gates, monotone lock) | `engine/policy_general.py` (`AD6Policy`) | K-agnostic |
| State-space learner: Rescorla–Wagner criterion dynamics + exponential log-σ relaxation with process noise | `training/learner_sim.py` (truth), `training/training_filter.py` (belief) | filter propagation mirrors the simulator's transition kernel T |
| Bootstrap particle filter for a MOVING state (reweight → systematic resample → propagate; **no MH**) | `training/training_filter.py` (`TaskFilter`) | MH replay targets the wrong distribution once the state moves (F1) |
| Exact conditional transition kernel (Gauss–Hermite marginalization) | `TaskFilter(exact_kernel=True)` | restores nominal filter calibration (F53; SBC-verified F57) |
| Bayesian model averaging over a discrete ceiling grid (σ_∞ hypotheses), prequential evidence weights | `training/mixture_filter.py` (`SigmaInfMixtureFilter`) | the honest trainability posterior P(ℓ_∞ > ℓ* | data) (D22); discreteness avoids the F4 path-degeneracy |
| Hidden-Markov regime shift at sitting boundaries: ε-contaminated state jump + fixed-share weight re-mix (Herbster–Warmuth) | `boundary_jump` / `boundary_shift` (same module) | Gate 4 (F80/F81); exact Bayes for a per-boundary prior re-draw with hazard ρ |
| Evidence-adaptive hazard: Beta posterior-mean ρ_n = ρ0·c0/(c0+n) over survive-and-reconfirm boundaries | `declared_hazard_scale`, `AdaptiveBoundaryHazard` | linear-in-ρ updates make the plug-in exact; validated opt-in, default OFF (D42 — the gain is coupled to a sharp-regression blind spot) |
| Anytime-valid e-processes (Ville's inequality): below-bar refutation gate; KT-predictor contact test; stale-mastery tripwire | `trainer_policy.EProcessGate`; `sandbox/contact.py`; the F90 stale gate | all three are nonnegative supermartingales under their nulls; P(false alarm ever) ≤ α with no correction |
| POMDP policy tiers: myopic expected-reward (tier 1), mode-gated heuristic with A-optimal probes (tier 2, shipped), H-step CRN Monte-Carlo rollout (tier 3) | `trainer_greedy.py`, `trainer_policy.py`, `trainer_rollout.py` | tier 2 is the validated default; tiers 1/3 are comparators (F56/F87: no dominance under the honest belief) |
| Deficiency-weighted cross-task allocation + finish-first + trainability discount | `trainer_policy.DeficiencyScheduler` | D3/D32/D36; expected-progress variants retained opt-in (D40/D41) |
| Spaced retention (SM-2-style doubling) + measurement-based gap re-anchoring | `trainer_policy.RetentionScheduler`, `training/gap_anchor.py` | retention fraction mixture marginalized at session close (F61/F68) |
| Terminal confirmation: confirmed tasks leave training for the retention layer; outcome-based stale revocation | `TrainerPolicy(terminal_confirmation=)` (F90) | removes the hazard re-polish tax without weakening the hazard |
| Offline dynamics estimation (penalized ML on prequential filter evidence; hierarchical shrinkage) | `training/dynamics_fit.py`, `studies/study_hier_shrink.py` | Phase-3; real-data priors F62/F65 |
| Windowed GLR learner-consistency changepoint monitor | `training/consistency.py` | shadow-only by design (D34) |

## 3. The domain interface (pluggability)

`training/domain.py` names the seam every consumer already uses:

```python
from training.domain import ArrayBank, Domain

bank = ArrayBank({0: dict(seg_id=…, s_mean=…, s_sd=…, y_star=…, margin=…)})
dom  = Domain(name="mydomain", tasks=(0,), task_names={0: "my-task"},
              ell_star={0: 0.40}, sigma_star={0: exp(-0.40)},
              assumed_params=lambda t: LearnerParams(…), bank=bank)
policy = dom.trainer(finish_first=True, probe_every=5)   # full stack
```

A domain supplies four things: an item pool with signed difficulty
signals and label confidence (`ItemBank`), per-task mastery cut-scores
(ℓ*, σ*), assumed learner dynamics (`LearnerParams`), and optionally a
belief-stack configuration. Everything else — filters, mixture, gates,
schedulers, probes, retention, the confirmation lifecycle — is
domain-independent and unchanged. `v15_domain()` constructs the shipped
instrument this way; `tests/test_m27.py` proves a synthetic toy domain
trains and graduates end-to-end with zero engine changes. The
stimulus/rendering layer is deliberately NOT part of the interface — it
is a delivery-vehicle concern (`sandbox/stimulus.py` owns the F73
ideal-observer render contract).

## 4. Conventions that govern every change

1. **Opt-in first.** New behavior ships behind a flag whose default is
   bit-identical to the previous checkpoint; defaults flip only on a
   dominating study, recorded as a dated D# decision.
2. **PECR loops.** Propose → execute → check (an end-to-end priced
   study with FG guardrails) → record (findings F#, decisions D#, a §6
   status row, tombstones for superseded text). Negative verdicts are
   recorded with the same care as wins (see M24–M26: three consecutive
   honest NOs).
3. **Priorities are normative:** false graduation ≥ lateness > speed >
   retention (D28); FG is priced at the CONFIRMED level (D41).
4. **One bridge.** All parameter conversions live in
   `training/bridge_conventions.py` (σ = e^{−ℓ}, **t = −θ** — the sign
   flip that reverses every bias update if forgotten).
5. **No pickle** in persisted state (npz + json only); state is keyed
   per learner (F72); RNG streams are not statistics and are not
   persisted.
6. **Script-style tests** (`python3 -m tests.test_*`), one behavioral
   pin per shipped mechanism; the full suite must be green before a
   checkpoint closes.

## 5. Reading order for a new reviewer

1. `docs/PROJECT_MEMORY.md` header + §1 (system map) — 15 minutes.
2. `docs/HOW_THE_TEST_WORKS.md` then `docs/learning_algorithm_plan.md`
   — the measurement and training designs.
3. `training/mixture_filter.py` and `training/trainer_policy.py`
   docstrings — the two central modules; every parameter cites the
   study that pinned it.
4. `docs/ADVISOR_BRIEF.md` — results, honest OCs, open decision asks.
5. The latest M# analysis docs (M21–M27) for the evidence chain on real
   testers.
