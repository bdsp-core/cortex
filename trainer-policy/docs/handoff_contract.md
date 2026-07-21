# Handoff contract: adaptive testing engine ↔ learning engine

Version 1.1 (2026-07-16; v1.0 2026-07-13). v1.1 adds §2a (raw trial
stream + seeding exclusivity) and the §4 `floor_joint` block — additive
optional fields, minor bump per §7. This contract binds the adaptive TESTING engine
(`testing-algo-cleaned/`, package `adaptive_testing`) to the LEARNING
engine (`sim/msengine.py`) and the population learner (M1,
`sim/multimodel.py`). It is written against the testing package's actual
dataclasses (`SessionResult`, `TrialRecord`, `PosteriorSummary`,
`ItemCandidate`, `Prior`) as shipped, not against an idealization.

```
                 ┌─────────────────────────────────────────────┐
   pilot data ──►│ M1 population learner (multimodel)          │
                 │  rates, gate, floors, transfer, chronometry │
                 └──────┬───────────────────────────┬──────────┘
                        │ §5 prior blocks           │ §4 population artifact
                        ▼                           ▼
                 ┌──────────────┐   §2 posterior   ┌──────────────┐
   respondent ──►│ TESTING      │ ───────────────► │ LEARNING     │──► items,
                 │ engine       │   (result JSON   │ engine (M2)  │    readiness
                 │ (particle    │    + cloud npz)  │ (RBPF belief │    flag
                 │  cloud, MH)  │                  │  + dynamics) │
                 └──────┬───────┘                  └──────┬───────┘
                        │            §1 item bank         │
                        └───────────────┬─────────────────┘
                                        ▼
                          crowd-anchored per-signal axes
                                        │ §6 session logs
                                        ▼
                              back to the M1 repository
```

## 0. Shared conventions

- **Dimensions = signals.** The testing engine's `dimension_names` and the
  learning engine's `signals` list must be byte-identical, same order.
- **The one shared constant.** Both engines default the lapse rate to
  0.025 (`ModelConfig.lapse_rate` there, `LAPSE_DESIGN` here). A deployment
  overriding one side must override the other.
- **Parameter mapping.** The testing engine estimates per-dimension
  response offsets `o_k` and log sensitivities `l_k` in the model
  `P(y=1) = λ + (1−2λ) Φ(exp(l_k)(s + o_k) / sqrt(1 + (exp(l_k) u)^2))`.
  The learning engine's state is criterion `t_k` and log-noise
  `u_k = log σ_k` in `P = λ + (1−2λ) Φ((s − t_k)/σ_k)`. The exact bijection:

  ```
  t_k = −o_k          σ_k = exp(−l_k)          u_k = −l_k
  ```

  The testing engine's signal-uncertainty marginalization (its `u` =
  `signal_sd`) has no learning-engine counterpart yet; the learning engine
  treats `signal_mean` as the item's evidence coordinate. Item banks built
  under §1 carry `signal_sd` so the testing side keeps its exactness.

## 1. Item bank (shared input)

Producer: this project's ingestion layer (`sim/realdata.py`). Consumer:
both engines.

One JSON file per task, the testing package's native schema
(`ItemBank.from_json` compatible):

```json
{
  "dimension_names": ["signal-2", "...", "signal-7"],
  "candidates": [
    {"item_id": "case-12841", "dimension": "signal-4",
     "signal_mean": -0.412, "signal_sd": 0.121}
  ]
}
```

- `signal_mean` = the crowd-anchored evidence axis `s_jm` (probit of the
  EB-shrunken leave-one-out crowd rate, D28) of item j on its GOLD channel.
- `signal_sd` = the delta-method standard error of that probit rate.
- An `item_id` may appear once per dimension; serving an item retires
  every occurrence (the testing package enforces this; the learning engine
  must respect the same rule via its `served` set).
- The learning engine additionally needs the full evidence ROW `s_j`
  (all M channels) per item; the extended bank file adds an optional
  `"signal_row": [..M floats..]` field per candidate that the testing
  engine ignores.

## 2. Testing → learning: the posterior handoff

Producer: `adaptive_testing.storage.write_result_json(result, path)` and,
when configured with `SessionConfig(capture_clouds=True)`,
`write_cloud_snapshots(result, path)`. Consumer:
`msengine.belief_seed_from_testing_result`.

- **Preferred payload**: the final cloud snapshot — arrays
  `response_offset (N, K)`, `log_sensitivity (N, K)`, `weights (N,)`.
  The learning belief resamples its particles from this weighted cloud
  through the §0 mapping. Cross-dimension posterior correlations (the
  testing prior's information sharing) survive the handoff exactly.
- **Fallback payload**: `result["final_summary"]` (per-dimension means and
  variances). The learning belief seeds an independent Gaussian per
  dimension; correlations are lost. Deployments SHOULD capture clouds.
- Required result fields: `dimension_names`, `final_summary`,
  `stop_reason`, `n_questions`, `served_item_ids` (the learning engine
  must not re-serve them), `seed`.
- The learning engine ignores `score_*` fields (the testing package's
  reporting transform is presentation-layer by its own contract).

### 2a. The raw trial stream (replay seeding, v1.1)

Producer: the testing package's per-trial log (`TrialRecord` /
`JsonLinesTrialWriter`). Consumer:
`msengine.MSBelief.seed_from_test_replay`, reachable through
`MSSessionEngine.run(testing_trials=...)`.

The test's EXACT question sequence, one record per trial in served order:

```json
{"k": 0, "item_id": "case-12841", "response": 3,
 "displayed_at": "2026-07-16T10:02:11Z", "submitted_at": "..."}
```

- `response` is the FULL n-way pick index — never binarized (binarized
  replay measurably corrupts criterion estimates, `sim/exp_v0.py` /
  D48 §2.6). Timestamps are optional until true RT logging lands
  (`next_cohort_design.md` §4.1); they are carried for the future
  chronometric replay channel, not consumed today.
- `item_id` must join the §1 bank. A record MAY instead carry an explicit
  `"s"` evidence row for consumers that cannot join the bank.
- The learning engine replays the stream through its OWN observation
  model, reweight-only: the test shows no feedback, so under the
  feedback-gated learning law the dynamics do not advance. Replayed
  `item_id`s join the engine's served set (the no-repeat rule).
- **Seeding exclusivity.** A deployment seeds a belief from the §2
  posterior payload OR the §2a raw stream, NEVER both — they encode the
  same responses (double counting). When both are available, replay is
  preferred (D49 §2.6 adjudication: matches-or-beats cloud on state
  fidelity and attainability calibration).

## 3. Learning-engine outputs

- **Item requests**: the selected candidate's `item_id` (the application
  presents it and returns the categorical response index).
- **Readiness flag**: `{"ready": bool, "at_question": int,
  "pass_prob": float, "rule": "double-eta-quantile", "eta": float}`.
  The flag semantics (D36): P(pass | θ) ≥ 1−η holds with posterior
  probability ≥ 1−η under a belief whose particles carry their own skill
  floors. The mixture-mean rule is dishonest on heterogeneous populations
  (measured: pass-given-ready 0.67–0.86 vs the 0.90 target).
- **Session log**: one JSON line per trial (crash-tolerant, mirroring the
  testing package's `JsonLinesTrialWriter`): `{"k": int, "item_id": str,
  "gold": int, "response": int}` plus optional response-time seconds.

## 4. M1 → learning engine: the population artifact

Producer: the M1 fit (`realfit_st_soft*` checkpoint). Schema consumed by
`MSBelief`:

```json
{
  "alpha_t": [..M..], "alpha_s": [..M..], "lam": 0.0096,
  "q_t": 0.0, "q_s": 0.006,
  "w_coef": [..8 simplex weights..],
  "floor_prior": [[..M means..], [..M sds..]],
  "floor_joint": {"slope": [..M..], "intercept": [..M..],
                  "resid_sd": [..M..]},
  "state_prior": {"mu_t0": [..M..], "tau_t0": [..M..],
                   "mu_u0": [..M..], "tau_u0": [..M..],
                   "gamma": [..M..]}
}
```

- `floor_prior` is on log σ_∞, computed from POOLED posterior draws
  (draws × participants), not the spread of per-participant means (the
  means-only spread understates the shallow-floor tail that breaks the
  readiness flag, D36).
- `w_coef` is the learned-gate simplex (D23/D35 posterior mean or a
  per-session Thompson draw); omit to fall back to the Wilson gate.
- `floor_joint` (v1.1, optional): the per-signal regression of the floor
  `u_inf` on the starting skill `u0` across pooled posterior draws
  (draws × participants). When present, the belief draws per-particle
  floors CONDITIONALLY on each particle's skill — at construction and
  again after posterior seeding replaces the skill particles — instead of
  from the marginal `floor_prior`. This repairs the state–floor joint
  that wholesale cloud replacement severs (the D49 attainability-optimism
  mechanism). Absent → marginal behavior, unchanged.

## 5. Learning → testing: the prior handoff (closing the loop)

Producer: `msengine.prior_blocks_from_population(artifact)`. Consumer:
`adaptive_testing.Prior(...)`.

The fitted population becomes the testing engine's declared prior:

```
offset_mean            = −mu_t0
offset_covariance      = diag(tau_t0²)
log_sensitivity_mean   = −mu_u0
log_sensitivity_covariance = γ γᵀ + diag(tau_u0²)
```

The one-factor transfer structure (D30) is exactly the cross-dimension
information sharing the testing engine's unstructured covariance was
designed to accept. A new cohort's testing sessions then start from what
the population has already taught us, and their results feed the next M1
refit (§6).

## 6. Session logs → M1 repository

Both engines' per-trial logs append to the response repository in the
ingestion layer's schema (participant, item, response, timestamp). The M1
refit consumes them exactly as it consumed the pilot (D28 ingestion), so
the loop testing → learning → repository → M1 → priors → testing closes
with no schema translation.

## 7. Versioning and validation

- This document carries the contract version; both engines embed it in
  their outputs (`"contract_version": "1.1"`).
- Conformance checks live in `sim/exp_msengine.py` (learning side): a
  golden testing-engine session is ingested, the mapping round-trips
  (t = −o to numerical precision), and the closed loop runs end to end.
- Breaking changes to either side's schemas bump the major version;
  additive optional fields bump the minor version.
