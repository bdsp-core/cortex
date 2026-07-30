# Phase 0/1/2 drop notes: instrumentation, shadow, and the live engine trainer

Historical record: these notes describe the staged rollout as it existed on
2026-07-16. The server-side engine became the sole trainer on 2026-07-17; the
incumbent browser-local trainer and its fallback semantics are retired. The
server exposure setting still exists, but disabling it does not restore a
browser model; use the current operations baseline rather than the historical
instructions below. See
[`../../docs/PRODUCTION_BASELINE.md`](../../docs/PRODUCTION_BASELINE.md).
See `INTEGRATION_PLAN_MIXED_TASKS.md` for the current request path.

2026-07-16. This note documents the Phase-0 (instrumentation), Phase-1
(read-only shadow), and Phase-2 (server-side engine trainer) changes that
put the learning engine into the web pipeline for real-human testing, per
the adoption roadmap and handoff contract v1.1. Terminology is generic;
code identifiers are quoted verbatim.

## Phase 2 at the time: engine trainer live behind a flag

The vendored engine now serves per-question training decisions through
the web API. At the date of these notes the incumbent client-side trainer was
still the default; that statement is historical and no longer describes
production.

- **Endpoints** (`services/api/routers/training_engine.py` +
  `engine_trainer.py`): `POST /api/training-engine/start` builds the
  sitting's belief — replay-seeded SERVER-side from the participant's
  latest certification sitting (contract §2a; posterior XOR replay) —
  and returns the first item; `POST /api/training-engine/record`
  applies an answer and returns the next item + per-task snapshot
  (skill/criterion/SD/pass-mass/trainability/mastered).
- **Pure decision-maker**: the engine writes nothing; the per-trial
  ledger keeps its single validated writer (the client checkpoint
  outbox). The belief is a deterministic function of (artifact, test
  stream, recorded responses), so a server restart REBUILDS the sitting
  from the ledger (`rebuiltSeq` in the start response; ordering via the
  new `training_trials.seq_in_session` column).
- **Policy** = the validated adapter stack (`LETrainerPolicy`):
  evidence-gated bias mode at the label boundary, balanced-sign
  gate-peak skill placement, value-per-item allocation, LCB mastery
  retirement, ESS-gated futility — with cut targets read from the
  bundle manifest's `ellStar` (v15), so trainer and exam share one
  source. Session-time imports are JAX-free (lazy calibration surface).
- **Client** (feature-flagged, same UI): `POST /api/training-sessions`
  now returns `engineMode`; when true the sitting runs on
  `ServerTrainerSession` (`trainer/serverSession.ts`) — a prefetching
  adapter behind the same controller surface (`submit` fires the record
  round trip while the participant reads the reveal; `waitForNext`
  bridges the async hop). The regimen deck becomes the engine's
  restrict set; the drawn pool's segIds bound what the engine may serve
  (media guaranteed loadable).
- **Enablement**: `CORTEX_TRAINER_ENGINE=off|cohort|all` (+
  `CORTEX_TRAINER_ENGINE_ALLOWLIST` for the pilot cohort) — the same
  reversible-flag pattern as `CORTEX_TRAINING_MODE`. Default off.
- **Serving posture** (live finding, 2026-07-17): the service defaults to
  PRACTICE MODE (`CORTEX_TRAINER_ENGINE_ALPHA=0`) — per-domain
  attainability is computed and RETURNED in the /start response (report-
  first doctrine) but never blocks serving. The first live learner
  measured below every v15 bar, so certification-mode futility (α=0.05)
  honestly refused to serve anything and surfaced as the client's
  "no new segments" screen. Set the env to 0.05 to restore the
  certification-economy gates. Seeding also runs on an expanded particle
  cloud contracted after replay (the depletion guard — raw replay
  collapsed the belief to a handful of ancestors and made the futility
  and bias gates per-sitting coin flips; `seedUnique` in the /start
  response tracks it).
- **To begin testing**: set `CORTEX_TRAINER_ENGINE=cohort` and list the
  pilot participants; they take a cert sitting, then train — every
  question now chosen by the learning engine, every response landing in
  the Phase-0 ledger, and the V4 shadow tool reads the same DB for
  analysis. Closed-loop verified end to end over HTTP
  (`services/api/test_engine_trainer.py`: replay seeding, no-repeat
  serving, snapshot sanity, restart rebuild, gating).

## Cut blocks: v15 is the decision (2026-07-16)

v15 is the updated metric set, everywhere: the web bundles already
certified at v15; the desktop/reference loader + policy-builder defaults
switched from v14 to v15 (`scripts/cortex_policy_k7.py`, mirror, viewer);
drift guards re-pinned (`test_ell_star_v15.py`,
`test_phase9_scaffolding.py`, `test_cortex_viewer.py`,
`test_trainer_g0_ell_star.py`). v14/v13 stay reachable by explicit
`block_name=` for historical replay. **One OC finding surfaced by the
switch, left visible as an xfail** (`test_trainer_g2_oc.py`): at v15
cuts, a strong simulated learner's verdicts are no longer fully robust
to 30% bank exclusion (6/7 tasks agree; 7/7 under v14, kept as the
strict mechanism guard). The lower v15 bars change which items are
informative near the cut — worth a near-cut item-coverage / info-gate
review before high-stakes use.

## What changed in the web pipeline (additive, no serving behavior change)

1. **Training responses are now persisted** (`training_trials` gained
   `pick, y_star, is_correct, feedback_shown, rt_ms, shown_client_utc,
   answered_client_utc`; idempotent boot migration). Previously only
   exposure + belief trajectories persisted — the participant's actual
   answers were discarded, which made any dynamics refit impossible.
   The client controller (`trainingController.ts`) now carries the
   response record per point, including the feedback text the reveal
   renders.
2. **Exam trials carry client wall times** (`trials.shown_client_utc /
   answered_client_utc`; `reaction_ms` stays the authoritative RT delta).
   These anchor session load, fatigue position, and between-session
   structure for the learning model.
3. **The regimen carries the raw test stream** (`plan.testStream`:
   ordered, contiguous fully-picked prefix of the source test sitting,
   full n-way pick, task index included) — the handoff contract §2a
   payload. A replay-capable trainer seeds its belief by re-observing
   this stream; a client must seed from EITHER `testStream` OR `prior`,
   never both (double counting).
4. Drift guards: legacy payloads without the new fields still land
   (NULLs); two new backend tests cover the response record and the
   test-stream round trip. Full backend suite: 175 passed.

## The vendored engine package

`learning-engine-cleaned/` (repo root) is the D51-state package:
mixed-link belief/engine, registry extensibility, replay seeding
(`MSBelief.seed_from_test_replay`, `MSSessionEngine.run(testing_trials=…)`),
`floor_joint` conditional floors, adapter, contract v1.1. Notes:

- **Session-time imports are JAX-free**: the package `__init__` resolves
  the calibration surface (NumPyro fit) lazily, so importing beliefs or
  the adapter never pulls JAX — compatible with the runtime BLAS/JAX
  isolation contract. Gated by `tests/test_package_smoke.py` (5 fast
  gates: JAX-free imports, bitwise determinism, hyperprior cold-start,
  replay seeding + double-count guard, contract version).
- **Artifacts**: `artifacts/nway_dynamics_v1_1.json` adds the
  `floor_joint` block (per-domain regression of skill floor on starting
  skill, computed from the same population posterior draws as v1;
  slopes ≈ 0.9–1.0 on this population — the floor tracks starting skill
  strongly, so conditional floor seeding is material here). v1 is kept
  unchanged as the validated-experiment input; deployments consume v1_1.

## Cut-block finding (RESOLVED 2026-07-16: v15 everywhere — see above)

Originally: web bundles certified at `ell_star_unified_v15` while the
desktop/reference path defaulted to `ell_star_unified_v14` — the same
wiring-trap family previously found in the trainer-side loader default.
Resolved by the v15 decision; the section above records the sweep.

## The V4 shadow (Phase 1, read-only)

`cortex_web/research/shadow/le_shadow.py`: batch tool that replays each
participant's latest finalized test sitting through the engine belief
(contract §2a), steps through their recorded training responses, and
emits per-participant JSONL: readiness mass per domain at seed and after
training, and the realized difficulty-gate weight of every served
training item (the placement-precision statistic). Deterministic; opens
the DB read-only; never runs in the serving path.

    python -m research.shadow.le_shadow --db services/api/cortex.db \
        --manifest apps/web/public/bundle/v1.6-k7-35k/manifest.json \
        --out shadow_report.jsonl

Pick coding CONFIRMED on the first real-data run (2026-07-17), and one
crash-class bug fixed before any live engine exposure: picks are 0-based
indices on the TASK axis (n-way answers 1..K-1; the binary task answers
0 = "yes" with sentinel K = "no") — NOT group-relative 0..5 as the
fabricated tests had assumed. The shadow, the engine service's replay
seeding, and both test fabrications now use the measured coding.

## Known items

- One pre-existing stochastic web-engine test failure
  (`engine/session.test.ts` — "clearly-unskilled rater FAILs the bulk of
  tasks": 3 FAILs vs ≥4 expected) reproduces WITHOUT these changes;
  flagged, not touched.
- The training UI stays binary one-vs-rest; native n-way training
  serving remains the UI-gated upgrade (integration plan §3.5 risk 3).
  The engine serves through the same one-vs-rest contract meanwhile.
- Consent/IRB scope for the new logging fields: confirmed defined
  (2026-07-16 directive).
- Engine-mode trajectory points (`param_trajectories`) carry the
  belief state ONE update behind (the client writes at answer time; the
  server's post-answer snapshot arrives with the next item). Response
  fields are exact; analyses needing exact post-answer states should
  join on the engine's record responses or recompute via the shadow.
- The Phase-2 alternative (a TypeScript port of the engine with golden
  parity tests, the house pattern) remains open as the long-term
  serving architecture; the server-side service is the pilot path.
