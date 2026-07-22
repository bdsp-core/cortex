# Certification-to-training handoff contract

Version 2.0 (2026-07-21). This contract describes the deployed server-driven
trainer. It supersedes Version 1.1, whose section 2a introduced exclusive raw
trial replay for the retired `adaptive_testing` and root-level trainer
prototypes.

## 1. Runtime owners

- Browser assembly: `cortex_web/apps/web/src/trainingSetup.ts`
- Browser API adapter: `cortex_web/apps/web/trainer/serverSession.ts`
- HTTP boundary: `cortex_web/services/api/routers/training_engine.py`
- Deployed session policy: `cortex_web/services/api/engine_trainer.py`
- Learning model: `cortex_web/learning-engine-cleaned/learning_engine/`
- Selection/update adapter: `cortex_web/learning-engine-cleaned/adapter/le_adapter.py`
- Population artifact:
  `cortex_web/learning-engine-cleaned/artifacts/nway_dynamics_v1_1.json`

The repository-root `trainer-policy/` package is an auditable Python mirror. It
is not a runtime dependency.

## 2. Shared bank and task coordinates

The server and browser use the same session-bank manifest. The following fields
are contract inputs:

- `taskCodes` fixes task order and dimension count;
- `ellStar` supplies the certification targets displayed by the trainer;
- `taskPatternWords` maps n-way gold classes onto the task axis;
- each segment's `sMean`, `sSd`, and `applicableTaskIdx` define its signal row,
  uncertainty, and eligible tasks.

The population artifact must contain one n-way domain for every manifest task
after the leading binary task. A mismatch stops session construction. The
trainer does not refit or reinterpret the certification prior.

## 3. Certification replay seed

At training start, the server reads the participant's latest finalized
certification session. It replays only the contiguous trial prefix beginning at
`trial_index == 0` and ending before the first gap or missing pick.

Pick coding is fixed by the live ledger:

- binary task: task-axis pick `0` means yes; any other task-axis pick means no;
- n-way task: task-axis pick `1..K-1` maps to group index `pick - 1`.

Certification replay is reweight-only because the test provides no answer
feedback. If no finalized session exists, the population prior remains the
seed. The browser deliberately does not also submit a posterior or test stream:
the seed source is server replay, never two competing representations.

The replay starts with an expanded particle cloud and contracts to the deployed
particle count after seeding. `seeded` and `seedUnique` expose the number of
replayed trials and surviving unique ancestors for audit.

## 4. Start endpoint

`POST /api/training-engine/start` requires authentication and ownership of the
training session.

Request:

```json
{
  "trainingId": "session identifier",
  "segIds": [101, 102],
  "restrictTaskKs": [0, 3]
}
```

`segIds` is the browser's drawn media pool; the server must not select media the
browser cannot load. `restrictTaskKs` is the optional regimen weak-task set.

Response:

```json
{
  "item": {
    "task": 3,
    "segId": 101,
    "s": 0.4,
    "sSd": 0.1,
    "yStar": 3,
    "mode": "skill",
    "link": "nway"
  },
  "snapshot": [],
  "allMastered": false,
  "seeded": 120,
  "rebuiltSeq": 0,
  "attainability": {},
  "seedUnique": 301
}
```

`item` may be null when no eligible candidate remains. `attainability` is a
report in practice mode; it does not silently change the certification rule.

## 5. Record endpoint and browser ordering

`POST /api/training-engine/record` accepts one pending answer:

```json
{
  "trainingId": "session identifier",
  "segId": 101,
  "taskK": 3,
  "pick": 2
}
```

The server rejects an unknown session, a segment that is not pending, or an
out-of-range n-way pick. It applies feedback dynamics, updates retention state,
and returns the next item plus the post-answer snapshot. `item: null` means the
sitting is done because of mastery, policy limits, or candidate exhaustion.

The browser sends this decision request immediately while showing feedback and
waits for it before advancing. Its synchronous `snapshot()` can therefore lag
one answer during the reveal; the snapshot returned by the record response is
authoritative.

## 6. Persistence and recovery

The decision endpoint does not write the response ledger. The existing browser
checkpoint outbox remains the single writer to `training_trials` and parameter
trajectories. Each persisted point carries the session sequence, task, segment,
pick, gold, correctness, feedback/timing fields, serving mode, and `binary` or
`nway` link.

The server holds an in-memory session cache, but the belief is rebuildable from:

1. the frozen population artifact;
2. the latest finalized certification trial stream; and
3. this training sitting's persisted response rows ordered by
   `seq_in_session`.

A restart can change side-alternation or item order, but persisted responses
must reconstruct the belief. A response still waiting in the client outbox is
not part of recovery until it is flushed.

## 7. Non-negotiable invariants

- Task order comes from the served manifest and is never inferred from labels.
- Certification replay and a client-supplied posterior are mutually exclusive.
- The server selects only from the submitted media pool and permitted tasks.
- One segment can be pending only under the server-issued item contract.
- Native n-way gold is the segment's true class; trained domain and gold class
  are not assumed to be the same.
- The checkpoint ledger has one writer and is idempotent by session sequence.
- Trainer mastery/attainability reporting cannot change certification cuts,
  stopping, likelihood, selector, particle settings, or precision policy.

## 8. Conformance checks

The focused production checks are:

```bash
cd cortex_web/services/api
python -m pytest -q \
  ../../learning-engine-cleaned/tests/test_package_smoke.py \
  test_engine_trainer.py

cd ../../apps/web
npx vitest run trainer/serverSession.test.ts \
  src/trainingController.test.ts src/trainingReveal.test.ts
```

The root mirror adds source-drift and deterministic behavioral-equivalence
checks under `trainer-policy/tests/`.
