# Isolated-to-production promotion map

This is a review map, not an instruction to copy files wholesale.

| Isolated implementation | Future reviewed production seam |
|---|---|
| `src/types.ts` | `cortex_web/apps/web/engine/types.ts` observation, profile, diagnostic, and compute-mode unions |
| `src/likelihood.ts` | `engine/likelihood.ts` binary-preserving F1 probability and log likelihood |
| `src/particles.ts` | `engine/particles.ts` update, categorical history replay, clone, and rejuvenation |
| `src/selector.ts` | `engine/choose_item.ts` full-vector candidates, six-outcome loss, shortlist, and regret instrumentation |
| `src/research_selector.ts` | Only after qualification: Fisher-augmented shortlist in `engine/choose_item.ts`; exact final loss remains authoritative |
| `src/research_artifact.ts` | Only after qualification: artifact-draw mixture in categorical update, replay, and selector paths |
| `src/integrated_protocol.ts` | Immutable selector-version dispatch; frozen profiles stay on their original selector |
| `src/session.ts` | `engine/advance.ts` and `engine/session.ts` raw-pick observation construction and direct-evidence bookkeeping |
| `src/precision_bridge.ts` | Preserve `engine/precision_policy.ts`; adapt categorical state and focal bank telemetry at its boundary |
| `src/snapshot.ts` | `engine/core_snapshot.ts`, including exact profile compatibility |
| `src/speculation.ts` | `engine/branch_protocol.ts`, `branch_executor.ts`, workers, and coordinator scheduling |
| `src/profile.ts` | Bundle validation, worker initialization, and resume/result profile checks |
| `server/nway_profile.py` | New fail-closed response rollout module plus `routers/testing.py` session stamping |
| `server/migration.sql` | Additive entries in `persistence/migrations.py` and `db.py` session writes/reads |
| `schemas/` | Bundle build/boot validation and artifact registry |
| `tests/` and `python/` | Production regression, parity, adaptive OC, browser, and rollout test suites |

## Promotion order

1. Land schemas and session database fields with binary defaults only.
2. Land profile parsing and replay/result checks while all sessions remain
   binary.
3. Land likelihood/state/history behind a build-time-disabled F1 profile.
4. Land selector and serial categorical execution; run parity and offline
   historical shadow.
5. Join the categorical state to the unchanged Precision policy and run the
   locked adaptive qualification.
6. Land ranked speculation only after serial mathematical qualification.
7. Enable internal and allowlisted sessions through the server-owned rollout.

At every step, legacy binary fixtures and active sessions must reproduce under
their stored profile. No promotion commit may reinterpret existing history.
