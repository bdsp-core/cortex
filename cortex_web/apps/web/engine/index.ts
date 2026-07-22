// The engine's public surface — everything the SPA is entitled to import,
// and nothing else.
//
// Before this file the engine had no declared boundary in the src -> engine
// direction: eleven modules under src/ reached individually into types.ts,
// mathfns.ts, session.ts, worker_protocol.ts, and compute_payload.ts, so the
// "public API" was whatever anyone had happened to import. The reverse
// direction was already enforced (architecture_boundary.test.ts forbids the
// engine importing React, src/, or the API client); this closes the pair, and
// engine_public_surface.test.ts fails if src/ reaches past it.
//
// Adding an export here is a deliberate widening of the contract. The frozen
// numerics — particle filter, policy, RNG, selection, stopping rules — stay
// internal: the SPA drives them through the worker protocol, never directly.
//
// NOT re-exported on purpose: engine/worker.ts and its helper workers. Those
// are loaded as worker entry points via `new URL("../engine/worker.ts",
// import.meta.url)`, which Vite resolves at build time; routing them through a
// barrel would break that.

// ── shared value/type vocabulary ──
export type {
  ComputeEngineInputs,
  EngineInputs,
  EnginePerformanceEvent,
  EngineStepTiming,
  ExecutionProfileEvent,
  RequestedComputeMode,
  RuntimePoolAdjustmentEvent,
  SegmentMeta,
  TerminationPolicyName,
  TrialDiag,
} from "./types";

// ── session result shape (the SPA persists and renders this) ──
export type { SessionResult } from "./session";

// ── worker protocol: the SPA's only channel to the running engine ──
export type {
  EngineWorkerRequest,
  EngineWorkerResponse,
} from "./worker_protocol";
export {
  computePayloadTransferables,
  packComputeInputs,
} from "./compute_payload";

// ── math shared with the SPA's own presentation code ──
// normCdf backs the results-screen ROC curve; it lives here so both engines
// use one implementation rather than the SPA rounding its own.
export { normCdf } from "./mathfns";
