import type { SessionResult } from "./session";
import type {
  EnginePerformanceEvent, TrialDiag, RequestedComputeMode,
} from "./types";
import type { PackedComputeInputs } from "./compute_payload";

/** Main-thread → certification-engine worker protocol. */
export type EngineWorkerRequest =
  | {
      type: "init";
      payload: PackedComputeInputs;
      sessionId: string;
      seed?: number;
      speculative?: boolean;
      requestedComputeMode?: RequestedComputeMode;
      /** Qualification-only device-tier override; omitted by the application. */
      qualificationHardwareConcurrency?: number;
      /** Qualification-only rank-expansion override; omitted by the application. */
      qualificationRankedSpeculation?: boolean;
    }
  | {
      type: "answer";
      pick: number;
      /** Main-realm high-resolution epoch. Telemetry/scheduling only. */
      submittedAtEpochMs?: number;
    }
  | { type: "abort" };

/** Certification-engine worker → main-thread protocol. */
export type EngineWorkerResponse =
  | { type: "item"; trialIndex: number; taskK: number; segId: number }
  | { type: "trial"; diag: TrialDiag }
  | { type: "performance"; event: EnginePerformanceEvent }
  | { type: "done"; result: SessionResult }
  | { type: "error"; message: string };

/** Exhaustiveness helper shared by protocol consumers. */
export function unreachableMessage(value: never): never {
  throw new Error(`unrecognized engine-worker message: ${String(value)}`);
}
