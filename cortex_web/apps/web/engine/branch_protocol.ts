import type { AdvanceParams, AdvanceResult } from "./advance";
import type { Chosen } from "./choose_item";
import {
  restoreCore, snapshotCore, snapshotTransferables, type SessionCoreSnapshot,
} from "./core_snapshot";
import type { ComputeEngineInputs, TrialDiag, EngineStepTiming } from "./types";
import type { PriorPair } from "./types";
import type { PackedComputeInputs } from "./compute_payload";

export type BranchWorkerRequest =
  | { type: "init"; payload: PackedComputeInputs }
  | {
      type: "advance";
      jobId: number;
      pick: number;
      snapshot: SessionCoreSnapshot;
      chosen: Chosen;
      params: AdvanceParams;
      trialIndex: number;
    };

export interface BranchAdvancePayload {
  core: SessionCoreSnapshot;
  diag: TrialDiag;
  trajSnapshot: { t: Float64Array; l: Float64Array; w: Float64Array };
  servedSegId: number;
  rejuv: boolean;
  nextChosen: Chosen;
  done: boolean;
  stopReason: string;
  timing: EngineStepTiming;
}

export type BranchWorkerResponse =
  | { type: "ready" }
  | { type: "result"; jobId: number; payload: BranchAdvancePayload }
  | { type: "error"; jobId: number | null; message: string };

export function serializeAdvanceResult(result: AdvanceResult): BranchAdvancePayload {
  return {
    core: snapshotCore(result.core),
    diag: result.diag,
    trajSnapshot: result.trajSnapshot,
    servedSegId: result.servedSegId,
    rejuv: result.rejuv,
    nextChosen: result.nextChosen,
    done: result.done,
    stopReason: result.stopReason,
    timing: result.timing,
  };
}

export function restoreAdvanceResult(
  payload: BranchAdvancePayload,
  inputs: ComputeEngineInputs,
  prior: PriorPair,
): AdvanceResult {
  return {
    ...payload,
    core: restoreCore(payload.core, inputs, prior),
  };
}

export function branchRequestTransferables(
  request: Extract<BranchWorkerRequest, { type: "advance" }>,
): Transferable[] {
  return snapshotTransferables(request.snapshot);
}

export function branchResultTransferables(payload: BranchAdvancePayload): Transferable[] {
  return [
    ...snapshotTransferables(payload.core),
    payload.trajSnapshot.t.buffer,
    payload.trajSnapshot.l.buffer,
    payload.trajSnapshot.w.buffer,
  ];
}
