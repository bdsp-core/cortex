import type {
  NWayCandidate, NWayScreeningMoments, NWaySelectionState,
} from "./nway_selector";
import type { PackedComputeInputs } from "./compute_payload";

export type NWaySelectorWorkerRequest =
  | { type: "init"; payload: PackedComputeInputs }
  | {
      type: "score";
      jobId: number;
      startIndex: number;
      state: NWaySelectionState;
      candidates: NWayCandidate[];
    }
  | {
      type: "screen";
      jobId: number;
      moments: NWayScreeningMoments;
      taskK: number;
      segIds: Float64Array;
    };

export type NWaySelectorWorkerResponse =
  | { type: "ready" }
  | { type: "result"; jobId: number; startIndex: number; losses: Float64Array }
  | {
      type: "screen_result";
      jobId: number;
      taskK: number;
      entropySegIds: number[];
      fisherSegIds: number[];
      entropyMs: number;
      fisherMs: number;
    }
  | { type: "error"; jobId: number | null; message: string };
