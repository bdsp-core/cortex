import { unpackComputeInputs } from "./compute_payload";
import {
  scoreNWayCandidateLosses, screenNWayDomain,
  type NWayCandidate,
} from "./nway_selector";
import type {
  NWaySelectorWorkerRequest, NWaySelectorWorkerResponse,
} from "./nway_selector_protocol";
import type { ComputeEngineInputs } from "./types";
import { logLikPackedHistory } from "./particles";

let inputs: ComputeEngineInputs | null = null;
let segmentsById: Map<number, ComputeEngineInputs["segments"][number]> | null = null;

const post = (message: NWaySelectorWorkerResponse, transfer: Transferable[] = []) =>
  self.postMessage(message, { transfer });

self.onmessage = (event: MessageEvent<NWaySelectorWorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      inputs = unpackComputeInputs(message.payload);
      segmentsById = new Map(inputs.segments.map((segment) => [segment.segId, segment]));
      post({ type: "ready" });
      return;
    }
    if (!inputs) throw new Error("n-way selector worker was not initialized");
    if (message.type === "screen") {
      if (!segmentsById) throw new Error("n-way selector segment index is unavailable");
      const candidates: NWayCandidate[] = Array.from(message.segIds, (segId) => {
        const segment = segmentsById!.get(segId);
        if (!segment) throw new Error(`n-way screen segment ${segId} is unavailable`);
        return { k: message.taskK, segment };
      });
      const result = screenNWayDomain(
        inputs, message.taskK, candidates, message.moments,
      );
      post({ type: "screen_result", jobId: message.jobId, ...result });
    } else if (message.type === "history_likelihood") {
      const logLikelihood = new Float64Array(message.N);
      logLikPackedHistory(
        message.history, message.N, message.K,
        message.t, message.l, logLikelihood,
      );
      post({
        type: "history_result", jobId: message.jobId,
        startIndex: message.startIndex, logLikelihood,
      }, [logLikelihood.buffer]);
    } else {
      const losses = scoreNWayCandidateLosses(
        message.state, inputs, message.candidates,
      );
      post(
        { type: "result", jobId: message.jobId, startIndex: message.startIndex, losses },
        [losses.buffer],
      );
    }
  } catch (error) {
    post({
      type: "error",
      jobId: message.type === "init" ? null : message.jobId,
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
};
