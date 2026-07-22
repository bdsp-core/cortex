import { unpackComputeInputs } from "./compute_payload";
import {
  scoreNWayCandidateLosses, screenNWayDomain,
  type NWayCandidate,
} from "./nway_selector";
import type {
  NWaySelectorWorkerRequest, NWaySelectorWorkerResponse,
} from "./nway_selector_protocol";
import type { ComputeEngineInputs } from "./types";
import {
  logLikPackedHistory, makePackedHistoryLikelihoodScratch,
  type PackedHistoryLikelihoodScratch,
} from "./particles";

let inputs: ComputeEngineInputs | null = null;
let segmentsById: Map<number, ComputeEngineInputs["segments"][number]> | null = null;
let cachedHistory: Extract<
  NWaySelectorWorkerRequest, { type: "history_likelihood" }
>["history"] = undefined;
let cachedHistoryVersion: number | null = null;
let historyLikelihoodScratch: PackedHistoryLikelihoodScratch | undefined;

const post = (message: NWaySelectorWorkerResponse, transfer: Transferable[] = []) =>
  self.postMessage(message, { transfer });

self.onmessage = (event: MessageEvent<NWaySelectorWorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      inputs = unpackComputeInputs(message.payload);
      segmentsById = new Map(inputs.segments.map((segment) => [segment.segId, segment]));
      cachedHistory = undefined;
      cachedHistoryVersion = null;
      historyLikelihoodScratch = makePackedHistoryLikelihoodScratch();
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
      if (!Number.isInteger(message.historyVersion) || message.historyVersion < 1) {
        throw new Error("n-way history version is invalid");
      }
      if (message.history) {
        cachedHistory = message.history;
        cachedHistoryVersion = message.historyVersion;
      } else if (!cachedHistory || cachedHistoryVersion !== message.historyVersion) {
        throw new Error("n-way history worker cache is unavailable or stale");
      }
      logLikPackedHistory(
        cachedHistory, message.N, message.K,
        message.t, message.l, message.logLikelihood, historyLikelihoodScratch,
      );
      post({
        type: "history_result", jobId: message.jobId,
        startIndex: message.startIndex,
        historyVersion: message.historyVersion,
        t: message.t, l: message.l, logLikelihood: message.logLikelihood,
      }, [message.t.buffer, message.l.buffer, message.logLikelihood.buffer]);
    } else {
      if (!segmentsById) throw new Error("n-way selector segment index is unavailable");
      if (message.taskKs.length !== message.segIds.length) {
        throw new Error("n-way selector candidate vectors are misaligned");
      }
      const candidates: NWayCandidate[] = Array.from(
        message.segIds, (segId, index) => {
          const segment = segmentsById!.get(segId);
          if (!segment) throw new Error(`n-way score segment ${segId} is unavailable`);
          const k = message.taskKs[index];
          if (k >= inputs!.taskCodes.length) {
            throw new Error(`n-way score task index ${k} is unavailable`);
          }
          return { k, segment };
        },
      );
      const losses = scoreNWayCandidateLosses(
        message.state, inputs, candidates,
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
