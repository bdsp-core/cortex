import { nwayWorkerInputs } from "../../engine/__testdata__/precision_fixture";
import type { SessionResult } from "../../engine/session";
import type {
  ComputeEngineInputs, EnginePerformanceEvent, RequestedComputeMode,
} from "../../engine/types";
import { EngineClient } from "../../src/engineClient";

interface HarnessResult {
  result: Omit<SessionResult, "traj"> & {
    traj: { t: number[]; l: number[]; w: number[] };
  };
  events: EnginePerformanceEvent[];
}

interface HarnessOptions {
  inputs?: ComputeEngineInputs;
  maxQuestions?: number;
  answerDelayMs?: number;
  answerPattern?: "yes" | "no" | "alternating";
}

declare global {
  interface Window {
    runWorkerHarness: (
      mode: RequestedComputeMode, options?: HarnessOptions,
    ) => Promise<HarnessResult>;
    workerHarnessReady: boolean;
  }
}

function portableResult(result: SessionResult): HarnessResult["result"] {
  return {
    ...result,
    traj: {
      t: Array.from(result.traj.t),
      l: Array.from(result.traj.l),
      w: Array.from(result.traj.w),
    },
  };
}

window.runWorkerHarness = (mode, options = {}) => new Promise((resolve, reject) => {
  const events: EnginePerformanceEvent[] = [];
  const client = new EngineClient({
    onItem: ({ trialIndex, taskK }) => {
      if (options.maxQuestions !== undefined && trialIndex >= options.maxQuestions) {
        queueMicrotask(() => client.abort());
        return;
      }
      const pattern = options.answerPattern ?? "yes";
      const answerYes = pattern === "yes"
        || (pattern === "alternating" && trialIndex % 2 === 0);
      const pick = answerYes ? taskK : (taskK === 0 ? 1 : taskK === 1 ? 2 : 1);
      window.setTimeout(() => client.answer(pick), options.answerDelayMs ?? 0);
    },
    onPerformance: (event) => events.push(event),
    onDone: (result) => {
      client.dispose();
      resolve({ result: portableResult(result), events });
    },
    onError: (message) => {
      client.dispose();
      reject(new Error(message));
    },
  });
  client.start(options.inputs ?? nwayWorkerInputs(), "browser-worker-parity", {
    seed: 31415,
    requestedComputeMode: mode,
  });
});

window.workerHarnessReady = true;
