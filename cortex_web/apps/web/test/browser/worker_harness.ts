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
  answerSequence?: number[];
  expectedItems?: { trialIndex: number; taskK: number; segId: number }[];
  seed?: number;
  sessionId?: string;
  deriveSeedFromSessionId?: boolean;
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
    onItem: ({ trialIndex, taskK, segId }) => {
      if (options.maxQuestions !== undefined && trialIndex >= options.maxQuestions) {
        queueMicrotask(() => client.abort());
        return;
      }
      const expected = options.expectedItems?.[trialIndex];
      if (options.expectedItems && (!expected
          || expected.trialIndex !== trialIndex
          || expected.taskK !== taskK
          || expected.segId !== segId)) {
        client.dispose();
        reject(new Error(
          `replay item mismatch at ${trialIndex}: expected ${JSON.stringify(expected)}, `
          + `received ${JSON.stringify({ trialIndex, taskK, segId })}`,
        ));
        return;
      }
      const replayPick = options.answerSequence?.[trialIndex];
      if (options.answerSequence && replayPick === undefined) {
        client.dispose();
        reject(new Error(`replay answer is missing at trial ${trialIndex}`));
        return;
      }
      const pattern = options.answerPattern ?? "yes";
      const answerYes = pattern === "yes"
        || (pattern === "alternating" && trialIndex % 2 === 0);
      const pick = replayPick
        ?? (answerYes ? taskK : (taskK === 0 ? 1 : taskK === 1 ? 2 : 1));
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
  client.start(options.inputs ?? nwayWorkerInputs(), options.sessionId ?? "browser-worker-parity", {
    ...(!options.deriveSeedFromSessionId ? { seed: options.seed ?? 31415 } : {}),
    requestedComputeMode: mode,
  });
});

window.workerHarnessReady = true;
