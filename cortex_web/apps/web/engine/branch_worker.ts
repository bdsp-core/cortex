import { advanceCore, precisionRemainingBank } from "./advance";
import {
  branchResultTransferables, serializeAdvanceResult,
  type BranchWorkerRequest, type BranchWorkerResponse,
} from "./branch_protocol";
import { restoreCore } from "./core_snapshot";
import { precomputePriorPair } from "./prior";
import type { ComputeEngineInputs, PriorPair } from "./types";
import { unpackComputeInputs } from "./compute_payload";

let inputs: ComputeEngineInputs | null = null;
let prior: PriorPair | null = null;

const post = (message: BranchWorkerResponse, transfer: Transferable[] = []) =>
  self.postMessage(message, { transfer });

self.onmessage = (event: MessageEvent<BranchWorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      inputs = unpackComputeInputs(message.payload);
      prior = precomputePriorPair(inputs.corrL, inputs.corrT);
      post({ type: "ready" });
      return;
    }
    if (!inputs || !prior) throw new Error("branch worker was not initialized");
    const core = restoreCore(message.snapshot, inputs, prior);
    const bankStartedAt = performance.now();
    const preparedBank = core.policy.name === "precision_v1"
      ? precisionRemainingBank(inputs, core.remaining, message.chosen.segId)
      : undefined;
    const bankPreparationMs = performance.now() - bankStartedAt;
    const result = advanceCore(
      core, inputs, message.chosen, message.y, message.params,
      message.trialIndex, preparedBank,
    );
    result.timing.bankPreparationMs = bankPreparationMs;
    result.timing.executionMode = "dual_branch";
    result.timing.speculative = true;
    const payload = serializeAdvanceResult(result);
    post(
      { type: "result", jobId: message.jobId, payload },
      branchResultTransferables(payload),
    );
  } catch (error) {
    post({
      type: "error",
      jobId: message.type === "advance" ? message.jobId : null,
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
};
