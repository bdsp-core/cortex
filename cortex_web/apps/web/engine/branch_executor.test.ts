import { describe, expect, it } from "vitest";

import { advanceCore, precisionRemainingBank } from "./advance";
import type { AdvanceParams, AdvanceResult, SessionCore } from "./advance";
import type { BranchExecutor } from "./branch_executor";
import type { Chosen } from "./choose_item";
import { restoreCore, snapshotCore } from "./core_snapshot";
import { precomputePriorPair } from "./prior";
import { WebCortexSession } from "./session";
import type { ComputeEngineInputs, EngineStepTiming } from "./types";
import { precisionGoldenInputs } from "./__testdata__/precision_fixture";

/** Structured-clone semantics without browser globals; real workers are gated separately. */
class SnapshotBranchExecutor implements BranchExecutor {
  private readonly prior;

  constructor(private readonly inputs: ComputeEngineInputs) {
    this.prior = precomputePriorPair(inputs.corrL, inputs.corrT);
  }

  advance(
    core: SessionCore,
    chosen: Chosen,
    params: AdvanceParams,
    trialIndex: number,
    pick: number,
  ): Promise<AdvanceResult> {
    const snapshot = snapshotCore(core);
    return Promise.resolve().then(() => {
      const restored = restoreCore(snapshot, this.inputs, this.prior);
      const prepared = precisionRemainingBank(
        this.inputs, restored.remaining, chosen.segId,
      );
      const result = advanceCore(
        restored, this.inputs, chosen, pick, params, trialIndex, prepared,
      );
      result.timing.executionMode = "dual_branch";
      result.timing.speculative = true;
      return result;
    });
  }

  dispose(): void {}
}

class RejectingBranchExecutor implements BranchExecutor {
  advance(): Promise<AdvanceResult> {
    return Promise.reject(new Error("injected worker failure"));
  }

  dispose(): void {}
}

async function run(mode: "serial" | "dual_branch" | "rejecting") {
  const inputs = precisionGoldenInputs();
  const timings: EngineStepTiming[] = [];
  const executor = mode === "dual_branch"
    ? new SnapshotBranchExecutor(inputs)
    : mode === "rejecting" ? new RejectingBranchExecutor() : undefined;
  const session = new WebCortexSession(inputs, "worker-parity", 31415, {
    onItem: ({ taskK }) => queueMicrotask(() => session.submitAnswer(taskK === 0 ? 0 : 1)),
    onPerformance: (timing) => timings.push(timing),
  }, {
    speculative: executor !== undefined,
    ...(executor ? { branchExecutor: executor } : {}),
  });
  const result = await session.run();
  executor?.dispose();
  return { result, timings };
}

describe("parallel answer-branch execution", () => {
  it("is exactly identical to the serial authoritative path", async () => {
    const serial = await run("serial");
    const parallel = await run("dual_branch");

    expect(parallel.result).toEqual(serial.result);
    expect(parallel.timings).toHaveLength(serial.result.nQuestions);
    expect(parallel.timings.every((x) => x.executionMode === "dual_branch")).toBe(true);
  }, 60_000);

  it("fails closed to an exact serial recomputation", async () => {
    const serial = await run("serial");
    const failedWorkers = await run("rejecting");

    expect(failedWorkers.result).toEqual(serial.result);
    expect(failedWorkers.timings[0].executionMode).toBe("serial_fallback");
    expect(failedWorkers.timings.slice(1).every(
      (timing) => timing.executionMode === "serial",
    )).toBe(true);
  }, 60_000);
});
