import type { AdvanceParams, AdvanceResult, SessionCore } from "./advance";
import type { BranchExecutor } from "./branch_executor";
import {
  branchRequestTransferables, restoreAdvanceResult,
  type BranchWorkerRequest, type BranchWorkerResponse,
} from "./branch_protocol";
import type { Chosen } from "./choose_item";
import { snapshotCore } from "./core_snapshot";
import { precomputePriorPair } from "./prior";
import type { ComputeEngineInputs } from "./types";
import {
  computePayloadTransferables, packComputeInputs,
} from "./compute_payload";

interface PendingJob {
  resolve: (result: AdvanceResult) => void;
  reject: (error: Error) => void;
  timeoutId: number;
  slot: BranchSlot;
}

interface BranchSlot {
  worker: Worker;
  ready: Promise<void>;
  busy: boolean;
}

/**
 * Persistent bounded pool for ranked immutable response branches. The
 * coordinator computes rank one; these helpers compute the next outcomes up
 * to measured device capacity. Only the observed branch may be adopted.
 */
export class BranchWorkerExecutor implements BranchExecutor {
  private static readonly READY_TIMEOUT_MS = 10_000;
  private static readonly JOB_TIMEOUT_MS = 30_000;
  readonly capacity: number;
  private readonly slots: BranchSlot[];
  private readonly pending = new Map<number, PendingJob>();
  private readonly prior;
  private nextJobId = 1;
  private disposed = false;

  constructor(private readonly inputs: ComputeEngineInputs, helperCount = 1) {
    if (!Number.isInteger(helperCount) || helperCount < 1 || helperCount > 5) {
      throw new Error(`invalid branch helper count: ${helperCount}`);
    }
    this.capacity = helperCount;
    this.prior = precomputePriorPair(inputs.corrL, inputs.corrT);
    this.slots = Array.from({ length: helperCount }, () => this.createSlot());
  }

  async ready(): Promise<void> {
    let timeoutId = 0;
    const timeout = new Promise<never>((_resolve, reject) => {
      timeoutId = self.setTimeout(() => reject(
        new Error("branch worker initialization timed out"),
      ), BranchWorkerExecutor.READY_TIMEOUT_MS);
    });
    try {
      await Promise.race([Promise.all(this.slots.map((slot) => slot.ready)), timeout]);
    } finally {
      self.clearTimeout(timeoutId);
    }
  }

  advance(
    core: SessionCore,
    chosen: Chosen,
    params: AdvanceParams,
    trialIndex: number,
    pick: number,
  ): Promise<AdvanceResult> {
    if (this.disposed) throw new Error("branch worker executor is disposed");
    const slot = this.slots.find((candidate) => !candidate.busy);
    if (!slot) throw new Error("branch worker pool has no idle slot");
    slot.busy = true;
    const jobId = this.nextJobId++;
    const request: Extract<BranchWorkerRequest, { type: "advance" }> = {
      type: "advance",
      jobId,
      pick,
      snapshot: snapshotCore(core),
      chosen,
      params,
      trialIndex,
    };
    const result = new Promise<AdvanceResult>((resolve, reject) => {
      const timeoutId = self.setTimeout(() => {
        this.pending.delete(jobId);
        slot.busy = false;
        reject(new Error(`branch worker job ${jobId} timed out`));
      }, BranchWorkerExecutor.JOB_TIMEOUT_MS);
      this.pending.set(jobId, { resolve, reject, timeoutId, slot });
    });
    slot.worker.postMessage(
      request, { transfer: branchRequestTransferables(request) },
    );
    return result;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots) slot.worker.terminate();
    const error = new Error("branch worker executor disposed");
    for (const pending of this.pending.values()) {
      self.clearTimeout(pending.timeoutId);
      pending.reject(error);
    }
    this.pending.clear();
  }

  private createSlot(): BranchSlot {
    const worker = new Worker(new URL("./branch_worker.ts", import.meta.url), { type: "module" });
    const slot: BranchSlot = { worker, ready: Promise.resolve(), busy: false };
    let markReady!: () => void;
    let rejectReady!: (error: Error) => void;
    const ready = new Promise<void>((resolve, reject) => {
      markReady = resolve;
      rejectReady = reject;
    });
    worker.onmessage = (event: MessageEvent<BranchWorkerResponse>) => {
      const message = event.data;
      if (message.type === "ready") { markReady(); return; }
      if (message.type === "error") {
        const error = new Error(message.message);
        if (message.jobId === null) rejectReady(error);
        else {
          const pending = this.pending.get(message.jobId);
          this.pending.delete(message.jobId);
          if (pending) self.clearTimeout(pending.timeoutId);
          if (pending) pending.slot.busy = false;
          pending?.reject(error);
        }
        return;
      }
      const pending = this.pending.get(message.jobId);
      this.pending.delete(message.jobId);
      if (pending) self.clearTimeout(pending.timeoutId);
      if (pending) pending.slot.busy = false;
      pending?.resolve(restoreAdvanceResult(message.payload, this.inputs, this.prior));
    };
    worker.onerror = (event) => {
      event.preventDefault();
      const error = new Error(event.message || "branch worker crashed");
      rejectReady(error);
      for (const pending of this.pending.values()) {
        self.clearTimeout(pending.timeoutId);
        pending.slot.busy = false;
        pending.reject(error);
      }
      this.pending.clear();
    };
    const payload = packComputeInputs(this.inputs);
    const init: BranchWorkerRequest = { type: "init", payload };
    worker.postMessage(init, { transfer: computePayloadTransferables(payload) });
    slot.ready = ready;
    return slot;
  }
}
