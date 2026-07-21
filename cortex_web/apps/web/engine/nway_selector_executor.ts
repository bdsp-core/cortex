import {
  computePayloadTransferables, packComputeInputs,
} from "./compute_payload";
import type {
  NWayCandidate, NWayDomainScreenResult, NWayScreeningMoments,
  NWaySelectionState,
} from "./nway_selector";
import type {
  NWaySelectorWorkerRequest, NWaySelectorWorkerResponse,
} from "./nway_selector_protocol";
import type { ComputeEngineInputs } from "./types";

interface SelectorSlot {
  worker: Worker;
  ready: Promise<void>;
}

export interface NWaySelectionExecutor {
  readonly workerCount: number;
  ready(): Promise<void>;
  score(
    state: NWaySelectionState, candidates: readonly NWayCandidate[],
  ): Promise<Float64Array>;
  screen(
    moments: NWayScreeningMoments,
    domains: readonly { taskK: number; segIds: readonly number[] }[],
  ): Promise<NWayDomainScreenResult[]>;
  dispose(): void;
}

/** Persistent deterministic candidate-shard pool. Workers return only indexed
 * loss vectors; ordering, tie behavior, and selection remain centralized. */
export class NWaySelectorWorkerExecutor implements NWaySelectionExecutor {
  private static readonly READY_TIMEOUT_MS = 10_000;
  private static readonly JOB_TIMEOUT_MS = 30_000;
  readonly workerCount: number;
  private readonly slots: SelectorSlot[];
  private nextJobId = 1;
  private disposed = false;

  constructor(private readonly inputs: ComputeEngineInputs, workerCount: number) {
    if (!Number.isInteger(workerCount) || workerCount < 1 || workerCount > 12) {
      throw new Error(`invalid n-way selector worker count: ${workerCount}`);
    }
    this.workerCount = workerCount;
    this.slots = Array.from({ length: workerCount }, () => this.createSlot());
  }

  async ready(): Promise<void> {
    let timeoutId = 0;
    const timeout = new Promise<never>((_resolve, reject) => {
      timeoutId = self.setTimeout(() => reject(
        new Error("n-way selector worker initialization timed out"),
      ), NWaySelectorWorkerExecutor.READY_TIMEOUT_MS);
    });
    try {
      await Promise.race([Promise.all(this.slots.map((slot) => slot.ready)), timeout]);
    } finally {
      self.clearTimeout(timeoutId);
    }
  }

  async score(
    state: NWaySelectionState, candidates: readonly NWayCandidate[],
  ): Promise<Float64Array> {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    if (candidates.length === 0) return new Float64Array(0);
    const shardCount = Math.min(this.slots.length, candidates.length);
    const shardSize = Math.ceil(candidates.length / shardCount);
    const losses = new Float64Array(candidates.length);
    await Promise.all(Array.from({ length: shardCount }, async (_unused, shardIndex) => {
      const startIndex = shardIndex * shardSize;
      const endIndex = Math.min(candidates.length, startIndex + shardSize);
      const response = await this.runShard(
        this.slots[shardIndex], state, candidates.slice(startIndex, endIndex), startIndex,
      );
      if (response.losses.length !== endIndex - startIndex
          || response.startIndex !== startIndex) {
        throw new Error("n-way selector worker returned a misaligned shard");
      }
      losses.set(response.losses, startIndex);
    }));
    return losses;
  }

  async screen(
    moments: NWayScreeningMoments,
    domains: readonly { taskK: number; segIds: readonly number[] }[],
  ): Promise<NWayDomainScreenResult[]> {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    const results = new Array<NWayDomainScreenResult>(domains.length);
    await Promise.all(this.slots.map(async (slot, slotIndex) => {
      for (let domainIndex = slotIndex; domainIndex < domains.length;
        domainIndex += this.slots.length) {
        const domain = domains[domainIndex];
        results[domainIndex] = await this.runScreen(
          slot, moments, domain.taskK, domain.segIds,
        );
      }
    }));
    return results;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots) slot.worker.terminate();
  }

  private runShard(
    slot: SelectorSlot, state: NWaySelectionState,
    candidates: NWayCandidate[], startIndex: number,
  ): Promise<Extract<NWaySelectorWorkerResponse, { type: "result" }>> {
    const jobId = this.nextJobId++;
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        slot.worker.removeEventListener("message", onMessage);
        slot.worker.removeEventListener("error", onError);
        slot.worker.removeEventListener("messageerror", onMessageError);
        self.clearTimeout(timeoutId);
      };
      const timeoutId = self.setTimeout(() => {
        cleanup();
        reject(new Error(`n-way selector worker job ${jobId} timed out`));
      }, NWaySelectorWorkerExecutor.JOB_TIMEOUT_MS);
      const onMessage = (event: MessageEvent<NWaySelectorWorkerResponse>) => {
        const message = event.data;
        if (message.type === "ready" || message.jobId !== jobId) return;
        cleanup();
        if (message.type === "error") reject(new Error(message.message));
        else if (message.type !== "result") {
          reject(new Error("n-way selector worker returned the wrong job kind"));
        } else resolve(message);
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || `n-way selector worker job ${jobId} crashed`));
      };
      const onMessageError = () => {
        cleanup();
        reject(new Error(`n-way selector worker job ${jobId} returned malformed data`));
      };
      slot.worker.addEventListener("message", onMessage);
      slot.worker.addEventListener("error", onError);
      slot.worker.addEventListener("messageerror", onMessageError);
      const request: NWaySelectorWorkerRequest = {
        type: "score", jobId, startIndex,
        state: { N: state.N, K: state.K, t: state.t, l: state.l, w: state.w },
        candidates,
      };
      slot.worker.postMessage(request);
    });
  }

  private runScreen(
    slot: SelectorSlot, moments: NWayScreeningMoments,
    taskK: number, segIds: readonly number[],
  ): Promise<NWayDomainScreenResult> {
    const jobId = this.nextJobId++;
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        slot.worker.removeEventListener("message", onMessage);
        slot.worker.removeEventListener("error", onError);
        slot.worker.removeEventListener("messageerror", onMessageError);
        self.clearTimeout(timeoutId);
      };
      const timeoutId = self.setTimeout(() => {
        cleanup();
        reject(new Error(`n-way selector screen job ${jobId} timed out`));
      }, NWaySelectorWorkerExecutor.JOB_TIMEOUT_MS);
      const onMessage = (event: MessageEvent<NWaySelectorWorkerResponse>) => {
        const message = event.data;
        if (message.type === "ready" || message.jobId !== jobId) return;
        cleanup();
        if (message.type === "error") reject(new Error(message.message));
        else if (message.type !== "screen_result") {
          reject(new Error("n-way selector worker returned the wrong job kind"));
        } else {
          resolve({
            taskK: message.taskK,
            entropySegIds: message.entropySegIds,
            fisherSegIds: message.fisherSegIds,
            entropyMs: message.entropyMs,
            fisherMs: message.fisherMs,
          });
        }
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || `n-way selector screen job ${jobId} crashed`));
      };
      const onMessageError = () => {
        cleanup();
        reject(new Error(`n-way selector screen job ${jobId} returned malformed data`));
      };
      slot.worker.addEventListener("message", onMessage);
      slot.worker.addEventListener("error", onError);
      slot.worker.addEventListener("messageerror", onMessageError);
      const packedSegIds = Float64Array.from(segIds);
      const request: NWaySelectorWorkerRequest = {
        type: "screen", jobId, moments, taskK, segIds: packedSegIds,
      };
      slot.worker.postMessage(request, { transfer: [packedSegIds.buffer] });
    });
  }

  private createSlot(): SelectorSlot {
    const worker = new Worker(new URL("./nway_selector_worker.ts", import.meta.url), {
      type: "module",
    });
    let markReady!: () => void;
    let rejectReady!: (error: Error) => void;
    const ready = new Promise<void>((resolve, reject) => {
      markReady = resolve;
      rejectReady = reject;
    });
    worker.addEventListener("message", (event: MessageEvent<NWaySelectorWorkerResponse>) => {
      const message = event.data;
      if (message.type === "ready") markReady();
      else if (message.type === "error" && message.jobId === null) {
        rejectReady(new Error(message.message));
      }
    });
    worker.addEventListener("error", (event) => {
      event.preventDefault();
      rejectReady(new Error(event.message || "n-way selector worker crashed"));
    });
    const payload = packComputeInputs(this.inputs);
    const init: NWaySelectorWorkerRequest = { type: "init", payload };
    worker.postMessage(init, { transfer: computePayloadTransferables(payload) });
    return { worker, ready };
  }
}
