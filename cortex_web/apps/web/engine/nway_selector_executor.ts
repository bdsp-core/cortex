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
import {
  selectCalibratedWorkerCount, type WorkerCalibrationSample,
} from "./execution_profile";

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

export interface NWayWorkerCalibration {
  sampleCandidates: number;
  selectedWorkerCount: number;
  totalDurationMs: number;
  samples: WorkerCalibrationSample[];
  result: "measured_v1" | "no_candidates";
}

/** Persistent deterministic candidate-shard pool. Workers return only indexed
 * loss vectors; ordering, tie behavior, and selection remain centralized. */
export class NWaySelectorWorkerExecutor implements NWaySelectionExecutor {
  private static readonly READY_TIMEOUT_MS = 10_000;
  private static readonly JOB_TIMEOUT_MS = 30_000;
  private slots: SelectorSlot[];
  private nextJobId = 1;
  private disposed = false;
  private calibration: NWayWorkerCalibration | null = null;

  constructor(private readonly inputs: ComputeEngineInputs, workerCount: number) {
    if (!Number.isInteger(workerCount) || workerCount < 1 || workerCount > 12) {
      throw new Error(`invalid n-way selector worker count: ${workerCount}`);
    }
    this.slots = Array.from({ length: workerCount }, () => this.createSlot());
  }

  get workerCount(): number {
    return this.slots.length;
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

  /** Bounded, policy-independent startup probe over real immutable segment
   * metadata and a synthetic N×K cloud. It chooses pool size only; all probe
   * results are discarded before the session RNG/state is created. */
  async calibrate(): Promise<NWayWorkerCalibration> {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    if (this.calibration) return this.calibration;
    const calibrationStartedAt = performance.now();
    const candidates = this.calibrationCandidates(24);
    if (candidates.length === 0) {
      this.activateWorkerCount(1);
      this.calibration = {
        sampleCandidates: 0, selectedWorkerCount: 1,
        totalDurationMs: performance.now() - calibrationStartedAt, samples: [],
        result: "no_candidates",
      };
      return this.calibration;
    }
    const N = this.inputs.nParticles ?? 600;
    const K = this.inputs.taskCodes.length;
    const weights = new Float64Array(N);
    weights.fill(1 / N);
    const state: NWaySelectionState = {
      N, K,
      t: new Float64Array(N * K),
      l: new Float64Array(N * K),
      w: weights,
    };
    // Warm every slot before comparing counts so lazy module/JIT startup does
    // not systematically penalize the first measured profile.
    await this.scoreWithWorkerCount(
      state, candidates.slice(0, Math.min(candidates.length, this.slots.length)),
      this.slots.length,
    );
    const counts = [1, 2, 4, 6, 8, 12]
      .filter((count) => count <= this.slots.length);
    if (!counts.includes(this.slots.length)) counts.push(this.slots.length);
    counts.sort((a, b) => a - b);
    const samples: WorkerCalibrationSample[] = [];
    let reference: Float64Array | null = null;
    for (const workers of counts) {
      const startedAt = performance.now();
      const heartbeatIntervalMs = 10;
      let heartbeatExpectedAt = startedAt + heartbeatIntervalMs;
      let heartbeatMaxDelayMs = 0;
      const heartbeatId = self.setInterval(() => {
        const now = performance.now();
        heartbeatMaxDelayMs = Math.max(
          heartbeatMaxDelayMs, Math.max(0, now - heartbeatExpectedAt),
        );
        heartbeatExpectedAt = now + heartbeatIntervalMs;
      }, heartbeatIntervalMs);
      let losses: Float64Array;
      try {
        losses = await this.scoreWithWorkerCount(state, candidates, workers);
      } finally {
        self.clearInterval(heartbeatId);
      }
      const durationMs = performance.now() - startedAt;
      if (reference === null) reference = losses;
      else if (losses.length !== reference.length
        || losses.some((loss, index) => !Object.is(loss, reference![index]))) {
        throw new Error("n-way selector calibration changed an exact loss");
      }
      samples.push({ workers, durationMs, heartbeatMaxDelayMs });
    }
    const selectedWorkerCount = selectCalibratedWorkerCount(samples);
    this.activateWorkerCount(selectedWorkerCount);
    this.calibration = {
      sampleCandidates: candidates.length,
      selectedWorkerCount,
      totalDurationMs: performance.now() - calibrationStartedAt,
      samples,
      result: "measured_v1",
    };
    return this.calibration;
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

  private async scoreWithWorkerCount(
    state: NWaySelectionState, candidates: readonly NWayCandidate[], workerCount: number,
  ): Promise<Float64Array> {
    const shardCount = Math.min(workerCount, candidates.length);
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
        throw new Error("n-way selector worker returned a misaligned calibration shard");
      }
      losses.set(response.losses, startIndex);
    }));
    return losses;
  }

  private calibrationCandidates(limit: number): NWayCandidate[] {
    const K = this.inputs.taskCodes.length;
    const domains: NWayCandidate[][] = Array.from({ length: K }, () => []);
    for (const segment of this.inputs.segments) {
      const applicable = segment.applicableTaskIdx
        ?? Array.from({ length: K }, (_unused, k) => k);
      for (const k of applicable) {
        if (this.inputs.taskClasses?.[k] === "iiic"
            || this.inputs.taskClasses?.[k] === "spike") {
          domains[k].push({ k, segment });
        }
      }
    }
    const populated = domains.filter((domain) => domain.length > 0);
    if (populated.length === 0) return [];
    const perDomain = Math.max(1, Math.ceil(limit / populated.length));
    const result: NWayCandidate[] = [];
    for (const domain of populated) {
      const count = Math.min(perDomain, domain.length);
      for (let i = 0; i < count && result.length < limit; i++) {
        const index = count === 1 ? 0
          : Math.round(i * (domain.length - 1) / (count - 1));
        result.push(domain[index]);
      }
    }
    return result;
  }

  private activateWorkerCount(workerCount: number): void {
    const selected = Math.max(1, Math.min(this.slots.length, workerCount));
    for (const slot of this.slots.slice(selected)) slot.worker.terminate();
    this.slots = this.slots.slice(0, selected);
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
