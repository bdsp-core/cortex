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
import type { ComputeEngineInputs, PackedParticleHistory } from "./types";
import {
  selectCalibratedWorkerCount, type WorkerCalibrationSample,
} from "./execution_profile";
import {
  SpeculationCancelledError, type SpeculationCancellationPhase,
} from "./speculation_cancellation";

interface SelectorSlot {
  worker: Worker;
  ready: Promise<void>;
  historyVersion: number | null;
}

interface ActiveSelectorJob {
  phase: Exclude<SpeculationCancellationPhase, "between_phases">;
  cancel: (error: Error) => void;
}

export interface NWayCancellationReport {
  cancelledJobs: number;
  phases: Exclude<SpeculationCancellationPhase, "between_phases">[];
}

export interface NWayCancellationRestart {
  report: NWayCancellationReport;
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
  historyLikelihood(
    history: PackedParticleHistory, N: number, K: number,
    t: Float64Array, l: Float64Array,
  ): Promise<Float64Array>;
  restartAfterCancellation?(): NWayCancellationRestart;
  dispose(): void;
}

export interface NWayWorkerCalibration {
  sampleCandidates: number;
  sampleScreenCandidates: number;
  sampleHistoryObservations: number;
  selectedWorkerCount: number;
  totalDurationMs: number;
  samples: WorkerCalibrationSample[];
  result: "measured_v2" | "no_candidates";
}

export interface NWayCalibrationProfile {
  /** Exact-refinement candidates scored for every candidate worker count. */
  exactCandidates: number;
  /** Evenly-spaced screening segments sampled independently per runtime domain. */
  screenCandidatesPerDomain: number;
  /** Mixed binary/categorical history rows evaluated over all particles. */
  historyObservations: number;
}

/** Bounded startup work, deliberately named and centralized for future bank or
 * domain-count requalification. Domains, response classes, K, N, and segment
 * signals are always derived from the served runtime inputs. */
export const DEFAULT_NWAY_CALIBRATION_PROFILE: Readonly<NWayCalibrationProfile> = {
  exactCandidates: 96,
  screenCandidatesPerDomain: 1024,
  historyObservations: 48,
};

export interface NWaySelectorExecutorOptions {
  /** Qualification injection only; production uses the same-origin module worker. */
  workerFactory?: () => Worker;
  jobTimeoutMs?: number;
  calibrationProfile?: Partial<NWayCalibrationProfile>;
}

/** Persistent deterministic candidate-shard pool. Startup calibration selects
 * its session-fixed size. Workers return only indexed loss vectors; ordering,
 * tie behavior, and selection remain centralized. */
export class NWaySelectorWorkerExecutor implements NWaySelectionExecutor {
  private static readonly READY_TIMEOUT_MS = 10_000;
  private static readonly JOB_TIMEOUT_MS = 30_000;
  private slots: SelectorSlot[];
  private nextJobId = 1;
  private disposed = false;
  private calibration: NWayWorkerCalibration | null = null;
  private readonly workerFactory?: () => Worker;
  private readonly jobTimeoutMs: number;
  private readonly calibrationProfile: NWayCalibrationProfile;
  private readonly activeJobs = new Map<number, ActiveSelectorJob>();
  private readonly historyVersions = new WeakMap<
    PackedParticleHistory, { length: number; version: number }
  >();
  private nextHistoryVersion = 1;

  constructor(
    private readonly inputs: ComputeEngineInputs, workerCount: number,
    options: NWaySelectorExecutorOptions = {},
  ) {
    if (!Number.isInteger(workerCount) || workerCount < 1 || workerCount > 12) {
      throw new Error(`invalid n-way selector worker count: ${workerCount}`);
    }
    this.workerFactory = options.workerFactory;
    this.jobTimeoutMs = options.jobTimeoutMs ?? NWaySelectorWorkerExecutor.JOB_TIMEOUT_MS;
    this.calibrationProfile = {
      ...DEFAULT_NWAY_CALIBRATION_PROFILE,
      ...options.calibrationProfile,
    };
    for (const [name, value] of Object.entries(this.calibrationProfile)) {
      if (!Number.isInteger(value) || value < 1) {
        throw new Error(`invalid n-way calibration ${name}: ${value}`);
      }
    }
    this.slots = Array.from({ length: workerCount }, () => this.createSlot());
  }

  get workerCount(): number {
    return this.slots.length;
  }

  async ready(): Promise<void> {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_resolve, reject) => {
      timeoutId = globalThis.setTimeout(() => reject(
        new Error("n-way selector worker initialization timed out"),
      ), NWaySelectorWorkerExecutor.READY_TIMEOUT_MS);
    });
    try {
      await Promise.race([Promise.all(this.slots.map((slot) => slot.ready)), timeout]);
    } finally {
      if (timeoutId !== undefined) globalThis.clearTimeout(timeoutId);
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
   * metadata and a deterministic N×K cloud. It represents screening, exact
   * refinement, and categorical history replay. It chooses pool size only;
   * all probe results are discarded before the session RNG/state is created. */
  async calibrate(): Promise<NWayWorkerCalibration> {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    if (this.calibration) return this.calibration;
    const calibrationStartedAt = performance.now();
    const candidates = this.calibrationCandidates(
      this.calibrationProfile.exactCandidates,
    );
    if (candidates.length === 0) {
      this.activateWorkerCount(1);
      this.calibration = {
        sampleCandidates: 0, sampleScreenCandidates: 0,
        sampleHistoryObservations: 0, selectedWorkerCount: 1,
        totalDurationMs: performance.now() - calibrationStartedAt, samples: [],
        result: "no_candidates",
      };
      return this.calibration;
    }
    const N = this.inputs.nParticles ?? 600;
    const K = this.inputs.taskCodes.length;
    const weights = new Float64Array(N);
    const t = new Float64Array(N * K);
    const l = new Float64Array(N * K);
    let weightSum = 0;
    for (let n = 0; n < N; n++) {
      weights[n] = 1 + (n % 17);
      weightSum += weights[n];
      for (let k = 0; k < K; k++) {
        const index = n * K + k;
        t[index] = (((n * 17 + k * 13) % 101) - 50) / 80;
        l[index] = (((n * 11 + k * 19) % 97) - 48) / 96;
      }
    }
    for (let n = 0; n < N; n++) weights[n] /= weightSum;
    const state: NWaySelectionState = {
      N, K, t, l, w: weights,
    };
    const screenDomains = this.calibrationScreenDomains(
      this.calibrationProfile.screenCandidatesPerDomain,
    );
    const moments: NWayScreeningMoments = {
      K,
      tMean: Array.from({ length: K }, (_unused, k) => (k - K / 2) / 10),
      lMean: Array.from({ length: K }, (_unused, k) => (K / 2 - k) / 12),
      tSd: Array.from({ length: K }, (_unused, k) => 0.45 + (k % 3) * 0.1),
      lSd: Array.from({ length: K }, (_unused, k) => 0.35 + (k % 2) * 0.1),
    };
    const history = this.calibrationHistory(
      this.calibrationProfile.historyObservations,
    );
    const sampleScreenCandidates = screenDomains.reduce(
      (sum, domain) => sum + domain.segIds.length, 0,
    );
    // Warm every job kind before comparing counts so lazy module/JIT startup
    // does not systematically penalize the first measured profile.
    await this.scoreWithWorkerCount(
      state, candidates.slice(0, Math.min(candidates.length, this.slots.length)),
      this.slots.length,
    );
    await this.screenWithWorkerCount(
      moments,
      screenDomains.map((domain) => ({ ...domain, segIds: domain.segIds.slice(0, 1) })),
      this.slots.length,
    );
    await this.historyLikelihoodWithWorkerCount(
      history, N, K, t, l, this.slots.length,
    );
    const counts = [1, 2, 4, 6, 8, 12]
      .filter((count) => count <= this.slots.length);
    if (!counts.includes(this.slots.length)) counts.push(this.slots.length);
    counts.sort((a, b) => a - b);
    const samples: WorkerCalibrationSample[] = [];
    let referenceLosses: Float64Array | null = null;
    let referenceScreens: NWayDomainScreenResult[] | null = null;
    let referenceHistory: Float64Array | null = null;
    for (const workers of counts) {
      const startedAt = performance.now();
      const heartbeatIntervalMs = 10;
      let heartbeatExpectedAt = startedAt + heartbeatIntervalMs;
      let heartbeatMaxDelayMs = 0;
      const heartbeatId = globalThis.setInterval(() => {
        const now = performance.now();
        heartbeatMaxDelayMs = Math.max(
          heartbeatMaxDelayMs, Math.max(0, now - heartbeatExpectedAt),
        );
        heartbeatExpectedAt = now + heartbeatIntervalMs;
      }, heartbeatIntervalMs);
      let losses: Float64Array;
      let screens: NWayDomainScreenResult[];
      let historyLikelihood: Float64Array;
      try {
        screens = await this.screenWithWorkerCount(
          moments, screenDomains, workers,
        );
        losses = await this.scoreWithWorkerCount(state, candidates, workers);
        historyLikelihood = await this.historyLikelihoodWithWorkerCount(
          history, N, K, t, l, workers,
        );
      } finally {
        globalThis.clearInterval(heartbeatId);
      }
      const durationMs = performance.now() - startedAt;
      if (referenceLosses === null) referenceLosses = losses;
      else if (losses.length !== referenceLosses.length
        || losses.some((loss, index) => !Object.is(loss, referenceLosses![index]))) {
        throw new Error("n-way selector calibration changed an exact loss");
      }
      if (referenceScreens === null) referenceScreens = screens;
      else if (!this.sameScreenResults(screens, referenceScreens)) {
        throw new Error("n-way selector calibration changed an exact screen");
      }
      if (referenceHistory === null) referenceHistory = historyLikelihood;
      else if (historyLikelihood.length !== referenceHistory.length
        || historyLikelihood.some((value, index) =>
          !Object.is(value, referenceHistory![index]))) {
        throw new Error("n-way selector calibration changed an exact history likelihood");
      }
      samples.push({ workers, durationMs, heartbeatMaxDelayMs });
    }
    const selectedWorkerCount = selectCalibratedWorkerCount(samples);
    this.activateWorkerCount(selectedWorkerCount);
    this.calibration = {
      sampleCandidates: candidates.length,
      sampleScreenCandidates,
      sampleHistoryObservations: history.length,
      selectedWorkerCount,
      totalDurationMs: performance.now() - calibrationStartedAt,
      samples,
      result: "measured_v2",
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

  /** Particle-index sharding preserves each particle's original history order;
   * no floating-point reduction crosses a worker boundary. */
  async historyLikelihood(
    history: PackedParticleHistory, N: number, K: number,
    t: Float64Array, l: Float64Array,
  ): Promise<Float64Array> {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    if (t.length !== N * K || l.length !== N * K) {
      throw new Error("n-way history likelihood dimensions are invalid");
    }
    const shardCount = Math.min(this.slots.length, N);
    const shardSize = Math.ceil(N / shardCount);
    const logLikelihood = new Float64Array(N);
    const historyVersion = this.versionForHistory(history);
    await Promise.all(Array.from({ length: shardCount }, async (_unused, shardIndex) => {
      const startIndex = shardIndex * shardSize;
      const endIndex = Math.min(N, startIndex + shardSize);
      const response = await this.runHistoryShard(
        this.slots[shardIndex], history, historyVersion, K,
        t.slice(startIndex * K, endIndex * K),
        l.slice(startIndex * K, endIndex * K), startIndex,
      );
      if (response.startIndex !== startIndex
          || response.logLikelihood.length !== endIndex - startIndex) {
        throw new Error("n-way history worker returned a misaligned shard");
      }
      logLikelihood.set(response.logLikelihood, startIndex);
    }));
    return logLikelihood;
  }

  restartAfterCancellation(): NWayCancellationRestart {
    if (this.disposed) throw new Error("n-way selector worker executor is disposed");
    const workerCount = this.slots.length;
    const jobs = [...this.activeJobs.values()];
    const phases = [...new Set(jobs.map((job) => job.phase))];
    const cancellation = new SpeculationCancelledError();
    for (const job of jobs) job.cancel(cancellation);
    for (const slot of this.slots) slot.worker.terminate();
    this.slots = Array.from({ length: workerCount }, () => this.createSlot());
    return {
      report: { cancelledJobs: jobs.length, phases },
      ready: this.ready(),
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const disposal = new Error("n-way selector worker executor is disposed");
    for (const job of [...this.activeJobs.values()]) job.cancel(disposal);
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
        globalThis.clearTimeout(timeoutId);
        this.activeJobs.delete(jobId);
      };
      const timeoutId = globalThis.setTimeout(() => {
        cleanup();
        reject(new Error(`n-way selector worker job ${jobId} timed out`));
      }, this.jobTimeoutMs);
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
      this.activeJobs.set(jobId, {
        phase: "exact_refinement",
        cancel: (error) => {
          cleanup();
          reject(error);
        },
      });
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

  private async screenWithWorkerCount(
    moments: NWayScreeningMoments,
    domains: readonly { taskK: number; segIds: readonly number[] }[],
    workerCount: number,
  ): Promise<NWayDomainScreenResult[]> {
    const activeCount = Math.min(workerCount, this.slots.length);
    const results = new Array<NWayDomainScreenResult>(domains.length);
    await Promise.all(this.slots.slice(0, activeCount).map(async (slot, slotIndex) => {
      for (let domainIndex = slotIndex; domainIndex < domains.length;
        domainIndex += activeCount) {
        const domain = domains[domainIndex];
        results[domainIndex] = await this.runScreen(
          slot, moments, domain.taskK, domain.segIds,
        );
      }
    }));
    return results;
  }

  private async historyLikelihoodWithWorkerCount(
    history: PackedParticleHistory, N: number, K: number,
    t: Float64Array, l: Float64Array, workerCount: number,
  ): Promise<Float64Array> {
    const shardCount = Math.min(workerCount, this.slots.length, N);
    const shardSize = Math.ceil(N / shardCount);
    const logLikelihood = new Float64Array(N);
    const historyVersion = this.versionForHistory(history);
    await Promise.all(Array.from({ length: shardCount }, async (_unused, shardIndex) => {
      const startIndex = shardIndex * shardSize;
      const endIndex = Math.min(N, startIndex + shardSize);
      const response = await this.runHistoryShard(
        this.slots[shardIndex], history, historyVersion, K,
        t.slice(startIndex * K, endIndex * K),
        l.slice(startIndex * K, endIndex * K), startIndex,
      );
      if (response.startIndex !== startIndex
          || response.logLikelihood.length !== endIndex - startIndex) {
        throw new Error("n-way history calibration returned a misaligned shard");
      }
      logLikelihood.set(response.logLikelihood, startIndex);
    }));
    return logLikelihood;
  }

  private calibrationScreenDomains(
    limitPerDomain: number,
  ): { taskK: number; segIds: number[] }[] {
    const K = this.inputs.taskCodes.length;
    const domains: number[][] = Array.from({ length: K }, () => []);
    for (const segment of this.inputs.segments) {
      const applicable = segment.applicableTaskIdx
        ?? Array.from({ length: K }, (_unused, k) => k);
      for (const k of applicable) {
        if (this.inputs.taskClasses?.[k] === "iiic"
            || this.inputs.taskClasses?.[k] === "spike") {
          domains[k].push(segment.segId);
        }
      }
    }
    return domains.flatMap((domain, taskK) => {
      if (domain.length === 0) return [];
      const count = Math.min(limitPerDomain, domain.length);
      const segIds = Array.from({ length: count }, (_unused, index) => {
        const sourceIndex = count === 1 ? 0
          : Math.round(index * (domain.length - 1) / (count - 1));
        return domain[sourceIndex];
      });
      return [{ taskK, segIds }];
    });
  }

  private calibrationHistory(limit: number): PackedParticleHistory {
    const K = this.inputs.taskCodes.length;
    const binary: NWayCandidate[] = [];
    const categorical: NWayCandidate[] = [];
    for (const segment of this.inputs.segments) {
      const applicable = segment.applicableTaskIdx
        ?? Array.from({ length: K }, (_unused, k) => k);
      for (const k of applicable) {
        const candidate = { k, segment };
        if (this.inputs.taskClasses?.[k] === "spike") binary.push(candidate);
        else if (this.inputs.taskClasses?.[k] === "iiic") categorical.push(candidate);
      }
    }
    const capacity = Math.max(1, limit);
    const history: PackedParticleHistory = {
      K, length: 0, capacity,
      kind: new Uint8Array(capacity),
      taskK: new Int8Array(capacity),
      pick: new Int8Array(capacity),
      binaryS: new Float64Array(capacity),
      binarySd: new Float64Array(capacity),
      signalMean: new Float64Array(capacity * K),
      signalSd: new Float64Array(capacity * K),
    };
    const binaryCount = binary.length > 0
      ? Math.min(Math.max(1, Math.ceil(limit / 7)), limit) : 0;
    for (let index = 0; index < binaryCount; index++) {
      const sourceIndex = Math.floor(index * binary.length / binaryCount);
      const candidate = binary[sourceIndex];
      history.kind[index] = 0;
      history.taskK[index] = candidate.k;
      history.pick[index] = index % 2;
      history.binaryS[index] = candidate.segment.sMean[candidate.k];
      history.binarySd[index] = candidate.segment.sSd[candidate.k];
      history.length += 1;
    }
    const categoricalCount = categorical.length > 0 ? limit - history.length : 0;
    const rawOutcomes = this.inputs.taskClasses
      ?.flatMap((taskClass, k) => taskClass === "iiic" ? [k] : []) ?? [];
    for (let offset = 0; offset < categoricalCount; offset++) {
      const index = history.length;
      const sourceIndex = Math.floor(offset * categorical.length / categoricalCount);
      const candidate = categorical[sourceIndex];
      history.kind[index] = 1;
      history.taskK[index] = candidate.k;
      history.pick[index] = rawOutcomes[offset % rawOutcomes.length] ?? candidate.k;
      history.signalMean.set(candidate.segment.sMean, index * K);
      history.signalSd.set(candidate.segment.sSd, index * K);
      history.length += 1;
    }
    return history;
  }

  private sameScreenResults(
    left: readonly NWayDomainScreenResult[],
    right: readonly NWayDomainScreenResult[],
  ): boolean {
    return left.length === right.length && left.every((result, index) => {
      const expected = right[index];
      return result.taskK === expected.taskK
        && result.entropySegIds.length === expected.entropySegIds.length
        && result.entropySegIds.every((segId, i) => segId === expected.entropySegIds[i])
        && result.fisherSegIds.length === expected.fisherSegIds.length
        && result.fisherSegIds.every((segId, i) => segId === expected.fisherSegIds[i]);
    });
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
    if (this.activeJobs.size > 0) {
      throw new Error("cannot activate n-way selector workers during an active job");
    }
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
        globalThis.clearTimeout(timeoutId);
        this.activeJobs.delete(jobId);
      };
      const timeoutId = globalThis.setTimeout(() => {
        cleanup();
        reject(new Error(`n-way selector screen job ${jobId} timed out`));
      }, this.jobTimeoutMs);
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
      this.activeJobs.set(jobId, {
        phase: "screen",
        cancel: (error) => {
          cleanup();
          reject(error);
        },
      });
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

  private runHistoryShard(
    slot: SelectorSlot, history: PackedParticleHistory, historyVersion: number, K: number,
    t: Float64Array, l: Float64Array, startIndex: number,
  ): Promise<Extract<NWaySelectorWorkerResponse, { type: "history_result" }>> {
    const jobId = this.nextJobId++;
    const includeHistory = slot.historyVersion !== historyVersion;
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        slot.worker.removeEventListener("message", onMessage);
        slot.worker.removeEventListener("error", onError);
        slot.worker.removeEventListener("messageerror", onMessageError);
        globalThis.clearTimeout(timeoutId);
        this.activeJobs.delete(jobId);
      };
      const timeoutId = globalThis.setTimeout(() => {
        cleanup();
        reject(new Error(`n-way history worker job ${jobId} timed out`));
      }, this.jobTimeoutMs);
      const onMessage = (event: MessageEvent<NWaySelectorWorkerResponse>) => {
        const message = event.data;
        if (message.type === "ready" || message.jobId !== jobId) return;
        cleanup();
        if (message.type === "error") reject(new Error(message.message));
        else if (message.type !== "history_result") {
          reject(new Error("n-way history worker returned the wrong job kind"));
        } else if (message.historyVersion !== historyVersion) {
          reject(new Error("n-way history worker returned a stale history version"));
        } else {
          slot.historyVersion = historyVersion;
          resolve(message);
        }
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || `n-way history worker job ${jobId} crashed`));
      };
      const onMessageError = () => {
        cleanup();
        reject(new Error(`n-way history worker job ${jobId} returned malformed data`));
      };
      this.activeJobs.set(jobId, {
        phase: "mh_history",
        cancel: (error) => {
          cleanup();
          reject(error);
        },
      });
      slot.worker.addEventListener("message", onMessage);
      slot.worker.addEventListener("error", onError);
      slot.worker.addEventListener("messageerror", onMessageError);
      const request: NWaySelectorWorkerRequest = {
        type: "history_likelihood", jobId, startIndex,
        N: t.length / K, K, historyVersion,
        ...(includeHistory ? { history } : {}),
        t, l,
      };
      slot.worker.postMessage(request, { transfer: [t.buffer, l.buffer] });
    });
  }

  private createSlot(): SelectorSlot {
    const worker = this.workerFactory?.() ?? new Worker(
      new URL("./nway_selector_worker.ts", import.meta.url), { type: "module" },
    );
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
    return { worker, ready, historyVersion: null };
  }

  private versionForHistory(history: PackedParticleHistory): number {
    const existing = this.historyVersions.get(history);
    if (existing?.length === history.length) return existing.version;
    const version = this.nextHistoryVersion++;
    this.historyVersions.set(history, { length: history.length, version });
    return version;
  }
}
