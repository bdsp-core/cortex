// Web Worker entry — runs the SMC session off the main thread so the UI
// never blocks on particle-filter compute. The main thread talks to it with
// structured-clone messages.
//
// Protocol (main → worker):
//   { type: "init", payload, sessionId, seed }     start a session
//   { type: "answer", pick }                       submit a 0-based 6-way pick
//   { type: "abort" }
// Worker → main:
//   { type: "item", trialIndex, taskK, segId }     next question is chosen
//   { type: "trial", diag }                        post-answer telemetry
//   { type: "done", result }                       session finished
//   { type: "error", message }

import { WebCortexSession, seedFromSessionId } from "./session";
import type { ComputeEngineInputs } from "./types";
import type { EngineWorkerRequest, EngineWorkerResponse } from "./worker_protocol";
import { BranchWorkerExecutor } from "./branch_worker_executor";
import { NWaySelectorWorkerExecutor } from "./nway_selector_executor";
import { selectExecutionProfile } from "./execution_profile";
import { unpackComputeInputs } from "./compute_payload";

let session: WebCortexSession | null = null;
let branchExecutor: BranchWorkerExecutor | null = null;
let selectionExecutor: NWaySelectorWorkerExecutor | null = null;

function estimatedWorkerMemoryBytes(inputs: ComputeEngineInputs, workers: number): number {
  const K = inputs.taskCodes.length;
  const N = inputs.nParticles ?? 600;
  const bankBytes = inputs.segments.length * (8 + 4 + 16 * K);
  const particleBytes = N * K * 16 + N * 24;
  return workers * (bankBytes + particleBytes);
}

self.onmessage = async (ev: MessageEvent<EngineWorkerRequest>) => {
  const msg = ev.data;
  const post = (message: EngineWorkerResponse, transfer: Transferable[] = []) =>
    self.postMessage(message, { transfer });
  try {
    if (msg.type === "init") {
      const inputs: ComputeEngineInputs = unpackComputeInputs(msg.payload);
      const seed: number =
        typeof msg.seed === "number" ? msg.seed : await seedFromSessionId(msg.sessionId);
      branchExecutor?.dispose();
      branchExecutor = null;
      selectionExecutor?.dispose();
      selectionExecutor = null;
      const qualifiedOverride = msg.qualificationHardwareConcurrency;
      const hardwareConcurrency = Number.isFinite(qualifiedOverride)
        && qualifiedOverride !== undefined && qualifiedOverride >= 1
        ? qualifiedOverride : self.navigator.hardwareConcurrency || undefined;
      let profile = selectExecutionProfile({
        requested: msg.requestedComputeMode ?? "serial",
        policy: inputs.terminationPolicy ?? "ad6",
        hardwareConcurrency,
        workerAvailable: typeof Worker !== "undefined",
      });
      let calibrationResult = "not_run";
      if (profile.mode === "adaptive_pool") {
        const candidate = new NWaySelectorWorkerExecutor(inputs, profile.computeWorkers);
        try {
          await candidate.ready();
          const calibration = await candidate.calibrate();
          calibrationResult = JSON.stringify({
            version: 1,
            result: calibration.result,
            sampleCandidates: calibration.sampleCandidates,
            selectedWorkerCount: calibration.selectedWorkerCount,
            totalDurationMs: Math.round(calibration.totalDurationMs * 10) / 10,
            samples: calibration.samples.map((sample) => ({
              workers: sample.workers,
              durationMs: Math.round(sample.durationMs * 10) / 10,
              heartbeatMaxDelayMs:
                Math.round(sample.heartbeatMaxDelayMs * 10) / 10,
            })),
          });
          profile = {
            ...profile,
            computeWorkers: calibration.selectedWorkerCount,
            ...(calibration.selectedWorkerCount === 1 ? {
              mode: "serial" as const,
              reason: "calibration_selected_serial" as const,
            } : {}),
          };
          if (profile.mode === "adaptive_pool") selectionExecutor = candidate;
          else candidate.dispose();
        } catch {
          candidate.dispose();
          calibrationResult = "calibration_failed_closed";
          profile = {
            mode: "serial", computeWorkers: 1, reason: "worker_unavailable",
          };
        }
      }
      post({ type: "performance", event: {
        kind: "execution_profile",
        requested: msg.requestedComputeMode ?? "serial",
        executionMode: profile.mode,
        reason: profile.reason,
        hardwareConcurrency: hardwareConcurrency ?? null,
        selectedWorkerCount: profile.computeWorkers,
        calibrationResult,
        estimatedWorkerMemoryBytes: estimatedWorkerMemoryBytes(
          inputs, profile.computeWorkers,
        ),
      } });
      session = new WebCortexSession(inputs, msg.sessionId, seed, {
        onItem: (item) => post({ type: "item", ...item }),
        onTrial: (diag) => post({ type: "trial", diag }),
        onPerformance: (event) => post({ type: "performance", event }),
        onDone: (result) => {
          branchExecutor?.dispose();
          branchExecutor = null;
          selectionExecutor?.dispose();
          selectionExecutor = null;
          post({ type: "done", result }, [
            result.traj.t.buffer, result.traj.l.buffer, result.traj.w.buffer,
          ]);
        },
        // Speculative precompute ON by default in production (bit-identical to
        // inline; hides the N=1200 selection in think-time). An init message may
        // set speculative:false to fall back to inline compute.
      }, {
        speculative: msg.speculative ?? true,
        ...(branchExecutor ? { branchExecutor } : {}),
        ...(selectionExecutor ? { selectionExecutor } : {}),
      });
      session.run().catch((e) =>
        {
          branchExecutor?.dispose();
          branchExecutor = null;
          selectionExecutor?.dispose();
          selectionExecutor = null;
          post({ type: "error", message: String(e?.stack || e) });
        },
      );
    } else if (msg.type === "answer") {
      session?.submitAnswer(msg.pick);
    } else if (msg.type === "abort") {
      branchExecutor?.dispose();
      branchExecutor = null;
      selectionExecutor?.dispose();
      selectionExecutor = null;
      session?.abort();
    }
  } catch (e: any) {
    branchExecutor?.dispose();
    branchExecutor = null;
    selectionExecutor?.dispose();
    selectionExecutor = null;
    post({ type: "error", message: String(e?.stack || e) });
  }
};
