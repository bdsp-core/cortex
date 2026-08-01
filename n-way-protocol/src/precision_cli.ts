// Newline-delimited JSON sidecar exposing the UNCHANGED production Precision
// stopping policy (`precision_v1`) to the governed Python qualification
// harness (python/nway_protocol/qualification.py --stopping precision).
//
// Construction mirrors the production session exactly:
//   cortex_web/apps/web/engine/session.ts:277-296
//     policy = PrecisionPolicy.fromInputs(
//       inputs,
//       qualificationPrecisionPerDomainCap ?? PRECISION_PER_DOMAIN_CAP,  // 60
//       qualificationPrecisionNMin ?? PRECISION_N_MIN,                   // 20
//     );
//     policy.reset(K);
// with `inputs` shaped like the served manifest: the frozen seven-task order
// (precision_policy.ts:31-33), corrL/corrT prior correlation blocks, and
// `precisionBandEdges` derived from the FULL served bank BEFORE any
// per-session draw (cortex_web/services/api/session_bank.py:60-83;
// engine/types.ts:57-60). The harness therefore always sends explicit edges.
//
// Per-answer evaluation mirrors cortex_web/apps/web/engine/advance.ts:498-514:
//   core.remaining.delete(chosen.segId);
//   core.nPerTask[chosen.k] += 1;
//   policy.recordAdministered(chosen.k, chosen.s);
//   telemetry = policy.bankTelemetry(remainingBank, state.lastRejuvenation);
//   policy.evaluate(state, core.nPerTask, telemetry);
// The particle update (and any resample/rejuvenation) has already happened on
// the Python side; the wire carries the post-update cloud and the most recent
// rejuvenation telemetry, exactly what `evaluate` consumes.
//
// Wire format (one JSON object per line on stdin; one JSON reply per line on
// stdout):
//   {"op":"ping"}
//   {"op":"init","sessionId",corrL,corrT,precisionBandEdges,perDomainCap?,nMin?}
//   {"op":"evaluate","sessionId","administered":{k,signal}|null,
//    "state":{N,K,t,l,w,lastRejuvenation|null},"nPerTask":[7],
//    "bank":{askedK,segId,sMean,sSd}}   // parallel remaining-candidate views
// Replies: {"ok":true,...} or {"ok":false,"error":...}.

import {
  PRECISION_N_MIN, PRECISION_PER_DOMAIN_CAP, PRECISION_TASK_CODES,
  PrecisionPolicy,
} from "../../cortex_web/apps/web/engine/precision_policy";
import type { ComputeEngineInputs } from "../../cortex_web/apps/web/engine/types";
import type { PolicyResult } from "../../cortex_web/apps/web/engine/policy";
import { evaluateFrozenPrecision } from "./precision_bridge";
import type {
  Candidate, PriorPieces, ProtocolParticleState, RejuvenationTelemetry,
} from "./types";

// The sidecar runs under node but the package tsconfig deliberately stays on
// the browser lib (the protocol sources are production-engine-shaped). The
// few process surfaces the CLI touches are declared here instead of pulling
// @types/node into the whole program.
interface SidecarStdin {
  setEncoding(encoding: string): void;
  on(event: "data", listener: (chunk: string) => void): void;
  on(event: "end", listener: () => void): void;
}
declare const process: {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
  stdin: SidecarStdin;
  stdout: { write(text: string): boolean };
};

export interface WireRejuvenation {
  qIndex: number;
  acceptanceRate: number;
  distinctAncestors: number;
  distinctAncestorFraction: number;
}

export interface WireState {
  N: number;
  K: number;
  t: number[];
  l: number[];
  w: number[];
  lastRejuvenation: WireRejuvenation | null;
}

export interface WireBank {
  askedK: number[];
  segId: number[];
  sMean: number[];
  sSd: number[];
}

export interface InitRequest {
  op: "init";
  sessionId: string;
  corrL: number[][];
  corrT: number[][];
  precisionBandEdges: number[][];
  perDomainCap?: number;
  nMin?: number;
}

export interface EvaluateRequest {
  op: "evaluate";
  sessionId: string;
  administered: { k: number; signal: number } | null;
  state: WireState;
  nPerTask: number[];
  bank: WireBank;
}

export type SidecarRequest = { op: "ping" } | InitRequest | EvaluateRequest;
export type SidecarResponse = Record<string, unknown>;

const K = PRECISION_TASK_CODES.length;

function frozenInputs(request: InitRequest): ComputeEngineInputs {
  // Only taskCodes, corrL, corrT and precisionBandEdges are consumed by
  // PrecisionPolicy.fromInputs (varPrior = diag(corrL); explicit edges skip
  // the segment-derived terciles). The remaining fields satisfy the served
  // manifest shape.
  return {
    taskCodes: [...PRECISION_TASK_CODES],
    taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
    taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
    corrL: request.corrL,
    corrT: request.corrT,
    precisionBandEdges: request.precisionBandEdges,
    ellStar: new Array(K).fill(0),
    segments: [],
  };
}

function wireParticleState(state: WireState): ProtocolParticleState {
  if (!Number.isInteger(state.N) || state.N <= 0 || state.K !== K) {
    throw new Error(`state must carry N>0 and K=${K}`);
  }
  if (state.t.length !== state.N * K || state.l.length !== state.N * K
    || state.w.length !== state.N) {
    throw new Error("state arrays disagree with N*K");
  }
  // evaluate() reads only N/K/t/l/w (+ lastRejuvenation); the remaining
  // ProtocolParticleState fields are structural.
  const stubPieces: PriorPieces = { K, sigmaInv: [], logDet: 0, L: [] };
  const lastRejuvenation: RejuvenationTelemetry | undefined =
    state.lastRejuvenation ?? undefined;
  return {
    N: state.N,
    K,
    t: Float64Array.from(state.t),
    l: Float64Array.from(state.l),
    w: Float64Array.from(state.w),
    logPrior: new Float64Array(state.N),
    logLik: new Float64Array(state.N),
    history: [],
    prior: { tPieces: stubPieces, lPieces: stubPieces },
    ...(lastRejuvenation ? { lastRejuvenation } : {}),
  };
}

function wireCandidates(bank: WireBank): Candidate[] {
  const n = bank.askedK.length;
  if (bank.segId.length !== n || bank.sMean.length !== n || bank.sSd.length !== n) {
    throw new Error("bank candidate arrays must be parallel");
  }
  const candidates: Candidate[] = [];
  for (let i = 0; i < n; i += 1) {
    candidates.push({
      askedK: bank.askedK[i],
      segmentIndex: i,
      segId: bank.segId[i],
      focalSignal: bank.sMean[i],
      focalSignalSd: bank.sSd[i],
    });
  }
  return candidates;
}

/** One live session per sidecar, mirroring one WebCortexSession run. */
export class PrecisionSidecar {
  private sessionId: string | null = null;
  private policy: PrecisionPolicy | null = null;

  handle(request: SidecarRequest): SidecarResponse {
    if (request.op === "ping") return { ok: true, op: "pong" };
    if (request.op === "init") return this.init(request);
    if (request.op === "evaluate") return this.evaluate(request);
    throw new Error(`unknown op ${String((request as { op?: unknown }).op)}`);
  }

  private init(request: InitRequest): SidecarResponse {
    const perDomainCap = request.perDomainCap ?? PRECISION_PER_DOMAIN_CAP;
    const nMin = request.nMin ?? PRECISION_N_MIN;
    // The FROZEN production policy has no override surface — the wire keeps
    // the optional fields for shape compatibility, but any value other than
    // the frozen constants is refused loudly rather than silently applied
    // (the 3-argument fromInputs exists only on the WIP engine branch).
    if (perDomainCap !== PRECISION_PER_DOMAIN_CAP || nMin !== PRECISION_N_MIN) {
      throw new Error(
        `the frozen precision policy accepts only perDomainCap=${
          PRECISION_PER_DOMAIN_CAP} and nMin=${PRECISION_N_MIN}`,
      );
    }
    // session.ts:262 + policy.reset(K).
    const policy = PrecisionPolicy.fromInputs(frozenInputs(request));
    policy.reset(K);
    this.sessionId = request.sessionId;
    this.policy = policy;
    return { ok: true, sessionId: request.sessionId, K, perDomainCap, nMin };
  }

  private evaluate(request: EvaluateRequest): SidecarResponse {
    if (!this.policy || this.sessionId !== request.sessionId) {
      throw new Error(
        `evaluate for session ${request.sessionId} but active session is ${this.sessionId}`,
      );
    }
    if (request.nPerTask.length !== K) throw new Error(`nPerTask must have length ${K}`);
    // advance.ts:500 — the administered item is recorded before telemetry.
    if (request.administered) {
      this.policy.recordAdministered(request.administered.k, request.administered.signal);
    }
    const state = wireParticleState(request.state);
    const candidates = wireCandidates(request.bank);
    const result: PolicyResult = evaluateFrozenPrecision({
      policy: this.policy,
      state,
      nPerTask: request.nPerTask,
      remainingCandidates: candidates,
      remainingSegmentIds: new Set(candidates.map((candidate) => candidate.segId)),
    });
    return { ok: true, result };
  }
}

export function main(): void {
  const sidecar = new PrecisionSidecar();
  let pending = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk: string) => {
    pending += chunk;
    let newline = pending.indexOf("\n");
    while (newline >= 0) {
      const line = pending.slice(0, newline).trim();
      pending = pending.slice(newline + 1);
      newline = pending.indexOf("\n");
      if (!line) continue;
      let response: SidecarResponse;
      try {
        response = sidecar.handle(JSON.parse(line) as SidecarRequest);
      } catch (error) {
        response = { ok: false, error: error instanceof Error ? error.message : String(error) };
      }
      process.stdout.write(`${JSON.stringify(response)}\n`);
    }
  });
  process.stdin.on("end", () => process.exit(0));
}

// The Python client launches the bundle with this variable set; importing the
// module (vitest golden test) must not start the stdin loop.
if (typeof process !== "undefined" && process.env.CORTEX_PRECISION_SIDECAR === "1") {
  main();
}
