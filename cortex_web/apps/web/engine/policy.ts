// AD6 termination + verdict policy — port of scripts/cortex_policy.py.
//
// Per task k after each trial:
//   π_k    = Σ_i w_i · 1[ℓ_k^(i) > ℓ*_k]                 pass-mass
//   mcse_k = √( π_k(1−π_k) / ESS )                        MC error
//   R_k    = 1 − Var_post(ℓ_k) / Var_prior(ℓ_k)            info gate
// RESOLVED iff n_k ≥ N_min ∧ R_k ≥ R* and
//   PASS : π_k − Z·mcse_k ≥ 1 − α
//   FAIL : π_k + Z·mcse_k ≤ α
// Stop when all K resolved. finalize() maps leftover PENDING →
// REFER_BORDERLINE (gate opened) / REFER_UNINFORMATIVE (gate never opened).

import { ParticleState, TerminationPolicyName } from "./types";
import { ess } from "./particles";

export const VERDICT = {
  PASS: "PASS",
  FAIL: "FAIL",
  PENDING: "PENDING",
  REFER_BORDERLINE: "REFER_BORDERLINE",
  REFER_UNINFORMATIVE: "REFER_UNINFORMATIVE",
} as const;

// Paper-grade thresholds — synced with the v1.3.6 instrument freeze
// (data/INSTRUMENT_FREEZE_v1_3_6.json). Tighter α + higher N_min trade some
// questions for confidence; matches the desktop's shipped config.
export const DEFAULT_N_MIN = 20;
export const DEFAULT_R_STAR = 0.3;
export const DEFAULT_ALPHA = 0.05;
export const DEFAULT_Z = 2.0;

export interface PolicyResult {
  policyName: TerminationPolicyName;
  stop: boolean;
  stopReason: string;
  // Selection state is deliberately policy-neutral: AD6 emits its monotone
  // verdict locks; Precision emits fresh ACTIVE/ESTIMATE_COMPLETE/terminal
  // statuses. The selector only tests whether a domain remains active.
  selectionStates: string[];
  verdicts: string[];
  pi: number[];
  mcse: number[];
  R: number[];
  ess: number;
  domainStatuses?: string[];
  determinations?: string[];
  terminalReasons?: (string | null)[];
  streakCounts?: number[];
  diagnostics?: Record<string, unknown>;
}

export interface FinalPolicyResult {
  verdicts: string[];
  domainStatuses?: string[];
  determinations?: string[];
  terminalReasons?: (string | null)[];
  skillIntervals?: [number, number][];
  biasIntervals?: [number, number][];
}

export interface EngineTerminationPolicy {
  readonly name: TerminationPolicyName;
  readonly activeLabel: string;
  reset(K: number): void;
  evaluate(
    st: ParticleState,
    nPerTask: number[],
    telemetry?: Record<string, unknown>,
  ): PolicyResult;
  finalizeResult(ellStar: number[]): FinalPolicyResult;
  clone(): EngineTerminationPolicy;
  snapshot(): PolicySnapshot;
  restore(snapshot: PolicySnapshot): void;
}

export interface AD6PolicySnapshot {
  name: "ad6";
  verdicts: string[];
  lastR: number[];
}

export interface PrecisionPolicySnapshot {
  name: "precision_v1";
  statuses: string[];
  streaks: number[];
  terminalReasons: (string | null)[];
  bandAdministered: number[][];
  // Kept structurally typed at the policy boundary to avoid making the generic
  // policy interface depend on Precision's diagnostic implementation module.
  lastDiag: Record<string, unknown> | null;
}

export type PolicySnapshot = AD6PolicySnapshot | PrecisionPolicySnapshot;

export class AD6Policy implements EngineTerminationPolicy {
  readonly name = "ad6" as const;
  readonly activeLabel = VERDICT.PENDING;
  private ellStar: number[];
  private varPrior: number[];
  private nMin: number;
  private rStar: number;
  private alpha: number;
  private Z: number;
  private verdicts: string[];
  private lastR: number[];

  constructor(
    ellStar: number[],
    varPrior: number[],
    opts: { nMin?: number; rStar?: number; alpha?: number; Z?: number } = {},
  ) {
    this.ellStar = ellStar;
    this.varPrior = varPrior;
    this.nMin = opts.nMin ?? DEFAULT_N_MIN;
    this.rStar = opts.rStar ?? DEFAULT_R_STAR;
    this.alpha = opts.alpha ?? DEFAULT_ALPHA;
    this.Z = opts.Z ?? DEFAULT_Z;
    this.verdicts = new Array(ellStar.length).fill(VERDICT.PENDING);
    this.lastR = new Array(ellStar.length).fill(0);
  }

  // Var_prior from diag(Corr_l); ℓ* from cert_config v13.
  static fromInputs(
    ellStar: number[],
    corrL: number[][],
    opts = {},
  ): AD6Policy {
    const varPrior = corrL.map((row, i) => row[i]);
    return new AD6Policy(ellStar, varPrior, opts);
  }

  evaluate(st: ParticleState, nPerTask: number[]): PolicyResult {
    const { N, K, l, w } = st;
    const e = ess(w);
    const pi = new Array(K).fill(0);
    const mcse = new Array(K).fill(0);
    const R = new Array(K).fill(0);
    for (let k = 0; k < K; k++) {
      let piK = 0;
      let mu = 0;
      for (let n = 0; n < N; n++) {
        const lk = l[n * K + k];
        if (lk > this.ellStar[k]) piK += w[n];
        mu += w[n] * lk;
      }
      let varPost = 0;
      for (let n = 0; n < N; n++) {
        const d = l[n * K + k] - mu;
        varPost += w[n] * d * d;
      }
      pi[k] = piK;
      mcse[k] = Math.sqrt(Math.max(piK * (1 - piK), 0) / Math.max(e, 1));
      R[k] = 1 - varPost / this.varPrior[k];
    }
    // monotonic verdict update
    for (let k = 0; k < K; k++) {
      if (this.verdicts[k] !== VERDICT.PENDING) continue;
      if (nPerTask[k] < this.nMin || R[k] < this.rStar) continue;
      if (pi[k] - this.Z * mcse[k] >= 1 - this.alpha) this.verdicts[k] = VERDICT.PASS;
      else if (pi[k] + this.Z * mcse[k] <= this.alpha) this.verdicts[k] = VERDICT.FAIL;
    }
    this.lastR = R;
    const allResolved = this.verdicts.every((v) => v !== VERDICT.PENDING);
    return {
      policyName: this.name,
      stop: allResolved,
      stopReason: allResolved ? "all_resolved" : "continue",
      selectionStates: [...this.verdicts],
      verdicts: [...this.verdicts],
      pi,
      mcse,
      R,
      ess: e,
    };
  }

  reset(K: number): void {
    if (K !== this.ellStar.length) {
      throw new Error(`K=${K} != configured domains=${this.ellStar.length}`);
    }
    this.verdicts.fill(VERDICT.PENDING);
    this.lastR.fill(0);
  }

  finalize(): string[] {
    return this.verdicts.map((v, k) =>
      v !== VERDICT.PENDING
        ? v
        : this.lastR[k] >= this.rStar
          ? VERDICT.REFER_BORDERLINE
          : VERDICT.REFER_UNINFORMATIVE,
    );
  }

  finalizeResult(_ellStar: number[]): FinalPolicyResult {
    return { verdicts: this.finalize() };
  }

  // Deep copy of the monotonic verdict state — for speculative branch isolation
  // (engine/advance.ts). Params (ℓ*, varPrior, thresholds) are immutable and
  // shared; only verdicts + lastR are per-branch mutable.
  clone(): AD6Policy {
    const p = new AD6Policy(this.ellStar, this.varPrior, {
      nMin: this.nMin, rStar: this.rStar, alpha: this.alpha, Z: this.Z,
    });
    p.verdicts = this.verdicts.slice();
    p.lastR = this.lastR.slice();
    return p;
  }

  snapshot(): AD6PolicySnapshot {
    return {
      name: this.name,
      verdicts: this.verdicts.slice(),
      lastR: this.lastR.slice(),
    };
  }

  restore(snapshot: PolicySnapshot): void {
    if (snapshot.name !== this.name) throw new Error("AD6 policy snapshot mismatch");
    if (snapshot.verdicts.length !== this.ellStar.length
      || snapshot.lastR.length !== this.ellStar.length) {
      throw new Error("AD6 policy snapshot domain mismatch");
    }
    this.verdicts = snapshot.verdicts.slice();
    this.lastR = snapshot.lastR.slice();
  }
}
