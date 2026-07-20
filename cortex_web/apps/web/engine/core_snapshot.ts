import type { SessionCore } from "./advance";
import { AD6Policy, type PolicySnapshot } from "./policy";
import { PrecisionPolicy } from "./precision_policy";
import { Rng, type RngSnapshot } from "./rng";
import type {
  ComputeEngineInputs, ParticleState, RejuvenationTelemetry,
} from "./types";
import type { PriorPair } from "./types";

export interface ParticleStateSnapshot {
  N: number;
  K: number;
  t: Float64Array;
  l: Float64Array;
  w: Float64Array;
  logPrior: Float64Array;
  logLik: Float64Array;
  history: { k: number; s: number; y: 0 | 1; sSd: number }[];
  lastRejuvenation?: RejuvenationTelemetry;
}

/** Plain structured-clone representation of all branch-dependent state. */
export interface SessionCoreSnapshot {
  state: ParticleStateSnapshot;
  rng: RngSnapshot;
  policy: PolicySnapshot;
  remaining: Int32Array;
  nPerTask: number[];
  cappedTasks: Int32Array;
  lastOutcomes: string[];
  lastTaskK: number;
  streakCount: number;
}

export function snapshotCore(core: SessionCore): SessionCoreSnapshot {
  const state = core.state;
  return {
    state: {
      N: state.N,
      K: state.K,
      t: state.t.slice(),
      l: state.l.slice(),
      w: state.w.slice(),
      logPrior: state.logPrior.slice(),
      logLik: state.logLik.slice(),
      history: state.history.map((x) => ({ ...x })),
      ...(state.lastRejuvenation
        ? { lastRejuvenation: { ...state.lastRejuvenation } }
        : {}),
    },
    rng: core.rng.snapshot(),
    policy: core.policy.snapshot(),
    remaining: Int32Array.from(core.remaining),
    nPerTask: core.nPerTask.slice(),
    cappedTasks: Int32Array.from(core.cappedTasks),
    lastOutcomes: core.lastOutcomes.slice(),
    lastTaskK: core.lastTaskK,
    streakCount: core.streakCount,
  };
}

export function restoreCore(
  snapshot: SessionCoreSnapshot,
  inputs: ComputeEngineInputs,
  prior: PriorPair,
): SessionCore {
  const { state: raw } = snapshot;
  const expectedK = inputs.taskCodes.length;
  if (raw.K !== expectedK || raw.t.length !== raw.N * raw.K
    || raw.l.length !== raw.N * raw.K || raw.w.length !== raw.N
    || raw.logPrior.length !== raw.N || raw.logLik.length !== raw.N
    || snapshot.nPerTask.length !== expectedK
    || snapshot.lastOutcomes.length !== expectedK) {
    throw new Error("invalid session-core snapshot dimensions");
  }
  const state: ParticleState = {
    N: raw.N,
    K: raw.K,
    t: raw.t,
    l: raw.l,
    w: raw.w,
    logPrior: raw.logPrior,
    logLik: raw.logLik,
    history: raw.history.map((x) => ({ ...x })),
    prior,
    ...(raw.lastRejuvenation
      ? { lastRejuvenation: { ...raw.lastRejuvenation } }
      : {}),
  };
  const policy = snapshot.policy.name === "precision_v1"
    ? PrecisionPolicy.fromInputs(inputs)
    : AD6Policy.fromInputs(inputs.ellStar, inputs.corrL);
  policy.reset(expectedK);
  policy.restore(snapshot.policy);
  return {
    state,
    rng: Rng.fromSnapshot(snapshot.rng),
    policy,
    remaining: new Set(snapshot.remaining),
    nPerTask: snapshot.nPerTask.slice(),
    cappedTasks: new Set(snapshot.cappedTasks),
    lastOutcomes: snapshot.lastOutcomes.slice(),
    lastTaskK: snapshot.lastTaskK,
    streakCount: snapshot.streakCount,
  };
}

/** Transfer ownership of every large numeric buffer without JSON conversion. */
export function snapshotTransferables(snapshot: SessionCoreSnapshot): Transferable[] {
  return [
    snapshot.state.t.buffer,
    snapshot.state.l.buffer,
    snapshot.state.w.buffer,
    snapshot.state.logPrior.buffer,
    snapshot.state.logLik.buffer,
    snapshot.remaining.buffer,
    snapshot.cappedTasks.buffer,
  ];
}
