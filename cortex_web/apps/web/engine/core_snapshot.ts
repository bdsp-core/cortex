import type { SessionCore } from "./advance";
import {
  type DistractorMonitorState, cloneDistractorMonitor,
} from "./misspec_monitor";
import { nwayResponseRuntimeFor } from "./nway_likelihood";
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
  history: ParticleState["history"];
  lastRejuvenation?: RejuvenationTelemetry;
  /** Per-particle artifact-atom lineage (draw-latent sessions only). */
  atomIndex?: Int32Array;
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
  // Optional so pre-monitor snapshots restore unchanged.
  distractorMonitor?: DistractorMonitorState;
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
      history: state.history.map((x) => x.kind === "categorical_f1"
        ? { ...x, sMean: x.sMean.slice(), sSd: x.sSd.slice() }
        : { ...x }),
      ...(state.lastRejuvenation
        ? { lastRejuvenation: { ...state.lastRejuvenation } }
        : {}),
      ...(state.atomIndex ? { atomIndex: state.atomIndex.slice() } : {}),
    },
    rng: core.rng.snapshot(),
    policy: core.policy.snapshot(),
    remaining: Int32Array.from(core.remaining),
    nPerTask: core.nPerTask.slice(),
    cappedTasks: Int32Array.from(core.cappedTasks),
    lastOutcomes: core.lastOutcomes.slice(),
    lastTaskK: core.lastTaskK,
    streakCount: core.streakCount,
    ...(core.distractorMonitor
      ? { distractorMonitor: cloneDistractorMonitor(core.distractorMonitor) }
      : {}),
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
  // Fail-closed cross-mode validation: a draw-latent session must never
  // restore a snapshot that lost its atom lineage, and a mixture session
  // must never adopt one that carries lineage from another mode.
  const responseRuntime = nwayResponseRuntimeFor(inputs);
  if (responseRuntime) {
    if (!raw.atomIndex || raw.atomIndex.length !== raw.N) {
      throw new Error(
        "draw-latent session-core snapshot is missing per-particle atom lineage",
      );
    }
    for (let n = 0; n < raw.N; n++) {
      if (raw.atomIndex[n] < 0 || raw.atomIndex[n] >= responseRuntime.atoms.length) {
        throw new Error("snapshot atom lineage is outside the artifact atoms");
      }
    }
  } else if (raw.atomIndex) {
    throw new Error("mixture session-core snapshot must not carry atom lineage");
  }
  const state: ParticleState = {
    N: raw.N,
    K: raw.K,
    t: raw.t,
    l: raw.l,
    w: raw.w,
    logPrior: raw.logPrior,
    logLik: raw.logLik,
    history: raw.history.map((x) => x.kind === "categorical_f1"
      ? { ...x, sMean: x.sMean.slice(), sSd: x.sSd.slice() }
      : { ...x }),
    prior,
    ...(raw.lastRejuvenation
      ? { lastRejuvenation: { ...raw.lastRejuvenation } }
      : {}),
    ...(raw.atomIndex ? { atomIndex: raw.atomIndex } : {}),
    ...(responseRuntime ? { responseRuntime } : {}),
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
    ...(snapshot.distractorMonitor
      ? { distractorMonitor: cloneDistractorMonitor(snapshot.distractorMonitor) }
      : {}),
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
    ...(snapshot.state.atomIndex ? [snapshot.state.atomIndex.buffer] : []),
    snapshot.remaining.buffer,
    snapshot.cappedTasks.buffer,
  ];
}
