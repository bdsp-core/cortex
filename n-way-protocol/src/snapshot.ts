import { assertReplayCompatible } from "./profile";
import type {
  Observation, PriorPair, ProfileStamp, ProtocolParticleState, RejuvenationTelemetry,
} from "./types";

export interface ProtocolStateSnapshot {
  schemaVersion: 1;
  profile: ProfileStamp;
  N: number;
  K: number;
  t: Float64Array;
  l: Float64Array;
  w: Float64Array;
  logPrior: Float64Array;
  logLik: Float64Array;
  history: Observation[];
  lastRejuvenation?: RejuvenationTelemetry;
  /** Per-particle artifact-atom lineage (draw-latent aggregation only). */
  atomIndex?: Int32Array;
}

export function snapshotProtocolState(
  state: ProtocolParticleState,
  profile: ProfileStamp,
): ProtocolStateSnapshot {
  return {
    schemaVersion: 1,
    profile: { ...profile },
    N: state.N,
    K: state.K,
    t: state.t.slice(),
    l: state.l.slice(),
    w: state.w.slice(),
    logPrior: state.logPrior.slice(),
    logLik: state.logLik.slice(),
    history: state.history.map((observation) => ({ ...observation })),
    ...(state.lastRejuvenation
      ? { lastRejuvenation: { ...state.lastRejuvenation } }
      : {}),
    ...(state.atomIndex ? { atomIndex: state.atomIndex.slice() } : {}),
  };
}

export function restoreProtocolState(
  snapshot: ProtocolStateSnapshot,
  offeredProfile: ProfileStamp,
  prior: PriorPair,
): ProtocolParticleState {
  if (snapshot.schemaVersion !== 1) throw new Error("unsupported protocol snapshot schema");
  assertReplayCompatible(snapshot.profile, offeredProfile);
  const expectedParameters = snapshot.N * snapshot.K;
  if (snapshot.t.length !== expectedParameters || snapshot.l.length !== expectedParameters
      || snapshot.w.length !== snapshot.N || snapshot.logPrior.length !== snapshot.N
      || snapshot.logLik.length !== snapshot.N || prior.tPieces.K !== snapshot.K
      || prior.lPieces.K !== snapshot.K) {
    throw new Error("invalid protocol snapshot dimensions");
  }
  if (snapshot.profile.responseAggregation === "draw_latent") {
    if (!snapshot.atomIndex || snapshot.atomIndex.length !== snapshot.N) {
      throw new Error("draw-latent snapshot is missing per-particle atom lineage");
    }
  } else if (snapshot.atomIndex) {
    throw new Error("mixture snapshot must not carry atom lineage");
  }
  return {
    N: snapshot.N,
    K: snapshot.K,
    t: snapshot.t.slice(),
    l: snapshot.l.slice(),
    w: snapshot.w.slice(),
    logPrior: snapshot.logPrior.slice(),
    logLik: snapshot.logLik.slice(),
    history: snapshot.history.map((observation) => ({ ...observation })),
    prior,
    ...(snapshot.lastRejuvenation
      ? { lastRejuvenation: { ...snapshot.lastRejuvenation } }
      : {}),
    ...(snapshot.atomIndex ? { atomIndex: snapshot.atomIndex.slice() } : {}),
  };
}

