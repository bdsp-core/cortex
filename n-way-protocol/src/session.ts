import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { makeObservation } from "./likelihood";
import {
  ess, resampleAndRejuvenateProtocol, updateProtocol,
} from "./particles";
import type {
  Candidate, ConditionalF1ResponseArtifact, EngineProfile, Observation,
  ProtocolParticleState, ProtocolSegment,
} from "./types";

export interface DirectEvidenceLedger {
  nPerTask: number[];
  bandAdministered: number[][];
  remainingSegmentIds: Set<number>;
}

export interface AdvanceProtocolParams {
  essThresholdFraction: number;
  nMhSteps: number;
  proposalScale: number;
  bandEdges: readonly (readonly [number, number])[];
}

export interface ProtocolTrialDiagnostic {
  trialIndex: number;
  askedK: number;
  segId: number;
  rawPick: number;
  responseKind: Observation["kind"];
  matchedAskedTask: boolean;
  ess: number;
  rejuvenated: boolean;
  nPerTask: number[];
  bandAdministered: number[][];
}

export function makeLedger(taskCount: number, segmentIds: readonly number[]): DirectEvidenceLedger {
  return {
    nPerTask: new Array<number>(taskCount).fill(0),
    bandAdministered: Array.from({ length: taskCount }, () => [0, 0, 0]),
    remainingSegmentIds: new Set(segmentIds),
  };
}

function bandIndex(signal: number, edges: readonly [number, number]): number {
  return signal < edges[0] ? 0 : signal < edges[1] ? 1 : 2;
}

export function advanceProtocol(args: {
  state: ProtocolParticleState;
  ledger: DirectEvidenceLedger;
  profile: EngineProfile;
  artifact: ConditionalF1ResponseArtifact | undefined;
  segments: readonly ProtocolSegment[];
  chosen: Candidate;
  rawPick: number;
  trialIndex: number;
  rng: Rng;
  params: AdvanceProtocolParams;
  precisionPolicy?: { recordAdministered: (taskK: number, signal: number) => void };
}): ProtocolTrialDiagnostic {
  if (!args.ledger.remainingSegmentIds.has(args.chosen.segId)) {
    throw new Error(`segment ${args.chosen.segId} is not available`);
  }
  const segment = args.segments[args.chosen.segmentIndex];
  if (!segment || segment.segId !== args.chosen.segId) throw new Error("chosen segment mismatch");
  const observation = makeObservation(
    args.profile, args.chosen.askedK, args.chosen.segmentIndex, args.rawPick,
  );
  updateProtocol(args.state, args.profile, args.artifact, args.segments, observation);
  let rejuvenated = false;
  if (ess(args.state.w) < args.params.essThresholdFraction * args.state.N) {
    resampleAndRejuvenateProtocol(
      args.state, args.profile, args.artifact, args.segments, args.rng,
      args.params.nMhSteps, args.params.proposalScale, args.trialIndex,
    );
    rejuvenated = true;
  }

  // Cross-domain likelihood evidence is deliberately not counted as a direct
  // administration or content-band exposure.
  args.ledger.remainingSegmentIds.delete(args.chosen.segId);
  args.ledger.nPerTask[args.chosen.askedK] += 1;
  args.precisionPolicy?.recordAdministered(
    args.chosen.askedK, args.chosen.focalSignal,
  );
  const edges = args.params.bandEdges[args.chosen.askedK];
  args.ledger.bandAdministered[args.chosen.askedK][
    bandIndex(args.chosen.focalSignal, edges)
  ] += 1;

  return {
    trialIndex: args.trialIndex,
    askedK: args.chosen.askedK,
    segId: args.chosen.segId,
    rawPick: args.rawPick,
    responseKind: observation.kind,
    matchedAskedTask: args.rawPick === args.chosen.askedK,
    ess: ess(args.state.w),
    rejuvenated,
    nPerTask: args.ledger.nPerTask.slice(),
    bandAdministered: args.ledger.bandAdministered.map((row) => row.slice()),
  };
}
