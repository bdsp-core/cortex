import { IIIC_TASK_INDICES } from "./nway_likelihood";
import {
  NWAY_ARTIFACT, NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT, type ArtifactDraw,
} from "./nway_profile";
import type {
  BinaryParticleObservation, CategoricalParticleObservation, ParticleObservation,
  ParticleState,
} from "./types";
import {
  PosteriorUpdateError, logLikPackedHistory, packParticleHistory,
} from "./particles";

// Distractor-misspecification monitor. The fitted lapse sits on the zero
// boundary, so a population whose wrong picks stop following signal collapses
// the categorical posterior (skill coverage 0.26–0.47 even floored) while the
// binary reduction is untouched by construction. This monitor scores every
// wrong pick against the uniform alternative and fails closed to binary
// updates when the evidence trips.
//
// Fixed-input discipline: the statistic depends only on the asked class, the
// pick, the frozen segment response axes (raw sMean — the frame the artifact
// was fitted in, deliberately not the particle-dependent signalZ frame), and
// the frozen artifact draws. No session posterior, no RNG.
// Reference implementation and OC evidence: n-way-protocol
// python/nway_protocol/misspec_monitor.py, reports/misspec_monitor_oc_floor015.json.

const UNIFORM_LOG_PROBABILITY = -Math.log(IIIC_TASK_INDICES.length - 1);

// Calibrated on 2,000 simulated sessions x 90 wrong-picks over real bank axes
// at a 1% per-session false-trip target for the floor-0.15 ensemble; shorter
// sessions trip strictly less often under the null. Uniform-world detection:
// 100% of sessions, median 14 wrong picks.
export const DISTRACTOR_MONITOR_THRESHOLD = 5.213009866319716;

// Same OC harness re-run for the qualified nesting34 draw-latent artifact
// (n-way-protocol reports/misspec_monitor_oc_nesting34.json, Phase 1e):
// 1% false-trip target on the raw served-bank axes; uniform-world detection
// 100% of sessions, median 15 wrong picks.
export const NESTING34_MONITOR_THRESHOLD = 5.101352318244237;

/** The monitor reference mixture and trip threshold for a session's stamped
 * response artifact. The monitor stays fixed-input: the reference is the
 * artifact's own frozen draw table, never the session posterior. Unknown or
 * absent artifacts keep the deployed floor015 calibration — the previous
 * behavior for every existing session. */
export function monitorConfigFor(artifactId?: string): {
  draws: readonly ArtifactDraw[]; threshold: number;
} {
  if (artifactId === NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId) {
    return {
      draws: NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws,
      threshold: NESTING34_MONITOR_THRESHOLD,
    };
  }
  return { draws: NWAY_ARTIFACT.draws, threshold: DISTRACTOR_MONITOR_THRESHOLD };
}

export interface DistractorMonitorState {
  statistic: number;
  tripped: boolean;
  wrongPicks: number;
  trippedAt: number | null;
}

export function makeDistractorMonitor(): DistractorMonitorState {
  return { statistic: 0, tripped: false, wrongPicks: 0, trippedAt: null };
}

export function cloneDistractorMonitor(
  monitor: DistractorMonitorState,
): DistractorMonitorState {
  return { ...monitor };
}

/** Per-wrong-pick evidence for the deployed mixture against uniform
 * allocation. Positive under a signal-following world; bounded below by
 * log(robustnessFloor) because every deployed draw carries the lapse floor. */
export function monitorIncrement(
  askedK: number, pickK: number, sMean: readonly number[],
  draws: readonly ArtifactDraw[] = NWAY_ARTIFACT.draws,
): number {
  if (pickK === askedK || !IIIC_TASK_INDICES.includes(pickK)
      || !IIIC_TASK_INDICES.includes(askedK)) {
    throw new Error("the distractor monitor consumes IIIC wrong picks only");
  }
  const distractors = IIIC_TASK_INDICES.filter((k) => k !== askedK);
  let probability = 0;
  let weightSum = 0;
  for (const draw of draws) {
    let maxLogit = -Infinity;
    for (const k of distractors) {
      maxLogit = Math.max(maxLogit, draw.beta * sMean[k]);
    }
    let denominator = 0;
    let pickExp = 0;
    for (const k of distractors) {
      const value = Math.exp(draw.beta * sMean[k] - maxLogit);
      denominator += value;
      if (k === pickK) pickExp = value;
    }
    probability += draw.weight * (
      draw.distractorLapse / distractors.length
      + (1 - draw.distractorLapse) * (pickExp / denominator)
    );
    weightSum += draw.weight;
  }
  return Math.log(probability / weightSum) - UNIFORM_LOG_PROBABILITY;
}

/** Page's CUSUM against the model; the trip is one-way within a session. */
export function observeDistractorMonitor(
  monitor: DistractorMonitorState, evidence: number,
  threshold: number = DISTRACTOR_MONITOR_THRESHOLD,
): boolean {
  if (monitor.tripped) return true;
  monitor.wrongPicks += 1;
  monitor.statistic = Math.max(0, monitor.statistic - evidence);
  if (monitor.statistic > threshold) {
    monitor.tripped = true;
    monitor.trippedAt = monitor.wrongPicks;
  }
  return monitor.tripped;
}

/** The binary margin of a categorical observation: asked-class hit/miss with
 * the raw pick retained for audit. Exactly the reduction the shipped binary
 * engine would have applied, so it is always a valid refuge. */
export function binaryReduction(
  observation: CategoricalParticleObservation,
): BinaryParticleObservation {
  return {
    kind: "binary",
    k: observation.askedK,
    s: observation.sMean[observation.askedK],
    sSd: observation.sSd[observation.askedK],
    y: observation.pickK === observation.askedK ? 1 : 0,
    rawPick: observation.pickK,
  };
}

/** Fail-closed rebuild after a trip: every categorical observation in history
 * is suspect, not just future ones. Keep the particle positions (the current
 * weighted cloud targets the contaminated posterior), importance-reweight to
 * the binary-reduced history target, and swap the history so the next
 * ESS-triggered rejuvenation replays binary evidence only. Transactional:
 * state changes commit only after every particle validates. */
export function rebuildBinaryReducedHistory(st: ParticleState): void {
  const reduced: ParticleObservation[] = st.history.map((observation) =>
    observation.kind === "categorical_f1"
      ? binaryReduction(observation)
      : { ...observation });
  const packed = packParticleHistory(reduced, st.K);
  const reducedLogLik = new Float64Array(st.N);
  logLikPackedHistory(packed, st.N, st.K, st.t, st.l, reducedLogLik);
  let maxLogWeight = -Infinity;
  for (let n = 0; n < st.N; n++) {
    if (!Number.isFinite(reducedLogLik[n])) {
      throw new PosteriorUpdateError("binary-reduced history likelihood is non-finite");
    }
    if (st.w[n] > 0) {
      maxLogWeight = Math.max(
        maxLogWeight, Math.log(st.w[n]) + reducedLogLik[n] - st.logLik[n],
      );
    }
  }
  if (!Number.isFinite(maxLogWeight)) {
    throw new PosteriorUpdateError("binary-reduced reweight has zero mass");
  }
  const nextW = new Float64Array(st.N);
  let sumW = 0;
  for (let n = 0; n < st.N; n++) {
    const value = st.w[n] === 0
      ? 0
      : Math.exp(Math.log(st.w[n]) + reducedLogLik[n] - st.logLik[n] - maxLogWeight);
    nextW[n] = value;
    sumW += value;
  }
  if (!Number.isFinite(sumW) || sumW <= 0) {
    throw new PosteriorUpdateError("binary-reduced reweight has invalid mass");
  }
  for (let n = 0; n < st.N; n++) nextW[n] /= sumW;
  st.w = nextW;
  st.logLik = reducedLogLik;
  st.history = reduced;
  st.packedHistory = packed;
}
