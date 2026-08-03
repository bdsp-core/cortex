// Read a sanitized production replay fixture from stdin and verify the
// sitting's MATH server-side: posterior evolution, stopping, and verdicts.
//
// FORCED-PATH SEMANTICS (2026-08-03). The original free-running replay
// assumed the node re-run reproduces the participant's browser bit-for-bit.
// That assumption is unsound across JS engines: transcendental functions
// (Math.exp/log/...) legally differ in the last bit between engines, the
// drift compounds through the particle filter, and the first flipped argmin
// sends item selection down a different path — every real sitting
// eventually crashed or diverged (zero passes in production). Instead we
// now FORCE the stored item sequence and picks through the engine
// (deferSelection=true skips the 35k-candidate scans entirely, which also
// cuts replay from tens of minutes to seconds) and verify what is
// verifiable across engines:
//   * discrete outcomes exactly - domain statuses, determinations,
//     terminal reasons, question count, stop reason;
//   * verdicts with a boundary allowance - a cut classification may flip
//     only when the replayed interval endpoint is within the drift
//     allowance of the cut, never by O(1) fabrication;
//   * continuous outputs (intervals, estimates) within an MC-drift
//     tolerance sized far below tamper scale;
//   * the stored item sequence's structural legality - segments must exist
//     in the manifest, respect the exclusion list, never repeat, and
//     reconcile with the per-domain counts and cap.
// A tampered client can fabricate verdicts under a valid stamp; it cannot
// make the forced re-derivation of its own answer stream agree with them.
//
// Fixture shape (unchanged):
//   {sessionId, sampleSeed, exclusion, trials:[{pick,diag}], result,
//    stopReason?, nwayProfile?, precisionRecalibration?}
//
// Usage:
//   fixture_command | node replay_precision_session.mjs manifest.json

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  advanceCore, chooseNext, precisionRemainingBank, SessionCore,
} from "../engine/advance";
import type { Chosen } from "../engine/choose_item";
import { nwayResponseRuntimeFor } from "../engine/nway_likelihood";
import { isNWaySession, validateNWayInputs } from "../engine/nway_profile";
import { makeState } from "../engine/particles";
import { PrecisionPolicy } from "../engine/precision_policy";
import { precomputePriorPair } from "../engine/prior";
import { Rng } from "../engine/rng";
import {
  ESS_THRESHOLD_FRAC, FIRST_ITEM_TOPN, MAX_CONSECUTIVE_SAME_DOMAIN,
  PRECISION_N_MH_STEPS, PRECISION_N_PARTICLES, PRECISION_N_SUBSAMPLE,
  seedFromSessionId,
} from "../engine/session";
import type { EngineInputs, TrialDiag } from "../engine/types";

interface ReplayFixture {
  sessionId: string;
  sampleSeed: number;
  exclusion: number[];
  trials: Array<{ pick: number; diag: TrialDiag }>;
  result: Record<string, any>;
  stopReason?: string;
  /** The session's server-stamped n-way profile. Absent = the pre-n-way
   *  binary precision flow, exactly as before. */
  nwayProfile?: Record<string, any>;
  /** The session's server-stamped stopping recalibration ("c1" = n_min 0 /
   *  persistence 3). Absent = the shipped 20/2 configuration. */
  precisionRecalibration?: "c1";
  /** The sitting's execution placement. dual_branch_auto (every production
   *  precision sitting, and the default when absent) updates WITHOUT the
   *  distractor monitor — advanceCoreWithSelectionExecutor never consults
   *  it; only the serial path does. */
  computeMode?: string;
}

interface ComparisonStats {
  meaningfulDifferences: string[];
  numericBitDifferences: number;
  maxAbsoluteDifference: number;
  maxAbsoluteDifferencePath: string | null;
  firstDrifts: Array<{ path: string; difference: number }>;
}

// Cross-engine forced-path drift: last-bit transcendental differences,
// amplified by occasional discrete resampling/rejuvenation flips, move the
// replayed posterior by up to ~MC noise (SD/sqrt(ESS) ~ 0.04). Fabricating
// an outcome moves intervals by O(1). The tolerance sits far above the
// former and far below the latter; anything between is flagged for review.
const FORCED_REPLAY_TOLERANCE = 0.15;
// A verdict flip is drift-explainable only when the replayed interval
// endpoint sits this close to the certification cut.
const VERDICT_BOUNDARY_TOLERANCE = 0.05;
// Post-update ESS agreement required between the stored and replayed clouds
// (absolute, particles). Drift moves ESS by well under one particle until a
// same-schedule resample; a forged schedule departs by hundreds.
const ESS_DRIFT_ALLOWANCE = 60;
// Bit-level accounting threshold (diagnostic only, never gates).
const NUMERIC_ABSOLUTE_TOLERANCE = 1e-12;

function linearQuantile(values: number[], q: number): number {
  const ordered = values.slice().sort((a, b) => a - b);
  const position = (ordered.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.min(low + 1, ordered.length - 1);
  const fraction = position - low;
  return ordered[low] + fraction * (ordered[high] - ordered[low]);
}

// The frozen band edges are terciles over the COMPLETE served manifest —
// exactly session_bank.py's derivation — never over the exclusion-filtered
// per-session pool (the engine's own fallback would use the latter).
function bandEdges(manifest: any): number[][] {
  if (manifest.precisionBandEdges) return manifest.precisionBandEdges;
  return manifest.taskCodes.map((_: string, k: number) => {
    const values = manifest.segments
      .filter((segment: any) => (
        !segment.applicableTaskIdx || segment.applicableTaskIdx.includes(k)
      ))
      .map((segment: any) => Number(segment.sMean[k]));
    return [linearQuantile(values, 1 / 3), linearQuantile(values, 2 / 3)];
  });
}

function recordNumericDrift(
  expected: unknown, actual: unknown, path: string, stats: ComparisonStats,
): void {
  if (typeof expected === "number" && typeof actual === "number") {
    const difference = Math.abs(expected - actual);
    if (difference > 0) {
      stats.numericBitDifferences += 1;
      if (stats.firstDrifts.length < 10) {
        stats.firstDrifts.push({ path, difference });
      }
      if (difference > stats.maxAbsoluteDifference) {
        stats.maxAbsoluteDifference = difference;
        stats.maxAbsoluteDifferencePath = path;
      }
    }
    return;
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    for (let i = 0; i < Math.min(expected.length, actual.length); i++) {
      recordNumericDrift(expected[i], actual[i], `${path}[${i}]`, stats);
    }
    return;
  }
  if (expected && actual
      && typeof expected === "object" && typeof actual === "object") {
    for (const key of Object.keys(expected as Record<string, unknown>)) {
      recordNumericDrift(
        (expected as Record<string, unknown>)[key],
        (actual as Record<string, unknown>)[key],
        `${path}.${key}`, stats,
      );
    }
  }
}

function within(
  expected: number[] | number[][], actual: number[] | number[][],
  tolerance: number,
): boolean {
  const flatExpected = (expected as any[]).flat(2) as number[];
  const flatActual = (actual as any[]).flat(2) as number[];
  if (flatExpected.length !== flatActual.length) return false;
  return flatExpected.every(
    (value, i) => Math.abs(value - flatActual[i]) <= tolerance,
  );
}

function classify(interval: [number, number], cut: number): string {
  if (interval[0] > cut) return "ABOVE_CUT";
  if (interval[1] < cut) return "BELOW_CUT";
  return "INDETERMINATE_AT_CUT";
}

function nearBoundary(interval: [number, number], cut: number): boolean {
  return Math.abs(interval[0] - cut) <= VERDICT_BOUNDARY_TOLERANCE
    || Math.abs(interval[1] - cut) <= VERDICT_BOUNDARY_TOLERANCE;
}

function trajectoryHash(trials: TrialDiag[]): string {
  const hash = createHash("sha256");
  hash.update(JSON.stringify(trials.map((trial) => [
    trial.segId, trial.taskK, trial.pick,
  ])));
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const manifestPath = process.argv[2];
  if (!manifestPath) throw new Error("manifest path argument is required");
  let fixtureText = "";
  for await (const chunk of process.stdin) fixtureText += chunk;
  const fixture = JSON.parse(fixtureText) as ReplayFixture;
  const manifest = JSON.parse(readFileSync(resolve(manifestPath), "utf8"));
  const excluded = new Set(fixture.exclusion.map(Number));
  const inputs = {
    ...manifest,
    precisionBandEdges: bandEdges(manifest),
    terminationPolicy: "precision_v1",
    segments: manifest.segments.filter(
      (segment: any) => !excluded.has(Number(segment.segId))),
    // n-way sessions replay under their stored stamp (validated by the same
    // validateNWayInputs gate the browser ran); a fixture without a stamp is
    // the pre-n-way binary flow.
    ...(fixture.nwayProfile ? { nwayProfile: fixture.nwayProfile } : {}),
    // A c1 sitting replays under its stored recalibration stamp; a fixture
    // without one replays the shipped 20/2 stopping configuration.
    ...(fixture.precisionRecalibration
      ? { precisionRecalibration: fixture.precisionRecalibration } : {}),
  } as EngineInputs;

  const stats: ComparisonStats = {
    meaningfulDifferences: [],
    numericBitDifferences: 0,
    maxAbsoluteDifference: 0,
    maxAbsoluteDifferencePath: null,
    firstDrifts: [],
  };
  const meaningful = (message: string) => {
    stats.meaningfulDifferences.push(message);
  };

  const K = inputs.taskCodes.length;
  const nway = isNWaySession(inputs);
  if (nway) validateNWayInputs(inputs);
  const engineSessionId = `web-${fixture.sampleSeed}`;
  const engineSeed = await seedFromSessionId(engineSessionId);
  const rng = new Rng(engineSeed);
  const prior = precomputePriorPair(inputs.corrL, inputs.corrT);
  const responseRuntime = nway ? nwayResponseRuntimeFor(inputs) : undefined;
  const state = makeState(PRECISION_N_PARTICLES, K, prior, rng, responseRuntime);
  const policy = PrecisionPolicy.fromInputs(inputs);
  policy.reset(K);
  const spikeIdx = inputs.taskClasses
    ? inputs.taskClasses.findIndex((c) => c === "spike") : -1;
  const core: SessionCore = {
    state,
    rng,
    policy,
    remaining: new Set(inputs.segments.map((s) => s.segId)),
    nPerTask: new Array(K).fill(0),
    cappedTasks: new Set<number>(),
    lastOutcomes: new Array(K).fill(policy.activeLabel),
    lastTaskK: -1,
    streakCount: 0,
  };
  const params = {
    K,
    nParticles: PRECISION_N_PARTICLES,
    perDomainCap: policy.perDomainCap,
    nMhSteps: PRECISION_N_MH_STEPS,
    essThresholdFrac: ESS_THRESHOLD_FRAC,
    proposalScale: 2.38 / Math.sqrt(2 * K),
    firstItemTopN: FIRST_ITEM_TOPN,
    maxConsecutiveSameDomain: MAX_CONSECUTIVE_SAME_DOMAIN,
    k7Spike: spikeIdx >= 0,
    spikeIdx,
    nSubsample: PRECISION_N_SUBSAMPLE,
    uncertaintyAwareSubsample: true,
  };

  const monitorInactive = (fixture.computeMode ?? "dual_branch_auto")
    === "dual_branch_auto";

  // The live session's ONLY selection-side RNG consumption is the trial-0
  // top-N opener. Run the real selection once (its item choice is discarded
  // in favor of the stored sequence) so the shared RNG stream sits at the
  // exact position the browser's did for every subsequent rejuvenation draw.
  chooseNext(core, inputs, params, 0);

  // Structural legality of the stored sequence, then the forced re-derivation.
  const servedSegIds = new Set<number>();
  const replayedTrials: TrialDiag[] = [];
  let earlyStopAt: number | null = null;
  let lastDone = false;
  let lastStopReason = "";
  for (let index = 0; index < fixture.trials.length; index++) {
    const stored = fixture.trials[index];
    const segId = Number(stored.diag.segId);
    const taskK = Number(stored.diag.taskK);
    const segment = (inputs.segments as any[]).find(
      (candidate) => candidate.segId === segId);
    if (!segment) {
      meaningful(`trials[${index}]: stored segment ${segId} is not in the `
        + "exclusion-filtered manifest");
      break;
    }
    if (servedSegIds.has(segId)) {
      meaningful(`trials[${index}]: stored segment ${segId} repeats`);
      break;
    }
    servedSegIds.add(segId);
    if (taskK < 0 || taskK >= K) {
      meaningful(`trials[${index}]: stored task ${taskK} out of range`);
      break;
    }
    if (core.nPerTask[taskK] >= policy.perDomainCap) {
      meaningful(`trials[${index}]: domain ${taskK} exceeds the per-domain cap`);
      break;
    }
    const chosen: Chosen = {
      k: taskK,
      s: segment.sMean[taskK],
      sSd: segment.sSd[taskK],
      segId,
      loss: 0,
      segment,
    };
    const result = advanceCore(
      core, inputs, chosen, stored.pick, params, index,
      precisionRemainingBank(inputs, core.remaining, segId),
      /* deferSelection */ true,
      {
        // Replay the RECORDED rejuvenation schedule: a near-threshold ESS
        // can legally sit on either side across JS engines, and a single
        // flipped trigger forks the whole stochastic path. The schedule
        // itself is integrity-checked against the replayed ESS below.
        rejuvenationOverride: Boolean(stored.diag.rejuv),
        // Mirror the sitting's execution placement: dual_branch_auto (all
        // production precision sittings; the default) never consults the
        // distractor monitor.
        distractorMonitorInactive: monitorInactive,
      },
    );
    replayedTrials.push(result.diag);
    recordNumericDrift(stored.diag, result.diag, `trials[${index}]`, stats);
    if (process.env.REPLAY_DEBUG) {
      const monitor = (core as any).distractorMonitor;
      process.stderr.write(`trial ${index} k=${taskK} rejuv=${stored.diag.rejuv} `
        + `dEss=${(result.diag.ess - stored.diag.ess).toExponential(2)} `
        + `monitor=${monitor ? JSON.stringify(monitor) : "none"} `
        + `storedMonitor=${JSON.stringify(stored.diag.distractorMonitor ?? null)}\n`);
    }
    // A fabricated schedule cannot hide: rejuvenation is only accepted where
    // the replayed post-update ESS is drift-consistent with the trigger.
    const essFloor = ESS_THRESHOLD_FRAC * PRECISION_N_PARTICLES;
    if (Math.abs(stored.diag.ess - result.diag.ess) > ESS_DRIFT_ALLOWANCE) {
      meaningful(`trials[${index}].ess: ${stored.diag.ess} departs replayed `
        + `${result.diag.ess} beyond the +/-${ESS_DRIFT_ALLOWANCE} allowance`);
      break;
    }
    // With no rejuvenation this trial, post-update ESS IS the trigger input;
    // a stored schedule that skipped a clearly-required rejuvenation is
    // inconsistent with the engine (the converse — a spurious extra
    // rejuvenation — is caught by the |dESS| gate above and the final
    // outcome gates).
    if (!stored.diag.rejuv
        && result.diag.ess < essFloor - ESS_DRIFT_ALLOWANCE) {
      meaningful(`trials[${index}]: stored schedule skips a rejuvenation the `
        + "replayed ESS requires");
      break;
    }
    lastDone = result.done;
    lastStopReason = result.stopReason;
    if (result.done && index < fixture.trials.length - 1 && earlyStopAt === null) {
      earlyStopAt = index;
    }
  }

  if (replayedTrials.length !== fixture.trials.length) {
    meaningful(`replayed ${replayedTrials.length} of ${fixture.trials.length} `
      + "stored trials");
  }
  if (earlyStopAt !== null) {
    meaningful(`policy declared the session complete at trial ${earlyStopAt} `
      + `but ${fixture.trials.length} trials were served`);
  }
  if (!lastDone && fixture.stopReason === "all_estimated_or_undeterminable") {
    meaningful("stored session stopped on policy completion but the replayed "
      + "policy is still active after the full answer stream");
  }
  if (fixture.stopReason !== undefined && lastDone
      && fixture.stopReason !== lastStopReason) {
    meaningful(`stopReason: ${JSON.stringify(fixture.stopReason)} != `
      + JSON.stringify(lastStopReason));
  }

  const finalized = policy.finalizeResult(inputs.ellStar);
  const expected = fixture.result;
  for (const key of ["domainStatuses", "determinations", "terminalReasons"]) {
    const expectedValue = JSON.stringify(expected[key] ?? null);
    const actualValue = JSON.stringify((finalized as any)[key] ?? null);
    if (expectedValue !== actualValue) {
      meaningful(`result.${key}: ${expectedValue} != ${actualValue}`);
    }
  }
  if (Number(expected.trials?.length ?? expected.nQuestions)
      !== replayedTrials.length) {
    meaningful("result.nQuestions: "
      + `${expected.trials?.length ?? expected.nQuestions} != `
      + `${replayedTrials.length}`);
  }

  // Continuous outputs within drift tolerance.
  for (const [key, replayedValue] of [
    ["skillIntervals", finalized.skillIntervals],
    ["biasIntervals", finalized.biasIntervals],
  ] as const) {
    const submitted = expected[key];
    if (!submitted || !replayedValue) {
      if (Boolean(submitted) !== Boolean(replayedValue)) {
        meaningful(`result.${key}: presence mismatch`);
      }
      continue;
    }
    if (!within(submitted, replayedValue, FORCED_REPLAY_TOLERANCE)) {
      meaningful(`result.${key}: outside the +/-${FORCED_REPLAY_TOLERANCE} `
        + "forced-replay tolerance");
    }
  }

  // Verdicts: exact, except drift-explainable boundary flips.
  const submittedVerdicts: string[] = expected.verdicts ?? [];
  const replayedVerdicts = finalized.verdicts;
  for (let k = 0; k < K; k++) {
    const submitted = submittedVerdicts[k];
    const replayed = replayedVerdicts[k];
    if (submitted === replayed) continue;
    const interval = finalized.skillIntervals?.[k];
    const cutClassifications = new Set(["ABOVE_CUT", "BELOW_CUT",
      "INDETERMINATE_AT_CUT"]);
    const bothCutClassifications = cutClassifications.has(submitted)
      && cutClassifications.has(replayed);
    if (bothCutClassifications && interval
        && classify(interval, inputs.ellStar[k]) === replayed
        && nearBoundary(interval, inputs.ellStar[k])) {
      continue; // drift-explainable boundary flip
    }
    meaningful(`result.verdicts[${k}]: ${JSON.stringify(submitted)} != `
      + JSON.stringify(replayed));
  }

  const summary = {
    replayMode: "forced_path",
    meaningfullyEquivalent: stats.meaningfulDifferences.length === 0,
    expectedQuestions: fixture.trials.length,
    replayedQuestions: replayedTrials.length,
    // Under forcing this reports the stored sequence's structural legality
    // (existence, exclusion, uniqueness, cap) rather than free-running
    // selection agreement, which is unverifiable across JS engines.
    exactSelectedItems: replayedTrials.length === fixture.trials.length,
    meaningfulDifferences: stats.meaningfulDifferences,
    numericBitDifferences: stats.numericBitDifferences,
    numericAbsoluteTolerance: NUMERIC_ABSOLUTE_TOLERANCE,
    forcedReplayTolerance: FORCED_REPLAY_TOLERANCE,
    verdictBoundaryTolerance: VERDICT_BOUNDARY_TOLERANCE,
    maxAbsoluteDifference: stats.maxAbsoluteDifference,
    maxAbsoluteDifferencePath: stats.maxAbsoluteDifferencePath,
    firstDrifts: stats.firstDrifts,
    stopReason: lastStopReason,
    domainStatuses: finalized.domainStatuses,
    verdicts: finalized.verdicts,
    trajectorySha256: trajectoryHash(replayedTrials),
  };
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  if (!summary.meaningfullyEquivalent) process.exitCode = 1;
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
