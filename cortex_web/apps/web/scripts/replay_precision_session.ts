// Read a sanitized production replay fixture from stdin and verify that the
// current browser engine reproduces it exactly against a local served-bank
// manifest. The fixture shape is intentionally generic and contains no
// account identifiers:
//   {sessionId, sampleSeed, exclusion, trials:[{pick,diag}], result}
//
// Usage:
//   fixture_command | vite-node scripts/replay_precision_session.ts manifest.json

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { seedFromSessionId, WebCortexSession } from "../engine/session";
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
}

interface ComparisonStats {
  meaningfulDifferences: string[];
  numericBitDifferences: number;
  maxAbsoluteDifference: number;
  maxAbsoluteDifferencePath: string | null;
  maxScaledDifference: number;
  maxScaledDifferencePath: string | null;
}

// Numeric values recorded by the production browser can differ from a local
// Node replay in their last floating-point bit. Such differences are immaterial
// only while they stay well below every guard/tie threshold and all structural
// and categorical outcomes remain exact.
const NUMERIC_ABSOLUTE_TOLERANCE = 1e-12;
const NUMERIC_RELATIVE_TOLERANCE = 1e-12;

function linearQuantile(values: number[], q: number): number {
  const ordered = values.slice().sort((a, b) => a - b);
  const position = (ordered.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.min(low + 1, ordered.length - 1);
  const fraction = position - low;
  return ordered[low] + fraction * (ordered[high] - ordered[low]);
}

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

function compareEquivalent(
  expected: unknown,
  actual: unknown,
  path: string,
  stats: ComparisonStats,
): void {
  if (Object.is(expected, actual)) return;
  if (typeof expected === "number" && typeof actual === "number") {
    stats.numericBitDifferences += 1;
    const absoluteDifference = Math.abs(expected - actual);
    if (absoluteDifference > stats.maxAbsoluteDifference) {
      stats.maxAbsoluteDifference = absoluteDifference;
      stats.maxAbsoluteDifferencePath = path;
    }
    const scale = Math.max(1, Math.abs(expected), Math.abs(actual));
    const scaledDifference = absoluteDifference / scale;
    if (scaledDifference > stats.maxScaledDifference) {
      stats.maxScaledDifference = scaledDifference;
      stats.maxScaledDifferencePath = path;
    }
    const allowedDifference = NUMERIC_ABSOLUTE_TOLERANCE
      + NUMERIC_RELATIVE_TOLERANCE * scale;
    if (!Number.isFinite(absoluteDifference) || absoluteDifference > allowedDifference) {
      stats.meaningfulDifferences.push(
        `${path}: ${expected} != ${actual} `
          + `(abs ${absoluteDifference}, scaled ${scaledDifference})`,
      );
    }
    return;
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    if (expected.length !== actual.length) {
      stats.meaningfulDifferences.push(
        `${path}.length: ${expected.length} != ${actual.length}`,
      );
      return;
    }
    for (let i = 0; i < expected.length; i++) {
      compareEquivalent(expected[i], actual[i], `${path}[${i}]`, stats);
    }
    return;
  }
  if (expected && actual && typeof expected === "object" && typeof actual === "object") {
    const expectedKeys = Object.keys(expected as Record<string, unknown>).sort();
    const actualKeys = Object.keys(actual as Record<string, unknown>).sort();
    compareEquivalent(expectedKeys, actualKeys, `${path}.keys`, stats);
    for (const key of expectedKeys) {
      compareEquivalent(
        (expected as Record<string, unknown>)[key],
        (actual as Record<string, unknown>)[key],
        `${path}.${key}`,
        stats,
      );
    }
    return;
  }
  stats.meaningfulDifferences.push(
    `${path}: ${JSON.stringify(expected)} != ${JSON.stringify(actual)}`,
  );
}

function trajectoryHash(result: Awaited<ReturnType<WebCortexSession["run"]>>): string {
  const hash = createHash("sha256");
  for (const values of [result.traj.t, result.traj.l, result.traj.w]) {
    hash.update(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
  }
  hash.update(JSON.stringify(result.traj.shape));
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const manifestPath = process.argv[2];
  if (!manifestPath) throw new Error("manifest path argument is required");
  let fixtureText = "";
  for await (const chunk of process.stdin) fixtureText += chunk;
  const fixture = JSON.parse(fixtureText) as ReplayFixture;
  const manifest = JSON.parse(readFileSync(resolve(manifestPath), "utf8"));
  // App.tsx starts the browser engine with this stable identity and leaves the
  // explicit engine seed undefined; worker.ts therefore hashes the identity.
  // The database session UUID and sampleSeed serve different purposes and are
  // not themselves the engine RNG seed.
  const engineSessionId = `web-${fixture.sampleSeed}`;
  const engineSeed = await seedFromSessionId(engineSessionId);
  const excluded = new Set(fixture.exclusion.map(Number));
  const inputs = {
    ...manifest,
    precisionBandEdges: bandEdges(manifest),
    terminationPolicy: "precision_v1",
    segments: manifest.segments.filter((segment: any) => !excluded.has(Number(segment.segId))),
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
    maxScaledDifference: 0,
    maxScaledDifferencePath: null,
  };
  let cursor = 0;
  let lastDiag: TrialDiag | undefined;
  const session = new WebCortexSession(
    inputs,
    engineSessionId,
    engineSeed,
    {
      onItem: ({ segId }) => {
        const expected = fixture.trials[cursor];
        if (!expected) {
          throw new Error(JSON.stringify({
            reason: "engine requested an item after the stored trajectory ended",
            cursor,
            segId,
            stats,
            lastDiag,
          }, null, 2));
        }
        compareEquivalent(expected.diag.segId, segId, `items[${cursor}]`, stats);
        queueMicrotask(() => session.submitAnswer(expected.pick));
      },
      onTrial: (diag) => {
        compareEquivalent(fixture.trials[cursor].diag, diag, `trials[${cursor}]`, stats);
        lastDiag = diag;
        cursor += 1;
      },
    },
    // Inline runs the same adopted branch without spending time on an answer
    // branch that is irrelevant to deterministic replay.
    { speculative: false },
  );
  const result = await session.run();

  const expected = fixture.result;
  for (const key of [
    "terminationPolicy", "domainStatuses", "determinations", "terminalReasons",
    "skillIntervals", "biasIntervals", "verdicts", "servedSegIds",
  ]) compareEquivalent(expected[key], (result as any)[key], `result.${key}`, stats);
  compareEquivalent(expected.trials, result.trials, "result.trials", stats);
  compareEquivalent(expected.perTask?.map((row: any) => row.auroc), result.finalAuroc,
    "result.finalAuroc", stats);
  compareEquivalent(expected.perTask?.map((row: any) => row.aurocHw), result.finalAurocHw,
    "result.finalAurocHw", stats);
  compareEquivalent(expected.perTask?.map((row: any) => row.ell), result.trials.at(-1)?.lMean,
    "result.lMean", stats);
  compareEquivalent(expected.perTask?.map((row: any) => row.theta), result.trials.at(-1)?.tMean,
    "result.tMean", stats);
  compareEquivalent(expected.trials?.length, result.nQuestions, "result.nQuestions", stats);
  if (fixture.stopReason !== undefined) {
    compareEquivalent(fixture.stopReason, result.stopReason, "result.stopReason", stats);
  }

  const summary = {
    meaningfullyEquivalent: stats.meaningfulDifferences.length === 0
      && cursor === fixture.trials.length,
    expectedQuestions: fixture.trials.length,
    replayedQuestions: cursor,
    exactSelectedItems: result.servedSegIds.every(
      (segId, index) => segId === fixture.trials[index]?.diag.segId,
    ),
    meaningfulDifferences: stats.meaningfulDifferences,
    numericBitDifferences: stats.numericBitDifferences,
    numericAbsoluteTolerance: NUMERIC_ABSOLUTE_TOLERANCE,
    numericRelativeTolerance: NUMERIC_RELATIVE_TOLERANCE,
    maxAbsoluteDifference: stats.maxAbsoluteDifference,
    maxAbsoluteDifferencePath: stats.maxAbsoluteDifferencePath,
    maxScaledDifference: stats.maxScaledDifference,
    maxScaledDifferencePath: stats.maxScaledDifferencePath,
    stopReason: result.stopReason,
    domainStatuses: result.domainStatuses,
    verdicts: result.verdicts,
    trajectorySha256: trajectoryHash(result),
  };
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  if (!summary.meaningfullyEquivalent) process.exitCode = 1;
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
