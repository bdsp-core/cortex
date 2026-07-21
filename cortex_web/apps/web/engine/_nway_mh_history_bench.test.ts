import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { makeResponseObservation } from "./nway_likelihood";
import { makeState, resampleAndRejuvenate } from "./particles";
import { precomputePriorPair } from "./prior";
import { Rng } from "./rng";
import type {
  ComputeEngineInputs, ComputeSegmentMeta, ParticlePhaseTimingV2,
} from "./types";

const manifestPath = process.env.CORTEX_PROD_MANIFEST;
const fixturePath = process.env.CORTEX_NATIVE_REPLAY_FIXTURE;
const enabled = !!manifestPath && !!fixturePath && process.env.CORTEX_MH_BENCH === "1";

function emptyTiming(): ParticlePhaseTimingV2 {
  return {
    categoricalUpdateMs: 0,
    essMs: 0,
    resamplingMs: 0,
    mhProposalGenerationMs: 0,
    mhPriorMs: 0,
    mhHistoryLikelihoodMs: 0,
    mhAcceptanceMs: 0,
    mhCopyingMs: 0,
  };
}

function stateHash(state: ReturnType<typeof makeState>, rng: Rng): string {
  const hash = createHash("sha256");
  for (const values of [
    state.t, state.l, state.w, state.logPrior, state.logLik,
  ]) {
    hash.update(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
  }
  const snapshot = rng.snapshot();
  hash.update(JSON.stringify({
    s0: snapshot.s0.toString(), s1: snapshot.s1.toString(),
    s2: snapshot.s2.toString(), s3: snapshot.s3.toString(),
    gaussSpare: snapshot.gaussSpare,
    lastRejuvenation: state.lastRejuvenation,
  }));
  return hash.digest("hex");
}

describe.runIf(enabled)("native 120-observation MH history benchmark", () => {
  it("replays the complete categorical history with the production MH profile", () => {
    const manifest = JSON.parse(readFileSync(manifestPath!, "utf8")) as ComputeEngineInputs;
    const fixture = JSON.parse(readFileSync(fixturePath!, "utf8")) as {
      trials: { segId: number; taskK: number; rawPick: number }[];
    };
    const segments = new Map<number, ComputeSegmentMeta>(
      manifest.segments.map((segment) => [segment.segId, segment]),
    );
    const observations = fixture.trials.filter((trial) => trial.taskK !== 0).map((trial) => {
      const segment = segments.get(trial.segId);
      if (!segment) throw new Error(`fixture segment ${trial.segId} is unavailable`);
      return makeResponseObservation(trial.taskK, segment, trial.rawPick, "iiic");
    });
    expect(observations).toHaveLength(120);
    const repeats = Math.max(1, Number(process.env.CORTEX_MH_BENCH_REPEATS || 1));
    const results = [];
    for (let repeat = 0; repeat < repeats; repeat++) {
      const rng = new Rng(20260720);
      const state = makeState(
        1200, manifest.taskCodes.length,
        precomputePriorPair(manifest.corrL, manifest.corrT), rng,
      );
      state.history = observations.map((observation) => observation.kind === "categorical_f1"
        ? { ...observation, sMean: observation.sMean.slice(), sSd: observation.sSd.slice() }
        : { ...observation });
      const timing = emptyTiming();
      const startedAt = performance.now();
      const acceptanceRate = resampleAndRejuvenate(
        state, rng, 30, 2.38 / Math.sqrt(2 * state.K), 120, timing,
      );
      const durationMs = performance.now() - startedAt;
      results.push({
        repeat,
        durationMs,
        acceptanceRate,
        stateHash: stateHash(state, rng),
        timing,
      });
    }
    expect(new Set(results.map((result) => result.stateHash)).size).toBe(1);
    console.info(`NWAY_MH_HISTORY_BENCH ${JSON.stringify({
      observations: observations.length,
      nParticles: 1200,
      nMhSteps: 30,
      results,
    })}`);
  }, 180_000);
});
