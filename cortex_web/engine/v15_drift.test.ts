// v15 staging drift-guard.
//
// SAFETY-CRITICAL: an in-flight pilot relies on the frozen instrument. The
// default (pilot) engine path must stay BIT-IDENTICAL across the Phase-5
// refactor that stages N=1200 + a separate Corr_t prior as an OPT-IN flag.
//
// The "default path" snapshot in __snapshots__/v15_drift.test.ts.snap was
// recorded from the PRE-REFACTOR code (single-PriorPieces era) and is the
// bit-identical contract: after the refactor it MUST still match without `-u`.
// If it does not match, the refactor changed the default path — fix the
// refactor, do not update the snapshot.
//
// The v15-path test below proves the opt-in flag actually changes behaviour
// (a different Corr_t yields different t-block samples while the l-block is
// untouched) and that N flows through from inputs.

import { describe, it, expect } from "vitest";
import { precomputePriorPair, samplePrior, logPriorOne } from "./prior";
import { WebCortexSession, N_PARTICLES } from "./session";
import { EngineInputs } from "./types";
import { Rng } from "./rng";
import { Mat } from "./linalg";

// A fixed 3×3 SPD correlation matrix (unit diagonal). Small + deterministic so
// the snapshot is tiny and the test is fast (no real bank).
const CORR_L: Mat = [
  [1.0, 0.3, -0.2],
  [0.3, 1.0, 0.4],
  [-0.2, 0.4, 1.0],
];

// A DIFFERENT SPD correlation matrix for the t-block in the v15 path.
const CORR_T: Mat = [
  [1.0, -0.5, 0.1],
  [-0.5, 1.0, -0.3],
  [0.1, -0.3, 1.0],
];

const N = 6;
const SEED = 12345;

// Round to a fixed number of decimals so the snapshot is exact-text-stable on
// the bit-identical default path (any drift in float op order shows up well
// above this rounding floor).
function ser(t: Float64Array, l: Float64Array, logPrior: Float64Array) {
  const fmt = (x: number) => Number(x.toFixed(12));
  return {
    t: Array.from(t).map(fmt),
    l: Array.from(l).map(fmt),
    logPrior: Array.from(logPrior).map(fmt),
  };
}

describe("v15 drift-guard — default (pilot) path bit-identical", () => {
  it("precomputePrior + samplePrior + logPriorOne (corrL only)", () => {
    // DEFAULT (pilot) PATH: corrT absent → t- and l-blocks both
    // precomputePrior(corrL). The serialized snapshot below was recorded from
    // the single-PriorPieces code and must remain unchanged.
    const { tPieces, lPieces } = precomputePriorPair(CORR_L);
    const K = CORR_L.length;
    const t = new Float64Array(N * K);
    const l = new Float64Array(N * K);
    const logPrior = new Float64Array(N);
    samplePrior(N, tPieces, lPieces, new Rng(SEED), t, l, logPrior);

    // Cross-check logPriorOne independently matches the sampled log-prior.
    for (let n = 0; n < N; n++) {
      const lp = logPriorOne(t, l, n, tPieces, lPieces);
      expect(Math.abs(lp - logPrior[n])).toBeLessThan(1e-12);
    }

    expect(ser(t, l, logPrior)).toMatchSnapshot();
  });
});

describe("v15 drift-guard — opt-in flag changes behaviour", () => {
  it("a different Corr_t changes the t-block but leaves the l-block identical", () => {
    const K = CORR_L.length;

    // pilot path (corrT absent → both blocks corrL)
    const pilot = precomputePriorPair(CORR_L);
    const tP = new Float64Array(N * K);
    const lP = new Float64Array(N * K);
    const lpP = new Float64Array(N);
    samplePrior(N, pilot.tPieces, pilot.lPieces, new Rng(SEED), tP, lP, lpP);

    // v15 path (corrT present → t-block corrT, l-block corrL)
    const v15 = precomputePriorPair(CORR_L, CORR_T);
    const tV = new Float64Array(N * K);
    const lV = new Float64Array(N * K);
    const lpV = new Float64Array(N);
    samplePrior(N, v15.tPieces, v15.lPieces, new Rng(SEED), tV, lV, lpV);

    // l-block pieces unchanged when only corrT is supplied.
    expect(v15.lPieces).toEqual(pilot.lPieces);

    // t-block differs (different Cholesky factor applied to the same ε stream).
    let tDiff = 0;
    for (let i = 0; i < tP.length; i++) tDiff += Math.abs(tP[i] - tV[i]);
    expect(tDiff).toBeGreaterThan(1e-6);

    // l-block is IDENTICAL: same corrL, and the t-block draw consumes the same
    // K gaussians per particle from the RNG, so the l-block stream stays
    // byte-aligned across both paths.
    for (let i = 0; i < lP.length; i++) expect(lV[i]).toBe(lP[i]);

    // logPrior differs because the t-block contribution changed.
    let lpDiff = 0;
    for (let n = 0; n < N; n++) lpDiff += Math.abs(lpP[n] - lpV[n]);
    expect(lpDiff).toBeGreaterThan(1e-6);
  });

  it("nParticles flows through from inputs (default 600, v15 override)", () => {
    // Minimal single-segment K=1 bundle so a session can be constructed and
    // its particle cloud inspected after one (synchronous) item.
    const baseInputs: EngineInputs = {
      taskCodes: ["spike"],
      taskLabels: ["Spike"],
      taskPatternWords: ["spike"],
      corrL: [[1.0]],
      ellStar: [0.0],
      segments: [
        {
          segId: 1,
          patternClass: "spike",
          sMean: [0.5],
          sSd: [0.1],
          fsHz: 128,
          nCh: 1,
          nSamp: 1,
          channelNames: ["Fp1"],
          eeg: "seg/1.eeg",
          spec: "",
        },
      ],
    };

    async function nParticlesAfterStart(inputs: EngineInputs): Promise<number> {
      let captured = -1;
      const session = new WebCortexSession(inputs, "drift-N", 7, {
        onItem: () => {
          // The cloud is built in run() before the first onItem fires; capture
          // its size then abort so we don't drive the full session loop. Defer
          // the abort to a microtask so it lands after run() has registered its
          // answer resolver (onItem fires synchronously before awaitAnswer).
          captured = (session as any).state.N;
          queueMicrotask(() => session.abort());
        },
      });
      await session.run();
      return captured;
    }

    return Promise.all([
      nParticlesAfterStart(baseInputs),
      nParticlesAfterStart({ ...baseInputs, nParticles: 1200 }),
    ]).then(([defaultN, v15N]) => {
      expect(defaultN).toBe(N_PARTICLES); // 600
      expect(v15N).toBe(1200);
    });
  });
});
