// Bit-identity proof for speculative precompute (engine/advance.ts + session.ts).
//
// A SPECULATIVE session MUST produce byte-for-byte the same result as the INLINE
// session on the same (seed, answers): identical item sequence, verdicts,
// per-trial diagnostics (every field), and the full particle-cloud trajectory.
// Speculation only changes WHEN the engine computes the next item (idle
// think-time vs after the answer), never WHAT it computes.

import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";
import { Rng } from "./rng";

const WORDS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"];
const CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];
const CLASSES = ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"] as ("spike" | "iiic")[];

// K=7 bank: spike (task 0) + 6 IIIC (tasks 1..6), `perClass` segs each, signals
// informative on the true task. corrL + corrT supplied → exercises the v15
// separate-t-block prior path too.
function makeBank(perClass: number, nParticles: number, perDomainCap: number): EngineInputs {
  const rng = new Rng(2024);
  const segments: EngineInputs["segments"] = [];
  let id = 0;
  for (let k = 0; k < 7; k++) {
    const applicable = CLASSES[k] === "spike" ? [0] : [1, 2, 3, 4, 5, 6];
    for (let i = 0; i < perClass; i++) {
      const sMean = new Array(7).fill(0);
      const sSd = new Array(7).fill(0);
      for (const j of applicable) {
        sMean[j] = 0.2 + 1.6 * (rng.int(1000) / 1000);
        sSd[j] = 0.25 + 0.1 * (rng.int(100) / 100);
      }
      segments.push({
        segId: id++, patternClass: WORDS[k], testClass: CLASSES[k],
        applicableTaskIdx: applicable, sMean, sSd,
        fsHz: 200, nCh: 1, nSamp: 1, channelNames: [], eeg: "", spec: "",
      });
    }
  }
  const mat = (off: number) =>
    Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => (i === j ? 1 : off)));
  return {
    taskCodes: CODES, taskLabels: CODES, taskPatternWords: WORDS, taskClasses: CLASSES,
    corrL: mat(0.15), corrT: mat(-0.1), nParticles, perDomainCap,
    ellStar: new Array(7).fill(0.8), segments,
  };
}

// The real frozen-pilot bank, if it's been built locally (gitignored) — a
// gold-standard bit-identity check on real signals + the real Corr_l.
function loadRealInputs(): EngineInputs | null {
  const p = fileURLToPath(new URL("../public/bundle/v1.1-local/manifest.json", import.meta.url));
  if (!existsSync(p)) return null;
  const m = JSON.parse(readFileSync(p, "utf8"));
  return {
    taskCodes: m.taskCodes, taskLabels: m.taskLabels, taskPatternWords: m.taskPatternWords,
    taskClasses: m.taskClasses, corrL: m.corrL, corrT: m.corrT,
    nParticles: m.nParticles, ellStar: m.ellStar, segments: m.segments,
  };
}

interface Capture {
  items: { trialIndex: number; taskK: number; segId: number }[];
  trials: unknown[];
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  servedSegIds: number[];
  finalAuroc: number[];
  finalAurocHw: number[];
  traj: { t: number[]; l: number[]; w: number[]; shape: number[] };
}

async function runSession(inputs: EngineInputs, seed: number, speculative: boolean): Promise<Capture> {
  const K = inputs.taskCodes.length;
  // Item-INDEPENDENT answers (one draw per trial, same seed) → both modes see
  // the identical answer at each trial index; any engine divergence shows up in
  // the captured comparison rather than via the answer feedback loop.
  const ansRng = new Rng((seed ^ 0xa5a5a5) >>> 0);
  const items: Capture["items"] = [];
  const trials: unknown[] = [];
  const session = new WebCortexSession(inputs, `bit-${seed}`, seed, {
    onItem: (it) => {
      items.push({ ...it });
      queueMicrotask(() => session.submitAnswer(ansRng.int(K)));
    },
    onTrial: (d) => trials.push(d),
  }, { speculative });
  const r = await session.run();
  return {
    items, trials,
    nQuestions: r.nQuestions, stopReason: r.stopReason, verdicts: r.verdicts,
    servedSegIds: r.servedSegIds, finalAuroc: r.finalAuroc, finalAurocHw: r.finalAurocHw,
    traj: {
      t: Array.from(r.traj.t), l: Array.from(r.traj.l), w: Array.from(r.traj.w), shape: r.traj.shape,
    },
  };
}

// Exact float mismatch count (0 = bit-identical).
function mismatches(a: number[], b: number[]): number {
  if (a.length !== b.length) return -1;
  let m = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) m++;
  return m;
}

describe("speculative precompute is bit-identical to inline", () => {
  const cases = [
    // Fast always-on cases — bit-identity is N-independent, so these prove it in
    // CI (normal resolution + the forced per-domain-cap path).
    { name: "N=300 normal", perClass: 10, N: 300, cap: 6, seeds: [1, 2], bench: false },
    { name: "N=300 forced caps", perClass: 12, N: 300, cap: 4, seeds: [3], bench: false },
    // Production particle count — slower; run with CORTEX_BENCH=1.
    { name: "N=1200 prod count", perClass: 10, N: 1200, cap: 4, seeds: [7], bench: true },
  ];
  for (const c of cases) {
    for (const seed of c.seeds) {
      const t = c.bench ? it.runIf(!!process.env.CORTEX_BENCH) : it;
      t(`${c.name} — seed ${seed}`, async () => {
        const inputs = makeBank(c.perClass, c.N, c.cap);
        const ref = await runSession(inputs, seed, false);  // inline
        const spec = await runSession(inputs, seed, true);  // speculative

        // a real adaptive session ran (not a degenerate 0/1-trial case)
        expect(ref.nQuestions).toBeGreaterThan(5);

        // EVERYTHING identical
        expect(spec.items).toEqual(ref.items);                 // item sequence
        expect(spec.nQuestions).toBe(ref.nQuestions);
        expect(spec.stopReason).toBe(ref.stopReason);
        expect(spec.verdicts).toEqual(ref.verdicts);
        expect(spec.servedSegIds).toEqual(ref.servedSegIds);
        expect(spec.trials).toEqual(ref.trials);               // every diag field, exact
        expect(spec.finalAuroc).toEqual(ref.finalAuroc);
        expect(spec.finalAurocHw).toEqual(ref.finalAurocHw);
        // full particle-cloud trajectory, exact (the strongest bit-identity check)
        expect(spec.traj.shape).toEqual(ref.traj.shape);
        expect(mismatches(spec.traj.t, ref.traj.t)).toBe(0);
        expect(mismatches(spec.traj.l, ref.traj.l)).toBe(0);
        expect(mismatches(spec.traj.w, ref.traj.w)).toBe(0);
      }, 180000);
    }
  }

  // Gold-standard: bit-identity on the REAL frozen-pilot bank (real signals +
  // Corr_l). Local + CORTEX_BENCH only (the bundle is gitignored / slow).
  it.runIf(!!process.env.CORTEX_BENCH && !!loadRealInputs())(
    "real frozen-pilot bank — bit-identical on gold-standard signals", async () => {
      const inputs = loadRealInputs()!;
      const ref = await runSession(inputs, 11, false);
      const spec = await runSession(inputs, 11, true);
      expect(ref.nQuestions).toBeGreaterThan(20);
      expect(spec.items).toEqual(ref.items);
      expect(spec.verdicts).toEqual(ref.verdicts);
      expect(spec.servedSegIds).toEqual(ref.servedSegIds);
      expect(spec.trials).toEqual(ref.trials);
      expect(mismatches(spec.traj.t, ref.traj.t)).toBe(0);
      expect(mismatches(spec.traj.l, ref.traj.l)).toBe(0);
      expect(mismatches(spec.traj.w, ref.traj.w)).toBe(0);
    }, 300000);
});
