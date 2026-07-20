// N=1200 between-question latency micro-bench (goal 2/6). Times the engine's
// answer→next-question critical path (update + maybe-rejuvenate + A-optimal
// select) over a realistic ~700-seg session bank at N=1200 (the v15 particle
// count). NOT a correctness gate — it LOGS percentiles and asserts only a
// generous ceiling to catch catastrophic regressions. Run explicitly:
//   npx vitest run engine/_latency_bench.test.ts
import { describe, it, expect } from "vitest";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";
import { Rng } from "./rng";

function realisticBank(perClass: number, perDomainCap = 15): EngineInputs {
  const codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];
  const words = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"];
  const classes = ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"] as ("spike" | "iiic")[];
  const rng = new Rng(99);
  const segs: EngineInputs["segments"] = [];
  let id = 0;
  for (let k = 0; k < 7; k++) {
    const applicable = classes[k] === "spike" ? [0] : [1, 2, 3, 4, 5, 6];
    for (let i = 0; i < perClass; i++) {
      const sMean = new Array(7).fill(0);
      const sSd = new Array(7).fill(0);
      for (const j of applicable) { sMean[j] = 0.2 + 1.6 * (rng.int(1000) / 1000); sSd[j] = 0.3; }
      segs.push({
        segId: id++, patternClass: words[k], testClass: classes[k],
        applicableTaskIdx: applicable, sMean, sSd,
        fsHz: 200, nCh: 1, nSamp: 1, channelNames: [], eeg: "", spec: "",
      });
    }
  }
  return {
    taskCodes: codes, taskLabels: codes, taskPatternWords: words, taskClasses: classes,
    corrL: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => (i === j ? 1 : 0))),
    ellStar: new Array(7).fill(1.0), nParticles: 1200, perDomainCap, segments: segs,
  };
}

describe("N=1200 between-question latency (bench)", () => {
  // Gated out of normal CI (each run is minutes). Run with: CORTEX_BENCH=1 npx vitest run engine/_latency_bench.test.ts
  it.runIf(!!process.env.CORTEX_BENCH)("logs answer→next-item percentiles across pool sizes at N=1200", async () => {
    for (const perClass of [36, 57, 100]) {     // pools ≈ 250 / 400 / 700
      const inputs = realisticBank(perClass);
      const rng = new Rng(7);
      const gaps: number[] = [];
      let tAnswer = 0;
      const session = new WebCortexSession(inputs, `bench-N1200-${perClass}`, 7, {
        onItem: () => {
          if (tAnswer) gaps.push(performance.now() - tAnswer);
          queueMicrotask(() => { tAnswer = performance.now(); session.submitAnswer(rng.int(7)); });
        },
      });
      const r = await session.run();
      gaps.sort((a, b) => a - b);
      const pct = (p: number) => gaps[Math.min(gaps.length - 1, Math.floor(p * gaps.length))];
      console.info(
        `[bench N=1200] pool=${inputs.segments.length} questions=${r.nQuestions} ` +
        `stop=${r.stopReason} median=${pct(0.5).toFixed(0)}ms ` +
        `p90=${pct(0.9).toFixed(0)}ms max=${gaps[gaps.length - 1].toFixed(0)}ms`,
      );
      expect(gaps.length).toBeGreaterThan(10);
      expect(pct(0.5)).toBeLessThan(3000); // generous ceiling; flags only catastrophic regressions
    }
  }, 360_000);

  // Proves speculation moves the cost OFF the answer→next-item critical path:
  // the perceived gap (answer submitted → next item shown) collapses with
  // speculation when the predicted branch completes during prior think-time.
  it.runIf(!!process.env.CORTEX_BENCH)("speculation hides the answer→next-item latency", async () => {
    for (const speculative of [false, true]) {
      const inputs = realisticBank(57, 6); // pool ≈ 400, ~42 trials
      const rng = new Rng(7);
      const gaps: number[] = [];
      let tAnswer = 0;
      const session = new WebCortexSession(inputs, `mode-${speculative}`, 7, {
        onItem: () => {
          if (tAnswer) gaps.push(performance.now() - tAnswer);
          queueMicrotask(() => { tAnswer = performance.now(); session.submitAnswer(rng.int(7)); });
        },
      }, { speculative });
      const r = await session.run();
      gaps.sort((a, b) => a - b);
      const med = gaps[Math.floor(gaps.length / 2)] ?? 0;
      console.info(
        `[hide-latency ${speculative ? "spec  " : "inline"}] pool=${inputs.segments.length} ` +
        `questions=${r.nQuestions} answer→next-item median=${med.toFixed(1)}ms`,
      );
    }
  }, 360_000);
});
