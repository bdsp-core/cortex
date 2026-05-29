import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";
import { pResponseYes, signalZ } from "./likelihood";
import { Rng } from "./rng";
import { VERDICT } from "./policy";

// End-to-end local test: drive the full adaptive loop with a SIMULATED rater
// of known per-task (t, l). Mirrors the desktop's
// session_controller.make_simulated_y_source smoke run. Prefers the real
// bank bundle if it's been generated locally
// (cortex_web/public/bundle/v1.1-local/manifest.json); otherwise builds
// synthetic inputs so the test stays self-contained in CI.

function loadRealInputs(): EngineInputs | null {
  const p = fileURLToPath(
    new URL("../public/bundle/v1.1-local/manifest.json", import.meta.url),
  );
  if (!existsSync(p)) return null;
  const m = JSON.parse(readFileSync(p, "utf8"));
  return {
    taskCodes: m.taskCodes,
    taskLabels: m.taskLabels,
    taskPatternWords: m.taskPatternWords,
    corrL: m.corrL,
    ellStar: m.ellStar,
    segments: m.segments,
  };
}

// Synthetic fallback: K=6, nSeg segments with signals spread across tasks,
// a realistic-ish unit-diagonal weakly-correlated prior, plausible ℓ*.
function syntheticInputs(nSeg = 96): EngineInputs {
  const K = 6;
  const taskCodes = ["sz", "lpd", "gpd", "lrda", "grda", "iic"];
  const words = ["seizure", "lpd", "gpd", "lrda", "grda", "other"];
  const corrL = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => (i === j ? 1 : 0.2)),
  );
  const ellStar = [0.45, 0.53, 0.33, 0.48, 0.49, 0.44];
  const rng = new Rng(123);
  const segments = [];
  for (let i = 0; i < nSeg; i++) {
    const trueTask = i % K;
    const sMean = Array.from({ length: K }, (_, k) =>
      // the true-task signal is positive (it IS that pattern), others near 0
      k === trueTask ? 1.0 + rng.random() * 1.5 : (rng.random() - 0.5) * 0.6,
    );
    const sSd = Array.from({ length: K }, () => 0.1 + rng.random() * 0.1);
    segments.push({
      segId: i + 1,
      patternClass: words[trueTask],
      sMean,
      sSd,
      fsHz: 200,
      nCh: 20,
      nSamp: 6000,
      channelNames: [],
      eeg: "",
      spec: "",
    });
  }
  return { taskCodes, taskLabels: taskCodes, taskPatternWords: words, corrL, ellStar, segments };
}

// Run one session against a simulated rater with per-task true (t, l).
async function runSim(
  inputs: EngineInputs,
  trueT: number[],
  trueL: number[],
  seed: number,
): Promise<{ nQ: number; verdicts: string[]; stopReason: string }> {
  const segById = new Map(inputs.segments.map((s) => [s.segId, s]));
  const simRng = new Rng(seed ^ 0x5eed);

  const session = new WebCortexSession(inputs, `sim-${seed}`, seed, {
    onItem: ({ taskK, segId }) => {
      // The engine asks task k at this segment. Draw the rater's binary Y at
      // the segment's task-k signal, then submit a 6-way pick whose
      // one-vs-rest reduction (pick==k) equals Y — exactly the desktop's
      // EngineWorker._y_source contract. Defer so the loop's awaitAnswer()
      // has installed its resolver first.
      const seg = segById.get(segId)!;
      const s = seg.sMean[taskK];
      const p = pResponseYes(signalZ(trueL[taskK], trueT[taskK], s));
      const y = simRng.random() < p ? 1 : 0;
      const pick = y === 1 ? taskK : (taskK + 1) % inputs.taskCodes.length;
      queueMicrotask(() => session.submitAnswer(pick));
    },
  });
  const res = await session.run();
  return { nQ: res.nQuestions, verdicts: res.verdicts, stopReason: res.stopReason };
}

const inputs = loadRealInputs() ?? syntheticInputs();
const source = loadRealInputs() ? "real bank bundle" : "synthetic";
const K = inputs.taskCodes.length;

describe(`full adaptive session (${source}, ${inputs.segments.length} segs)`, () => {
  it("terminates within the bank", async () => {
    const r = await runSim(inputs, new Array(K).fill(0), new Array(K).fill(0.6), 1);
    expect(r.nQ).toBeGreaterThan(0);
    expect(r.nQ).toBeLessThanOrEqual(inputs.segments.length);
    expect(["all_resolved", "bank_exhausted"]).toContain(r.stopReason);
  }, 30000);

  it("a clearly-skilled rater earns mostly PASS", async () => {
    // high discrimination (l≈1.0), unbiased (t=0) → should pass most tasks
    const r = await runSim(inputs, new Array(K).fill(0), new Array(K).fill(1.0), 2);
    const nPass = r.verdicts.filter((v) => v === VERDICT.PASS).length;
    // tolerant assertion: the simulated rater is strong, but REFER is allowed
    // on tasks the adaptive bank under-samples; just demand no FAILs and a
    // clear PASS majority among resolved tasks.
    const nFail = r.verdicts.filter((v) => v === VERDICT.FAIL).length;
    expect(nFail).toBe(0);
    expect(nPass).toBeGreaterThanOrEqual(3);
  }, 30000);

  it("a clearly-unskilled rater earns no PASS", async () => {
    // at-chance discrimination (l≈0) → no task should certify PASS
    const r = await runSim(inputs, new Array(K).fill(0), new Array(K).fill(-0.2), 3);
    const nPass = r.verdicts.filter((v) => v === VERDICT.PASS).length;
    expect(nPass).toBe(0);
  }, 30000);
});
