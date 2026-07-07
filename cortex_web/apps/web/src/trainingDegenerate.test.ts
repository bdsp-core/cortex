import { describe, it, expect } from "vitest";
import { ArrayBank, TrainerSession, buildFilters } from "../trainer/session";
import type { FilterParams } from "../trainer/filter";
import { buildTrainerBank, cutScores, seedClouds } from "./trainerBank";
import { TrainingController } from "./trainingController";

// Regression guard for the "Resume training" blank-screen bug: when a small or
// fully-spaced-out candidate draw yields empty/tiny per-task pools, the trainer
// must degrade gracefully (reach a valid phase without throwing), never crash
// the render. Mirrors App.tsx startTraining's session construction.
const PARAMS: FilterParams = {
  alphaT: 0.097, alphaSigma: 0.047, sigmaInf: 0.4,
  qT: 0.05, qSigma: 0.02, rho: 0.5, rule: "soft",
};
const ELLSTAR = [0.3, 0.155, 0.36, 0.36, 0.36, 0.36, 0.36];
const LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other/IIC"];

function buildSession(segments: unknown[]) {
  const { ellStars, sigmaStars, sigmaInf } = cutScores(ELLSTAR);
  const clouds = seedClouds(ellStars.map(() => 0.0), 200, 1);
  const filters = buildFilters(clouds, PARAMS, sigmaInf, ellStars, { seed: 7, useMixture: false });
  return new TrainerSession(
    filters, ellStars, sigmaStars,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    new ArrayBank(buildTrainerBank({ segments: segments as any })), { seed: 7 });
}

// every method the TrainingRunner render calls
function driveRenderPath(c: TrainingController) {
  c.snapshot(); c.allMastered(); c.progress(); c.drainTrajectory();
}

describe("training start — degenerate draws never throw", () => {
  it("empty draw → resting 'done' phase (renders the no-segments screen)", () => {
    const c = new TrainingController(buildSession([]), LABELS, { total: 120 });
    expect(() => driveRenderPath(c)).not.toThrow();
    expect(c.phase).toBe("done");
    expect(c.progress().count).toBe(0);
  });

  it("tiny draw → runs and finishes cleanly", () => {
    const segments = [
      { segId: 101, patternClass: "seizure", applicableTaskIdx: [1, 2, 3, 4, 5, 6],
        sMean: [0, 0.9, -0.4, -0.5, -0.6, -0.3, 0.2], sSd: [0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4] },
      { segId: 102, patternClass: "gpd", applicableTaskIdx: [1, 2, 3, 4, 5, 6],
        sMean: [0, -0.5, -0.3, 0.8, -0.4, -0.2, 0.1], sSd: [0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4] },
      { segId: 103, patternClass: "spike", applicableTaskIdx: [0],
        sMean: [1.1, 0, 0, 0, 0, 0, 0], sSd: [0.4, 0, 0, 0, 0, 0, 0] },
    ];
    const c = new TrainingController(buildSession(segments), LABELS, { total: 120 });
    expect(() => {
      driveRenderPath(c);
      for (let i = 0; i < 12 && c.phase !== "done"; i++) {
        if (c.phase === "question") { c.markShown(0); c.answer(i % 2 === 0, 50); }
        else if (c.phase === "result") c.continue();
        driveRenderPath(c);
      }
    }).not.toThrow();
    expect(["question", "result", "done"]).toContain(c.phase);
  });
});
