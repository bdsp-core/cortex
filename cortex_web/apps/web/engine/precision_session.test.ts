import { describe, expect, it } from "vitest";

import { WebCortexSession } from "./session";
import type { EngineInputs, TrialDiag } from "./types";

function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Number(i === j)));
}

function goldenInputs(): EngineInputs {
  const codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];
  const signals = Array.from({ length: 20 }, (_, i) => -2 + 4 * i / 19);
  return {
    taskCodes: codes,
    taskLabels: codes,
    taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
    taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
    corrL: identity(7),
    corrT: identity(7),
    nParticles: 1200,
    perDomainCap: 60,
    terminationPolicy: "precision_v1",
    precisionBandEdges: Array.from({ length: 7 }, () => [-0.5, 0.5]),
    ellStar: new Array(7).fill(0),
    segments: signals.map((signal, i) => ({
      segId: i + 1,
      patternClass: "spike",
      testClass: "spike" as const,
      applicableTaskIdx: [0],
      sMean: [signal, 0, 0, 0, 0, 0, 0],
      sSd: [0.2, 0, 0, 0, 0, 0, 0],
      fsHz: 200,
      nCh: 1,
      nSamp: 1,
      channelNames: [],
      eeg: "",
      spec: "",
    })),
  };
}

function rounded(values: number[] | undefined): number[] | undefined {
  return values?.map((x) => Number(x.toFixed(12)));
}

describe("Precision per-engine deterministic golden session", () => {
  it("pins the production profile independently of Python RNG", async () => {
    const items: number[] = [];
    let last: TrialDiag | undefined;
    const session = new WebCortexSession(goldenInputs(), "precision-golden", 2718, {
      onItem: ({ segId }) => {
        items.push(segId);
        queueMicrotask(() => session.submitAnswer(0));
      },
      onTrial: (diag) => { last = diag; },
    });
    const result = await session.run();
    const golden = {
      nQuestions: result.nQuestions,
      stopReason: result.stopReason,
      items,
      verdicts: result.verdicts,
      domainStatuses: result.domainStatuses,
      determinations: result.determinations,
      finalAuroc: rounded(result.finalAuroc),
      lMean: rounded(last?.lMean),
      radii: rounded(last?.skillPointCenteredRadius),
      radiusMcse: rounded(last?.skillPointCenteredRadiusMcse),
      streaks: last?.precisionStreakCounts,
    };
    expect(golden).toEqual({
      nQuestions: 20,
      stopReason: "all_estimated_or_undeterminable",
      items: [13, 10, 12, 11, 4, 7, 8, 6, 1, 3, 2, 5, 9, 14, 15, 16, 17, 18, 19, 20],
      verdicts: new Array(7).fill("UNDETERMINABLE_BANK"),
      domainStatuses: new Array(7).fill("UNDETERMINABLE_BANK"),
      determinations: new Array(7).fill("UNDETERMINABLE_BANK"),
      finalAuroc: [
        0.886224152996, 0.808394411791, 0.809996355607, 0.80876406091,
        0.808633671687, 0.810540045215, 0.813492628681,
      ],
      lMean: [
        0.807796814039, -0.010923878676, 0.014772225831, 0.00080607016,
        -0.026312796347, 0.018443317374, 0.055124611465,
      ],
      radii: [
        1.545752190313, 1.936975635024, 1.940976525386, 2.0818848863,
        1.949532229156, 1.954307156512, 1.985080091331,
      ],
      radiusMcse: [
        0.050426640969, 0.075069070094, 0.091941275482, 0.102798386481,
        0.091625901491, 0.104731594263, 0.066901173299,
      ],
      streaks: new Array(7).fill(0),
    });
  }, 60_000);

  it("keeps the new Precision clone seam bit-identical to inline execution", async () => {
    async function run(speculative: boolean) {
      const items: number[] = [];
      const session = new WebCortexSession(
        goldenInputs(), `precision-clone-${speculative}`, 31415,
        {
          onItem: ({ segId }) => {
            items.push(segId);
            queueMicrotask(() => session.submitAnswer(0));
          },
        },
        { speculative },
      );
      const result = await session.run();
      return {
        items,
        nQuestions: result.nQuestions,
        stopReason: result.stopReason,
        verdicts: result.verdicts,
        domainStatuses: result.domainStatuses,
        determinations: result.determinations,
        servedSegIds: result.servedSegIds,
        trials: result.trials,
        finalAuroc: result.finalAuroc,
        finalAurocHw: result.finalAurocHw,
        trajectory: {
          t: Array.from(result.traj.t),
          l: Array.from(result.traj.l),
          w: Array.from(result.traj.w),
          shape: result.traj.shape,
        },
      };
    }

    expect(await run(true)).toEqual(await run(false));
  }, 60_000);
});
