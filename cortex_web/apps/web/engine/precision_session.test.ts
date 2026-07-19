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
    // Golden regenerated 2026-07-18 for the mc-guard v2 default (30 MH
    // rejuvenation + requalified inflation 1.3090533918867642). Deterministic
    // (JS engine); verified identical across repeated runs.
    expect(golden).toEqual({
      nQuestions: 20,
      stopReason: "all_estimated_or_undeterminable",
      items: [13, 10, 12, 5, 8, 9, 7, 1, 4, 3, 2, 6, 11, 14, 15, 16, 17, 18, 19, 20],
      verdicts: new Array(7).fill("UNDETERMINABLE_BANK"),
      domainStatuses: new Array(7).fill("UNDETERMINABLE_BANK"),
      determinations: new Array(7).fill("UNDETERMINABLE_BANK"),
      finalAuroc: [
        0.887441282001, 0.805521924498, 0.807925398904, 0.809039855684,
        0.814195326743, 0.806795792622, 0.815216617389,
      ],
      lMean: [
        0.832037541266, -0.055032653519, -0.014967141147, 0.007167667373,
        0.042178529543, -0.006808585652, 0.056747515935,
      ],
      radii: [
        1.455189563025, 2.039682015564, 1.998461408126, 1.964979902026,
        1.918893634779, 2.086133089494, 1.963331043901,
      ],
      radiusMcse: [
        0.059771121067, 0.10179637015, 0.110353091087, 0.103555240577,
        0.093900656031, 0.107774204025, 0.090189922441,
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
