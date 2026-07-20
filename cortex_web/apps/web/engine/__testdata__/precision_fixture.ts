import type { ComputeEngineInputs } from "../types";

function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Number(i === j)));
}

/** Small deterministic Precision bank used by execution-path drift guards. */
export function precisionGoldenInputs(): ComputeEngineInputs {
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
      applicableTaskIdx: [0],
      sMean: [signal, 0, 0, 0, 0, 0, 0],
      sSd: [0.2, 0, 0, 0, 0, 0, 0],
    })),
  };
}
