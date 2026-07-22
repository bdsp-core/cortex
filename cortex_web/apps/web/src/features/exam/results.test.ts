import { describe, expect, it } from "vitest";

import { buildPerTaskRecords, cloudSdPerTask, perTaskRoc } from "./results";

// [T,N,K] row-major cloud; only the LAST time step is read.
function cloud(step: { l: number[][]; w: number[] }[], K: number) {
  const T = step.length, N = step[0].w.length;
  const l = new Float32Array(T * N * K);
  const w = new Float32Array(T * N);
  step.forEach((s, t) => {
    s.w.forEach((weight, i) => {
      w[t * N + i] = weight;
      s.l[i].forEach((value, k) => { l[(t * N + i) * K + k] = value; });
    });
  });
  return { shape: [T, N, K] as [number, number, number], l, w };
}

describe("cloudSdPerTask", () => {
  it("computes the weighted SD of the final step", () => {
    // Final step: values 1 and 3 at equal weight -> mean 2, SD 1.
    const traj = cloud([
      { l: [[9], [9]], w: [1, 1] },        // earlier step must be ignored
      { l: [[1], [3]], w: [1, 1] },
    ], 1);
    const [sd] = cloudSdPerTask(traj, 1);
    expect(sd).toBeCloseTo(1, 10);
  });

  it("honours unequal weights", () => {
    // Values 0 and 10 with weights 9:1 -> mean 1, var = .9*1 + .1*81 = 9.
    const traj = cloud([{ l: [[0], [10]], w: [9, 1] }], 1);
    const [sd] = cloudSdPerTask(traj, 1);
    expect(sd).toBeCloseTo(3, 10);
  });

  it("returns null rather than a fabricated zero when weights vanish", () => {
    const traj = cloud([{ l: [[1], [3]], w: [0, 0] }], 1);
    expect(cloudSdPerTask(traj, 1)).toEqual([null]);
  });

  it("keeps tasks independent", () => {
    const traj = cloud([{ l: [[1, 5], [3, 5]], w: [1, 1] }], 2);
    const sds = cloudSdPerTask(traj, 2);
    expect(sds[0]).toBeCloseTo(1, 10);
    expect(sds[1]).toBeCloseTo(0, 10);   // second task is constant
  });
});

describe("perTaskRoc", () => {
  const inputs = {
    taskPatternWords: ["spike", "gpd"],
    taskClasses: ["spike", "iiic"] as ("iiic" | "spike")[],
  };

  it("carries AUROC and half-width per task", () => {
    const roc = perTaskRoc([0.8, 0.6], [0.05, 0.1], [], inputs, () => undefined);
    expect(roc.map((r) => r.auroc)).toEqual([0.8, 0.6]);
    expect(roc.map((r) => r.hw)).toEqual([0.05, 0.1]);
  });

  it("defaults a missing half-width to zero and no trials to a null point", () => {
    const roc = perTaskRoc([0.7], undefined, [], inputs, () => undefined);
    expect(roc[0].hw).toBe(0);
    expect(roc[0].opFar).toBeNull();
    expect(roc[0].opHr).toBeNull();
  });

  it("is empty when the engine reported no AUROC", () => {
    expect(perTaskRoc(undefined, undefined, [], inputs, () => undefined)).toEqual([]);
  });
});

describe("buildPerTaskRecords", () => {
  const inputs = {
    taskCodes: ["spike", "gpd"],
    taskLabels: ["Spike", "GPD"],
    ellStar: [0.15, 0.33],
  };
  const roc = [
    { auroc: 0.9, hw: 0.02, opFar: null, opHr: null },
    { auroc: 0.5, hw: 0.2, opFar: null, opHr: null },
  ];

  it("assembles the persisted per-task record", () => {
    const rows = buildPerTaskRecords({
      inputs,
      lastDiag: { lMean: [1.1, 0.2], tMean: [0.0, -0.3] },
      sdPerTask: [0.12, null],
      roc,
      verdicts: ["PASS", "FAIL"],
    });
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      taskK: 0, code: "spike", label: "Spike",
      ell: 1.1, theta: 0.0, sd: 0.12, ellStar: 0.15,
      auroc: 0.9, aurocHw: 0.02, verdict: "PASS",
    });
    expect(rows[1]).toMatchObject({ sd: null, verdict: "FAIL" });
  });

  it("falls back to PENDING and nulls when the engine left a task unresolved", () => {
    const rows = buildPerTaskRecords({
      inputs, lastDiag: null, sdPerTask: [], roc: [],
    });
    expect(rows[0]).toMatchObject({
      ell: null, theta: null, sd: null, auroc: null, verdict: "PENDING",
    });
  });

  it("labels a task by its code when no label is supplied", () => {
    const rows = buildPerTaskRecords({
      inputs: { taskCodes: ["xyz"], taskLabels: [], ellStar: [] },
      lastDiag: null, sdPerTask: [], roc: [],
    });
    expect(rows[0].label).toBe("xyz");
  });
});
