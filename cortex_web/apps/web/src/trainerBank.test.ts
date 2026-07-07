import { describe, it, expect } from "vitest";
import { buildTrainerBank, cutScores, seedClouds, seedCloudsFromPrior, type TaskPrior } from "./trainerBank";

function mean(a: number[]) { return a.reduce((x, y) => x + y, 0) / a.length; }
function sd(a: number[]) { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); }

const SD = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
const manifest = {
  segments: [
    { segId: 0, sMean: [1.0, 0, 0, 0, 0, 0, 0], sSd: SD, applicableTaskIdx: [0] },
    { segId: 1, sMean: [0, -0.5, 0.8, 0, 0, 0, 0], sSd: SD, applicableTaskIdx: [1, 2, 3, 4, 5, 6] },
    { segId: 2, sMean: [0, 0.4, -0.2, 0, 0, 0, 0], sSd: SD, applicableTaskIdx: [1, 2, 3, 4, 5, 6] },
  ],
};

describe("trainerBank", () => {
  it("builds per-task SDT-frame candidates from the manifest", () => {
    const bank = buildTrainerBank(manifest);
    expect(bank.length).toBe(7);
    expect(bank[0].seg).toEqual([0]);          // spike seg only in task 0
    expect(bank[0].yStar).toEqual([1]);        // sMean[0]=1>0
    const i = bank[1].seg.indexOf(1);
    expect(bank[1].yStar[i]).toBe(0);          // task1 sMean[1]=-0.5<0
    const j = bank[2].seg.indexOf(1);
    expect(bank[2].yStar[j]).toBe(1);          // task2 sMean[2]=0.8>0
    expect(bank.every((p) => p.coherent.every((c) => c))).toBe(true);  // SDT ⇒ coherent
  });

  it("cut-scores: σ* = exp(−ℓ*), ceiling above the cut", () => {
    const { sigmaStars, sigmaInf } = cutScores([0.3, 0.2, 0.5, 0.3, 0.3, 0.35, 0.3]);
    expect(sigmaStars[0]).toBeCloseTo(Math.exp(-0.3));
    expect(sigmaInf[0]).toBeLessThan(sigmaStars[0]);  // σ_∞ < σ* ⇒ ceiling skill > cut skill
  });

  it("seedClouds: N finite particles per task, deterministic", () => {
    const clouds = seedClouds([0, 0, 0, 0, 0, 0, 0], 50, 1);
    expect(clouds.length).toBe(7);
    expect(clouds[0].theta.length).toBe(50);
    expect(clouds[0].ell.every(Number.isFinite)).toBe(true);
    expect(seedClouds([0], 50, 1)[0].theta[0]).toBe(clouds[0].theta[0]);
  });

  it("seedCloudsFromPrior: centers each cloud on the measured ℓ/θ, inflates the ℓ SD", () => {
    const prior: TaskPrior[] = [
      { taskK: 0, ell: 1.2, theta: -0.3, sd: 0.5 },   // strong, biased, sd above floor
      { taskK: 2, ell: -0.1, theta: 0.0, sd: 0.1 },   // weak, sd below floor
    ];
    const clouds = seedCloudsFromPrior(prior, 7, 4000, 1, { inflate: 1.6, ellSdFloor: 0.25, thetaSd: 0.4 });
    expect(clouds.length).toBe(7);
    // task 0 (has a prior): ℓ mean ≈ 1.2, θ mean ≈ −0.3
    expect(mean(clouds[0].ell)).toBeCloseTo(1.2, 1);
    expect(mean(clouds[0].theta)).toBeCloseTo(-0.3, 1);
    // ℓ SD ≈ posterior sd × inflate = 0.5 × 1.6 = 0.80 (sd is above the floor)
    expect(sd(clouds[0].ell)).toBeCloseTo(0.80, 1);
    // task 2's sd=0.1 is below the floor → floored to 0.25 then ×1.6 = 0.40
    expect(sd(clouds[2].ell)).toBeCloseTo(0.40, 1);
    // task 1 (no prior): falls back to the generic ℓ~N(0,·), θ~N(0,·)
    expect(mean(clouds[1].ell)).toBeCloseTo(0, 1);
    // weak task 2 seeded low so the trainer targets it
    expect(mean(clouds[2].ell)).toBeCloseTo(-0.1, 1);
    // deterministic for a fixed seed
    expect(seedCloudsFromPrior(prior, 7, 4000, 1)[0].ell[0]).toBe(clouds[0].ell[0]);
  });
});
