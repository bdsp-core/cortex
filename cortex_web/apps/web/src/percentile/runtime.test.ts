import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { scoreDomain } from "./runtime";

describe("percentile runtime scorer", () => {
  it("propagates candidate and reference uncertainty", () => {
    const curves = new Float32Array([
      -1, 0, 1, -1.1, 0, 1.1,
    ]);
    const metadata = {
      intervalLevel: 0.95,
      intervalResolutionPercentagePoints: 0.01,
      policyStatus: "provisional_historical_calibration_cohort",
      domains: {
        spike: {
          bootstrap: { offset: 0, count: 6, rows: 2, cols: 3 },
          sensitivity: {},
        },
      },
    };
    const result = scoreDomain(
      metadata as never, curves, "spike", [-1, 1], [0.5, 0.5],
    );
    expect(result.estimate).toBeCloseTo(50);
    expect(result.lower).toBeLessThan(50);
    expect(result.upper).toBeGreaterThan(50);
    expect(result.candidateMethod).toBe("posterior_particles");
  });

  it("matches the Python scorer on the shipped immutable artifact", () => {
    const base = new URL(
      "../../public/norms/historical-calibration-k7-provisional-v1",
      import.meta.url,
    );
    const metadata = JSON.parse(
      readFileSync(fileURLToPath(new URL(`${base.href}.json`)), "utf8"),
    );
    const bytes = readFileSync(fileURLToPath(new URL(`${base.href}.bin`)));
    const values = new Float32Array(
      bytes.buffer, bytes.byteOffset, bytes.byteLength / 4,
    );
    const result = scoreDomain(
      metadata, values, "spike", [-0.25, 0.1, 0.7], [0.2, 0.5, 0.3],
    );
    expect(result.estimate).toBeCloseTo(62.044597563171386, 7);
    expect(result.lower).toBe(29.48);
    expect(result.upper).toBe(93.45);
    expect(result.sensitivityRange?.[0]).toBeCloseTo(59.76856975317356, 7);
    expect(result.sensitivityRange?.[1]).toBeCloseTo(61.716726964972786, 7);
  });

  it("scores a full seven-domain 1,200-particle cloud within the result budget", () => {
    const base = new URL(
      "../../public/norms/historical-calibration-k7-provisional-v1",
      import.meta.url,
    );
    const metadata = JSON.parse(
      readFileSync(fileURLToPath(new URL(`${base.href}.json`)), "utf8"),
    );
    const bytes = readFileSync(fileURLToPath(new URL(`${base.href}.bin`)));
    const values = new Float32Array(
      bytes.buffer, bytes.byteOffset, bytes.byteLength / 4,
    );
    const samples = Float64Array.from(
      { length: 1_200 }, (_, i) => -1.5 + (3 * i) / 1_199,
    );
    const weights = new Float64Array(1_200).fill(1 / 1_200);
    const started = performance.now();
    for (const domain of [
      "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic",
    ] as const) {
      scoreDomain(metadata, values, domain, samples, weights);
    }
    // A generous non-flaky ceiling. The observed local runtime is reported in
    // the validation report; this protects against accidental O(B*N*Q) drift.
    expect(performance.now() - started).toBeLessThan(5_000);
  });
});
