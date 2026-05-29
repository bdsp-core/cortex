import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { normCdf, logNdtr, logSumExp2 } from "./mathfns";

const ref = JSON.parse(
  readFileSync(fileURLToPath(new URL("./__testdata__/reference.json", import.meta.url)), "utf8"),
);

// erfc() is the ~1e-7 Numerical-Recipes form, so Φ matches scipy to ~1e-7
// (relative) in the body. In the deep tail Φ is tiny so we compare relative
// magnitude with a generous floor. logΦ uses the stable left-tail asymptotic.

describe("normCdf vs scipy.stats.norm.cdf", () => {
  for (const { x, v } of ref.normCdf as { x: number; v: number }[]) {
    it(`Φ(${x}) ≈ ${v.toExponential(3)}`, () => {
      const got = normCdf(x);
      const tol = Math.max(1e-9, Math.abs(v) * 2e-4);
      expect(Math.abs(got - v)).toBeLessThan(tol);
    });
  }
});

describe("logNdtr vs scipy.special.log_ndtr", () => {
  for (const { x, v } of ref.logNdtr as { x: number; v: number }[]) {
    it(`logΦ(${x}) ≈ ${v.toFixed(4)}`, () => {
      const got = logNdtr(x);
      // relative tolerance on the (large-magnitude) log value
      const tol = Math.max(1e-6, Math.abs(v) * 1e-3);
      expect(Math.abs(got - v)).toBeLessThan(tol);
    });
  }
});

describe("logSumExp2 vs scipy.special.logsumexp", () => {
  for (const { a, b, v } of ref.logSumExp2 as { a: number; b: number; v: number }[]) {
    it(`logsumexp(${a}, ${b}) ≈ ${v}`, () => {
      expect(Math.abs(logSumExp2(a, b) - v)).toBeLessThan(1e-9);
    });
  }
});
