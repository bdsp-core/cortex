import { describe, expect, it } from "vitest";

import { Rng } from "./rng";

describe("Rng.fillGaussian", () => {
  it("is exactly equivalent to repeated scalar draws for every spare parity", () => {
    for (const startWithSpare of [false, true]) {
      for (let length = 0; length <= 17; length++) {
        const scalar = new Rng(20260721);
        const bulk = new Rng(20260721);
        if (startWithSpare) {
          expect(bulk.gaussian()).toBe(scalar.gaussian());
        }
        const expected = Float64Array.from(
          { length }, () => scalar.gaussian(),
        );
        const actual = new Float64Array(length);
        bulk.fillGaussian(actual);
        expect(actual).toEqual(expected);
        expect(bulk.snapshot()).toEqual(scalar.snapshot());
      }
    }
  });

  it("preserves the scalar stream across consecutive mixed-size fills", () => {
    const scalar = new Rng(9182);
    const bulk = new Rng(9182);
    for (const length of [1, 14, 3, 0, 8, 7, 2]) {
      const expected = Float64Array.from(
        { length }, () => scalar.gaussian(),
      );
      const actual = new Float64Array(length);
      bulk.fillGaussian(actual);
      expect(actual).toEqual(expected);
      expect(bulk.snapshot()).toEqual(scalar.snapshot());
    }
  });
});
