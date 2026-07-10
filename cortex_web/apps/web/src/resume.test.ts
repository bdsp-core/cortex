import { describe, expect, it } from "vitest";
import { ReplayDriver } from "./resume";

const TRIALS = [
  { trialIndex: 0, segId: 101, pick: 2 },
  { trialIndex: 1, segId: 55, pick: 0 },
];

describe("ReplayDriver", () => {
  it("feeds recorded picks while items match, then goes live", () => {
    const d = new ReplayDriver(TRIALS);
    expect(d.remaining).toBe(2);
    expect(d.next(101)).toEqual({ kind: "answer", pick: 2 });
    expect(d.next(55)).toEqual({ kind: "answer", pick: 0 });
    expect(d.remaining).toBe(0);
    expect(d.next(999)).toEqual({ kind: "live" });
  });

  it("flags a mismatch when the engine serves an unexpected item", () => {
    const d = new ReplayDriver(TRIALS);
    expect(d.next(102)).toEqual({ kind: "mismatch" });
  });

  it("is immediately live with no trials", () => {
    const d = new ReplayDriver([]);
    expect(d.remaining).toBe(0);
    expect(d.next(101)).toEqual({ kind: "live" });
  });

  it("does not mutate the caller's trial array", () => {
    const trials = [...TRIALS];
    const d = new ReplayDriver(trials);
    d.next(101);
    expect(trials).toHaveLength(2);
  });
});
