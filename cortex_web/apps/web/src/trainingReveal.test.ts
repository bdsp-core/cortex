import { describe, expect, it } from "vitest";
import { buildRevealRows } from "./trainingReveal";
import type { TaskSnapshot } from "../trainer/types";

function snap(task: number, over: Partial<TaskSnapshot> = {}): TaskSnapshot {
  return {
    task, mastered: false, skill: 0.2, theta: 0, sd: 0.3,
    passMass: 0.4, trainability: null, ...over,
  };
}

const LABELS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other", "Spike"];
const ELL = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

describe("buildRevealRows", () => {
  it("includes only domains trained this session", () => {
    const start = LABELS.map((_, k) => snap(k));
    const end = LABELS.map((_, k) => snap(k, { skill: 0.25 }));
    const rows = buildRevealRows(start, end, LABELS, new Map([[2, 8], [4, 3]]), ELL);
    expect(rows.map((r) => r.taskK).sort()).toEqual([2, 4]);
    expect(rows.every((r) => r.items > 0)).toBe(true);
  });

  it("computes before/after movement from the snapshots", () => {
    const start = [snap(0, { skill: 0.20, sd: 0.30, passMass: 0.40 })];
    const end = [snap(0, { skill: 0.31, sd: 0.22, passMass: 0.55 })];
    const [r] = buildRevealRows(start, end, LABELS, new Map([[0, 12]]), ELL);
    expect(r.before).toEqual({ skill: 0.20, sd: 0.30, pass: 0.40 });
    expect(r.after).toEqual({ skill: 0.31, sd: 0.22, pass: 0.55 });
    expect(r.label).toBe("Seizure");
    expect(r.items).toBe(12);
  });

  it("flags a domain newly mastered today", () => {
    const start = [snap(0, { mastered: false })];
    const end = [snap(0, { mastered: true, skill: 0.6 })];
    const [r] = buildRevealRows(start, end, LABELS, new Map([[0, 5]]), ELL);
    expect(r.newlyMastered).toBe(true);
    expect(r.mastered).toBe(true);
  });

  it("does not flag an already-mastered domain as newly mastered", () => {
    const start = [snap(0, { mastered: true })];
    const end = [snap(0, { mastered: true })];
    const [r] = buildRevealRows(start, end, LABELS, new Map([[0, 5]]), ELL);
    expect(r.newlyMastered).toBe(false);
    expect(r.mastered).toBe(true);
  });

  it("flags near-bar when the 1-sigma interval touches ell-star", () => {
    // skill 0.42 + sd 0.10 = 0.52 >= 0.5 → touches the bar
    const touch = buildRevealRows(
      [snap(0)], [snap(0, { skill: 0.42, sd: 0.10 })], LABELS, new Map([[0, 4]]), ELL)[0];
    expect(touch.nearBar).toBe(true);
    // skill 0.30 + sd 0.10 = 0.40 < 0.5 → not near
    const far = buildRevealRows(
      [snap(0)], [snap(0, { skill: 0.30, sd: 0.10 })], LABELS, new Map([[0, 4]]), ELL)[0];
    expect(far.nearBar).toBe(false);
    // mastered domains are never "near the bar"
    const done = buildRevealRows(
      [snap(0, { mastered: true })],
      [snap(0, { mastered: true, skill: 0.55, sd: 0.10 })],
      LABELS, new Map([[0, 4]]), ELL)[0];
    expect(done.nearBar).toBe(false);
  });

  it("orders newly-mastered, then near-bar, then by items answered", () => {
    const start = [snap(0), snap(1), snap(2), snap(3)];
    const end = [
      snap(0, { skill: 0.1, sd: 0.05 }),                    // plain, 10 items
      snap(1, { mastered: true, skill: 0.6 }),              // newly mastered, 2 items
      snap(2, { skill: 0.45, sd: 0.10 }),                   // near bar, 3 items
      snap(3, { skill: 0.1, sd: 0.05 }),                    // plain, 6 items
    ];
    const rows = buildRevealRows(start, end, LABELS,
      new Map([[0, 10], [1, 2], [2, 3], [3, 6]]), ELL);
    expect(rows.map((r) => r.taskK)).toEqual([1, 2, 0, 3]);
  });

  it("tolerates missing labels and ell-stars", () => {
    const [r] = buildRevealRows([snap(0)], [snap(0)], [], new Map([[0, 1]]), []);
    expect(r.label).toBe("task 0");
    expect(r.ellStar).toBe(0);
  });
});
