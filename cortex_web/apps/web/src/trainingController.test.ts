import { describe, it, expect } from "vitest";
import { ArrayBank, TrainerSession, buildFilters, type CandidateArrays } from "../trainer/session";
import type { FilterParams } from "../trainer/filter";
import { TrainingController } from "./trainingController";

function makeController(total = 6) {
  const per: CandidateArrays[] = [0, 1].map((k) => {
    const a: CandidateArrays = { seg: [], sMean: [], sSd: [], yStar: [], margin: [], coherent: [] };
    for (let i = 0; i < 40; i++) {
      const s = -1.5 + (3 * i) / 39;
      const y = s > 0 ? 1 : 0;
      a.seg.push(k * 100 + i); a.sMean.push(s); a.sSd.push(0.4);
      a.yStar.push(y); a.margin.push(1); a.coherent.push(true);
    }
    return a;
  });
  const bank = new ArrayBank(per);
  const params: FilterParams = { alphaT: 0.1, alphaSigma: 0.15, sigmaInf: 0.4, qT: 0.05, qSigma: 0.02, rho: 0.5, rule: "soft" };
  const clouds = [0, 1].map(() => ({
    theta: new Array(80).fill(0), ell: new Array(80).fill(0), w: new Array(80).fill(1 / 80),
  }));
  const filters = buildFilters(clouds, params, [0.4, 0.4], [0.3, 0.3], { seed: 1, useMixture: false });
  const session = new TrainerSession(filters, [0.3, 0.3], [Math.exp(-0.3), Math.exp(-0.3)], bank, { seed: 1 });
  return new TrainingController(session, ["Spike", "Seizure"], { total });
}

describe("TrainingController", () => {
  it("question → result → question, with correctness, label, and a trajectory point", () => {
    const c = makeController(6);
    expect(c.phase).toBe("question");
    const it = c.item!;
    c.markShown(0);
    c.answer(true, 100);                       // "yes"
    expect(c.phase).toBe("result");
    expect(c.lastResult!.correct).toBe(it.yStar === 1);
    expect(c.lastResult!.patternLabel).toBe(["Spike", "Seizure"][it.task]);
    expect(c.progress().count).toBe(1);
    const pts = c.drainTrajectory();
    expect(pts.length).toBe(1);
    expect(pts[0].segId).toBe(it.segId);
    expect(Number.isFinite(pts[0].ell)).toBe(true);
    c.continue();
    expect(c.phase).toBe("question");
  });

  it("finishes at the item budget", () => {
    const c = makeController(3);
    for (let i = 0; i < 20 && c.phase !== "done"; i++) {
      if (c.phase === "question") { c.markShown(0); c.answer(i % 2 === 0, 50); }
      else if (c.phase === "result") c.continue();
    }
    expect(c.phase).toBe("done");
    expect(c.progress().count).toBeGreaterThan(0);
    expect(c.progress().count).toBeLessThanOrEqual(3);
  });

  it("ignores an answer outside the question phase", () => {
    const c = makeController(6);
    c.markShown(0);
    c.answer(true, 10);
    const cnt = c.progress().count;
    c.answer(false, 20);                       // in result phase → no-op
    expect(c.progress().count).toBe(cnt);
  });
});
