// Phase L3: the server-driven trainer session behind the controller —
// prefetch protocol (submit → whenReady → next), snapshot refresh,
// graceful end on a record failure, and the controller integration
// (waitForNext + continue advances the question flow).
import { describe, expect, it, vi } from "vitest";

vi.mock("../src/api", () => {
  const snap = (skill: number) => [0, 1, 2].map((task) => ({
    task, mastered: false, skill, theta: 0.0, sd: 0.3,
    passMass: 0.2, trainability: 0.9,
  }));
  return {
    engineStart: vi.fn(async () => ({
      item: { task: 1, segId: 10, s: 0.4, sSd: 0.1, yStar: 1, mode: "skill" },
      snapshot: snap(0.1), allMastered: false, seeded: 8, rebuiltSeq: 0,
    })),
    engineRecord: vi.fn(async (_tid: string, segId: number) => {
      if (segId === 99) throw new Error("boom");
      return {
        item: segId === 10
          ? { task: 2, segId: 11, s: -0.2, sSd: 0.1, yStar: 0, mode: "bias" }
          : null,
        snapshot: snap(0.2), allMastered: segId !== 10,
      };
    }),
  };
});

import { ServerTrainerSession } from "./serverSession";
import { TrainingController } from "../src/trainingController";
import * as api from "../src/api";

describe("ServerTrainerSession", () => {
  it("prefetches the next item through submit/whenReady", async () => {
    const s = await ServerTrainerSession.start("t1", [10, 11], [1, 2]);
    expect(api.engineStart).toHaveBeenCalledWith("t1", [10, 11], [1, 2]);
    const first = s.next();
    expect(first?.segId).toBe(10);
    expect(s.snapshot()[1].skill).toBeCloseTo(0.1);

    s.submit(first!, 1);
    expect(s.next()).toBeNull();          // consumed until the server answers
    await s.whenReady();
    expect(s.next()?.segId).toBe(11);
    expect(s.snapshot()[1].skill).toBeCloseTo(0.2);   // refreshed post-answer

    s.submit(s.next()!, 0);
    await s.whenReady();
    expect(s.next()).toBeNull();          // server says done
    expect(s.allMastered()).toBe(true);
  });

  it("ends gracefully when a record call fails", async () => {
    const s = await ServerTrainerSession.start("t2", [99]);
    s.submit({ task: 0, mode: "skill", segId: 99, s: 0, sSd: 0, yStar: 1,
               margin: 0, info: {}, now: 0 }, 1);
    await s.whenReady();                  // must not reject
    expect(s.next()).toBeNull();
  });

  it("drives the TrainingController question flow", async () => {
    const s = await ServerTrainerSession.start("t3", [10, 11]);
    const c = new TrainingController(s, ["A", "B", "C"], { total: 5 });
    expect(c.phase).toBe("question");
    expect(c.item?.segId).toBe(10);
    c.markShown(0, "2026-07-16T10:00:00Z");
    c.answer(true, 850, "2026-07-16T10:00:01Z");
    expect(c.phase).toBe("result");
    const pts = c.drainTrajectory();
    expect(pts[0]).toMatchObject({ segId: 10, pick: 1, yStar: 1,
                                   isCorrect: true,
                                   shownClientUtc: "2026-07-16T10:00:00Z" });
    await c.waitForNext();                // engine round trip
    c.continue();
    expect(c.phase).toBe("question");
    expect(c.item?.segId).toBe(11);
  });
});
