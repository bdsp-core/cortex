import { describe, expect, it } from "vitest";
import { TrainingController, type TrainerSessionLike } from "./trainingController";
import type { Choice, TaskSnapshot } from "../trainer/types";

// Controller state-machine tests over a plain TrainerSessionLike fake.
// Rewritten 2026-07-17 for the engine-only trainer (the incumbent client
// TrainerSession was removed from the codebase).
const LABELS = ["Spike", "Seizure", "LPD"];

function item(task: number, yStar: number, extra: Partial<Choice> = {}): Choice {
  return { task, mode: "skill", segId: 10 + task, s: 0.4, sSd: 0.1,
           yStar, margin: 0, info: {}, now: 0, ...extra };
}

function fakeSession(items: Choice[]): TrainerSessionLike {
  let i = 0;
  const snap = (): TaskSnapshot[] => LABELS.map((_, task) => ({
    task, mastered: false, skill: 0.1 * i, theta: 0, sd: 0.3,
    passMass: 0.2, trainability: 0.9,
  }));
  return {
    next: () => (i < items.length ? items[i++] : null),
    submit: () => {},
    snapshot: snap,
    allMastered: () => false,
    policy: { ellStars: LABELS.map(() => 0.3) },
  };
}

describe("TrainingController", () => {
  it("runs the binary question flow and records the L2 response", () => {
    const c = new TrainingController(
      fakeSession([item(1, 1), item(2, 0)]), LABELS, { total: 5 });
    expect(c.phase).toBe("question");
    c.markShown(0, "2026-07-17T10:00:00Z");
    c.answer(true, 850, "2026-07-17T10:00:01Z");
    expect(c.phase).toBe("result");
    expect(c.lastResult).toMatchObject({ correct: true, taskK: 1 });
    const [pt] = c.drainTrajectory();
    expect(pt).toMatchObject({
      pick: 1, yStar: 1, isCorrect: true, link: "binary", rt: 850,
      shownClientUtc: "2026-07-17T10:00:00Z",
      feedbackShown: "Correct — This is Seizure.",
    });
    c.continue();
    expect(c.phase).toBe("question");
    expect(c.item?.task).toBe(2);
  });

  it("runs the n-way flow via answerPick with task-axis coding", () => {
    const c = new TrainingController(
      fakeSession([item(2, 1, { link: "nway" } as Partial<Choice>)]),
      LABELS, { total: 5 });
    c.markShown(0);
    c.answerPick(2, 400);   // picked LPD; gold is Seizure
    expect(c.lastResult).toMatchObject({
      correct: false, nway: true, patternLabel: "Seizure",
      pickedLabel: "LPD",
    });
    const [pt] = c.drainTrajectory();
    expect(pt).toMatchObject({ pick: 2, yStar: 1, isCorrect: false,
                               link: "nway" });
    expect(pt.feedbackShown).toBe("Incorrect — This is Seizure, not LPD.");
  });

  it("finishes at the session bound and reports done", () => {
    const c = new TrainingController(
      fakeSession([item(0, 1), item(1, 0), item(2, 1)]), LABELS,
      { total: 2 });
    for (let k = 0; k < 2; k++) {
      c.markShown(0);
      c.answer(true, 100);
      c.continue();
    }
    expect(c.phase).toBe("done");
    expect(c.progress()).toEqual({ count: 2, total: 2 });
  });
});
