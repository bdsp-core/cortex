import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { TrainingRunner } from "./components/TrainingRunner";
import type { TrainerSessionLike } from "./trainingController";
import type { Choice, TaskSnapshot } from "../trainer/types";

// SSR probe: renders TrainingRunner's FIRST synchronous frame (seg not yet
// loaded — effects don't run under renderToStaticMarkup). Catches any
// synchronous render throw that would blank the SPA before its first paint.
// Rewritten 2026-07-17 for the engine-only trainer (the incumbent client
// trainer was removed): the session is a plain TrainerSessionLike fake.
const LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other/IIC"];

function snap(): TaskSnapshot[] {
  return LABELS.map((_, task) => ({
    task, mastered: false, skill: -1.0, theta: 0, sd: 0.3,
    passMass: 0.1, trainability: 0.5,
  }));
}

function fakeSession(items: Choice[]): TrainerSessionLike {
  let i = 0;
  return {
    next: () => (i < items.length ? items[i++] : null),
    submit: () => {},
    snapshot: snap,
    allMastered: () => false,
    policy: { ellStars: LABELS.map(() => 0.3) },
  };
}

const ITEM: Choice = {
  task: 6, mode: "skill", segId: 1, s: 0.4, sSd: 0.1, yStar: 6,
  margin: 0, info: {}, now: 0,
};

// A minimal Bundle stand-in: the first frame only reads inputs metadata.
const BUNDLE = {
  inputs: { taskLabels: LABELS, taskClasses:
    ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"] },
  seg: async () => null,
// eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

describe("TrainingRunner SSR first frame", () => {
  it("renders the first question frame without throwing", () => {
    const html = renderToStaticMarkup(createElement(TrainingRunner, {
      bundle: BUNDLE, session: fakeSession([ITEM]), trainingId: "t",
      labels: LABELS, onExit: () => {},
    }));
    expect(html.length).toBeGreaterThan(0);
  });

  it("renders the empty-session frame without throwing", () => {
    const html = renderToStaticMarkup(createElement(TrainingRunner, {
      bundle: BUNDLE, session: fakeSession([]), trainingId: "t",
      labels: LABELS, onExit: () => {},
    }));
    expect(html).toContain("No new segments");
  });
});
