import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { TrainingRunner } from "./components/TrainingRunner";
import { ArrayBank, TrainerSession, buildFilters } from "../trainer/session";
import type { FilterParams } from "../trainer/filter";
import { buildTrainerBank, cutScores, seedClouds } from "./trainerBank";

// SSR probe: renders TrainingRunner's FIRST synchronous frame (seg not yet
// loaded — effects don't run under renderToStaticMarkup). Catches any
// synchronous render throw that would blank the SPA before its first paint.
const PARAMS: FilterParams = {
  alphaT: 0.097, alphaSigma: 0.047, sigmaInf: 0.4, qT: 0.05, qSigma: 0.02, rho: 0.5, rule: "soft",
};
const ELLSTAR = [0.3, 0.155, 0.36, 0.36, 0.36, 0.36, 0.36];
const LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other/IIC"];

function session(segments: unknown[]) {
  const { ellStars, sigmaStars, sigmaInf } = cutScores(ELLSTAR);
  const clouds = seedClouds(ellStars.map(() => 0.0), 200, 1);
  const filters = buildFilters(clouds, PARAMS, sigmaInf, ellStars, { seed: 7, useMixture: false });
  return new TrainerSession(filters, ellStars, sigmaStars,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    new ArrayBank(buildTrainerBank({ segments: segments as any })), { seed: 7 });
}

// a Bundle stand-in whose media never resolves (effect is a no-op in SSR anyway)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockBundle: any = { segment: () => new Promise(() => {}) };

function render(sess: TrainerSession) {
  return renderToStaticMarkup(createElement(TrainingRunner, {
    bundle: mockBundle, session: sess, trainingId: "t1", labels: LABELS, onExit: () => {},
  }));
}

describe("TrainingRunner first-frame SSR (blank-screen probe)", () => {
  it("QUESTION frame (tiny draw) renders without throwing", () => {
    const segments = [
      { segId: 102, patternClass: "gpd", applicableTaskIdx: [1, 2, 3, 4, 5, 6],
        sMean: [0, -0.5, -0.3, 0.8, -0.4, -0.2, 0.1], sSd: [0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4] },
    ];
    let html = "";
    expect(() => { html = render(session(segments)); }).not.toThrow();
    expect(html).toContain("In training");
  });

  it("DONE frame (empty draw) renders without throwing", () => {
    let html = "";
    expect(() => { html = render(session([])); }).not.toThrow();
    expect(html).toContain("No new segments");
  });
});
