import { describe, expect, it } from "vitest";

import { Bundle, type SessionBank } from "./bundle";

describe("server-stamped session profile bridge", () => {
  it("preserves the policy, per-domain cap, and full-bank terciles", () => {
    const bank = {
      bundleUrl: "/bundle/test",
      version: "test",
      eegScale: 4,
      specDbRange: [-10, 25],
      taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
      taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
      taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
      corrL: Array.from({ length: 7 }, (_, i) =>
        Array.from({ length: 7 }, (_, j) => Number(i === j))),
      corrT: Array.from({ length: 7 }, (_, i) =>
        Array.from({ length: 7 }, (_, j) => Number(i === j))),
      nParticles: 1200,
      perDomainCap: 60,
      terminationPolicy: "precision_v1",
      precisionBandEdges: Array.from({ length: 7 }, () => [-0.5, 0.5]),
      ellStar: new Array(7).fill(0),
      segments: [],
    } as SessionBank;

    const profile = Bundle.fromSessionBank(bank).inputs;
    expect(profile.terminationPolicy).toBe("precision_v1");
    expect(profile.nParticles).toBe(1200);
    expect(profile.perDomainCap).toBe(60);
    expect(profile.precisionBandEdges).toEqual(bank.precisionBandEdges);
  });

  it("keeps rendering metadata in Bundle and out of the compute payload", () => {
    const bank = {
      bundleUrl: "/bundle/test",
      version: "test",
      eegScale: 4,
      specDbRange: [-10, 25],
      taskCodes: ["spike"], taskLabels: ["Spike"],
      taskPatternWords: ["spike"], taskClasses: ["spike"],
      corrL: [[1]], ellStar: [0],
      segments: [{
        segId: 17, patternClass: "spike", testClass: "spike",
        applicableTaskIdx: [0], sMean: [0.4], sSd: [0.2],
        fsHz: 200, nCh: 2, nSamp: 400, channelNames: ["Fp1", "Fp2"],
        eeg: "seg/17.eeg", spec: "seg/17.spec", specShape: [20, 40],
      }],
    } as SessionBank;

    const bundle = Bundle.fromSessionBank(bank);
    expect(bundle.manifest.segments[0].eeg).toBe("seg/17.eeg");
    expect(bundle.manifest.segments[0].spec).toBe("seg/17.spec");
    expect(bundle.computeInputs.segments[0]).toEqual({
      segId: 17, applicableTaskIdx: [0], sMean: [0.4], sSd: [0.2],
    });
    expect(bundle.computeInputs).toBe(bundle.computeInputs);
  });
});
