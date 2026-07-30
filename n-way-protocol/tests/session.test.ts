import { describe, expect, it } from "vitest";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { advanceProtocol, makeLedger } from "../src/session";
import { artifact, manualState, nwayProfile, segments } from "./fixtures";

describe("protocol direct-evidence bookkeeping", () => {
  it("updates all relevant posterior coordinates but counts only the asked domain", () => {
    const bank = segments(2);
    const state = manualState(32);
    const ledger = makeLedger(state.K, bank.map((segment) => segment.segId));
    const chosen = {
      askedK: 2,
      segmentIndex: 1,
      segId: bank[1].segId,
      focalSignal: bank[1].sMean[2],
      focalSignalSd: bank[1].sSd[2],
    };
    const diagnostic = advanceProtocol({
      state,
      ledger,
      profile: nwayProfile,
      artifact,
      segments: bank,
      chosen,
      rawPick: 5,
      trialIndex: 0,
      rng: new Rng(90),
      params: {
        essThresholdFraction: 0,
        nMhSteps: 0,
        proposalScale: 0,
        bandEdges: Array.from({ length: state.K }, () => [-0.5, 0.5] as const),
      },
    });
    expect(diagnostic.rawPick).toBe(5);
    expect(diagnostic.matchedAskedTask).toBe(false);
    expect(diagnostic.responseKind).toBe("categorical_f1");
    expect(diagnostic.nPerTask).toEqual([0, 0, 1, 0, 0, 0, 0]);
    expect(diagnostic.bandAdministered.flat().reduce((a, b) => a + b, 0)).toBe(1);
    expect(ledger.remainingSegmentIds.has(chosen.segId)).toBe(false);
  });
});

