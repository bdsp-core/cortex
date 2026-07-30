import { describe, expect, it } from "vitest";
import {
  approximateFisherUtility,
  auditResearchShortlist,
  chooseResearchCandidate,
  fisherAugmentedShortlist,
  responseMutualInformation,
} from "../src/research_selector";
import { shortlistCandidates } from "../src/selector";
import type { Candidate, ProtocolSegment } from "../src/types";
import { artifact, nwayProfile, randomState, segments } from "./fixtures";

function candidatesFor(bank: readonly ProtocolSegment[]): Candidate[] {
  return bank.flatMap((segment) => [1, 2, 3, 4, 5, 6].map((askedK) => ({
    askedK,
    segmentIndex: segment.segmentIndex,
    segId: segment.segId,
    focalSignal: segment.sMean[askedK],
    focalSignalSd: segment.sSd[askedK],
  })));
}

describe("opt-in research selectors", () => {
  it("adds deterministic Fisher-screened candidates without dropping the baseline shortlist", () => {
    const state = randomState(48);
    const bank = segments(30);
    const candidates = candidatesFor(bank);
    const options = { fullScanLimit: 10, coarsePerTask: 3, entropyPerTask: 1, fisherPerTask: 2 };
    const baseline = shortlistCandidates(state, nwayProfile, artifact, candidates, bank, options);
    const challenger = fisherAugmentedShortlist(
      state, nwayProfile, artifact, candidates, bank, options,
    );
    expect(challenger.length).toBeGreaterThanOrEqual(baseline.length);
    expect(baseline.every((candidate) => challenger.includes(candidate))).toBe(true);
    expect(challenger.map((candidate) => `${candidate.askedK}/${candidate.segId}`)).toEqual(
      fisherAugmentedShortlist(
        state, nwayProfile, artifact, candidates, bank, options,
      ).map((candidate) => `${candidate.askedK}/${candidate.segId}`),
    );
  });

  it("never has more total-variance regret than its baseline-subset shortlist", () => {
    const state = randomState(64);
    const bank = segments(24);
    const audit = auditResearchShortlist(
      state, nwayProfile, artifact, candidatesFor(bank), bank,
      { fullScanLimit: 10, coarsePerTask: 2, entropyPerTask: 1, fisherPerTask: 2 },
    );
    expect(audit.challengerRegret).toBeLessThanOrEqual(audit.baselineRegret + 1e-12);
  });

  it("returns finite Fisher, mutual-information, and bias-weighted objectives", () => {
    const state = randomState(64);
    const bank = segments(4);
    const candidates = candidatesFor(bank);
    const candidate = candidates[0];
    expect(approximateFisherUtility(
      state, nwayProfile, artifact, candidate, bank[candidate.segmentIndex],
    )).toBeGreaterThanOrEqual(0);
    expect(responseMutualInformation(
      state, nwayProfile, artifact, candidate, bank[candidate.segmentIndex],
    )).toBeGreaterThanOrEqual(0);
    for (const objective of ["bias_weighted", "mutual_information"] as const) {
      const chosen = chooseResearchCandidate(
        state, nwayProfile, artifact, candidates, bank,
        { objective, skillWeight: 1, biasWeight: 1.5 },
      );
      expect(Number.isFinite(chosen.loss)).toBe(true);
    }
  });
});
