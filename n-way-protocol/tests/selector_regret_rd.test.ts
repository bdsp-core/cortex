import { describe, expect, it } from "vitest";
import { auditResearchShortlist } from "../src/research_selector";
import type { Candidate, ProtocolSegment } from "../src/types";
import { artifact, K, nwayProfile, randomState } from "./fixtures";

function syntheticBank(segmentCount: number): { bank: ProtocolSegment[]; candidates: Candidate[] } {
  const bank: ProtocolSegment[] = [];
  const candidates: Candidate[] = [];
  for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
    const sMean = Array.from(
      { length: K },
      (_, k) => ((segmentIndex * 19 + k * 31) % 503 - 251) / 95,
    );
    const sSd = Array.from(
      { length: K },
      (_, k) => 0.02 + ((segmentIndex * 7 + k * 3) % 17) / 100,
    );
    const segment: ProtocolSegment = {
      segmentIndex,
      segId: 80_000 + segmentIndex,
      sMean,
      sSd,
      applicableTaskIdx: [1, 2, 3, 4, 5, 6],
    };
    bank.push(segment);
    for (let askedK = 1; askedK <= 6; askedK += 1) {
      candidates.push({
        askedK,
        segmentIndex,
        segId: segment.segId,
        focalSignal: sMean[askedK],
        focalSignalSd: sSd[askedK],
      });
    }
  }
  return { bank, candidates };
}

describe("selector regret R&D", () => {
  it.runIf(process.env.CORTEX_NWAY_SELECTOR_RD === "1")(
    "compares the baseline and Fisher-augmented shortlists to exhaustive scoring",
    () => {
      const { bank, candidates } = syntheticBank(96);
      const rows = [];
      for (let stateIndex = 0; stateIndex < 8; stateIndex += 1) {
        const state = randomState(192);
        const scale = 0.65 + stateIndex * 0.1;
        for (let i = 0; i < state.t.length; i += 1) {
          state.t[i] = state.t[i] * scale + (stateIndex - 3.5) * 0.08;
          state.l[i] = state.l[i] * (1.25 - stateIndex * 0.06);
        }
        rows.push(auditResearchShortlist(
          state, nwayProfile, artifact, candidates, bank,
          {
            fullScanLimit: 10,
            coarsePerTask: 8,
            entropyPerTask: 3,
            fisherPerTask: 4,
          },
        ));
      }
      const summary = {
        states: rows.length,
        meanBaselineRegret: rows.reduce((sum, row) => sum + row.baselineRegret, 0) / rows.length,
        meanChallengerRegret: rows.reduce((sum, row) => sum + row.challengerRegret, 0) / rows.length,
        p95BaselineRegret: rows.map((row) => row.baselineRegret).sort((a, b) => a - b)[
          Math.floor(0.95 * (rows.length - 1))
        ],
        p95ChallengerRegret: rows.map((row) => row.challengerRegret).sort((a, b) => a - b)[
          Math.floor(0.95 * (rows.length - 1))
        ],
        exactChoiceRecallBaseline: rows.filter((row) => row.baselineRegret <= 1e-12).length / rows.length,
        exactChoiceRecallChallenger: rows.filter((row) => row.challengerRegret <= 1e-12).length / rows.length,
        meanBaselineShortlist: rows.reduce((sum, row) => sum + row.baselineShortlistSize, 0) / rows.length,
        meanChallengerShortlist: rows.reduce((sum, row) => sum + row.challengerShortlistSize, 0) / rows.length,
      };
      console.log(`SELECTOR_REGRET_RD ${JSON.stringify(summary)}`);
      expect(summary.meanChallengerRegret).toBeLessThanOrEqual(summary.meanBaselineRegret + 1e-12);
      expect(summary.p95ChallengerRegret).toBeLessThanOrEqual(summary.p95BaselineRegret + 1e-12);
    },
    120_000,
  );
});
