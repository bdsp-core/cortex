import { describe, expect, it } from "vitest";
import { performance } from "node:perf_hooks";
import { fisherAugmentedShortlist } from "../src/research_selector";
import { chooseCandidate, shortlistCandidates } from "../src/selector";
import type { Candidate, ProtocolSegment } from "../src/types";
import {
  artifact, ensembleArtifact, integratedNwayProfile, K, nwayProfile, randomState,
} from "./fixtures";

describe("full-bank protocol performance", () => {
  it.runIf(process.env.CORTEX_NWAY_BENCH === "1")(
    "keeps the 35k-bank shortlist and exact categorical refinement bounded",
    () => {
      const segmentCount = 35_000;
      const bank: ProtocolSegment[] = new Array(segmentCount);
      const candidates: Candidate[] = [];
      for (let i = 0; i < segmentCount; i += 1) {
        const sMean = Array.from({ length: K }, (_, k) => ((i * 17 + k * 13) % 401 - 200) / 80);
        const sSd = Array.from({ length: K }, (_, k) => 0.02 + ((i + k) % 11) / 100);
        bank[i] = {
          segmentIndex: i, segId: 50_000 + i, sMean, sSd,
          applicableTaskIdx: [1, 2, 3, 4, 5, 6],
        };
        const askedK = 1 + (i % 6);
        candidates.push({
          askedK, segmentIndex: i, segId: bank[i].segId,
          focalSignal: sMean[askedK], focalSignalSd: sSd[askedK],
        });
      }
      const state = randomState(1200);
      const options = { fullScanLimit: 512, coarsePerTask: 32, entropyPerTask: 12 };
      const started = performance.now();
      const shortlist = shortlistCandidates(
        state, nwayProfile, artifact, candidates, bank, options,
      );
      const shortlistedAt = performance.now();
      const chosen = chooseCandidate(state, nwayProfile, artifact, candidates, bank, options);
      const completedAt = performance.now();
      // At most one coarse and one min-sd representative per bin plus the
      // entropy quota, per six IIIC tasks. Duplicates can only lower this.
      expect(shortlist.length).toBeLessThanOrEqual(6 * (2 * 32 + 12));
      expect(Number.isFinite(chosen.loss)).toBe(true);
      expect(shortlistedAt - started).toBeLessThan(5_000);
      expect(completedAt - shortlistedAt).toBeLessThan(20_000);
    },
    30_000,
  );

  it.runIf(process.env.CORTEX_NWAY_BENCH === "1")(
    "keeps the integrated ensemble/Fisher 35k-bank selection bounded",
    () => {
      const segmentCount = 35_000;
      const bank: ProtocolSegment[] = new Array(segmentCount);
      const candidates: Candidate[] = [];
      for (let i = 0; i < segmentCount; i += 1) {
        const sMean = Array.from({ length: K }, (_, k) => ((i * 17 + k * 13) % 401 - 200) / 80);
        const sSd = Array.from({ length: K }, (_, k) => 0.02 + ((i + k) % 11) / 100);
        bank[i] = {
          segmentIndex: i, segId: 90_000 + i, sMean, sSd,
          applicableTaskIdx: [1, 2, 3, 4, 5, 6],
        };
        const askedK = 1 + (i % 6);
        candidates.push({
          askedK, segmentIndex: i, segId: bank[i].segId,
          focalSignal: sMean[askedK], focalSignalSd: sSd[askedK],
        });
      }
      const state = randomState(1200);
      const options = {
        fullScanLimit: 512, coarsePerTask: 32, entropyPerTask: 12, fisherPerTask: 8,
      };
      const started = performance.now();
      const shortlist = fisherAugmentedShortlist(
        state, integratedNwayProfile, ensembleArtifact, candidates, bank, options,
      );
      const screenedAt = performance.now();
      const chosen = chooseCandidate(
        state, integratedNwayProfile, ensembleArtifact, shortlist, bank,
        { ...options, fullScanLimit: Number.POSITIVE_INFINITY },
      );
      const completedAt = performance.now();
      expect(shortlist.length).toBeLessThanOrEqual(6 * (2 * 32 + 12 + 8));
      expect(Number.isFinite(chosen.loss)).toBe(true);
      expect(screenedAt - started).toBeLessThan(20_000);
      expect(completedAt - screenedAt).toBeLessThan(20_000);
    },
    60_000,
  );
});
