import { describe, expect, it } from "vitest";

import type { EngineInputs, SegmentMeta } from "../engine/types";
import { answerIsCorrect } from "./tasks";

const inputs = {
  taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
  taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
  taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
} as EngineInputs;

function segment(patternClass: string, spikeSignal = 1): SegmentMeta {
  return {
    segId: 1, patternClass, fsHz: 1, nCh: 1, nSamp: 1,
    channelNames: [], eeg: "", spec: "", sMean: [spikeSignal, 0, 0, 0, 0, 0, 0],
    sSd: new Array(7).fill(0),
  };
}

describe("gold response scoring", () => {
  it("does not confuse a wrong-category response with the asked-task marginal", () => {
    const lpd = segment("lpd");
    expect(answerIsCorrect(inputs, lpd, 1, 2)).toBe(true);
    expect(answerIsCorrect(inputs, lpd, 1, 1)).toBe(false);
  });

  it("scores spike yes/no against the signal sign", () => {
    expect(answerIsCorrect(inputs, segment("spike", 1), 0, 0)).toBe(true);
    expect(answerIsCorrect(inputs, segment("other", -1), 0, 7)).toBe(true);
  });
});
