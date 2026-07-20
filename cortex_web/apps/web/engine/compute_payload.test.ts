import { describe, expect, it } from "vitest";

import {
  computePayloadTransferables, packComputeInputs, packedSegmentBytes,
  unpackComputeInputs,
} from "./compute_payload";
import { precisionGoldenInputs } from "./__testdata__/precision_fixture";

describe("compact worker calculation payload", () => {
  it("round-trips the exact engine inputs without rendering fields", () => {
    const inputs = precisionGoldenInputs();
    const packed = packComputeInputs(inputs);
    expect(unpackComputeInputs(packed)).toEqual(inputs);
    expect(computePayloadTransferables(packed)).toHaveLength(4);
    expect(packedSegmentBytes(packed)).toBe(
      inputs.segments.length * (8 + 4 + 8 * inputs.taskCodes.length * 2),
    );
    expect("eeg" in (packed.segments as unknown as Record<string, unknown>)).toBe(false);
  });

  it("retains omitted-applicability semantics as all tasks", () => {
    const inputs = precisionGoldenInputs();
    delete inputs.segments[0].applicableTaskIdx;
    expect(unpackComputeInputs(packComputeInputs(inputs)).segments[0].applicableTaskIdx)
      .toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
});
