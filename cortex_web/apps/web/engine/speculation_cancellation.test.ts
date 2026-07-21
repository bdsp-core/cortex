import { describe, expect, it } from "vitest";

import {
  SpeculationCancelledError, speculationCancellationCheckpoint,
} from "./speculation_cancellation";

describe("speculation cancellation checkpoints", () => {
  it("rejects an already-cancelled branch before more work starts", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(speculationCancellationCheckpoint(controller.signal))
      .rejects.toBeInstanceOf(SpeculationCancelledError);
  });

  it("yields so an answer task can cancel before the next phase", async () => {
    const controller = new AbortController();
    globalThis.setTimeout(() => controller.abort(), 0);
    await expect(speculationCancellationCheckpoint(controller.signal))
      .rejects.toBeInstanceOf(SpeculationCancelledError);
  });
});
