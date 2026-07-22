import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  TRAJECTORY_STATUS_DELAY_MS, TrajectoryOptimizationStatus,
} from "./TrajectoryOptimizationStatus";

describe("long next-question wait status", () => {
  it("uses a delayed, accessible teal trajectory message", () => {
    const html = renderToStaticMarkup(createElement(TrajectoryOptimizationStatus));

    expect(TRAJECTORY_STATUS_DELAY_MS).toBe(1_000);
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain("Optimizing Question Trajectory");
    expect(html).toContain("Preparing the next question");
    expect(html).toContain("var(--teal-deep)");
  });
});
