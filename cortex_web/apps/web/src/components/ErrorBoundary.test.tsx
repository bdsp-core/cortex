import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";
import { resetTelemetryForTests } from "../telemetry";

// Boundary-caught render errors never reach the global window "error" hook in
// production builds, so the boundary must report explicitly — this guards the
// componentDidCatch → reportClientError wiring (logic-level: the instance is
// driven directly, no DOM render).
describe("ErrorBoundary crash reporting", () => {
  const calls: unknown[][] = [];

  beforeEach(() => {
    calls.length = 0;
    resetTelemetryForTests();
    vi.stubGlobal("fetch", (...args: unknown[]) => {
      calls.push(args);
      return Promise.resolve(new Response("{}"));
    });
    vi.stubGlobal("navigator", { userAgent: "test-agent" });
    vi.stubGlobal("window", { location: { pathname: "/train" } });
  });

  afterEach(() => vi.unstubAllGlobals());

  it("componentDidCatch posts the error to /api/client-error", () => {
    const eb = new ErrorBoundary({ children: null });
    Object.assign(eb, { setState: () => {} });   // detached instance — no React tree
    const err = new Error("render exploded");
    eb.componentDidCatch(err, { componentStack: "\n    at Broken" });
    expect(calls).toHaveLength(1);
    expect(calls[0][0]).toBe("/api/client-error");
    const body = JSON.parse((calls[0][1] as RequestInit).body as string);
    expect(body).toMatchObject({
      message: "render exploded",
      surface: "error-boundary",
      url: "/train",
    });
    expect(body.stack).toContain("at Broken");
  });
});
