import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { reportClientError, resetTelemetryForTests } from "./telemetry";

describe("reportClientError", () => {
  const calls: unknown[][] = [];

  beforeEach(() => {
    calls.length = 0;
    resetTelemetryForTests();
    vi.stubGlobal("fetch", (...args: unknown[]) => {
      calls.push(args);
      return Promise.resolve(new Response("{}"));
    });
    vi.stubGlobal("navigator", { userAgent: "test-agent" });
    vi.stubGlobal("window", { location: { pathname: "/" } });
  });

  afterEach(() => vi.unstubAllGlobals());

  it("posts a report once per distinct message", () => {
    reportClientError("boom", "at line 1", "desktop");
    reportClientError("boom", "at line 1", "desktop");
    expect(calls).toHaveLength(1);
    const body = JSON.parse((calls[0][1] as RequestInit).body as string);
    expect(body).toMatchObject({ message: "boom", surface: "desktop", ua: "test-agent" });
  });

  it("caps total reports per page load", () => {
    for (let i = 0; i < 20; i++) reportClientError(`err ${i}`);
    expect(calls).toHaveLength(5);
  });

  it("truncates oversized fields to the server caps", () => {
    reportClientError("x".repeat(2000), "y".repeat(9000));
    const body = JSON.parse((calls[0][1] as RequestInit).body as string);
    expect(body.message).toHaveLength(500);
    expect(body.stack).toHaveLength(4000);
  });

  it("never throws, even when fetch is broken", () => {
    vi.stubGlobal("fetch", () => { throw new Error("no network"); });
    expect(() => reportClientError("still fine")).not.toThrow();
  });
});
