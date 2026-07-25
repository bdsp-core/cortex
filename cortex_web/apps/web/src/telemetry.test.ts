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
    const init = calls[0][1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body).toMatchObject({ message: "boom", surface: "desktop", ua: "test-agent" });
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("attaches the session bearer token when one is available", () => {
    vi.stubGlobal("sessionStorage", {
      getItem: (key: string) => key === "cortex_token" ? "signed-token" : null,
    });
    reportClientError("authenticated boom");
    const init = calls[0][1] as RequestInit;
    expect(init.headers).toEqual({
      "Content-Type": "application/json",
      Authorization: "Bearer signed-token",
    });
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
