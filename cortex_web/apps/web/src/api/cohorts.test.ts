import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, String(value)); },
    removeItem: (key) => { values.delete(key); },
    clear: () => values.clear(),
    key: (index) => Array.from(values.keys())[index] ?? null,
    get length() { return values.size; },
  } as Storage;
}

beforeEach(() => {
  Object.assign(globalThis, {
    localStorage: memoryStorage(),
    sessionStorage: memoryStorage(),
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  delete (globalThis as { localStorage?: Storage }).localStorage;
  delete (globalThis as { sessionStorage?: Storage }).sessionStorage;
});

describe("cohort API boundary", () => {
  it("encodes identifiers and preserves the requested performance window", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ members: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const core = await import("./core");
    const cohorts = await import("./cohorts");
    core.setToken("cohort-token");

    await cohorts.getCohortPerformance("team/alpha", 30);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/cohorts/team%2Falpha/performance?days=30");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer cohort-token");
  });

  it("sends cohort mutations as explicit JSON requests", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ cohortId: "c-1", name: "Reviewers" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const cohorts = await import("./cohorts");

    await cohorts.createCohort("Reviewers");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/cohorts");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ name: "Reviewers" }));
  });
});
