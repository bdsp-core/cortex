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

describe("API core boundary", () => {
  it("keeps credentials tab-scoped and clears all session identity on logout", async () => {
    const core = await import("./core");
    core.setToken("token-1");
    core.setDisplayName("Reader");
    sessionStorage.setItem("cortex-welcome-seen", "1");

    expect(core.isAuthed()).toBe(true);
    expect(core.getDisplayName()).toBe("Reader");
    expect(localStorage.getItem("cortex_token")).toBeNull();

    core.logout();
    expect(core.isAuthed()).toBe(false);
    expect(core.getDisplayName()).toBeNull();
    expect(sessionStorage.getItem("cortex-welcome-seen")).toBeNull();
  });

  it("adds the bearer token and fails closed by clearing it on 401", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer token-2");
      return new Response(JSON.stringify({ error: "expired" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const core = await import("./core");
    core.setToken("token-2");

    await expect(core.authedFetch("/api/private")).rejects.toMatchObject({
      status: 401,
      message: "expired",
    });
    expect(core.getToken()).toBeNull();
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});

describe("session-expiry announcement", () => {
  function respond(status: number) {
    return vi.fn(async () => new Response(
      JSON.stringify(status === 200 ? { ok: true } : { error: "expired" }),
      { status, headers: { "Content-Type": "application/json" } },
    ));
  }

  it("announces once when an authenticated call is rejected", async () => {
    vi.stubGlobal("fetch", respond(401));
    const core = await import("./core");
    core.setToken("t0");
    const seen = vi.fn();
    core.onSessionExpired(seen);

    await expect(core.authedFetch("/api/dashboard")).rejects.toThrow();
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("stays quiet for a 401 on a call that carried no token", async () => {
    vi.stubGlobal("fetch", respond(401));
    const core = await import("./core");
    const seen = vi.fn();
    core.onSessionExpired(seen);

    await expect(core.authedFetch("/api/dashboard")).rejects.toThrow();
    expect(seen).not.toHaveBeenCalled();
  });

  it("announces once even when several requests fail together", async () => {
    vi.stubGlobal("fetch", respond(401));
    const core = await import("./core");
    core.setToken("t0");
    const seen = vi.fn();
    core.onSessionExpired(seen);

    await Promise.allSettled([
      core.authedFetch("/api/dashboard"),
      core.authedFetch("/api/history"),
      core.authedFetch("/api/regimen"),
    ]);
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("stays quiet on a successful call", async () => {
    vi.stubGlobal("fetch", respond(200));
    const core = await import("./core");
    core.setToken("t0");
    const seen = vi.fn();
    core.onSessionExpired(seen);

    await core.authedFetch("/api/dashboard");
    expect(seen).not.toHaveBeenCalled();
    expect(core.getToken()).toBe("t0");
  });

  it("still delivers the API error when a listener throws", async () => {
    vi.stubGlobal("fetch", respond(401));
    const core = await import("./core");
    core.setToken("t0");
    core.onSessionExpired(() => { throw new Error("listener blew up"); });

    await expect(core.authedFetch("/api/dashboard"))
      .rejects.toMatchObject({ status: 401, message: "expired" });
  });

  it("stops notifying after unsubscribe", async () => {
    vi.stubGlobal("fetch", respond(401));
    const core = await import("./core");
    core.setToken("t0");
    const seen = vi.fn();
    core.onSessionExpired(seen)();       // subscribe, then immediately drop it

    await expect(core.authedFetch("/api/dashboard")).rejects.toThrow();
    expect(seen).not.toHaveBeenCalled();
  });
});
