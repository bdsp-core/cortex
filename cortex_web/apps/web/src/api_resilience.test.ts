// G4 robustness: submitResults must still DELIVER the result when the durable
// localStorage copy can't be written (quota / private mode) — the direct POST
// carries it from memory. Previously delivery was routed through
// flushPendingResults(), which re-reads localStorage, so a failed write meant
// the result was never sent. The storage/network globals don't exist in the
// node test env, so we install minimal stubs and dynamic-import the module
// after they're in place.

import { afterEach, describe, expect, it, vi } from "vitest";

function memStorage(): Storage {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => (m.has(k) ? m.get(k)! : null),
    setItem: (k: string, v: string) => void m.set(k, String(v)),
    removeItem: (k: string) => void m.delete(k),
    clear: () => m.clear(),
    key: (i: number) => Array.from(m.keys())[i] ?? null,
    get length() { return m.size; },
  } as Storage;
}

function install(opts: { localStorageThrows: boolean; fetchStatus: number }) {
  const g = globalThis as any;
  g.sessionStorage = memStorage();
  g.sessionStorage.setItem("cortex_token", "test-token");
  const ls = memStorage();
  if (opts.localStorageThrows) {
    ls.setItem = () => { throw new DOMException("QuotaExceededError"); };
  }
  g.localStorage = ls;
  const calls: string[] = [];
  g.fetch = vi.fn(async (url: string) => {
    calls.push(url);
    return new Response("{}", { status: opts.fetchStatus });
  });
  return calls;
}

afterEach(() => {
  const g = globalThis as any;
  delete g.sessionStorage; delete g.localStorage; delete g.fetch;
  vi.resetModules();
});

describe("submitResults storage resilience", () => {
  it("still delivers when localStorage.setItem throws", async () => {
    const calls = install({ localStorageThrows: true, fetchStatus: 200 });
    const api = await import("./api");
    const delivered = await api.submitResults("sess-1", { ok: 1 }, "done", 3);
    expect(delivered).toBe(true);
    expect(calls.some((u) => u.endsWith("/api/results"))).toBe(true);
  });

  it("returns false and retains when the POST fails (storage OK)", async () => {
    const calls = install({ localStorageThrows: false, fetchStatus: 500 });
    const api = await import("./api");
    const delivered = await api.submitResults("sess-2", { ok: 1 }, "done", 3);
    expect(delivered).toBe(false);
    expect(calls.some((u) => u.endsWith("/api/results"))).toBe(true);
    // The undelivered result is retained for the next flush.
    expect((globalThis as any).localStorage.getItem("cortex_pending_results"))
      .toContain("sess-2");
  });
});
