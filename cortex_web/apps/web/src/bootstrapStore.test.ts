import { describe, expect, it } from "vitest";
import { bootstrapOnce, invalidateBootstrap } from "./bootstrapStore";
import type { BootstrapData } from "./api";

const EMPTY: BootstrapData = {
  dashboard: null, trajectories: null, regimen: null,
  activity: null, session: null, cohorts: null,
};

describe("bootstrapStore", () => {
  it("shares one fetch across concurrent consumers", async () => {
    invalidateBootstrap();
    let calls = 0;
    const fetcher = () => { calls++; return Promise.resolve(EMPTY); };
    const [a, b, c] = await Promise.all([
      bootstrapOnce(fetcher), bootstrapOnce(fetcher), bootstrapOnce(fetcher),
    ]);
    expect(calls).toBe(1);
    expect(a).toBe(b);
    expect(b).toBe(c);
  });

  it("re-fetches after invalidation", async () => {
    invalidateBootstrap();
    let calls = 0;
    const fetcher = () => { calls++; return Promise.resolve(EMPTY); };
    await bootstrapOnce(fetcher);
    invalidateBootstrap();
    await bootstrapOnce(fetcher);
    expect(calls).toBe(2);
  });

  it("does not cache a rejected fetch", async () => {
    invalidateBootstrap();
    let calls = 0;
    const failing = () => { calls++; return Promise.reject(new Error("net")); };
    await expect(bootstrapOnce(failing)).rejects.toThrow("net");
    // the rejection cleared the cache synchronously via its catch handler;
    // give the microtask queue one turn to be safe
    await Promise.resolve();
    const ok = () => { calls++; return Promise.resolve(EMPTY); };
    await expect(bootstrapOnce(ok)).resolves.toBe(EMPTY);
    expect(calls).toBe(2);
  });

  it("a later invalidation does not clobber a newer in-flight fetch", async () => {
    invalidateBootstrap();
    // first fetch fails slowly; a second (fresh) fetch starts after
    // invalidation and must not be cleared by the first one's rejection
    let rejectFirst: (e: Error) => void = () => {};
    const slowFail = () => new Promise<BootstrapData>((_res, rej) => { rejectFirst = rej; });
    const p1 = bootstrapOnce(slowFail);
    p1.catch(() => { /* consumed */ });
    invalidateBootstrap();
    const p2 = bootstrapOnce(() => Promise.resolve(EMPTY));
    rejectFirst(new Error("stale"));
    await Promise.resolve();
    // p2 is still the cached one: a third call joins it without a new fetch
    let calls = 0;
    await bootstrapOnce(() => { calls++; return Promise.resolve(EMPTY); });
    expect(calls).toBe(0);
    await expect(p2).resolves.toBe(EMPTY);
  });
});
