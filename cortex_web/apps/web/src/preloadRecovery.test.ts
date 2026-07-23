import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installPreloadRecovery, recoverFromStaleAsset } from "./preloadRecovery";

// A stale tab whose deploy rotated the chunk hashes must auto-refresh once,
// but a genuine outage must NOT reload-loop. Drive the window listener
// directly with an injected clock (logic-level, no DOM), mirroring the
// telemetry test's global-stub style.
describe("installPreloadRecovery", () => {
  let listener: ((e: unknown) => void) | null;
  let store: Record<string, string>;
  let reloads: number;
  let clock: number;

  function firePreloadError(): boolean {
    let prevented = false;
    listener?.({
      type: "vite:preloadError",
      payload: new Error("Failed to fetch dynamically imported module"),
      preventDefault() { prevented = true; },
    });
    return prevented;
  }

  beforeEach(() => {
    listener = null;
    store = {};
    reloads = 0;
    clock = 1_000_000;
    vi.stubGlobal("window", {
      addEventListener: (type: string, cb: (e: unknown) => void) => {
        if (type === "vite:preloadError") listener = cb;
      },
      location: { reload: () => { reloads++; } },
    });
    vi.stubGlobal("sessionStorage", {
      getItem: (k: string) => (k in store ? store[k] : null),
      setItem: (k: string, v: string) => { store[k] = v; },
    });
    installPreloadRecovery(() => clock);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("reloads once on the first preload error and stamps the recovery", () => {
    const prevented = firePreloadError();
    expect(prevented).toBe(true);          // Vite's throw is pre-empted
    expect(reloads).toBe(1);
    expect(store["cortex:stale-asset-reloaded-at"]).toBe(String(clock));
  });

  it("stands down on a repeat within the guard window (no reload loop)", () => {
    firePreloadError();                    // reload #1
    clock += 5_000;                        // still inside the 15s guard
    const prevented = firePreloadError();
    expect(prevented).toBe(false);         // let it surface to the ErrorBoundary
    expect(reloads).toBe(1);               // did NOT reload again
  });

  it("recovers again for a fresh incident after the guard window", () => {
    firePreloadError();                    // reload #1 (first deploy)
    clock += 20_000;                       // past the guard: a SECOND deploy, same tab
    const prevented = firePreloadError();
    expect(prevented).toBe(true);
    expect(reloads).toBe(2);
  });

  it("still attempts one reload when sessionStorage is unavailable", () => {
    vi.stubGlobal("sessionStorage", {
      getItem: () => { throw new Error("private mode"); },
      setItem: () => { throw new Error("private mode"); },
    });
    const prevented = firePreloadError();
    expect(prevented).toBe(true);
    expect(reloads).toBe(1);
  });

  it("exposes the same guarded recovery to non-Vite lazy assets", () => {
    expect(recoverFromStaleAsset(() => clock)).toBe(true);
    expect(recoverFromStaleAsset(() => clock + 5_000)).toBe(false);
    expect(recoverFromStaleAsset(() => clock + 20_000)).toBe(true);
    expect(reloads).toBe(2);
  });
});
