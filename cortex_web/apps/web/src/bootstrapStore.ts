// Shared fetch-once cache for GET /api/bootstrap. The dashboard mounts
// several independent consumers at once (App's washout/resume status, the
// Shell's DashboardSurface, the cohort-invite banner); each calls
// bootstrapOnce() and the first one triggers the single network round-trip —
// the rest share the same promise. App invalidates on every dashboard EXIT
// (phase-effect cleanup), so each dashboard entry re-fetches fresh while tab
// hops within the Shell reuse the cached payload.
//
// A rejected fetch clears the cache immediately: every consumer's catch
// falls back to its empty state (same as a failed standalone call), and the
// next mount retries instead of caching the failure.

import * as api from "./api";

let inflight: Promise<api.BootstrapData> | null = null;

export function bootstrapOnce(
  fetcher: () => Promise<api.BootstrapData> = api.bootstrap,
): Promise<api.BootstrapData> {
  if (!inflight) {
    const p = fetcher();
    inflight = p;
    p.catch(() => { if (inflight === p) inflight = null; });
  }
  return inflight;
}

export function invalidateBootstrap(): void {
  inflight = null;
}
