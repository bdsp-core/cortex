// Deploy-race recovery for the code-split SPA. Navigating to training setup,
// cohorts, the mobile surface, or a locale triggers a dynamic import of a
// content-hashed chunk (e.g. trainingSetup-<hash>.js). Each deploy rebuilds
// dist/ under fresh hashes and does not keep the previous build's files, so a
// tab opened BEFORE a deploy asks the server for a chunk that no longer
// exists. Caddy's SPA fallback answers that /assets/* miss with index.html
// (200 text/html); the browser rejects HTML as a module and the lazy import
// rejects with "Failed to fetch dynamically imported module" — which surfaces
// as the ErrorBoundary crash screen. A manual refresh fixes it (fresh
// index.html -> current hashes); this does that refresh automatically before
// the user ever sees the error.
//
// Vite fires `vite:preloadError` on window when a dynamic import's preload
// fails. We reload once to pull the current index.html. A sessionStorage
// stamp guards against a reload LOOP: if we already reloaded moments ago and
// the import STILL fails, it is not a stale tab (genuine outage / broken
// deploy) — we stand down and let Vite throw so the ErrorBoundary shows AND
// reports the real failure. The stamp is a timestamp, not a boolean, so a
// SECOND deploy during the same long-lived tab session recovers again (these
// sittings are long and the test/train washout invites leaving the tab open).

const STAMP_KEY = "cortex:stale-asset-reloaded-at";
// If a reload does not fix it within this window, stop looping and surface.
const LOOP_GUARD_MS = 15_000;

/**
 * Reload once when a lazy, content-hashed asset may have disappeared during
 * a deploy. Shared by Vite chunk failures and the separately loaded engine
 * worker. Returns true when recovery took ownership of the failure.
 */
export function recoverFromStaleAsset(now: () => number = Date.now): boolean {
  let last = 0;
  try { last = Number(sessionStorage.getItem(STAMP_KEY)) || 0; }
  catch { /* sessionStorage can throw in private mode — treat as no stamp */ }
  const t = now();
  if (t - last < LOOP_GUARD_MS) return false;
  try { sessionStorage.setItem(STAMP_KEY, String(t)); }
  catch { /* private mode: skip the stamp, still worth one reload attempt */ }
  window.location.reload();
  return true;
}

export function installPreloadRecovery(now: () => number = Date.now): void {
  window.addEventListener("vite:preloadError", (event) => {
    if (!recoverFromStaleAsset(now)) {
      // Already reloaded recently and it still failed: not a stale tab. Let
      // Vite throw so the ErrorBoundary surfaces + reports the real failure.
      return;
    }
    event.preventDefault();   // pre-empt Vite's re-throw; we recover by reloading
  });
}
