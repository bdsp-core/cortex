// The mobile/desktop module boundary, enforced (same spirit as the repo's D1
// engine/deployment separation): the phone surface may only import from its
// own tree plus an explicit shared allowlist, and no desktop module may
// import from mobile/ (main.tsx's boot branch is the single exception).
// This is what keeps "future adjustments clean": a change to Shell/Viewer
// can never break the phone surface, and phone layout tweaks can never leak
// into the desktop-layout web app.

import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, relative } from "node:path";

const mobileDir = dirname(fileURLToPath(import.meta.url));
const srcDir = join(mobileDir, "..");

// Modules the mobile tree may import from OUTSIDE src/mobile/. Everything
// else (App.tsx, Shell, Viewer, charts, engine client…) is desktop-only by
// contract. AuthFlow is deliberately shared: auth is one flow with subtle
// logic (deep links, bounce polling, did-you-mean) that must never fork.
const SHARED_ALLOW = new Set([
  "../api",
  "../deepLink",
  "../device",
  "../i18n/LanguageProvider",
  "../theme/ThemeProvider",
  "../../ui/theme",
  "../components/AuthFlow",
  "../components/charts",   // self-contained data-viz (Heatmap on the home)
  "../components/CohortInviteBanner",  // shared floating invite banner
  "../components/MilestoneBanner",     // shared floating milestone banner
  "../profileFields",
]);

function importsOf(path: string): string[] {
  const src = readFileSync(path, "utf8");
  return [...src.matchAll(/(?:from|import)\s+["']([^"']+)["']/g)].map((m) => m[1]);
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      return name === "mobile" || name === "__snapshots__" ? [] : walk(p);
    }
    return /\.(ts|tsx)$/.test(name) ? [p] : [];
  });
}

describe("mobile/desktop module boundary", () => {
  it("mobile imports only its own tree + the shared allowlist", () => {
    const files = readdirSync(mobileDir)
      .filter((f) => /\.(ts|tsx)$/.test(f) && !f.includes(".test."));
    expect(files.length).toBeGreaterThan(0);
    for (const f of files) {
      for (const imp of importsOf(join(mobileDir, f))) {
        if (!imp.startsWith(".")) continue;       // package imports
        if (imp.startsWith("./")) continue;       // own tree
        expect(SHARED_ALLOW.has(imp),
          `${f} imports "${imp}" — not in the mobile shared allowlist`).toBe(true);
      }
    }
  });

  it("no desktop module imports from mobile/ (except the main.tsx boot branch)", () => {
    for (const p of walk(srcDir)) {
      const rel = relative(srcDir, p);
      if (rel === "main.tsx") continue;           // the single boot-time branch
      for (const imp of importsOf(p)) {
        expect(imp.includes("mobile/") || imp.endsWith("/mobile"),
          `${rel} imports "${imp}" — desktop must not depend on mobile/`).toBe(false);
      }
    }
  });
});
