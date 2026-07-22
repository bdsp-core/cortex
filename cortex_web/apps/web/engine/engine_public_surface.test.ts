// The src -> engine half of the module boundary.
//
// architecture_boundary.test.ts already enforces the other direction (the
// engine may not import React, src/, or the API client). Nothing enforced
// this one, so the engine's "public API" had become whatever any of eleven
// src/ modules had happened to reach for: types.ts, mathfns.ts, session.ts,
// worker_protocol.ts, compute_payload.ts. This test makes engine/index.ts the
// only door, so widening the contract is a deliberate edit to that file
// rather than a new deep import someone adds in passing.
//
// Same spirit as src/mobile/boundary.test.ts.

import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, relative } from "node:path";

const engineDir = dirname(fileURLToPath(import.meta.url));
const srcDir = join(engineDir, "..", "src");

// Worker entry points are loaded by URL, not imported, so they never route
// through the barrel: `new Worker(new URL("../engine/worker.ts", ...))` is
// resolved by Vite at build time and a re-export would break it.
const WORKER_URL_REFERENCE = /new URL\(\s*["'][^"']*engine\/[a-z_]+\.ts["']/;

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      return name === "__snapshots__" ? [] : walk(p);
    }
    return /\.(ts|tsx)$/.test(name) ? [p] : [];
  });
}

describe("engine public surface", () => {
  it("is the only path src/ imports the engine through", () => {
    const offenders: string[] = [];
    for (const file of walk(srcDir)) {
      const source = readFileSync(file, "utf8");
      for (const match of source.matchAll(
        /(?:from|import)\s*\(?\s*["']((?:\.\.\/)+engine[^"']*)["']/g,
      )) {
        const specifier = match[1];
        // "../engine" / "../../engine" are the barrel; anything deeper is a
        // reach past it.
        if (/^(?:\.\.\/)+engine$/.test(specifier)) continue;
        offenders.push(`${relative(srcDir, file)} -> ${specifier}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("still lets worker entry points be referenced by URL", () => {
    const client = readFileSync(join(srcDir, "engineClient.ts"), "utf8");
    expect(WORKER_URL_REFERENCE.test(client)).toBe(true);
  });

  it("exports the symbols src/ actually consumes", async () => {
    const surface = await import("./index");
    // Value exports (types are erased, so only these are observable here).
    expect(typeof surface.normCdf).toBe("function");
    expect(typeof surface.packComputeInputs).toBe("function");
    expect(typeof surface.computePayloadTransferables).toBe("function");
  });
});
