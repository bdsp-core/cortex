// Lean-bank hydration: rebuild the session pool from the immutable manifest,
// byte-hash-verified against the server stamp, in manifest order.
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { hydrateSessionBank, type WireSessionBank } from "./api";

const MANIFEST = {
  taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
  segments: [
    { segId: 1, sMean: [0.1], sSd: [0.2] },
    { segId: 2, sMean: [0.3], sSd: [0.4] },
    { segId: 3, sMean: [0.5], sSd: [0.6] },
  ],
};
const BYTES = new TextEncoder().encode(JSON.stringify(MANIFEST));
const SHA = createHash("sha256").update(BYTES).digest("hex");

function fetchStub(status = 200, bytes: Uint8Array = BYTES): typeof fetch {
  return (async () => ({
    ok: status === 200,
    status,
    arrayBuffer: async () => bytes.buffer.slice(
      bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  })) as unknown as typeof fetch;
}

function leanBank(overrides: Partial<Record<string, unknown>> = {}): WireSessionBank {
  return {
    bundleUrl: "/bundle/v-test",
    sampleSeed: 7,
    nPool: 2,
    leanBank: true,
    manifestSha256: SHA,
    exclusion: [2],
    ...overrides,
  } as unknown as WireSessionBank;
}

describe("hydrateSessionBank", () => {
  it("passes a full (or old-server) payload through untouched", async () => {
    const fat = { bundleUrl: "/b", segments: [{ segId: 9 }] } as never;
    expect(await hydrateSessionBank(fat, fetchStub())).toBe(fat);
  });

  it("reconstructs the pool from the manifest, exclusion-filtered in order", async () => {
    const bank = await hydrateSessionBank(leanBank(), fetchStub());
    expect(bank.segments.map((s) => s.segId)).toEqual([1, 3]);
    expect(bank.sampleSeed).toBe(7);
    // Wire-only fields do not leak into the hydrated bank.
    expect("leanBank" in bank).toBe(false);
    expect("manifestSha256" in bank).toBe(false);
    expect("exclusion" in bank).toBe(false);
  });

  it("refuses a manifest whose bytes do not hash to the stamp", async () => {
    const tampered = new TextEncoder().encode(
      JSON.stringify({ ...MANIFEST, segments: [] }));
    await expect(hydrateSessionBank(leanBank(), fetchStub(200, tampered)))
      .rejects.toThrow(/hash mismatch/);
  });

  it("surfaces a failed manifest fetch", async () => {
    await expect(hydrateSessionBank(leanBank(), fetchStub(503)))
      .rejects.toThrow(/manifest fetch failed: 503/);
  });
});
