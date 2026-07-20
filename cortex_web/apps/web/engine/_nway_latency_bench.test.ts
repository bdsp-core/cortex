import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { BankArrays } from "./choose_item";
import { chooseNWayItem } from "./nway_selector";
import { expectedNWayProfile } from "./nway_profile";
import { makeState } from "./particles";
import { precomputePriorPair } from "./prior";
import { Rng } from "./rng";
import type { ComputeEngineInputs, ComputeSegmentMeta } from "./types";

const manifestPath = process.env.CORTEX_PROD_MANIFEST;

describe.runIf(!!manifestPath)("native n-way served-bank latency", () => {
  it("selects from the real manifest with the production particle cloud", () => {
    const bytes = readFileSync(manifestPath!);
    const manifest = JSON.parse(bytes.toString()) as ComputeEngineInputs;
    const inputs: ComputeEngineInputs = {
      ...manifest,
      terminationPolicy: "precision_v1",
      nwayProfile: expectedNWayProfile(createHash("sha256").update(bytes).digest("hex")),
    };
    const K = inputs.taskCodes.length;
    const bank: BankArrays = {
      sMean: Array.from({ length: K }, () => []),
      sSd: Array.from({ length: K }, () => []),
      segId: Array.from({ length: K }, () => []),
      segment: Array.from({ length: K }, () => [] as ComputeSegmentMeta[]),
    };
    for (const segment of inputs.segments) {
      for (const k of segment.applicableTaskIdx ?? Array.from({ length: K }, (_, i) => i)) {
        bank.sMean[k].push(segment.sMean[k]);
        bank.sSd[k].push(segment.sSd[k]);
        bank.segId[k].push(segment.segId);
        bank.segment![k].push(segment);
      }
    }
    for (let k = 0; k < K; k++) {
      const order = bank.sMean[k].map((value, index) => ({ value, index }))
        .sort((a, b) => (a.value - b.value) || (a.index - b.index));
      bank.sMean[k] = order.map(({ index }) => bank.sMean[k][index]);
      bank.sSd[k] = order.map(({ index }) => bank.sSd[k][index]);
      bank.segId[k] = order.map(({ index }) => bank.segId[k][index]);
      bank.segment![k] = order.map(({ index }) => bank.segment![k][index]);
    }
    const rng = new Rng(20260720);
    const state = makeState(1200, K, precomputePriorPair(inputs.corrL, inputs.corrT), rng);
    const started = performance.now();
    const chosen = chooseNWayItem(state, inputs, bank, new Set([0]));
    const durationMs = performance.now() - started;
    expect(chosen.k).toBeGreaterThanOrEqual(1);
    expect(chosen.segId).toBeGreaterThanOrEqual(0);
    expect(durationMs).toBeLessThan(60_000);
    console.info(JSON.stringify({
      kind: "nway_real_bank_selection", candidates: bank.segId.slice(1)
        .reduce((sum, ids) => sum + ids.length, 0), durationMs, chosen,
    }));
  }, 90_000);
});
