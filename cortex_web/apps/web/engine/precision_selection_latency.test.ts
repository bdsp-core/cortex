import { describe, expect, it } from "vitest";

import { chooseItem, type BankArrays } from "./choose_item";
import { makeState } from "./particles";
import { precomputePriorPair } from "./prior";
import { Rng } from "./rng";

function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Number(i === j)));
}

function phaseTwoFullBank(nCandidates = 30_000): BankArrays {
  const bank: BankArrays = { sMean: [[]], sSd: [[]], segId: [[]] };
  for (let k = 1; k < 7; k++) {
    const signal = new Array<number>(nCandidates);
    const uncertainty = new Array<number>(nCandidates);
    const ids = new Array<number>(nCandidates);
    for (let i = 0; i < nCandidates; i++) {
      signal[i] = -3 + 6 * i / (nCandidates - 1);
      uncertainty[i] = 0.15 + 0.2 * ((i * 37 + k * 11) % 101) / 100;
      ids[i] = i;
    }
    bank.sMean.push(signal);
    bank.sSd.push(uncertainty);
    bank.segId.push(ids);
  }
  return bank;
}

describe("Precision full-bank browser selector latency", () => {
  it("keeps one production-sized phase-II selection below the safety ceiling", () => {
    const prior = precomputePriorPair(identity(7), identity(7));
    const state = makeState(1200, 7, prior, new Rng(73));
    const bank = phaseTwoFullBank();
    const started = performance.now();
    const selected = chooseItem(state, bank, new Set([0]), {
      nSubsample: 128,
      uncertaintyAware: true,
    });
    const elapsedMs = performance.now() - started;

    expect(selected.segId).toBeGreaterThanOrEqual(0);
    expect(selected.k).toBeGreaterThanOrEqual(1);
    // This is deliberately a broad hardware-independent guard, not a product
    // SLA. An accidental exhaustive 6×30k scan at N=1200 breaches it by a
    // wide margin; the frozen coarse-to-fine implementation does not.
    expect(elapsedMs).toBeLessThan(15_000);
  }, 30_000);
});
