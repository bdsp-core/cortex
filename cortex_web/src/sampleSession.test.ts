// Tests for the per-session stratified sampler. Validates the properties the
// design promises: target size, class balance, difficulty spread, determinism
// per seed, fresh draw per seed, and graceful pool<target behavior.

import { describe, expect, it } from "vitest";
import { sampleSession } from "./sampleSession";
import { EngineInputs, SegmentMeta } from "../engine/types";

const WORDS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"];
const CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"];

// Build a synthetic pool: `perClass` segments per class, each segment's
// true-task signal swept 0→1 across the class so difficulty spans the range.
function makePool(perClass: number): EngineInputs {
  const segments: SegmentMeta[] = [];
  let id = 0;
  WORDS.forEach((word, k) => {
    for (let i = 0; i < perClass; i++) {
      const sMean = new Array(6).fill(-1);
      sMean[k] = (i / (perClass - 1)) * 2; // 0 (hard) → 2 (easy) on the true task
      segments.push({
        segId: id++,
        patternClass: word,
        sMean,
        sSd: new Array(6).fill(0.5),
        fsHz: 200,
        nCh: 20,
        nSamp: 6000,
        channelNames: [],
        eeg: "",
        spec: "",
      });
    }
  });
  return {
    taskCodes: CODES,
    taskLabels: CODES,
    taskPatternWords: WORDS,
    corrL: [],
    ellStar: [],
    segments,
  };
}

describe("sampleSession", () => {
  it("draws the target count, class-balanced", () => {
    const pool = makePool(200); // 1200 total
    const { inputs, info } = sampleSession(pool, 300, 12345);
    expect(inputs.segments.length).toBe(300);
    expect(info.nSampled).toBe(300);
    // 6 classes → ~50 each
    const counts = Object.values(info.perClass);
    for (const c of counts) expect(c).toBeGreaterThanOrEqual(45);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(300);
  });

  it("spans easy→hard within each class (not all textbook cases)", () => {
    const pool = makePool(200);
    const { inputs } = sampleSession(pool, 300, 7);
    // for the seizure class (k=0), check the sampled true-task signals span a
    // wide range rather than clustering at the easy end
    const sz = inputs.segments.filter((s) => s.patternClass === "seizure")
                              .map((s) => s.sMean[0]);
    const lo = Math.min(...sz), hi = Math.max(...sz);
    expect(lo).toBeLessThan(0.5); // some hard examples
    expect(hi).toBeGreaterThan(1.5); // some easy examples
  });

  it("is deterministic for a fixed seed and differs across seeds", () => {
    const pool = makePool(200);
    const a = sampleSession(pool, 300, 42).inputs.segments.map((s) => s.segId);
    const b = sampleSession(pool, 300, 42).inputs.segments.map((s) => s.segId);
    const c = sampleSession(pool, 300, 43).inputs.segments.map((s) => s.segId);
    expect(a).toEqual(b);
    // different seed → different ordering and/or membership
    expect(a).not.toEqual(c);
  });

  it("rarely repeats the same set across sittings when pool >> target", () => {
    const pool = makePool(500); // 3000 total
    const s1 = new Set(sampleSession(pool, 300, 1).inputs.segments.map((s) => s.segId));
    const s2 = new Set(sampleSession(pool, 300, 2).inputs.segments.map((s) => s.segId));
    let overlap = 0;
    s1.forEach((id) => { if (s2.has(id)) overlap++; });
    // two independent 300-draws from 3000 should overlap far below identical
    expect(overlap).toBeLessThan(150);
  });

  it("uses the whole pool when pool <= target", () => {
    const pool = makePool(50); // 300 total
    const { inputs } = sampleSession(pool, 500, 9);
    expect(inputs.segments.length).toBe(300);
  });
});
