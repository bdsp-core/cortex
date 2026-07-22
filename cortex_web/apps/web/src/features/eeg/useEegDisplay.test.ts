import { describe, expect, it } from "vitest";

import { buildDisplayRows, nextMontage, steppedGain } from "./useEegDisplay";
import { parseBandpass } from "../../dsp";
import { BANDPASS_OPTIONS, GAIN_LADDER, MONTAGES, NOTCH_OPTIONS } from "../../../ui/theme";
import type { SegmentData } from "../../bundle";

// A short synthetic segment carrying a strong DC offset plus high-frequency
// content, so a real 0.5-70 Hz bandpass measurably changes the samples. Each
// channel gets its own offset/amplitude/phase — the montages subtract channel
// pairs, so identical channels would cancel to all-zero rows and hide any
// filtering difference.
function fakeSegment(nSamp = 512, fsHz = 200): SegmentData {
  const channelNames = ["Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2",
                        "F7", "T3", "T5", "F8", "T4", "T6", "Fz", "Cz", "Pz", "EKG"];
  const eeg = new Float32Array(channelNames.length * nSamp);
  for (let c = 0; c < channelNames.length; c++) {
    for (let i = 0; i < nSamp; i++) {
      const t = i / fsHz;
      eeg[c * nSamp + i] =
        (150 + 12 * c)                                        // per-channel DC
        + (30 + 2 * c) * Math.sin(2 * Math.PI * 10 * t + c)    // in-band
        + 15 * Math.sin(2 * Math.PI * 95 * t + c);             // above the 70 Hz edge
    }
  }
  return { eeg, nCh: channelNames.length, nSamp, fsHz, channelNames, spec: null };
}

describe("shared EEG display pipeline", () => {
  it("returns no rows without a segment", () => {
    expect(buildDisplayRows(null, "bipolar", BANDPASS_OPTIONS[0], NOTCH_OPTIONS[0])).toEqual([]);
  });

  it("applies the default bandpass instead of passing the signal through", () => {
    // Regression: the trainer requested "0.5–70 Hz" (EN-DASH), parseBandpass
    // returned null, and the cascade came back empty — so the trainer rendered
    // raw unfiltered EEG while the exam rendered a filtered trace.
    const seg = fakeSegment();
    const filtered = buildDisplayRows(seg, "bipolar", BANDPASS_OPTIONS[0], NOTCH_OPTIONS[0]);
    const unfiltered = buildDisplayRows(seg, "bipolar", "off", "off");

    const withData = filtered.findIndex((r) => r.data);
    expect(withData).toBeGreaterThanOrEqual(0);
    const a = filtered[withData].data!;
    const b = unfiltered[withData].data!;
    expect(a.length).toBe(b.length);
    // The DC offset alone guarantees a large difference once the high-pass runs.
    const maxDelta = a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
    expect(maxDelta).toBeGreaterThan(1);
  });

  it("passes the montage through untouched when filtering is off", () => {
    const seg = fakeSegment();
    const rows = buildDisplayRows(seg, "bipolar", "off", "off");
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.some((r) => r.data)).toBe(true);
  });

  it("keeps blank separator rows as separators", () => {
    const rows = buildDisplayRows(fakeSegment(), "bipolar", BANDPASS_OPTIONS[0], NOTCH_OPTIONS[0]);
    expect(rows.some((r) => r.data === null)).toBe(true);
  });
});

describe("bandpass string parsing", () => {
  it("parses every shipped dropdown option", () => {
    for (const opt of BANDPASS_OPTIONS) {
      if (opt === "off") {
        expect(parseBandpass(opt)).toBeNull();
      } else {
        expect(parseBandpass(opt)).not.toBeNull();
      }
    }
  });

  it("accepts typographic dashes as well as the ASCII hyphen", () => {
    expect(parseBandpass("0.5-70 Hz")).toEqual([0.5, 70]);
    expect(parseBandpass("0.5–70 Hz")).toEqual([0.5, 70]);   // en-dash
    expect(parseBandpass("0.5—70 Hz")).toEqual([0.5, 70]);   // em-dash
    expect(parseBandpass("0.5 - 70 Hz")).toEqual([0.5, 70]);
  });
});

describe("display control stepping", () => {
  it("cycles montages and wraps", () => {
    let m: string = MONTAGES[0];
    for (let i = 1; i < MONTAGES.length; i++) {
      m = nextMontage(m);
      expect(m).toBe(MONTAGES[i]);
    }
    expect(nextMontage(m)).toBe(MONTAGES[0]);
  });

  it("restarts the montage cycle from an unknown value", () => {
    expect(nextMontage("not-a-montage")).toBe(MONTAGES[0]);
  });

  it("steps the gain ladder and clamps at both ends", () => {
    expect(steppedGain(GAIN_LADDER[2], -1)).toBe(GAIN_LADDER[1]);
    expect(steppedGain(GAIN_LADDER[2], 1)).toBe(GAIN_LADDER[3]);
    expect(steppedGain(GAIN_LADDER[0], -1)).toBe(GAIN_LADDER[0]);
    expect(steppedGain(GAIN_LADDER[GAIN_LADDER.length - 1], 1))
      .toBe(GAIN_LADDER[GAIN_LADDER.length - 1]);
  });
});
