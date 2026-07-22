// Shared EEG display state and the montage→filter pipeline behind every
// waveform surface: the exam Viewer, the exam SpikeViewer, and the training
// TrainingRunner.
//
// Each of those three used to re-declare the same montage/gain/bandpass/notch
// state, the same applyMontage→buildCascade→filtfilt derivation, and the same
// gain-ladder / montage-cycle keyboard math. Keeping three copies let them
// drift: the trainer's copy passed an EN-DASH filter string ("0.5–70 Hz"),
// which parseBandpass could not match, so the trainer silently rendered
// UNFILTERED EEG while the exam rendered 0.5–70 Hz + 60 Hz notch. Its copy
// also skipped useMemo, so the whole cascade re-ran on unrelated renders.
//
// Display only: filtering changes what the participant sees, never any
// response, belief, or engine state.

import { useCallback, useMemo, useState } from "react";
import type { SegmentData } from "../../bundle";
import { applyMontage, type MontageRow } from "../../montage";
import { buildCascade, filtfilt } from "../../dsp";
import {
  BANDPASS_OPTIONS, GAIN_LADDER, MONTAGES, NOTCH_OPTIONS,
} from "../../../ui/theme";

export const DEFAULT_MONTAGE = "bipolar";
export const DEFAULT_GAIN = 100;

// The montage + zero-phase filter cascade for one segment. Pure, so the
// filter actually reaching the canvas is unit-testable without a DOM.
export function buildDisplayRows(
  seg: SegmentData | null,
  montage: string,
  bandpass: string,
  notchHz: string,
): MontageRow[] {
  if (!seg) return [];
  const base = applyMontage(montage, seg.eeg, seg.channelNames, seg.nSamp);
  const cascade = buildCascade(bandpass, notchHz, seg.fsHz);
  if (!cascade.length) return base;
  return base.map((r) => (r.data ? { ...r, data: filtfilt(r.data, cascade) } : r));
}

// Ctrl cycles montages; an unknown value restarts the cycle at index 0.
export function nextMontage(current: string): string {
  const i = MONTAGES.indexOf(current as (typeof MONTAGES)[number]);
  return MONTAGES[(i + 1) % MONTAGES.length];
}

// ↑ steps one rung finer (delta -1), ↓ one coarser (+1), clamped at both ends.
export function steppedGain(current: number, delta: number): number {
  const i = GAIN_LADDER.indexOf(current) + delta;
  return GAIN_LADDER[Math.min(GAIN_LADDER.length - 1, Math.max(0, i))];
}

export function useEegDisplay(seg: SegmentData | null) {
  const [montage, setMontage] = useState<string>(DEFAULT_MONTAGE);
  const [gain, setGain] = useState(DEFAULT_GAIN);
  const [bandpass, setBandpass] = useState(BANDPASS_OPTIONS[0]);
  const [notchHz, setNotchHz] = useState(NOTCH_OPTIONS[0]);

  const rows = useMemo(
    () => buildDisplayRows(seg, montage, bandpass, notchHz),
    [seg, montage, bandpass, notchHz],
  );

  // Functional updates keep these referentially stable, so the once-bound
  // capture-phase key handlers can call them without re-binding or reading a
  // stale closure.
  const cycleMontage = useCallback(() => setMontage(nextMontage), []);
  const stepGain = useCallback(
    (delta: number) => setGain((g) => steppedGain(g, delta)), []);

  return {
    montage, setMontage, gain, setGain,
    bandpass, setBandpass, notchHz, setNotchHz,
    rows, cycleMontage, stepGain,
  };
}
