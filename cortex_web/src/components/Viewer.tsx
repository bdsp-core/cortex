// The Viewer screen — the heart of the test. Spectrogram (left) + EEG (right),
// a 6-button IIIC answer panel + Confirm, and the display controls, all driven
// by the engine Web Worker. Keyboard: 1–6 select, Enter confirm, ←/→ pan,
// ↑/↓ gain ladder, Ctrl cycle montage.

import { useEffect, useMemo, useState } from "react";
import { Bundle, SegmentData } from "../bundle";
import { applyMontage, MontageRow } from "../montage";
import { buildCascade, filtfilt } from "../dsp";
import { EegCanvas } from "./EegCanvas";
import { SpecCanvas } from "./SpecCanvas";
import {
  COLORS, FONTS, IIIC_OPTIONS, GAIN_LADDER, MONTAGES, BANDPASS_OPTIONS,
  NOTCH_OPTIONS, WINDOW_OPTIONS,
} from "../../ui/theme";

export interface Item {
  trialIndex: number;
  taskK: number;
  segId: number;
}

export function Viewer({
  bundle,
  item,
  onAnswer,
}: {
  bundle: Bundle;
  item: Item | null;
  onAnswer: (pick: number) => void;
}) {
  const [seg, setSeg] = useState<SegmentData | null>(null);
  const [montage, setMontage] = useState<string>("bipolar");
  const [gain, setGain] = useState(100);
  const [bandpass, setBandpass] = useState(BANDPASS_OPTIONS[0]);
  const [notchHz, setNotchHz] = useState(NOTCH_OPTIONS[0]);
  const [windowS, setWindowS] = useState(10);
  const [panStart, setPanStart] = useState(0);
  const [pick, setPick] = useState<number | null>(null);

  // fetch the segment whenever the item changes; reset per-question UI state
  useEffect(() => {
    if (!item) return;
    let alive = true;
    setSeg(null);
    setPick(null);
    setPanStart(0);
    bundle.segment(item.segId).then((s) => alive && setSeg(s));
    return () => { alive = false; };
  }, [item, bundle]);

  // filtered montage rows (recompute on seg / montage / filter change)
  const rows: MontageRow[] = useMemo(() => {
    if (!seg) return [];
    const base = applyMontage(montage, seg.eeg, seg.channelNames, seg.nSamp);
    const cascade = buildCascade(bandpass, notchHz, seg.fsHz);
    if (!cascade.length) return base;
    return base.map((r) => (r.data ? { ...r, data: filtfilt(r.data, cascade) } : r));
  }, [seg, montage, bandpass, notchHz]);

  const confirm = () => {
    if (pick === null || !item) return;
    onAnswer(pick);
    setSeg(null); // clear until the next item loads
  };

  // keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key >= "1" && e.key <= "6") setPick(parseInt(e.key, 10) - 1);
      else if (e.key === "Enter") confirm();
      else if (e.key === "ArrowLeft") setPanStart((p) => Math.max(0, p - windowS / 2));
      else if (e.key === "ArrowRight")
        setPanStart((p) => (seg ? Math.min(seg.nSamp / seg.fsHz - windowS, p + windowS / 2) : p));
      else if (e.key === "ArrowUp")
        setGain((g) => GAIN_LADDER[Math.max(0, GAIN_LADDER.indexOf(g) - 1)]);
      else if (e.key === "ArrowDown")
        setGain((g) => GAIN_LADDER[Math.min(GAIN_LADDER.length - 1, GAIN_LADDER.indexOf(g) + 1)]);
      else if (e.key === "Control")
        setMontage((m) => MONTAGES[(MONTAGES.indexOf(m as any) + 1) % MONTAGES.length]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pick, item, windowS, seg]);

  const dur = seg ? seg.nSamp / seg.fsHz : 0;
  const sel = (v: boolean) => ({
    outline: v ? `3px solid ${COLORS.accent}` : "none",
    borderRadius: 6,
  });

  return (
    <div style={{ background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
                  height: "100vh", display: "flex", flexDirection: "column", padding: 12, boxSizing: "border-box" }}>
      {/* top: question + answer buttons + confirm */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600, marginRight: 8 }}>
          Question {item ? item.trialIndex + 1 : "—"}
        </span>
        {IIIC_OPTIONS.map((o, i) => (
          <button key={o.code} onClick={() => setPick(i)}
            style={{ minWidth: 135, padding: "10px 8px", ...sel(pick === i) }}>
            {i + 1} · {o.label}
          </button>
        ))}
        <button onClick={confirm} disabled={pick === null}
          style={{ minWidth: 120, padding: "10px 8px", marginLeft: 8 }}>
          Confirm ⏎
        </button>
      </div>

      {/* middle: spectrogram (left) + EEG (right) */}
      <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8 }}>
        <div style={{ width: 280, background: "#fff", borderRadius: 4 }}>
          <SpecCanvas spec={seg?.spec ?? null} width={280} height={760} />
        </div>
        <div style={{ flex: 1, background: "#fff", borderRadius: 4 }}>
          {seg ? (
            <EegCanvas rows={rows} fsHz={seg.fsHz} gainUv={gain} windowS={windowS}
              panStartS={panStart} width={1140} height={760} />
          ) : (
            <div style={{ color: "#888", padding: 20 }}>loading EEG…</div>
          )}
        </div>
      </div>

      {/* bottom: display controls */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginTop: 8, fontSize: 13 }}>
        <label>Montage{" "}
          <select value={montage} onChange={(e) => setMontage(e.target.value)}>
            {MONTAGES.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label>Gain{" "}
          <select value={gain} onChange={(e) => setGain(parseInt(e.target.value, 10))}>
            {GAIN_LADDER.map((g) => <option key={g} value={g}>{g} µV/div</option>)}
          </select>
        </label>
        <label>Bandpass{" "}
          <select value={bandpass} onChange={(e) => setBandpass(e.target.value)}>
            {BANDPASS_OPTIONS.map((b) => <option key={b}>{b}</option>)}
          </select>
        </label>
        <label>Notch{" "}
          <select value={notchHz} onChange={(e) => setNotchHz(e.target.value)}>
            {NOTCH_OPTIONS.map((n) => <option key={n}>{n}</option>)}
          </select>
        </label>
        <label>Window{" "}
          <select value={windowS} onChange={(e) => setWindowS(parseInt(e.target.value, 10))}>
            {WINDOW_OPTIONS.map((w) => <option key={w} value={w}>{w} s</option>)}
          </select>
        </label>
        <button onClick={() => setPanStart((p) => Math.max(0, p - windowS / 2))}>◀ Pan</button>
        <button onClick={() => setPanStart((p) => Math.min(Math.max(0, dur - windowS), p + windowS / 2))}>Pan ▶</button>
        <span style={{ color: COLORS.textTertiary }}>
          {item && seg ? `EEG ${panStart.toFixed(1)}–${(panStart + windowS).toFixed(1)} s of ${dur.toFixed(1)} s · ${montage} · ${gain} µV/div` : ""}
        </span>
      </div>
    </div>
  );
}
