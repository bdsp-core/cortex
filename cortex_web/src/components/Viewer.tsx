// The Viewer screen — the heart of the test. Spectrogram (left) + EEG (right),
// a 6-button IIIC answer panel, and the display controls, all driven by the
// engine Web Worker. Keyboard: 1–6 pick-and-advance (a keypress both selects
// and submits), ←/→ pan, ↑/↓ gain ladder, Ctrl cycle montage. Clicking a
// choice also advances.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bundle, SegmentData } from "../bundle";
import { applyMontage, MontageRow } from "../montage";
import { buildCascade, filtfilt } from "../dsp";
import { Progress } from "../progress";
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
  progress,
  onAnswer,
}: {
  bundle: Bundle;
  item: Item | null;
  progress: Progress;
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

  // Refs let the once-bound keydown handler read fresh values without stale
  // closures (the bug in the previous version). `answered` guards against a
  // double-submit while the next item loads.
  const segRef = useRef<SegmentData | null>(null);
  segRef.current = seg;
  const itemRef = useRef<Item | null>(null);
  itemRef.current = item;
  const windowSRef = useRef(windowS);
  windowSRef.current = windowS;
  const answered = useRef(false);

  // fetch the segment whenever the item changes; reset per-question UI state
  useEffect(() => {
    if (!item) return;
    let alive = true;
    setSeg(null);
    setPick(null);
    setPanStart(0);
    answered.current = false; // new question → allow a new answer
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

  // Pick-and-advance: a single choice (key or click) selects AND submits.
  const submit = useCallback(
    (k: number) => {
      if (answered.current || !itemRef.current || !segRef.current) return;
      answered.current = true;
      setPick(k);
      onAnswer(k);
      setSeg(null); // clear until the next item loads
    },
    [onAnswer],
  );

  // keyboard. Bound ONCE for the component's life (empty deps) with a stable
  // listener; reads the latest submit + state via refs so it never goes
  // stale. Capture phase ('true') so the event is handled at the window
  // before any focused <select>/<button> can consume it.
  //
  // NB: a key-grabbing browser extension (Vimium/Surfingkeys-style navigator,
  // tab-switcher) that registers a capture-phase listener at document_start
  // and stopImmediatePropagation()s number keys will pre-empt this — no page
  // code can recover the event. Confirmed during dev: keys work in an
  // extension-free (incognito) profile; click-to-answer is the fallback.
  const submitRef = useRef(submit);
  submitRef.current = submit;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key >= "1" && e.key <= "6") {
        e.preventDefault();
        submitRef.current(parseInt(e.key, 10) - 1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setPanStart((p) => Math.max(0, p - windowSRef.current / 2));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setPanStart((p) => {
          const s = segRef.current;
          return s ? Math.min(s.nSamp / s.fsHz - windowSRef.current, p + windowSRef.current / 2) : p;
        });
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setGain((g) => GAIN_LADDER[Math.max(0, GAIN_LADDER.indexOf(g) - 1)]);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setGain((g) => GAIN_LADDER[Math.min(GAIN_LADDER.length - 1, GAIN_LADDER.indexOf(g) + 1)]);
      } else if (e.key === "Control") {
        setMontage((m) => MONTAGES[(MONTAGES.indexOf(m as any) + 1) % MONTAGES.length]);
      }
    };
    window.addEventListener("keydown", onKey, true); // capture phase
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  const dur = seg ? seg.nSamp / seg.fsHz : 0;
  const sel = (v: boolean) => ({
    outline: v ? `3px solid ${COLORS.accent}` : "none",
    borderRadius: 6,
  });

  // Ensure the page holds keyboard focus (a window opened via `open` can land
  // with focus off the document). A focusable root + focus-on-mount fixes it.
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => { rootRef.current?.focus(); }, []);

  return (
    <div ref={rootRef} tabIndex={0}
      style={{ background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
                  height: "100vh", display: "flex", flexDirection: "column", padding: 12,
                  boxSizing: "border-box", outline: "none" }}>
      {/* top: question counter + answer buttons (pick-and-advance) */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600, marginRight: 8 }}>
          Question {item ? item.trialIndex + 1 : "—"} of up to {progress.maxQ || "—"}
        </span>
        {IIIC_OPTIONS.map((o, i) => (
          <button key={o.code} onClick={() => submit(i)}
            style={{ minWidth: 135, padding: "10px 8px", ...sel(pick === i) }}>
            {i + 1} · {o.label}
          </button>
        ))}
        <span style={{ marginLeft: 8, color: COLORS.textTertiary, fontSize: 12 }}>
          press 1–6 to answer
        </span>
        <span style={{ marginLeft: "auto", color: COLORS.textBody, fontSize: 13 }}>
          Est. chance of reaching a conclusion:{" "}
          <b style={{ color: COLORS.textPrimary }}>
            {progress.finishProb == null ? "—" : `${Math.round(progress.finishProb * 100)}%`}
          </b>
        </span>
      </div>

      {/* middle: spectrogram (left) + EEG (right) */}
      <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8 }}>
        <div style={{ width: 280, background: "#fff", borderRadius: 4 }}>
          <SpecCanvas spec={seg?.spec ?? null} width={280} height={760} markerFrac={0.5} />
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
