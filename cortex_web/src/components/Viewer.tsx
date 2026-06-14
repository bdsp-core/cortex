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
import { iiicTasks } from "../tasks";
import { EegCanvas } from "./EegCanvas";
import { SpecCanvas } from "./SpecCanvas";
import {
  COLORS, FONTS, GAIN_LADDER, MONTAGES, BANDPASS_OPTIONS,
  NOTCH_OPTIONS, WINDOW_OPTIONS,
  IIIC_LABEL_START_S, IIIC_LABEL_END_S,
  SPEC_CLIP_START_FRAC, SPEC_CLIP_END_FRAC,
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
  const [panStart, setPanStart] = useState(IIIC_LABEL_START_S);
  const [pick, setPick] = useState<number | null>(null);

  // IIIC answer options derived from the bundle's task list, so the button
  // index ↔ engine task index mapping is correct for both K=6 (idx 0..5)
  // and K=7 (idx 1..6 — spike at 0 has its own screen).
  const iiicOpts = useMemo(() => iiicTasks(bundle.inputs), [bundle]);

  // Refs let the once-bound keydown handler read fresh values without stale
  // closures (the bug in the previous version). `answered` guards against a
  // double-submit while the next item loads.
  const segRef = useRef<SegmentData | null>(null);
  segRef.current = seg;
  const iiicOptsRef = useRef(iiicOpts);
  iiicOptsRef.current = iiicOpts;
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
    // Open on the labeled epoch (clip-local 10–20 s), not 0 s — desktop
    // v1.3.8. Pan reveals the {0–10, 20–30}-s context windows.
    setPanStart(IIIC_LABEL_START_S);
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
      // 1–N selects the N-th IIIC button (N = iiicOpts.length, typically 6).
      // We submit the *engine task index* (iiicOpts[i].idx), not the button
      // position, so K=7 bundles where IIIC tasks live at indices 1..6 also
      // produce y = (pick === chosen.k) correctly.
      const opts = iiicOptsRef.current;
      if (e.key >= "1" && e.key <= String(opts.length)) {
        e.preventDefault();
        const i = parseInt(e.key, 10) - 1;
        submitRef.current(opts[i].idx);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        // Full-window pan (v1.3.8) → cycles through {0, 10, 20}-s context
        // windows on a 30-s clip with a 10-s window.
        setPanStart((p) => Math.max(0, p - windowSRef.current));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setPanStart((p) => {
          const s = segRef.current;
          return s ? Math.min(s.nSamp / s.fsHz - windowSRef.current, p + windowSRef.current) : p;
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

  // Track the EEG pane's actual size so the canvas fills the available space
  // (the spectrogram on the left is fixed 280 px, the EEG pane is flex:1 —
  // pre-resize-observer the canvas was hard-coded 1140×760 and left a strip
  // of white on wide displays).
  const eegBoxRef = useRef<HTMLDivElement>(null);
  const [eegSize, setEegSize] = useState({ w: 1140, h: 760 });
  useEffect(() => {
    const el = eegBoxRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const cr = e.contentRect;
        if (cr.width > 0 && cr.height > 0) {
          setEegSize({ w: Math.floor(cr.width), h: Math.floor(cr.height) });
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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
        {iiicOpts.map((o, i) => (
          <button key={o.code} onClick={() => submit(o.idx)}
            style={{ minWidth: 135, padding: "10px 8px", ...sel(pick === o.idx) }}>
            {i + 1} · {o.label}
          </button>
        ))}
        <span style={{ marginLeft: 8, color: COLORS.textTertiary, fontSize: 12 }}>
          press 1–{iiicOpts.length} to answer
        </span>
        <span style={{ marginLeft: "auto", color: COLORS.textBody, fontSize: 13 }}>
          Confidence of reaching a verdict (most-uncertain task):{" "}
          <b style={{ color: COLORS.textPrimary }}>
            {progress.resolveConf == null ? "—" : `${Math.round(progress.resolveConf * 100)}%`}
          </b>
        </span>
      </div>

      {/* IIIC scoring hint — desktop v1.3.8 text */}
      <div style={{ fontSize: 12, color: COLORS.textTertiary, marginTop: 6 }}>
        Classify the pattern found <b style={{ color: COLORS.fail }}>within the red
        box</b>. The box marks the 10-second region scored by the panel; pan
        with ◀/▶ or ←/→ to see neighbouring context.
      </div>

      {/* middle: spectrogram (left) + EEG (right) */}
      <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8 }}>
        <div style={{ width: 280, background: "#fff", borderRadius: 4 }}>
          <SpecCanvas spec={seg?.spec ?? null} width={280} height={760}
            clipBoundsFrac={[SPEC_CLIP_START_FRAC, SPEC_CLIP_END_FRAC]} />
        </div>
        <div ref={eegBoxRef}
          style={{ flex: 1, background: "#fff", borderRadius: 4, minWidth: 0 }}>
          {seg ? (
            <EegCanvas rows={rows} fsHz={seg.fsHz} gainUv={gain} windowS={windowS}
              panStartS={panStart} width={eegSize.w} height={eegSize.h}
              labeledEpoch={{ startS: IIIC_LABEL_START_S, endS: IIIC_LABEL_END_S }} />
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
        <button onClick={() => setPanStart((p) => Math.max(0, p - windowS))}>◀ Pan</button>
        <button onClick={() => setPanStart((p) => Math.min(Math.max(0, dur - windowS), p + windowS))}>Pan ▶</button>
        <span style={{ color: COLORS.textTertiary }}>
          {item && seg ? `EEG ${panStart.toFixed(1)}–${(panStart + windowS).toFixed(1)} s of ${dur.toFixed(1)} s · ${montage} · ${gain} µV/div` : ""}
        </span>
      </div>
    </div>
  );
}
