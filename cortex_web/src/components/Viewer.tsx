// The Viewer screen — the heart of the test. Spectrogram (left) + EEG (right),
// a 6-button IIIC answer panel, and the display controls, all driven by the
// engine Web Worker. Keyboard: 1–6 pick-and-advance (a keypress both selects
// and submits), ←/→ pan, ↑/↓ gain ladder, Ctrl cycle montage. Clicking a
// choice also advances.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Bundle, SegmentData } from "../bundle";
import { applyMontage, MontageRow } from "../montage";
import { buildCascade, filtfilt } from "../dsp";
import { EegCanvas } from "./EegCanvas";
import { SpecCanvas } from "./SpecCanvas";
import { TutorialOverlay, TutorialStep } from "./TutorialOverlay";
import {
  COLORS, FONTS, IIIC_OPTIONS, GAIN_LADDER, MONTAGES, BANDPASS_OPTIONS,
  NOTCH_OPTIONS, WINDOW_OPTIONS,
} from "../../ui/theme";

export interface Item {
  trialIndex: number;
  taskK: number;
  segId: number;
}

// Track an element's content-box size so the canvases can be drawn to fit their
// flex containers (instead of a hardcoded 1140×760 that overflows small windows).
function useSize(ref: React.RefObject<HTMLElement>): { w: number; h: number } {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export function Viewer({
  bundle,
  item,
  onAnswer,
  tutorial,
}: {
  bundle: Bundle;
  item: Item | null;
  onAnswer: (pick: number) => void;
  // When set, the Viewer is the in-context tutorial backdrop: answering is
  // disabled and a coach-marks overlay walks the user through the UI regions.
  tutorial?: { onFinish: () => void };
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

  // Region refs the tutorial overlay spotlights (answer row, spectrogram, EEG,
  // controls) — mirror the desktop steps (eeg_bank_viewer.py:632-677).
  const answerRowRef = useRef<HTMLDivElement>(null);
  const specRef = useRef<HTMLDivElement>(null);
  const eegRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<HTMLDivElement>(null);
  const tut = !!tutorial;

  // Live container sizes → the canvases draw to fit (responsive to window resize).
  const specSize = useSize(specRef);
  const eegSize = useSize(eegRef);

  // fetch the segment whenever the item changes; reset per-question UI state
  useEffect(() => {
    if (!item) return;
    let alive = true;
    setSeg(null);
    setPick(null);
    setPanStart(0);
    answered.current = false; // new question → allow a new answer
    bundle.segment(item.segId).then((s) => {
      if (!alive) return;
      setSeg(s);
      // Desktop default (eeg_bank_viewer.py:1157-1162): a 10s window opening on
      // the labeled epoch [10,20] for IIIC 30s clips; pan one full window to the
      // 0-10 / 20-30 context. Spike 10s clips open at 0.
      const dur = s.nSamp / s.fsHz;
      setWindowS(10);
      setPanStart(dur >= 25 ? 10 : 0);
    });
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
  // `i` is the 0-based BUTTON index; we map it to the engine's `raw` value
  // exactly like the desktop _confirm_answer (eeg_bank_viewer.py:963-1013):
  //   IIIC  → raw = i + 1   (button i ∈ 0..5 → task k ∈ 1..6; y = (raw==k))
  //   spike → raw = 0 (Yes) / -1 (No)   (Yes → matches task 0 → y=1)
  const submit = useCallback(
    (i: number) => {
      if (tutorial) return; // tutorial backdrop: answering is disabled
      const s = segRef.current;
      if (answered.current || !itemRef.current || !s) return;
      const spike = s.family === "spike";
      const nOpts = spike ? 2 : 6;
      if (i < 0 || i >= nOpts) return; // ignore out-of-range keys (3-6 on spike)
      const raw = spike ? (i === 0 ? 0 : -1) : i + 1;
      answered.current = true;
      setPick(i);
      onAnswer(raw);
      setSeg(null); // clear until the next item loads
    },
    [onAnswer, tutorial],
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
        setPanStart((p) => Math.max(0, p - windowSRef.current)); // full-window step
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
  const isSpike = !!seg && seg.family === "spike"; // Yes/No, 10s, no spectrogram/box/banner
  const isIiic = !isSpike; // IIIC (or not-yet-loaded) → spectrogram + red box + banner
  // Answer options follow the desktop _SPIKE_OPTIONS / _IIIC_OPTIONS.
  const options = isSpike
    ? [{ code: "yes", label: "Yes (spike present)" }, { code: "no", label: "No (no spike)" }]
    : IIIC_OPTIONS;
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
      {/* top: question counter + answer buttons (pick-and-advance) + the red
          IIIC scoring banner inline to the right (eeg_bank_viewer.py:723-727). */}
      <div ref={answerRowRef} style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
              flexShrink: 0, paddingBottom: 8, borderBottom: `1px solid ${COLORS.borderInactive}` }}>
        <span style={{ fontWeight: 600, marginRight: 8 }}>
          {tut ? "Tutorial example" : `Question ${item ? item.trialIndex + 1 : "—"}`}
        </span>
        {options.map((o, i) => (
          <button key={o.code} onClick={() => submit(i)}
            style={{ minWidth: 135, padding: "10px 8px", ...sel(pick === i) }}>
            {i + 1} · {o.label}
          </button>
        ))}
        {isIiic && (
          <span style={{ color: "#ff5c5c", fontWeight: 600, marginLeft: 8 }}>
            Classify the pattern found within the red box. Pan left or right to gain context.
          </span>
        )}
      </div>

      {/* middle: spectrogram (left) + EEG (right). flex:1 + minHeight:0 +
          overflow:hidden contains the canvases so they NEVER spill into the
          controls; each canvas is drawn to its measured container size. */}
      <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8, overflow: "hidden" }}>
        {/* spectrogram — IIIC only; spike clips show the EEG on its own */}
        {!isSpike && (
          <div ref={specRef} style={{
            flex: "0 0 280px", minWidth: 0, minHeight: 0, overflow: "hidden",
            background: "#fff", borderRadius: 4, border: `1px solid ${COLORS.borderInactive}`,
          }}>
            <SpecCanvas spec={seg?.spec ?? null} width={specSize.w || 280}
              height={specSize.h || 600} markerFrac={0.5} />
          </div>
        )}
        <div ref={eegRef} style={{
          flex: 1, minWidth: 0, minHeight: 0, overflow: "hidden",
          background: "#fff", borderRadius: 4, border: `1px solid ${COLORS.borderInactive}`,
        }}>
          {seg ? (
            <EegCanvas rows={rows} fsHz={seg.fsHz} gainUv={gain} windowS={windowS}
              panStartS={panStart} width={eegSize.w || 900} height={eegSize.h || 600}
              labelStartS={isSpike ? null : 10}
              labelLenS={isSpike ? null : 10} />
          ) : (
            <div style={{ color: "#888", padding: 20 }}>loading EEG…</div>
          )}
        </div>
      </div>

      {/* bottom: display controls — fixed height (flexShrink:0) + wrap so they
          stay put and never get overrun by the EEG/spectrogram area above. */}
      <div ref={controlsRef} style={{ display: "flex", gap: 16, rowGap: 8, alignItems: "center",
              flexWrap: "wrap", flexShrink: 0, marginTop: 8, paddingTop: 8, fontSize: 13,
              borderTop: `1px solid ${COLORS.borderInactive}` }}>
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
          {seg ? `EEG ${panStart.toFixed(1)}–${(panStart + windowS).toFixed(1)} s of ${dur.toFixed(1)} s · ${montage} · ${gain} µV/div` : ""}
        </span>
      </div>

      {tutorial && seg && (
        <TutorialOverlay onFinish={tutorial.onFinish} steps={tutorialSteps(answerRowRef, specRef, eegRef, controlsRef)} />
      )}
    </div>
  );
}

// The 5 coach-marks, verbatim from the desktop (eeg_bank_viewer.py:632-677).
function tutorialSteps(
  answerRow: React.RefObject<HTMLDivElement>,
  spec: React.RefObject<HTMLDivElement>,
  eeg: React.RefObject<HTMLDivElement>,
  controls: React.RefObject<HTMLDivElement>,
): TutorialStep[] {
  return [
    {
      target: answerRow,
      title: "Choosing an answer",
      body: (
        <>
          For each recording, choose the pattern that best matches what you see.
          <br /><br />
          <b><u>Once you select an answer, the test immediately advances to the next
          recording — you cannot change your answer.</u></b>
          <br /><br />
          • Seizure: an electrographic seizure<br />
          • LPD / GPD: lateralized or generalized periodic discharges<br />
          • LRDA / GRDA: lateralized or generalized rhythmic delta activity<br />
          • Other: a pattern fitting none of the above<br /><br />
          Some recordings instead ask only whether an epileptiform spike is present.
        </>
      ),
    },
    {
      target: spec,
      title: "The spectrogram",
      body: (
        <>
          A compressed time-frequency summary of the recording. A quick way to spot
          rhythmic or evolving activity before reading the waveforms.
          <br /><br />
          The four panels are brain regions: LL and RL are the left and right temporal
          chains; LP and RP are the left and right parasagittal chains.
          <br /><br />
          The spectrogram appears only for these pattern-classification recordings. The
          spike-present questions show the EEG on its own, with no spectrogram panel.
        </>
      ),
    },
    {
      target: eeg,
      title: "The EEG",
      body: (
        <>
          The raw tracings. Each row is a derivation between two electrodes. Pan through
          the recording with the ◀ ▶ buttons or the left / right arrow keys. Adjust the
          gain with the ▲ ▼ buttons or the up / down arrow keys.
        </>
      ),
    },
    {
      target: controls,
      title: "Display controls",
      body: (
        <>
          These change how the EEG is displayed — never your answer:
          <br /><br />
          • Montage: how electrode pairs are combined (bipolar, average, Laplacian).{" "}
          <b><u>Press the Ctrl key to flip through the montages</u></b> without leaving
          the keyboard.<br />
          • Gain: vertical scale, in µV per division<br />
          • Bandpass: keeps a frequency band, removing slow drift and high-frequency noise<br />
          • Notch: removes 50 / 60 Hz mains interference<br />
          • Window: how many seconds of EEG are shown at once
        </>
      ),
    },
    {
      target: null,
      title: "Ready to begin",
      body: (
        <>
          That is the full interface. When you select Begin, the assessment starts and
          this example is replaced by the first recording.
        </>
      ),
    },
  ];
}
