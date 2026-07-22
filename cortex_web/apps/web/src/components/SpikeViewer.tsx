// Spike-block viewer — the binary spike-or-not screen the desktop runs FIRST
// at K=7 (scripts/session_controller.py "Phase A"). A 10-s clip @ 128 Hz, no
// spectrogram, no panning (the whole clip is the question), with two large
// Yes / No buttons (and 1 / 2 keys). The IIIC Viewer takes over for Phase B.
//
// Submit semantics: the engine reads y = (pick === chosen.k). For a spike
// question chosen.k is the spike task index. "Yes" submits that index → y=1
// (a spike). "No" submits an out-of-range sentinel (K) → y=0 (no spike).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bundle, SegmentData, Item } from "../bundle";
import { applyMontage, MontageRow } from "../montage";
import { buildCascade, filtfilt } from "../dsp";
import { Progress } from "../progress";
import { EegCanvas } from "./EegCanvas";
import {
  TrajectoryOptimizationStatus, useTrajectoryOptimizationStatus,
} from "./TrajectoryOptimizationStatus";
import {
  COLORS, FONTS, GAIN_LADDER, MONTAGES, BANDPASS_OPTIONS, NOTCH_OPTIONS,
} from "../../ui/theme";

export type { Item };

export function SpikeViewer({
  bundle, item, onAnswer, onMediaReady, spikeTaskIdx, totalTasks, onExit,
}: {
  bundle: Bundle;
  item: Item | null;
  progress: Progress;
  onAnswer: (pick: number) => void;
  onMediaReady?: (trialIndex: number) => void;
  spikeTaskIdx: number;        // engine task index for spike (typically 0)
  totalTasks: number;          // K — used as the "No" sentinel pick (out of range)
  // "Save & finish later": answers checkpoint server-side as they happen, so
  // exiting is always safe; the dashboard offers resume for 24 h.
  onExit?: () => void;
}) {
  const [seg, setSeg] = useState<SegmentData | null>(null);
  const [segError, setSegError] = useState(false);
  const [montage, setMontage] = useState<string>("bipolar");
  const [gain, setGain] = useState(100);
  const [bandpass, setBandpass] = useState(BANDPASS_OPTIONS[0]);
  const [notchHz, setNotchHz] = useState(NOTCH_OPTIONS[0]);
  const [pick, setPick] = useState<number | null>(null);

  const segRef = useRef<SegmentData | null>(null);
  segRef.current = seg;
  const itemRef = useRef<Item | null>(null);
  itemRef.current = item;
  const answered = useRef(false);
  const {
    beginWait: beginTrajectoryWait, visible: trajectoryStatusVisible,
  } = useTrajectoryOptimizationStatus(item?.trialIndex ?? null);

  useEffect(() => {
    if (!item) return;
    let alive = true;
    setSeg(null);
    setSegError(false);
    setPick(null);
    answered.current = false;
    bundle.segment(item.segId)
      .then((s) => {
        if (alive) {
          setSeg(s);
          onMediaReady?.(item.trialIndex);
        }
      })
      .catch(() => { if (alive) setSegError(true); });
    return () => { alive = false; };
  }, [item, bundle, onMediaReady]);

  const rows: MontageRow[] = useMemo(() => {
    if (!seg) return [];
    const base = applyMontage(montage, seg.eeg, seg.channelNames, seg.nSamp);
    const cascade = buildCascade(bandpass, notchHz, seg.fsHz);
    if (!cascade.length) return base;
    return base.map((r) => (r.data ? { ...r, data: filtfilt(r.data, cascade) } : r));
  }, [seg, montage, bandpass, notchHz]);

  // Pick-and-advance: Yes = spikeTaskIdx (y=1); No = totalTasks (out-of-range, y=0).
  const submit = useCallback(
    (k: number) => {
      if (answered.current || !itemRef.current || !segRef.current) return;
      answered.current = true;
      setPick(k);
      beginTrajectoryWait();
      onAnswer(k);
      setSeg(null);
    },
    [beginTrajectoryWait, onAnswer],
  );

  const submitRef = useRef(submit);
  submitRef.current = submit;
  const spikeIdxRef = useRef(spikeTaskIdx);
  spikeIdxRef.current = spikeTaskIdx;
  const totalRef = useRef(totalTasks);
  totalRef.current = totalTasks;

  // Keyboard: 1 = Yes, 2 = No. Capture phase so dropdowns/buttons don't eat it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "1" || e.key.toLowerCase() === "y") {
        e.preventDefault();
        submitRef.current(spikeIdxRef.current);
      } else if (e.key === "2" || e.key.toLowerCase() === "n") {
        e.preventDefault();
        submitRef.current(totalRef.current);
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
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  const dur = seg ? seg.nSamp / seg.fsHz : 0;
  const sel = (v: boolean) => ({
    outline: v ? `3px solid ${COLORS.accent}` : "none",
    borderRadius: 6,
  });

  // Page focus + responsive EEG sizing.
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => { rootRef.current?.focus(); }, []);
  const eegBoxRef = useRef<HTMLDivElement>(null);
  const [eegSize, setEegSize] = useState({ w: 1440, h: 760 });
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

  // The spike clip is the whole question; no labeled-epoch overlay, no pan.
  const yesPick = pick === spikeTaskIdx;
  const noPick = pick === totalTasks;
  const isYes = (v: boolean) => sel(v);

  return (
    <div ref={rootRef} tabIndex={0} className="cx-test"
      style={{ background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
               height: "100vh", display: "flex", flexDirection: "column", padding: 12,
               boxSizing: "border-box", outline: "none" }}>
      {/* top: question counter + Yes / No */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600, marginRight: 8 }}>
          Question {item ? item.trialIndex + 1 : ""}
        </span>
        <button onClick={() => submit(spikeTaskIdx)}
          style={{ minWidth: 180, padding: "12px 16px", fontWeight: 700, ...isYes(yesPick) }}>
          1 · Spike
        </button>
        <button onClick={() => submit(totalTasks)}
          style={{ minWidth: 180, padding: "12px 16px", fontWeight: 700, ...isYes(noPick) }}>
          2 · No spike
        </button>
      </div>

      <div style={{ fontSize: 12, color: COLORS.textTertiary, marginTop: 6 }}>
        Does this 10-second clip contain at least one <b style={{ color: COLORS.textPrimary }}>epileptiform spike or sharp wave</b>?
      </div>

      {/* full-width EEG (no spectrogram for spike). minHeight:0 is essential —
          without it column-flex auto-min lets the canvas push the box to
          grow each ResizeObserver cycle (slow "stretching" loop). */}
      <div ref={eegBoxRef}
        style={{ flex: 1, background: "#fff", borderRadius: 4,
                 minWidth: 0, minHeight: 0, marginTop: 8 }}>
        {seg ? (
          <EegCanvas rows={rows} fsHz={seg.fsHz} gainUv={gain} windowS={dur || 10}
            panStartS={0} width={eegSize.w} height={eegSize.h} />
        ) : segError ? (
          <div style={{ color: "#ff5c5c", padding: 20 }}>
            This recording failed to load. Check your connection and answer to
            continue, or reload the page.
          </div>
        ) : (
          <div style={{ color: "#888", padding: 20 }}>loading EEG…</div>
        )}
      </div>

      {/* bottom: display controls (no window/pan — the whole clip IS the question) */}
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
        <span style={{ color: COLORS.textTertiary, marginLeft: "auto" }}>
          {item && seg ? `EEG ${dur.toFixed(1)} s · ${montage} · ${gain} µV/div · ${seg.fsHz} Hz` : ""}
        </span>
        {onExit && (
          <button onClick={onExit}
            style={{ background: "none", cursor: "pointer",
              border: `1px solid ${COLORS.borderInactive}`, borderRadius: 4,
              color: COLORS.textFaint, fontFamily: FONTS.sans, fontSize: 12,
              padding: "8px 12px", whiteSpace: "nowrap" }}>
            Save &amp; finish later
          </button>
        )}
      </div>
      {trajectoryStatusVisible && <TrajectoryOptimizationStatus />}
    </div>
  );
}
