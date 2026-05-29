import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
// The Viewer screen — the heart of the test. Spectrogram (left) + EEG (right),
// a 6-button IIIC answer panel + Confirm, and the display controls, all driven
// by the engine Web Worker. Keyboard: 1–6 select, Enter confirm, ←/→ pan,
// ↑/↓ gain ladder, Ctrl cycle montage.
import { useEffect, useMemo, useState } from "react";
import { applyMontage } from "../montage";
import { buildCascade, filtfilt } from "../dsp";
import { EegCanvas } from "./EegCanvas";
import { SpecCanvas } from "./SpecCanvas";
import { COLORS, FONTS, IIIC_OPTIONS, GAIN_LADDER, MONTAGES, BANDPASS_OPTIONS, NOTCH_OPTIONS, WINDOW_OPTIONS, } from "../../ui/theme";
export function Viewer({ bundle, item, onAnswer, }) {
    const [seg, setSeg] = useState(null);
    const [montage, setMontage] = useState("bipolar");
    const [gain, setGain] = useState(100);
    const [bandpass, setBandpass] = useState(BANDPASS_OPTIONS[0]);
    const [notchHz, setNotchHz] = useState(NOTCH_OPTIONS[0]);
    const [windowS, setWindowS] = useState(10);
    const [panStart, setPanStart] = useState(0);
    const [pick, setPick] = useState(null);
    // fetch the segment whenever the item changes; reset per-question UI state
    useEffect(() => {
        if (!item)
            return;
        let alive = true;
        setSeg(null);
        setPick(null);
        setPanStart(0);
        bundle.segment(item.segId).then((s) => alive && setSeg(s));
        return () => { alive = false; };
    }, [item, bundle]);
    // filtered montage rows (recompute on seg / montage / filter change)
    const rows = useMemo(() => {
        if (!seg)
            return [];
        const base = applyMontage(montage, seg.eeg, seg.channelNames, seg.nSamp);
        const cascade = buildCascade(bandpass, notchHz, seg.fsHz);
        if (!cascade.length)
            return base;
        return base.map((r) => (r.data ? { ...r, data: filtfilt(r.data, cascade) } : r));
    }, [seg, montage, bandpass, notchHz]);
    const confirm = () => {
        if (pick === null || !item)
            return;
        onAnswer(pick);
        setSeg(null); // clear until the next item loads
    };
    // keyboard
    useEffect(() => {
        const onKey = (e) => {
            if (e.key >= "1" && e.key <= "6")
                setPick(parseInt(e.key, 10) - 1);
            else if (e.key === "Enter")
                confirm();
            else if (e.key === "ArrowLeft")
                setPanStart((p) => Math.max(0, p - windowS / 2));
            else if (e.key === "ArrowRight")
                setPanStart((p) => (seg ? Math.min(seg.nSamp / seg.fsHz - windowS, p + windowS / 2) : p));
            else if (e.key === "ArrowUp")
                setGain((g) => GAIN_LADDER[Math.max(0, GAIN_LADDER.indexOf(g) - 1)]);
            else if (e.key === "ArrowDown")
                setGain((g) => GAIN_LADDER[Math.min(GAIN_LADDER.length - 1, GAIN_LADDER.indexOf(g) + 1)]);
            else if (e.key === "Control")
                setMontage((m) => MONTAGES[(MONTAGES.indexOf(m) + 1) % MONTAGES.length]);
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pick, item, windowS, seg]);
    const dur = seg ? seg.nSamp / seg.fsHz : 0;
    const sel = (v) => ({
        outline: v ? `3px solid ${COLORS.accent}` : "none",
        borderRadius: 6,
    });
    return (_jsxs("div", { style: { background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
            height: "100vh", display: "flex", flexDirection: "column", padding: 12, boxSizing: "border-box" }, children: [_jsxs("div", { style: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }, children: [_jsxs("span", { style: { fontWeight: 600, marginRight: 8 }, children: ["Question ", item ? item.trialIndex + 1 : "—"] }), IIIC_OPTIONS.map((o, i) => (_jsxs("button", { onClick: () => setPick(i), style: { minWidth: 135, padding: "10px 8px", ...sel(pick === i) }, children: [i + 1, " \u00B7 ", o.label] }, o.code))), _jsx("button", { onClick: confirm, disabled: pick === null, style: { minWidth: 120, padding: "10px 8px", marginLeft: 8 }, children: "Confirm \u23CE" })] }), _jsxs("div", { style: { display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8 }, children: [_jsx("div", { style: { width: 280, background: "#fff", borderRadius: 4 }, children: _jsx(SpecCanvas, { spec: seg?.spec ?? null, width: 280, height: 760 }) }), _jsx("div", { style: { flex: 1, background: "#fff", borderRadius: 4 }, children: seg ? (_jsx(EegCanvas, { rows: rows, fsHz: seg.fsHz, gainUv: gain, windowS: windowS, panStartS: panStart, width: 1140, height: 760 })) : (_jsx("div", { style: { color: "#888", padding: 20 }, children: "loading EEG\u2026" })) })] }), _jsxs("div", { style: { display: "flex", gap: 16, alignItems: "center", marginTop: 8, fontSize: 13 }, children: [_jsxs("label", { children: ["Montage", " ", _jsx("select", { value: montage, onChange: (e) => setMontage(e.target.value), children: MONTAGES.map((m) => _jsx("option", { children: m }, m)) })] }), _jsxs("label", { children: ["Gain", " ", _jsx("select", { value: gain, onChange: (e) => setGain(parseInt(e.target.value, 10)), children: GAIN_LADDER.map((g) => _jsxs("option", { value: g, children: [g, " \u00B5V/div"] }, g)) })] }), _jsxs("label", { children: ["Bandpass", " ", _jsx("select", { value: bandpass, onChange: (e) => setBandpass(e.target.value), children: BANDPASS_OPTIONS.map((b) => _jsx("option", { children: b }, b)) })] }), _jsxs("label", { children: ["Notch", " ", _jsx("select", { value: notchHz, onChange: (e) => setNotchHz(e.target.value), children: NOTCH_OPTIONS.map((n) => _jsx("option", { children: n }, n)) })] }), _jsxs("label", { children: ["Window", " ", _jsx("select", { value: windowS, onChange: (e) => setWindowS(parseInt(e.target.value, 10)), children: WINDOW_OPTIONS.map((w) => _jsxs("option", { value: w, children: [w, " s"] }, w)) })] }), _jsx("button", { onClick: () => setPanStart((p) => Math.max(0, p - windowS / 2)), children: "\u25C0 Pan" }), _jsx("button", { onClick: () => setPanStart((p) => Math.min(Math.max(0, dur - windowS), p + windowS / 2)), children: "Pan \u25B6" }), _jsx("span", { style: { color: COLORS.textTertiary }, children: item && seg ? `EEG ${panStart.toFixed(1)}–${(panStart + windowS).toFixed(1)} s of ${dur.toFixed(1)} s · ${montage} · ${gain} µV/div` : "" })] })] }));
}
