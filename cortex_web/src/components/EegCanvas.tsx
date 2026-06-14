// Canvas EEG renderer — port of the desktop's pyqtgraph EEG plot. White bg,
// black traces (red EKG), per-row offset, ±clip·gain clipping with gain µV =
// 1 row-unit, time-window + pan, 1-sec/gain scale bar.

import { useEffect, useRef } from "react";
import { MontageRow } from "../montage";
import { COLORS, EEG_CLIP_MULT } from "../../ui/theme";

export interface EegCanvasProps {
  rows: MontageRow[];
  fsHz: number;
  gainUv: number;
  windowS: number;
  panStartS: number;
  width: number;
  height: number;
  // The expert-labeled scoring epoch in CLIP-time seconds (IIIC = [10,20]).
  // When set, a translucent red box marks it (matches the desktop
  // LinearRegionItem, eeg_bank_viewer.py:1256-1264). Omit for spike clips.
  labelStartS?: number | null;
  labelLenS?: number | null;
}

export function EegCanvas(props: EegCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = props.width * dpr;
    cv.height = props.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { rows, fsHz, gainUv, windowS, panStartS, width, height,
            labelStartS, labelLenS } = props;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // padR is now small (scale bar moved inline, not at the right edge) so the
    // traces fill the width; padL fits the larger channel labels.
    const padL = 70, padR = 16, padT = 8, padB = 28;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const nRows = rows.length;
    const rowH = plotH / Math.max(nRows, 1);
    const s0 = Math.round(panStartS * fsHz);
    const nWin = Math.round(windowS * fsHz);

    // Scoring-epoch box (clip-time [labelStartS, labelStartS+labelLenS]),
    // drawn behind the grid + traces. Translucent red fill + solid border,
    // matching the desktop. Clamped to the visible [panStartS, panStartS+windowS].
    if (labelStartS != null && labelLenS != null) {
      const lo = Math.max(labelStartS, panStartS);
      const hi = Math.min(labelStartS + labelLenS, panStartS + windowS);
      if (hi > lo) {
        const xLo = padL + ((lo - panStartS) / windowS) * plotW;
        const xHi = padL + ((hi - panStartS) / windowS) * plotW;
        ctx.fillStyle = "rgba(255,80,80,0.137)"; // (255,80,80,35/255)
        ctx.fillRect(xLo, padT, xHi - xLo, plotH);
        ctx.strokeStyle = "rgb(215,45,45)";
        ctx.lineWidth = 2;
        ctx.strokeRect(xLo, padT, xHi - xLo, plotH);
      }
    }

    // time grid (1-sec)
    ctx.strokeStyle = "rgba(0,0,0,0.12)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#333";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    for (let sec = 0; sec <= windowS; sec++) {
      const x = padL + (sec / windowS) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
      ctx.fillText(`${(panStartS + sec).toFixed(0)}`, x, height - 8);
    }

    const clip = EEG_CLIP_MULT * gainUv;
    ctx.textAlign = "right";
    for (let r = 0; r < nRows; r++) {
      const row = rows[r];
      const yMid = padT + (r + 0.5) * rowH;
      // channel label
      if (row.name) {
        ctx.fillStyle = "#222";
        ctx.font = "12px system-ui";
        ctx.fillText(row.name, padL - 6, yMid + 3);
      }
      if (!row.data) continue; // separator
      ctx.strokeStyle = row.isEkg ? COLORS.ekgTrace : COLORS.eegTrace;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const yScale = rowH / (2 * gainUv); // gain_uv µV → half a row
      for (let i = 0; i < nWin; i++) {
        const si = s0 + i;
        if (si < 0 || si >= row.data.length) continue;
        let v = row.data[si];
        if (v > clip) v = clip;
        else if (v < -clip) v = -clip;
        const x = padL + (i / nWin) * plotW;
        const y = yMid - v * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // scale bar (1 s + gain µV): placed at the Fz-Cz / Cz-Pz boundary of the
    // bipolar montage, within the second-to-last second of the window. Falls
    // back to a low-central position for montages without a Cz-Pz row.
    const idxCzPz = rows.findIndex((r) => r.name === "Cz-Pz");
    const yA = idxCzPz >= 0 ? padT + idxCzPz * rowH : padT + plotH - rowH * 1.5;
    const xR = padL + ((windowS - 1) / windowS) * plotW; // corner at the (last-1)s mark
    const xL = padL + ((windowS - 2) / windowS) * plotW; // 1 s wide to its left
    const gainPx = rowH / 2; // gain_uv µV in pixels (1 div = rowH; ½ div shown)
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xL, yA);
    ctx.lineTo(xR, yA); // horizontal: 1 s
    ctx.moveTo(xR, yA);
    ctx.lineTo(xR, yA - gainPx); // vertical: gain µV, up from the corner
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("1 s", (xL + xR) / 2, yA + 15);
    ctx.fillText(`${gainUv} µV`, xR, yA - gainPx - 5);
  }, [props]);

  return <canvas ref={ref} style={{ width: props.width, height: props.height, display: "block" }} />;
}
