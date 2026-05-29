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

    const { rows, fsHz, gainUv, windowS, panStartS, width, height } = props;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 64, padR = 70, padT = 8, padB = 24;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const nRows = rows.length;
    const rowH = plotH / Math.max(nRows, 1);
    const s0 = Math.round(panStartS * fsHz);
    const nWin = Math.round(windowS * fsHz);

    // time grid (1-sec)
    ctx.strokeStyle = "rgba(0,0,0,0.12)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#333";
    ctx.font = "10px system-ui";
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
        ctx.font = "10px system-ui";
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

    // scale bar: 1 sec + gain_uv µV, bottom-right
    const bx = padL + plotW - 4;
    const by = padT + plotH - 4;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const oneSecPx = plotW / windowS;
    ctx.moveTo(bx - oneSecPx, by);
    ctx.lineTo(bx, by);
    ctx.lineTo(bx, by - rowH / 2);
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.font = "10px system-ui";
    ctx.textAlign = "right";
    ctx.fillText("1 s", bx, by + 14);
    ctx.textAlign = "left";
    ctx.fillText(`${gainUv} µV`, bx + 4, by - rowH / 4);
  }, [props]);

  return <canvas ref={ref} style={{ width: props.width, height: props.height }} />;
}
