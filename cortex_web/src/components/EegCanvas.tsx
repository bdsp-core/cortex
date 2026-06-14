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
  // Optional labeled-epoch overlay (the desktop's "red box" — v1.3.8). When
  // provided AND the labeled region intersects the visible window, a
  // translucent red rectangle + red border is drawn so the rater knows which
  // portion of the clip is being scored. Hidden for spike clips.
  labeledEpoch?: { startS: number; endS: number };
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

    const { rows, fsHz, gainUv, windowS, panStartS, width, height, labeledEpoch } = props;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 64, padR = 70, padT = 8, padB = 24;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const nRows = rows.length;
    const rowH = plotH / Math.max(nRows, 1);
    const s0 = Math.round(panStartS * fsHz);
    const nWin = Math.round(windowS * fsHz);

    // labeled-epoch overlay (drawn BEFORE traces so lines stay crisp on top).
    // Intersect the labeled span with the visible window and paint a
    // translucent red rectangle + red border. Mirrors the desktop's
    // LinearRegionItem framing the central scored 10 s.
    if (labeledEpoch) {
      const visLo = Math.max(labeledEpoch.startS, panStartS);
      const visHi = Math.min(labeledEpoch.endS, panStartS + windowS);
      if (visHi > visLo) {
        const xL = padL + ((visLo - panStartS) / windowS) * plotW;
        const xR = padL + ((visHi - panStartS) / windowS) * plotW;
        ctx.save();
        ctx.fillStyle = "rgba(245, 90, 75, 0.10)";
        ctx.fillRect(xL, padT, xR - xL, plotH);
        ctx.strokeStyle = "rgba(216, 80, 70, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        // only stroke the visible edges (so panning hides edges that are off-screen)
        if (visLo >= panStartS && labeledEpoch.startS >= panStartS) {
          ctx.moveTo(xL, padT); ctx.lineTo(xL, padT + plotH);
        }
        if (visHi <= panStartS + windowS && labeledEpoch.endS <= panStartS + windowS) {
          ctx.moveTo(xR, padT); ctx.lineTo(xR, padT + plotH);
        }
        ctx.stroke();
        ctx.restore();
      }
    }

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

  // display:block prevents the inline-baseline descender that lets the
  // canvas's height drift up by a few pixels per ResizeObserver cycle in a
  // column-flex parent (SpikeViewer's "slowly stretching" bug).
  return <canvas ref={ref}
    style={{ display: "block", width: props.width, height: props.height }} />;
}
