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
  // When true, trace drawing is clipped to the plot rectangle so a hard
  // deflection (up to ±EEG_CLIP_MULT×gain ≈ 1.5 rows) can't overflow past the
  // bottom row into the time-axis labels. Default false → exam rendering is
  // byte-identical; the trainer opts in.
  clipTraces?: boolean;
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

    // padR is small (the scale bar is now inline at the Fz-Cz/Cz-Pz boundary,
    // not at the right edge) so the traces fill the width; padL fits the larger
    // channel labels.
    const padL = 70, padR = 16, padT = 8, padB = 28;
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
        ctx.fillStyle = "rgba(255,80,80,0.137)"; // (255,80,80,35/255)
        ctx.fillRect(xL, padT, xR - xL, plotH);
        ctx.strokeStyle = "rgb(215,45,45)";
        ctx.lineWidth = 2;
        ctx.strokeRect(xL, padT, xR - xL, plotH);
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

    // Channel labels — drawn in the left gutter BEFORE the trace clip. clipTraces
    // clips to the plot rect (x ≥ padL); the labels sit at x < padL, so drawing
    // them inside that clip (as they were) erased them for the trainer — the only
    // clipTraces caller. Labels never overlap the traces (x ≥ padL), so for the
    // unclipped exam path this is pixel-identical to the previous per-row draw.
    ctx.textAlign = "right";
    ctx.fillStyle = "#222";
    ctx.font = "12px system-ui";
    for (let r = 0; r < nRows; r++) {
      const row = rows[r];
      if (row.name) ctx.fillText(row.name, padL - 6, padT + (r + 0.5) * rowH + 3);
    }

    // keep traces inside the plot so they never paint over the time labels
    if (props.clipTraces) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();
    }
    for (let r = 0; r < nRows; r++) {
      const row = rows[r];
      const yMid = padT + (r + 0.5) * rowH;
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
    if (props.clipTraces) ctx.restore();

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
    ctx.moveTo(xL, yA); ctx.lineTo(xR, yA); // horizontal: 1 s
    ctx.moveTo(xR, yA); ctx.lineTo(xR, yA - gainPx); // vertical: gain µV
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("1 s", (xL + xR) / 2, yA + 15);
    ctx.fillText(`${gainUv} µV`, xR, yA - gainPx - 5);
    // Explicit deps (NOT `[props]`): every parent render hands us a fresh props
    // object, which re-ran this full canvas draw (montage/DSP loop over all
    // channels × samples) on any unrelated re-render. `rows` is memoized in the
    // Viewer, and labeledEpoch's fields are constants, so this only redraws when
    // something it actually paints changes.
  }, [props.rows, props.fsHz, props.gainUv, props.windowS, props.panStartS,
      props.width, props.height, props.labeledEpoch?.startS,
      props.labeledEpoch?.endS, props.clipTraces]);

  // display:block prevents the inline-baseline descender that lets the
  // canvas's height drift up by a few pixels per ResizeObserver cycle in a
  // column-flex parent (SpikeViewer's "slowly stretching" bug).
  return <canvas ref={ref}
    style={{ display: "block", width: props.width, height: props.height }} />;
}
