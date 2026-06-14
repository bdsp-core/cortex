// Canvas spectrogram renderer — 4 stacked region panels (LL/RL/LP/RP), jet
// colormap, fixed [-10,25] dB. The bundle ships sdata already dB-quantized to
// uint8 over that range, so we just map uint8 → jet (no recompute), matching
// the desktop pixel-for-pixel. Frequency (Hz) ticks on the left of each panel
// and a shared time (s) axis along the bottom mirror the desktop axes.
//
// sdata layout (from the precompute): (nTimes, nFreqs*4) with the four regions
// concatenated along the freq axis in LL,RL,LP,RP order. specShape = [nTimes,
// nFreqs*4].

import { useEffect, useRef } from "react";
import { JET_LUT, SPEC_REGIONS } from "../../ui/theme";

const SPEC_FREQ_FALLBACK: [number, number] = [0.5, 25.0]; // masked band (Hz)

// jet lookup: value u8 (0..255) → rgb, piecewise-linear over the 9 stops.
function jet(u8: number): [number, number, number] {
  const t = (u8 / 255) * (JET_LUT.length - 1);
  const i = Math.min(JET_LUT.length - 2, Math.floor(t));
  const f = t - i;
  const a = JET_LUT[i], b = JET_LUT[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

// "nice" round tick values spanning [lo,hi], ≤ ~target ticks.
function ticks(lo: number, hi: number, target = 5): number[] {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * mag >= raw)! * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

export interface SpecCanvasProps {
  spec: { data: Uint8Array; shape: number[]; time?: number[] | null; freq?: number[] | null } | null;
  width: number;
  height: number;
  // Fractional x-position (0..1) of an optional single dashed marker (legacy
  // centered "you are here" marker — superseded by clipBoundsFrac for IIIC).
  markerFrac?: number | null;
  // Fractional x-positions of the 30-s EEG clip's bounds within the 10-min
  // spectrogram (desktop v1.3.8). When provided, drawn across all 4 region
  // panels and takes priority over markerFrac.
  clipBoundsFrac?: [number, number] | null;
}

export function SpecCanvas({
  spec, width, height, markerFrac = null, clipBoundsFrac = null,
}: SpecCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    if (!spec || spec.shape.length !== 2) return;

    const [nTimes, nFreqAll] = spec.shape;
    const nReg = SPEC_REGIONS.length;
    const nFreq = Math.floor(nFreqAll / nReg);
    const labelW = 38;           // left margin: freq ticks + region label
    const padB = 24;             // bottom margin: shared time axis
    const plotW = width - labelW;
    const panelH = (height - padB) / nReg;

    const [f0, f1] = spec.freq && spec.freq.length === 2 ? (spec.freq as [number, number]) : SPEC_FREQ_FALLBACK;
    const [t0, t1] = spec.time && spec.time.length === 2 ? (spec.time as [number, number]) : [0, nTimes];

    for (let reg = 0; reg < nReg; reg++) {
      const y0 = reg * panelH;
      const img = ctx.createImageData(nTimes, nFreq);
      for (let ti = 0; ti < nTimes; ti++) {
        for (let fi = 0; fi < nFreq; fi++) {
          const u8 = spec.data[ti * nFreqAll + reg * nFreq + fi];
          const [r, g, b] = jet(u8);
          const py = nFreq - 1 - fi; // flip freq so low freq at bottom
          const o = (py * nTimes + ti) * 4;
          img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
        }
      }
      const tmp = document.createElement("canvas");
      tmp.width = nTimes; tmp.height = nFreq;
      tmp.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, labelW, y0 + 1, plotW, panelH - 2);

      // freq (Hz) ticks on the left edge of this panel
      ctx.fillStyle = "#222";
      ctx.font = "11px system-ui";
      ctx.textAlign = "right";
      for (const fv of ticks(f0, f1, 6)) { // ~5 Hz steps: 5,10,15,20,25
        const yy = y0 + (panelH - 2) * (1 - (fv - f0) / (f1 - f0)) + 1;
        ctx.fillText(`${fv.toFixed(0)}`, labelW - 4, yy + 4);
      }
      // region tag, top-left — large WHITE + dark outline for contrast.
      ctx.textAlign = "left";
      ctx.font = "bold 16px system-ui";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.strokeText(SPEC_REGIONS[reg], labelW + 5, y0 + 19);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(SPEC_REGIONS[reg], labelW + 5, y0 + 19);
    }

    // shared time (s) axis along the bottom
    const axisY = height - padB;
    ctx.fillStyle = "#222";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    for (const tv of ticks(t0, t1, 6)) { // ~100 s steps
      const xx = labelW + ((tv - t0) / (t1 - t0)) * plotW;
      ctx.fillText(`${tv.toFixed(0)}`, xx, axisY + 15);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#444";
    ctx.fillText("Time (s)", 2, axisY + 15); // far-left corner — clears the first tick
    ctx.save();
    ctx.translate(9, (height - padB) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#444";
    ctx.fillText("Freq (Hz)", 0, 0);
    ctx.restore();

    // dotted-white verticals at the 30-s clip bounds (desktop v1.3.8) or the
    // legacy single centered marker. Spans the four region panels (not the axis).
    const drawDottedV = (frac: number) => {
      const x = labelW + Math.max(0, Math.min(1, frac)) * plotW;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height - padB); ctx.stroke();
    };
    if (clipBoundsFrac) {
      ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.lineWidth = 1.5;
      drawDottedV(clipBoundsFrac[0]); drawDottedV(clipBoundsFrac[1]); ctx.restore();
    } else if (markerFrac != null) {
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.lineWidth = 1.5;
      drawDottedV(markerFrac); ctx.restore();
    }
  }, [spec, width, height, markerFrac, clipBoundsFrac]);

  return <canvas ref={ref} style={{ display: "block", width, height }} />;
}
