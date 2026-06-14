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
import { JET_LUT, SPEC_REGIONS, SPEC_FREQ_RANGE } from "../../ui/theme";

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

// "nice" round tick values spanning [lo,hi] (inclusive-ish), ≤ ~6 ticks.
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
  spec: { data: Uint8Array; shape: number[]; time: number[] | null; freq: number[] | null } | null;
  width: number;
  height: number;
  // Fractional x-position (0..1) of the dashed marker showing where the
  // displayed EEG segment sits in the 10-min spectrogram. The 30-sec EEG is
  // centered in the 600-sec spectrogram, so this is 0.5. null = no marker.
  markerFrac?: number | null;
}

export function SpecCanvas({ spec, width, height, markerFrac = null }: SpecCanvasProps) {
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
    const labelW = 38;            // left margin: freq ticks + region label
    const padB = 24;              // bottom margin: shared time axis
    const plotW = width - labelW;
    const panelH = (height - padB) / nReg;

    const [f0, f1] = spec.freq && spec.freq.length === 2 ? spec.freq : SPEC_FREQ_RANGE;
    const [t0, t1] = spec.time && spec.time.length === 2 ? spec.time : [0, nTimes];

    for (let reg = 0; reg < nReg; reg++) {
      const y0 = reg * panelH;
      // build an offscreen image (nFreq tall × nTimes wide), freq low→bottom
      const img = ctx.createImageData(nTimes, nFreq);
      for (let ti = 0; ti < nTimes; ti++) {
        for (let fi = 0; fi < nFreq; fi++) {
          const u8 = spec.data[ti * nFreqAll + reg * nFreq + fi];
          const [r, g, b] = jet(u8);
          const py = nFreq - 1 - fi; // flip freq so low freq at bottom
          const o = (py * nTimes + ti) * 4;
          img.data[o] = r;
          img.data[o + 1] = g;
          img.data[o + 2] = b;
          img.data[o + 3] = 255;
        }
      }
      const tmp = document.createElement("canvas");
      tmp.width = nTimes;
      tmp.height = nFreq;
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
        ctx.strokeStyle = "rgba(0,0,0,0.18)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(labelW - 2, yy);
        ctx.lineTo(labelW, yy);
        ctx.stroke();
      }
      // region tag, top-left — large WHITE with a dark outline so it reads on
      // any spectrogram color underneath.
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
    for (const tv of ticks(t0, t1, 6)) { // ~100 s steps: 100,200,…,500
      const xx = labelW + ((tv - t0) / (t1 - t0)) * plotW;
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xx, axisY);
      ctx.lineTo(xx, axisY + 3);
      ctx.stroke();
      ctx.fillText(`${tv.toFixed(0)}`, xx, axisY + 15);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#444";
    // Far-left corner of the axis row so it never overlaps the first time tick
    // ("100"), which sits ~40px in from the plot's left edge.
    ctx.fillText("Time (s)", 2, axisY + 15);
    // freq axis caption (rotated, far left)
    ctx.save();
    ctx.translate(9, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#444";
    ctx.fillText("Freq (Hz)", 0, 0);
    ctx.restore();

    // dashed white vertical marker — location of the displayed EEG segment
    // (centered in the 10-min spectrogram). Spans the four region panels.
    if (markerFrac != null) {
      const mx = labelW + Math.max(0, Math.min(1, markerFrac)) * plotW;
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mx, 0);
      ctx.lineTo(mx, height - padB);
      ctx.stroke();
      ctx.restore();
    }
  }, [spec, width, height, markerFrac]);

  return <canvas ref={ref} style={{ width, height, display: "block" }} />;
}
