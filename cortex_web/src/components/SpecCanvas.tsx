// Canvas spectrogram renderer — 4 stacked region panels (LL/RL/LP/RP), jet
// colormap, fixed [-10,25] dB. The bundle ships sdata already dB-quantized to
// uint8 over that range, so we just map uint8 → jet (no recompute), matching
// the desktop pixel-for-pixel.
//
// sdata layout (from the precompute): (nTimes, nFreqs*4) with the four regions
// concatenated along the freq axis in LL,RL,LP,RP order. specShape = [nTimes,
// nFreqs*4].

import { useEffect, useRef } from "react";
import { JET_LUT, SPEC_REGIONS } from "../../ui/theme";

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

export interface SpecCanvasProps {
  spec: { data: Uint8Array; shape: number[] } | null;
  width: number;
  height: number;
  // Fractional x-position (0..1) of an optional single dashed marker (legacy
  // centered "you are here" marker — superseded by clipBoundsFrac for IIIC).
  markerFrac?: number | null;
  // Fractional x-positions of the 30-s EEG clip's bounds within the 10-min
  // spectrogram (desktop v1.3.8: two dotted-white verticals at 285 s and
  // 315 s = 0.475 and 0.525). When provided, drawn across all 4 region
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
    const panelH = height / nReg;
    const labelW = 38;
    const plotW = width - labelW;

    ctx.fillStyle = "#222";
    ctx.font = "9px system-ui";

    for (let reg = 0; reg < nReg; reg++) {
      const y0 = reg * panelH;
      // build an offscreen image (nFreq tall × nTimes wide), freq low→bottom
      const img = ctx.createImageData(nTimes, nFreq);
      for (let ti = 0; ti < nTimes; ti++) {
        for (let fi = 0; fi < nFreq; fi++) {
          const u8 = spec.data[ti * nFreqAll + reg * nFreq + fi];
          const [r, g, b] = jet(u8);
          // flip freq so low freq at bottom
          const py = nFreq - 1 - fi;
          const o = (py * nTimes + ti) * 4;
          img.data[o] = r;
          img.data[o + 1] = g;
          img.data[o + 2] = b;
          img.data[o + 3] = 255;
        }
      }
      // blit scaled into the panel via a temp canvas
      const tmp = document.createElement("canvas");
      tmp.width = nTimes;
      tmp.height = nFreq;
      tmp.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, labelW, y0 + 1, plotW, panelH - 2);
      ctx.fillStyle = "#222";
      ctx.fillText(SPEC_REGIONS[reg], 4, y0 + 12);
    }

    // dotted-white verticals at the 30-s clip bounds (desktop v1.3.8) or
    // legacy single centered marker. Spans all four region panels.
    const drawDottedV = (frac: number) => {
      const x = labelW + Math.max(0, Math.min(1, frac)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    };
    if (clipBoundsFrac) {
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.5;
      drawDottedV(clipBoundsFrac[0]);
      drawDottedV(clipBoundsFrac[1]);
      ctx.restore();
    } else if (markerFrac != null) {
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.5;
      drawDottedV(markerFrac);
      ctx.restore();
    }
  }, [spec, width, height, markerFrac, clipBoundsFrac]);

  return <canvas ref={ref} style={{ width, height }} />;
}
