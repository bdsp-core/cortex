// Landing screen — CORTEX wordmark over a faint static multi-channel EEG
// backdrop, with a single "BEGIN ASSESSMENT" call to action. Matches the
// desktop landing (980×660, backdrop pen #2e425e, button 252×50).

import { useEffect, useRef } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { Button, Stage, Wordmark } from "./ui";

// 16 channels of deterministic pseudo-EEG drawn once as a decorative backdrop.
function EegBackdrop({ width, height }: { width: number; height: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const nCh = 16;
    const rowH = height / nCh;
    ctx.strokeStyle = COLORS.eegBackdrop;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.55;
    for (let ch = 0; ch < nCh; ch++) {
      const y0 = rowH * (ch + 0.5);
      ctx.beginPath();
      // a few mixed sinusoids + a deterministic jitter, seeded by channel
      const f1 = 0.06 + 0.01 * ch, f2 = 0.18 + 0.013 * ch, ph = ch * 1.7;
      for (let x = 0; x <= width; x += 2) {
        const jitter = Math.sin(x * 0.7 + ch * 3.1) * 1.5;
        const y =
          y0 +
          (Math.sin(x * f1 + ph) * rowH * 0.18 +
            Math.sin(x * f2 + ph * 2) * rowH * 0.1 +
            jitter);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }, [width, height]);
  return (
    <canvas
      ref={ref}
      style={{ position: "absolute", inset: 0, width, height, pointerEvents: "none" }}
    />
  );
}

export function Landing({ onBegin }: { onBegin: () => void }) {
  return (
    <Stage maxW={980}>
      <div
        style={{
          position: "relative",
          borderRadius: 12,
          overflow: "hidden",
          border: `1px solid ${COLORS.borderInactive}`,
          background: COLORS.bg,
          aspectRatio: "980 / 660",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <EegBackdrop width={980} height={660} />
        <div style={{ position: "relative", textAlign: "center" }}>
          <Wordmark size={84} />
          <div
            style={{
              fontFamily: FONTS.serif,
              color: COLORS.textBody,
              fontSize: 18,
              letterSpacing: 2,
              marginTop: 4,
              marginBottom: 48,
            }}
          >
            EEG Skill Certification
          </div>
          <Button onClick={onBegin} style={{ width: 252, height: 50, fontSize: 16, letterSpacing: 1 }}>
            BEGIN ASSESSMENT
          </Button>
        </div>
      </div>
    </Stage>
  );
}
