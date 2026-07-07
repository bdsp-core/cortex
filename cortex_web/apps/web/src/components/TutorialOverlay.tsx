// Coach-marks tutorial overlay — port of TutorialOverlay (scripts/eeg_bank_viewer.py
// 357-542). Dims the live viewer with a veil and spotlights one region at a time
// (accent border), with a stepping card (STEP i OF n → title → body → NEXT/BEGIN).
// The backdrop is the real Viewer showing an example segment, so the walkthrough
// happens IN CONTEXT rather than on a separate start screen.

import { ReactNode, RefObject, useLayoutEffect, useState } from "react";
import { COLORS, FONTS } from "../../ui/theme";

export interface TutorialStep {
  target: RefObject<HTMLElement> | null; // null = centered card, no spotlight
  title: string;
  body: ReactNode;
}

const VEIL = "rgba(8,9,12,0.745)"; // desktop QColor(8,9,12,190)
const PAD = 7; // spotlight padding (desktop .adjusted(-7,-7,7,7))
const GAP = 22;
const CARD_W = 420;
const EST_H = 320; // card-height estimate for placement

function veil(x: number, y: number, w: number, h: number): React.CSSProperties {
  return { position: "absolute", left: x, top: y, width: Math.max(0, w), height: Math.max(0, h), background: VEIL };
}

// Place the card beside the spotlight: prefer right, then left, then below,
// then above — whichever fits the viewport — else centered.
function placeCard(tr: { x: number; y: number; w: number; h: number } | null, W: number, H: number) {
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(v, Math.max(lo, hi)));
  if (!tr) return { x: (W - CARD_W) / 2, y: Math.max(20, (H - EST_H) / 2) };
  if (tr.x + tr.w + GAP + CARD_W <= W) // right
    return { x: tr.x + tr.w + GAP, y: clamp(tr.y, 12, H - EST_H - 12) };
  if (tr.x - GAP - CARD_W >= 0) // left
    return { x: tr.x - GAP - CARD_W, y: clamp(tr.y, 12, H - EST_H - 12) };
  if (tr.y + tr.h + GAP + EST_H <= H) // below
    return { x: clamp(tr.x + tr.w / 2 - CARD_W / 2, 12, W - CARD_W - 12), y: tr.y + tr.h + GAP };
  return { x: clamp(tr.x + tr.w / 2 - CARD_W / 2, 12, W - CARD_W - 12), y: clamp(tr.y - GAP - EST_H, 12, H - EST_H - 12) }; // above
}

export function TutorialOverlay({ steps, onFinish }: { steps: TutorialStep[]; onFinish: () => void }) {
  const [i, setI] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [tick, setTick] = useState(0); // bump to recompute on resize

  useLayoutEffect(() => {
    const el = steps[i]?.target?.current ?? null;
    setRect(el ? el.getBoundingClientRect() : null);
  }, [i, steps, tick]);

  useLayoutEffect(() => {
    const on = () => setTick((t) => t + 1);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);

  const last = i === steps.length - 1;
  const advance = () => (last ? onFinish() : setI((v) => v + 1));

  const W = typeof window !== "undefined" ? window.innerWidth : 1500;
  const H = typeof window !== "undefined" ? window.innerHeight : 950;
  const tr = rect ? { x: rect.left - PAD, y: rect.top - PAD, w: rect.width + 2 * PAD, h: rect.height + 2 * PAD } : null;
  const card = placeCard(tr, W, H);

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, fontFamily: FONTS.sans }}>
      {tr ? (
        <>
          <div style={veil(0, 0, W, tr.y)} />
          <div style={veil(0, tr.y + tr.h, W, H - tr.y - tr.h)} />
          <div style={veil(0, tr.y, tr.x, tr.h)} />
          <div style={veil(tr.x + tr.w, tr.y, W - tr.x - tr.w, tr.h)} />
          <div style={{
            position: "absolute", left: tr.x, top: tr.y, width: tr.w, height: tr.h,
            border: `2px solid ${COLORS.accent}`, pointerEvents: "none",
          }} />
        </>
      ) : (
        <div style={veil(0, 0, W, H)} />
      )}

      <div style={{
        position: "absolute", left: card.x, top: card.y, width: CARD_W, boxSizing: "border-box",
        background: COLORS.card, border: `1px solid ${COLORS.borderInactive}`, padding: "20px 24px",
      }}>
        <div style={{ color: "#6b7280", fontSize: 11, fontWeight: 500, letterSpacing: 1 }}>
          STEP {i + 1} OF {steps.length}
        </div>
        <div style={{ color: COLORS.textPrimary, fontSize: 18, fontWeight: 600, marginTop: 7 }}>
          {steps[i].title}
        </div>
        <div style={{ color: COLORS.textBody, fontSize: 13, marginTop: 10, lineHeight: 1.5 }}>
          {steps[i].body}
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
          <button onClick={advance} style={{
            height: 38, minWidth: last ? 132 : 100, cursor: "pointer",
            background: COLORS.accent, color: "#fff", border: "none", borderRadius: 4,
            fontWeight: 700, fontSize: 13, fontFamily: FONTS.sans,
          }}>
            {last ? "BEGIN" : "NEXT"}
          </button>
        </div>
      </div>
    </div>
  );
}
