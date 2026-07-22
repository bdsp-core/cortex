import { useCallback, useEffect, useRef, useState } from "react";

import { COLORS, FONTS } from "../../ui/theme";

// Keep ordinary next-question transitions quiet; surface status only after the
// wait crosses the one-second interaction threshold typical of rejuvenation.
export const TRAJECTORY_STATUS_DELAY_MS = 1_000;

export function useTrajectoryOptimizationStatus(questionIndex: number | null) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) globalThis.clearTimeout(timerRef.current);
    timerRef.current = null;
  }, []);

  const beginWait = useCallback(() => {
    clearTimer();
    setVisible(false);
    timerRef.current = globalThis.setTimeout(() => {
      timerRef.current = null;
      setVisible(true);
    }, TRAJECTORY_STATUS_DELAY_MS);
  }, [clearTimer]);

  useEffect(() => {
    clearTimer();
    setVisible(false);
    return clearTimer;
  }, [clearTimer, questionIndex]);

  return { beginWait, visible };
}

export function TrajectoryOptimizationStatus() {
  return (
    <div role="status" aria-live="polite" aria-atomic="true" aria-busy="true"
      style={{
        position: "fixed", inset: 0, zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: 20, background: "rgba(11, 31, 28, 0.58)",
      }}>
      <div className="cx-trajectory-status-card" style={{
        width: 390, maxWidth: "100%", boxSizing: "border-box",
        padding: "30px 28px 28px", textAlign: "center",
        background: COLORS.card, border: "1px solid var(--teal-mid)",
        borderTop: `3px solid ${COLORS.accent}`,
        borderRadius: "var(--radius-panel)",
        boxShadow: "0 18px 48px rgba(10, 48, 42, 0.34)",
      }}>
        <div className="cx-trajectory-status-spinner" aria-hidden="true" style={{
          width: 38, height: 38, margin: "0 auto 18px",
          border: "3px solid var(--teal-mid)",
          borderTopColor: COLORS.accent, borderRadius: "50%",
        }} />
        <div style={{
          color: "var(--teal-deep)", fontFamily: FONTS.sans,
          fontSize: 17, lineHeight: 1.35, fontWeight: 700,
        }}>
          Optimizing Question Trajectory
        </div>
        <div style={{
          marginTop: 8, color: COLORS.textBody, fontFamily: FONTS.sans,
          fontSize: 13, lineHeight: 1.45,
        }}>
          Preparing the next question…
        </div>
      </div>
      <style>{`
        @keyframes cxTrajectorySpin { to { transform: rotate(360deg); } }
        @keyframes cxTrajectoryEnter {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: none; }
        }
        .cx-trajectory-status-card { animation: cxTrajectoryEnter .18s ease-out; }
        .cx-trajectory-status-spinner { animation: cxTrajectorySpin .85s linear infinite; }
        @media (prefers-reduced-motion: reduce) {
          .cx-trajectory-status-card, .cx-trajectory-status-spinner { animation: none; }
        }
      `}</style>
    </div>
  );
}
