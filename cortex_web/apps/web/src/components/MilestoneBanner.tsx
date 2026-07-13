// Unannounced-milestone notice, mounted on the dashboard surfaces alongside
// CohortInviteBanner (desktop Shell + mobile home). Presentation matches the
// exam-washout and cohort-invitation notices: a card descends smoothly to the
// center of a dimmed screen, with the shared slow teal glow, so the moment is
// recognized rather than glanced past. "Noted" acknowledges server-side
// (acknowledged_utc), so each milestone is announced exactly once; the full
// history stays visible in Settings under Badges & milestones.
//
// Deliberately NO schedule is shown: milestones arrive at irregular ladders
// (awards.py) and the surprise is the design. Reads the shared dashboard
// bootstrap (no extra request); milestones only appear after the user's own
// completed sitting/test, so no poll is needed.

import { useEffect, useState } from "react";
import * as api from "../api";
import { bootstrapOnce } from "../bootstrapStore";
import { COLORS, FONTS } from "../../ui/theme";

export function MilestoneBanner() {
  const [pending, setPending] = useState<api.PendingMilestone[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    bootstrapOnce()
      .then((b) => {
        if (!live || !b.awards) return;
        setPending(b.awards.pending);
      })
      .catch(() => { /* best-effort surface; next dashboard entry retries */ });
    return () => { live = false; };
  }, []);

  async function acknowledge() {
    setBusy(true);
    try {
      await Promise.all(pending.map((m) => api.ackAward(m.awardId)));
    } catch { /* unacked rows simply return next visit */ }
    setPending([]);
    setBusy(false);
  }

  if (pending.length === 0) return null;

  return (
    <div role="dialog" aria-modal="true" aria-label="Milestone reached"
      style={{
        position: "fixed", inset: 0, zIndex: 60,
        background: "rgba(20, 28, 26, 0.45)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: FONTS.sans, color: COLORS.textPrimary,
        animation: "cx-milestone-dim 300ms ease-out",
      }}>
      <style>{`
        @keyframes cx-milestone-dim { from { background: rgba(20,28,26,0); }
                                      to { background: rgba(20,28,26,0.45); } }
        @keyframes cx-milestone-descend { from { transform: translateY(-60vh); opacity: 0.4; }
                                          to { transform: none; opacity: 1; } }
        @keyframes cxBannerGlow {
          0%, 100% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 0 0 rgba(47,143,131,0); }
          50% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 24px 5px rgba(47,143,131,0.4); }
        }
        .cx-milestone-card {
          animation: cx-milestone-descend 460ms cubic-bezier(0.22, 0.8, 0.36, 1),
                     cxBannerGlow 3s ease-in-out 0.6s infinite;
        }
        @media (prefers-reduced-motion: reduce) { .cx-milestone-card { animation: none; } }
      `}</style>
      <div className="cx-milestone-card" style={{
        width: 560, maxWidth: "92vw", background: COLORS.card,
        border: `1px solid ${COLORS.borderInactive}`,
        borderTop: "3px solid var(--teal)",
        boxShadow: "0 16px 44px rgba(15, 40, 36, 0.32)",
      }}>
        <div style={{ padding: "14px 18px", borderBottom: `1px solid ${COLORS.borderInactive}` }}>
          <span style={{ fontFamily: FONTS.serif, fontSize: 18, fontWeight: 600 }}>
            Milestone{pending.length > 1 ? "s" : ""} reached
          </span>
        </div>
        {pending.map((m) => (
          <div key={m.awardId} style={{
            display: "flex", alignItems: "baseline", gap: 10,
            padding: "13px 18px", borderBottom: `1px solid ${COLORS.borderInactive}`,
          }}>
            <span aria-hidden style={{ color: "var(--teal)", fontSize: 13 }}>●</span>
            <div style={{ minWidth: 0 }}>
              <b style={{ color: "var(--teal-deep)", fontSize: 14 }}>{m.label}</b>
              {m.detail && (
                <div style={{ fontSize: 13, color: COLORS.textBody, marginTop: 2 }}>
                  {m.detail}
                </div>
              )}
            </div>
          </div>
        ))}
        <div style={{ display: "flex", justifyContent: "flex-end", padding: "12px 18px" }}>
          <button disabled={busy} onClick={() => void acknowledge()}
            style={{
              border: "1px solid transparent", borderRadius: 4, cursor: "pointer",
              fontFamily: FONTS.sans, fontSize: 13, fontWeight: 600,
              padding: "8px 16px", background: "var(--teal)", color: "#fff",
            }}>
            Noted
          </button>
        </div>
      </div>
    </div>
  );
}
