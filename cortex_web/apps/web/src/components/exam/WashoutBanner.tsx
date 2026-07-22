import { reopenLabel } from "../../washout";
import { Button } from "../ui";
import { COLORS } from "../../../ui/theme";

// Post-training exam-washout notice (server-enforced 12h gate in
// routers/testing.py). A top sheet that descends over a dimmed dashboard and
// requires explicit acknowledgment: the participant must recognize that
// testing is closed, not glance past a corner card.
export function WashoutBanner({ reopensAtUtc, onAccept }: {
  reopensAtUtc: string;
  onAccept: () => void;
}) {
  const label = reopenLabel(reopensAtUtc);
  return (
    <div role="alertdialog" aria-modal="true"
      aria-label="Testing temporarily unavailable"
      style={{
        position: "fixed", inset: 0, zIndex: 80,
        background: "rgba(20, 28, 26, 0.45)",
        display: "flex", alignItems: "center", justifyContent: "center",
        animation: "cx-washout-dim 300ms ease-out",
      }}>
      <style>{`
        @keyframes cx-washout-dim { from { background: rgba(20,28,26,0); }
                                    to { background: rgba(20,28,26,0.45); } }
        @keyframes cx-washout-descend { from { transform: translateY(-60vh); opacity: 0.4; }
                                        to { transform: none; opacity: 1; } }
        @keyframes cxBannerGlow {
          0%, 100% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 0 0 rgba(47,143,131,0); }
          50% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 24px 5px rgba(47,143,131,0.4); }
        }
        .cx-washout-card {
          animation: cx-washout-descend 460ms cubic-bezier(0.22, 0.8, 0.36, 1),
                     cxBannerGlow 3s ease-in-out 0.6s infinite;
        }
        @media (prefers-reduced-motion: reduce) { .cx-washout-card { animation: none; } }
      `}</style>
      <div className="cx-washout-card" style={{
        width: 560, maxWidth: "92vw", background: COLORS.card,
        border: `1px solid ${COLORS.borderInactive}`,
        borderTop: `3px solid ${COLORS.fail}`,
        boxShadow: "0 16px 44px rgba(15, 40, 36, 0.32)",
        padding: "26px 28px", boxSizing: "border-box",
      }}>
        <div style={{
          fontSize: 15, lineHeight: 1.6, color: COLORS.textPrimary,
        }}>
          You trained earlier today; to keep the exam a clean measure,
          testing reopens at <b>{label}</b>.
        </div>
        <div style={{ marginTop: 18, display: "flex", justifyContent: "flex-end" }}>
          <Button onClick={onAccept}>I understand</Button>
        </div>
      </div>
    </div>
  );
}
