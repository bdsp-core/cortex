// "Pending cohort invitation" notice, mounted on the dashboard surfaces
// (desktop Shell + mobile home; shared component, allowlisted in the mobile
// boundary). Shows the caller's status='invited' cohorts and lets them
// Accept / Decline inline: the same consent gate as the Cohorts screen,
// just a faster path to it. Self-contained styling (no shellCss dependency)
// so it renders identically on both surfaces.
//
// Presentation matches the exam-washout notice: a box descends smoothly to
// the center of a dimmed screen so the invitation is recognized rather than
// glanced past. "Maybe later" defers for the session (sessionStorage; it
// returns next visit); slow-poll (60s) while mounted picks up new invites
// without a reload; the cohort a /?cohort=... email deep-linked to gets a
// brief highlight pulse.

import { useCallback, useEffect, useState } from "react";
import * as api from "../api";
import { bootstrapOnce } from "../bootstrapStore";
import { COLORS, FONTS } from "../../ui/theme";

const DISMISS_KEY = "cortex.inviteBannerDismissed";
const POLL_MS = 60_000;

const btnBase = {
  border: "1px solid transparent", borderRadius: 4, cursor: "pointer",
  fontFamily: FONTS.sans, fontSize: 13, fontWeight: 600 as const,
  padding: "8px 14px",
};

export function CohortInviteBanner({ highlightCohortId }: {
  // variant is accepted for caller compatibility; the centered modal renders
  // identically on both surfaces.
  variant: "desktop" | "mobile";
  highlightCohortId?: string | null;
}) {
  const [invites, setInvites] = useState<api.CohortSummary[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(() => {
    try { return sessionStorage.getItem(DISMISS_KEY) === "1"; } catch { return false; }
  });

  const refresh = useCallback(async () => {
    try {
      const { cohorts } = await api.listCohorts();
      setInvites(cohorts.filter((c) => c.status === "invited" && c.role === "member"));
    } catch { /* best-effort surface; the next poll retries */ }
  }, []);

  useEffect(() => {
    let live = true;
    // Initial paint rides the shared dashboard bootstrap (no extra request);
    // the standalone-list poll keeps invites fresh while mounted. A null
    // cohorts section just waits for the first poll tick.
    bootstrapOnce()
      .then((b) => {
        if (!live || !b.cohorts) return;
        setInvites(b.cohorts.cohorts.filter(
          (c) => c.status === "invited" && c.role === "member"));
      })
      .catch(() => { /* the poll below retries */ });
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => { live = false; window.clearInterval(id); };
  }, [refresh]);

  async function act(cohortId: string, fn: (id: string) => Promise<unknown>) {
    setBusy(cohortId);
    try { await fn(cohortId); } catch { /* refresh below re-syncs state */ }
    await refresh();
    setBusy(null);
  }

  function dismiss() {
    setDismissed(true);
    try { sessionStorage.setItem(DISMISS_KEY, "1"); } catch { /* private mode */ }
  }

  if (dismissed || invites.length === 0) return null;

  // Same treatment as the exam-washout notice: a box that descends smoothly
  // from the top to the CENTER of a dimmed screen, so a pending invitation
  // is recognized rather than glanced past. Teal rule (an offer, not a
  // restriction); "Maybe later" defers for the session.
  return (
    <div role="dialog" aria-modal="true" aria-label="Cohort invitation"
      style={{
        position: "fixed", inset: 0, zIndex: 70,
        background: "rgba(20, 28, 26, 0.45)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: FONTS.sans, color: COLORS.textPrimary,
        animation: "cx-invite-dim 300ms ease-out",
      }}>
      <style>{`
        @keyframes cx-invite-dim { from { background: rgba(20,28,26,0); }
                                   to { background: rgba(20,28,26,0.45); } }
        @keyframes cx-invite-descend { from { transform: translateY(-60vh); opacity: 0.4; }
                                       to { transform: none; opacity: 1; } }
        @keyframes cx-invite-pulse { 0%, 100% { background: transparent; }
                                     50% { background: var(--teal-weak); } }
        @keyframes cxBannerGlow {
          0%, 100% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 0 0 rgba(47,143,131,0); }
          50% { box-shadow: 0 16px 44px rgba(15,40,36,0.32), 0 0 24px 5px rgba(47,143,131,0.4); }
        }
        .cx-invite-card {
          animation: cx-invite-descend 460ms cubic-bezier(0.22, 0.8, 0.36, 1),
                     cxBannerGlow 3s ease-in-out 0.6s infinite;
        }
        @media (prefers-reduced-motion: reduce) { .cx-invite-card { animation: none; } }
      `}</style>
      <div className="cx-invite-card" style={{
        width: 560, maxWidth: "92vw", background: COLORS.card,
        border: `1px solid ${COLORS.borderInactive}`,
        borderTop: "3px solid var(--teal)",
        boxShadow: "0 16px 44px rgba(15, 40, 36, 0.32)",
      }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8,
        padding: "12px 16px", borderBottom: `1px solid ${COLORS.borderInactive}` }}>
        <span style={{ fontSize: 14, fontWeight: 700 }}>
          Cohort invitation{invites.length > 1 ? `s (${invites.length})` : ""}
        </span>
        <span style={{ marginLeft: "auto" }} />
        <button className="cx-invite-later" onClick={dismiss}
          style={{ background: "none", cursor: "pointer",
            border: `1px solid ${COLORS.borderInactive2}`, borderRadius: 4,
            color: COLORS.textBody, fontFamily: FONTS.sans, fontSize: 12,
            padding: "6px 10px" }}>
          Maybe later
        </button>
      </div>
      {invites.map((c) => (
        <div key={c.cohortId} style={{
          display: "flex", alignItems: "center", flexWrap: "wrap", gap: 10,
          padding: "14px 16px",
          animation: c.cohortId === highlightCohortId
            ? "cx-invite-pulse 1.2s ease-in-out 2" : undefined,
        }}>
          <div style={{ flex: "1 1 140px", minWidth: 0, fontSize: 14 }}>
            You&apos;ve been invited to join{" "}
            <b style={{ color: "var(--teal-deep)" }}>{c.name}</b>.
          </div>
          <span style={{ display: "flex", gap: 8 }}>
            <button disabled={busy === c.cohortId}
              onClick={() => void act(c.cohortId, api.acceptCohortInvite)}
              style={{ ...btnBase, background: "var(--teal)", color: "#fff" }}>
              Accept
            </button>
            <button disabled={busy === c.cohortId}
              onClick={() => void act(c.cohortId, api.declineCohortInvite)}
              style={{ ...btnBase, background: "none",
                borderColor: COLORS.borderInactive2, color: COLORS.textBody }}>
              Decline
            </button>
          </span>
        </div>
      ))}
      </div>
    </div>
  );
}
