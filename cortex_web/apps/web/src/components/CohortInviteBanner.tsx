// Floating "pending cohort invitation" banner, mounted on the dashboard
// surfaces (desktop Shell + mobile home; shared component, allowlisted in
// the mobile boundary). Shows the caller's status='invited' cohorts and
// lets them Accept / Decline inline: the same consent gate as the Cohorts
// screen, just a faster path to it. Self-contained styling (no shellCss
// dependency) so it renders identically on both surfaces.
//
// Behavior: slides in when invites exist, slow-polls (60s) while mounted so
// a new invite appears without a reload, session-dismissable via the ✕
// (sessionStorage; it returns next visit), and the cohort a /?cohort=...
// invite email deep-linked to gets a brief highlight pulse.

import { useCallback, useEffect, useState } from "react";
import * as api from "../api";
import { COLORS, FONTS } from "../../ui/theme";

const DISMISS_KEY = "cortex.inviteBannerDismissed";
const POLL_MS = 60_000;

const btnBase = {
  border: "1px solid transparent", borderRadius: 4, cursor: "pointer",
  fontFamily: FONTS.sans, fontSize: 13, fontWeight: 600 as const,
  padding: "8px 14px",
};

export function CohortInviteBanner({ variant, highlightCohortId }: {
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
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
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

  const place = variant === "desktop"
    ? { right: 24, bottom: 24, width: 380, maxWidth: "calc(100vw - 48px)" }
    : { left: 12, right: 12, bottom: 12 };

  return (
    <div role="status" style={{
      position: "fixed", zIndex: 60, ...place,
      background: COLORS.card, border: `1px solid ${COLORS.borderInactive2}`,
      borderLeft: "3px solid var(--teal)",
      boxShadow: "0 8px 28px rgba(15, 40, 36, 0.18)",
      fontFamily: FONTS.sans, color: COLORS.textPrimary,
      // slide in once, then a slow teal breathing glow to draw the eye
      animation: "cx-invite-in 240ms ease-out, cx-invite-glow 2.6s ease-in-out 300ms infinite",
    }}>
      <style>{`
        @keyframes cx-invite-in { from { opacity: 0; transform: translateY(14px); }
                                  to { opacity: 1; transform: none; } }
        @keyframes cx-invite-glow {
          0%, 100% { box-shadow: 0 8px 28px rgba(15, 40, 36, 0.18);
                     border-left-color: var(--teal); }
          50% { box-shadow: 0 8px 28px rgba(15, 40, 36, 0.18),
                            0 0 0 3px var(--teal-weak),
                            0 0 18px 2px rgba(47, 143, 131, 0.55);
                border-left-color: var(--teal-deep); }
        }
        @keyframes cx-invite-pulse { 0%, 100% { background: transparent; }
                                     50% { background: var(--teal-weak); } }
      `}</style>
      <div style={{ display: "flex", alignItems: "center", gap: 8,
        padding: "10px 14px", borderBottom: `1px solid ${COLORS.borderInactive}` }}>
        <span style={{ fontSize: 13, fontWeight: 700 }}>
          Cohort invitation{invites.length > 1 ? `s (${invites.length})` : ""}
        </span>
        <span style={{ marginLeft: "auto" }} />
        <button aria-label="Dismiss" onClick={dismiss}
          style={{ background: "none", border: "none", cursor: "pointer",
            color: COLORS.textFaint, fontSize: 16, lineHeight: 1, padding: 4 }}>
          ×
        </button>
      </div>
      {invites.map((c) => (
        <div key={c.cohortId} style={{
          display: "flex", alignItems: "center", flexWrap: "wrap", gap: 10,
          padding: "12px 14px",
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
  );
}
