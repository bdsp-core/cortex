// One-click auth links from the verification / password-reset emails
// (services/api/mailer.py) land on the SPA ROOT with query params — no
// router or Caddy fallback rule needed:
//   /?verifyEmail=..&verifyCode=..  → verify screen, code pre-filled
//                                     (the auto-submit effect completes it)
//   /?resetEmail=..&resetCode=..    → reset screen, code pre-filled (the
//                                     user only types their new password)
// Parsing is pure for testability; App.tsx consumes + strips the params at
// boot so the code never lingers in the address bar or browser history.

export interface AuthDeepLink {
  kind: "verify" | "reset";
  email: string;
  code: string;
}

const CODE_RE = /^\d{6}$/;

export function parseAuthDeepLink(search: string): AuthDeepLink | null {
  const p = new URLSearchParams(search);
  for (const kind of ["verify", "reset"] as const) {
    const email = (p.get(`${kind}Email`) || "").trim().toLowerCase();
    const code = (p.get(`${kind}Code`) || "").trim();
    if (email.includes("@") && CODE_RE.test(code)) return { kind, email, code };
  }
  return null;
}

/** Parse the current URL's params and strip them from the address bar. */
export function consumeAuthDeepLink(): AuthDeepLink | null {
  const link = parseAuthDeepLink(window.location.search);
  if (link) window.history.replaceState(null, "", window.location.pathname);
  return link;
}

// Cohort-invitation emails link to /?cohort=<id> so the recipient lands on
// the pending invite (the dashboard banner highlights it) instead of the
// front door. The id grants nothing: invites are only visible to the
// signed-in account they belong to.
const COHORT_ID_RE = /^ch-[A-Za-z0-9_-]{6,}$/;

export function parseCohortDeepLink(search: string): string | null {
  const v = (new URLSearchParams(search).get("cohort") || "").trim();
  return COHORT_ID_RE.test(v) ? v : null;
}

export function consumeCohortDeepLink(): string | null {
  const id = parseCohortDeepLink(window.location.search);
  if (id) window.history.replaceState(null, "", window.location.pathname);
  return id;
}

// Public navigation can open the first step of account creation without
// coupling a static/info page to AuthFlow's internal screen state. Unknown
// values are ignored, and consuming the flag preserves unrelated query params
// (notably cohort invitations).
export type AuthStartScreen = "signup";

export function parseAuthStartScreen(search: string): AuthStartScreen | null {
  return new URLSearchParams(search).get("auth") === "signup" ? "signup" : null;
}

export function consumeAuthStartScreen(): AuthStartScreen | null {
  const screen = parseAuthStartScreen(window.location.search);
  if (!screen) return null;

  const url = new URL(window.location.href);
  url.searchParams.delete("auth");
  window.history.replaceState(
    null,
    "",
    `${url.pathname}${url.search}${url.hash}`,
  );
  return screen;
}
