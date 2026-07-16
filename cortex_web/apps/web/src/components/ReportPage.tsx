// Public support/feedback page at /report (Caddy try_files → index.html;
// main.tsx renders this for the /report path). Follows the create-account card
// aesthetic: themed CSS vars, horizontal wordmark, theme toggle, teal accents,
// sharp panel radii. The form POSTs to /api/report, which emails the receiver
// (REPORT_TO) with the message + the browser diagnostics collected here. The
// page is English-only for now (not part of the i18n scope).

import { CSSProperties, ReactNode, useState } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";
import { useI18n } from "../i18n/LanguageProvider";
import { ApiError, submitReport } from "../api";

const wrap: CSSProperties = {
  minHeight: "100vh",
  background: COLORS.bg,
  color: COLORS.textBody,
  fontFamily: FONTS.sans,
  boxSizing: "border-box",
  position: "relative",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: "48px 24px",
};
const cardStyle: CSSProperties = {
  width: "100%",
  maxWidth: 480,
  background: COLORS.card,
  border: `1px solid ${COLORS.borderInactive}`,
  borderRadius: "var(--radius-panel)",
  padding: 32,
  boxSizing: "border-box",
};
const h1Style: CSSProperties = {
  fontFamily: FONTS.sans, fontSize: 22, fontWeight: 600, letterSpacing: "-0.01em",
  color: COLORS.textPrimary, margin: "0 0 4px",
};
const ledeStyle: CSSProperties = { fontSize: 13, color: COLORS.textBody, margin: "0 0 24px", lineHeight: 1.5 };
const labelStyle: CSSProperties = {
  display: "block", fontSize: 12, fontWeight: 600, color: COLORS.textBody,
  marginBottom: 4, letterSpacing: "0.02em",
};
const inputStyle: CSSProperties = {
  width: "100%", padding: "12px 12px", border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: "var(--radius-ctl)", background: COLORS.cardAlt, color: COLORS.textPrimary,
  fontFamily: FONTS.sans, fontSize: 14, lineHeight: 1.3, boxSizing: "border-box",
};
const hintStyle: CSSProperties = { fontSize: 11, color: COLORS.textFaint, marginTop: 6, lineHeight: 1.5 };
const linkStyle: CSSProperties = { color: "var(--teal-deep)", fontWeight: 600, textDecoration: "none", fontSize: 13 };

function Btn({ children, type = "button", disabled, onClick }: {
  children: ReactNode; type?: "button" | "submit"; disabled?: boolean; onClick?: () => void;
}) {
  const [h, setH] = useState(false);
  return (
    <button type={type} disabled={disabled} onClick={onClick}
      onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        width: "100%", fontFamily: FONTS.sans, fontSize: 14, fontWeight: 600,
        padding: "12px 16px", borderRadius: "var(--radius-ctl)",
        background: (h && !disabled) ? "var(--teal-hover)" : "var(--teal)",
        border: `1px solid ${(h && !disabled) ? "var(--teal-hover)" : "var(--teal)"}`,
        color: "#fff", cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.45 : 1,
        transition: "background 0.12s, border-color 0.12s", boxSizing: "border-box",
      }}>
      {children}
    </button>
  );
}

// Browser diagnostics for troubleshooting (no GPS prompt: location is inferred
// from timezone/locale + the server-side IP). Sent with the report.
function collectDiagnostics(appLang: string): Record<string, string> {
  const nav = navigator as Navigator & {
    userAgentData?: { platform?: string; brands?: { brand: string; version: string }[] };
  };
  const ua = nav.userAgentData;
  const brands = (ua?.brands || []).map((b) => `${b.brand} ${b.version}`).join(", ");
  return {
    os: ua?.platform || navigator.platform || "",
    browser: brands,
    platform: navigator.platform || "",
    userAgent: navigator.userAgent || "",
    language: [navigator.language, ...(navigator.languages || [])].filter(Boolean).join(", "),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    screen: `${window.screen.width}x${window.screen.height} @${window.devicePixelRatio}x`,
    viewport: `${window.innerWidth}x${window.innerHeight}`,
    clientTime: new Date().toString(),
    url: location.href,
    referrer: document.referrer || "",
    online: navigator.onLine ? "yes" : "no",
    appLang,
  };
}

export function ReportPage() {
  const { lang } = useI18n();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (!message.trim()) { setErr("Please describe the issue or suggestion."); return; }
    setBusy(true);
    try {
      await submitReport({
        username: username.trim(),
        email: email.trim(),
        message: message.trim(),
        client: collectDiagnostics(lang),
      });
      setDone(true);
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 429) setErr("Too many reports. Please wait a minute and try again.");
      else if (ex instanceof ApiError && ex.status === 0) setErr("Cannot reach the server.");
      else setErr(ex instanceof Error ? ex.message : "Could not send your report.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={wrap}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />

      <div style={{ width: "100%", maxWidth: 480, marginBottom: 24 }}>
        <a href="/" aria-label="Back to CORTEX" style={{ display: "block" }}>
          <img
            className="cortex-logo"
            src="/cortex_logo_word_horizontal@3x.png"
            srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
            alt="CORTEX EEG Skill Certification"
            style={{ width: "70%", height: "auto", display: "block", margin: "0 auto" }}
          />
        </a>
      </div>

      <div style={cardStyle}>
        {done ? (
          <div style={{ textAlign: "center" }}>
            <div style={{
              width: 56, height: 56, background: "var(--teal-weak)", border: "1px solid var(--teal-mid)",
              display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px",
            }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--teal)"
                strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h1 style={h1Style}>Report sent</h1>
            <p style={{ ...ledeStyle, marginBottom: 24 }}>
              Thank you. Your report has been sent to the CORTEX team
              {email.trim() ? `, and we may reply to ${email.trim()}.` : "."}
            </p>
            <a href="/" style={{ ...linkStyle, display: "inline-block" }}>Back to CORTEX</a>
          </div>
        ) : (
          <>
            <h1 style={h1Style}>Report a problem or suggestion</h1>
            <p style={ledeStyle}>
              Use this form to report a bug or suggest an improvement. To help us reproduce and
              diagnose issues, we automatically attach your browser and device details (operating
              system, browser, screen size, timezone) and the time of submission.
            </p>
            <form onSubmit={submit}>
              <label style={{ display: "block", marginBottom: 16 }}>
                <span style={labelStyle}>Profile username</span>
                <input value={username} onChange={(e) => setUsername(e.target.value)}
                  placeholder="Your CORTEX username (optional)" autoComplete="username" style={inputStyle} />
              </label>
              <label style={{ display: "block", marginBottom: 16 }}>
                <span style={labelStyle}>Email</span>
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.org (so we can reply)" autoComplete="email" style={inputStyle} />
              </label>
              <label style={{ display: "block", marginBottom: 8 }}>
                <span style={labelStyle}>Description <span style={{ color: "var(--teal)" }}>*</span></span>
                <textarea value={message} onChange={(e) => setMessage(e.target.value)} required rows={6}
                  placeholder="Describe the bug, or suggest an improvement…"
                  style={{ ...inputStyle, resize: "vertical", minHeight: 120, fontFamily: FONTS.sans }} />
              </label>
              <div style={hintStyle}>
                Your message, device details, and submission time are emailed to the CORTEX team. Do not
                include patient-identifying information.
              </div>
              {err && <div style={{ fontSize: 13, color: COLORS.fail, marginTop: 12 }}>{err}</div>}
              <div style={{ marginTop: 24 }}>
                <Btn type="submit" disabled={busy}>{busy ? "Sending…" : "Send report"}</Btn>
              </div>
            </form>
          </>
        )}
      </div>

      <div style={{ marginTop: 24 }}>
        <a href="/" style={linkStyle}>Back to CORTEX</a>
      </div>
    </div>
  );
}
