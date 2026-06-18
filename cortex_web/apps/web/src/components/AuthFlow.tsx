// Public-signup auth surface — a single component holding the six locked-v1
// auth screens (sign in / create account / verify email / forgot / reset /
// success) as internal state, mirroring the mockup's single-file screen
// switching. App.tsx only needs its existing "auth" phase; sign-in success
// stores the JWT and calls onAuthed() to proceed into the post-auth flow.
//
// Locked flow: create account → verify email → success → sign in;
// forgot → reset → success → sign in. We never auto-login after verify — the
// user signs in every time with credentials.
//
// All user-facing copy comes from the i18n catalogs (locales/*.json) via t();
// "CORTEX" stays literal. Visuals match cortex_web_design/mockups/locked-v1-auth.

import {
  CSSProperties, KeyboardEvent, ClipboardEvent, ReactNode, useRef, useState,
} from "react";
import {
  ApiError, EmailNotVerifiedError, login, register, verifyCode, resendCode,
  requestReset, resetPassword,
} from "../api";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";
import { useI18n, TFn, LANGS, Lang } from "../i18n/LanguageProvider";

type Screen = "signin" | "signup" | "verify" | "forgot" | "reset" | "success";

// Self-reported expertise dropdown shown at signup. [catalogKey, submittedValue]:
// the visible label is translated; the submitted value stays canonical English.
const EXPERTISE: [string, string][] = [
  ["auth.expertise.epileptologist", "Attending epileptologist"],
  ["auth.expertise.neurologist", "Attending neurologist (non-epilepsy)"],
  ["auth.expertise.fellow", "Clinical neurophysiology fellow"],
  ["auth.expertise.resident", "Neurology resident"],
  ["auth.expertise.tech", "EEG technologist"],
  ["auth.expertise.researcher", "Researcher / scientist"],
  ["auth.expertise.other", "Other"],
];

const RESEND_COOLDOWN_S = 30;

// ── shared inline styles (port of the mockup CSS) ────────────────────────
const cardStyle: CSSProperties = {
  width: "100%",
  maxWidth: 440,
  background: COLORS.card,
  border: `1px solid ${COLORS.borderInactive}`,
  borderRadius: "var(--radius-panel)",
  padding: 32,
  boxSizing: "border-box",
};
const h1Style: CSSProperties = {
  fontFamily: FONTS.sans,
  fontSize: 22,
  fontWeight: 600,
  letterSpacing: "-0.01em",
  color: COLORS.textPrimary,
  margin: "0 0 4px",
};
const ledeStyle: CSSProperties = {
  fontSize: 13,
  color: COLORS.textBody,
  margin: "0 0 24px",
};
const labelStyle: CSSProperties = {
  display: "block",
  fontSize: 12,
  fontWeight: 600,
  color: COLORS.textBody,
  marginBottom: 4,
  letterSpacing: "0.02em",
};
const inputStyle: CSSProperties = {
  width: "100%",
  padding: "12px 12px",
  border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: "var(--radius-ctl)",
  background: COLORS.cardAlt,
  color: COLORS.textPrimary,
  fontFamily: FONTS.sans,
  fontSize: 14,
  lineHeight: 1.3,
  boxSizing: "border-box",
};
const hintStyle: CSSProperties = { fontSize: 11, color: COLORS.textFaint, marginTop: 4 };

const linkBtnStyle: CSSProperties = {
  background: "none",
  border: "none",
  padding: 0,
  cursor: "pointer",
  fontFamily: FONTS.sans,
  fontSize: 13,
  fontWeight: 600,
  color: "var(--teal-deep)",
};
const switchStyle: CSSProperties = {
  marginTop: 24,
  paddingTop: 16,
  borderTop: `1px solid ${COLORS.borderInactive}`,
  textAlign: "center",
  fontSize: 13,
  color: COLORS.textBody,
};
const metaStyle: CSSProperties = {
  marginTop: 16,
  textAlign: "center",
  fontSize: 11,
  color: COLORS.textFaint,
  lineHeight: 1.5,
};

// Render an interpolated lede whose {email} should be visually emphasized:
// split the already-substituted string on the email and wrap it in a span.
function ledeWithEmail(text: string, email: string): ReactNode[] {
  if (!email) return [text];
  const parts = text.split(email);
  const out: ReactNode[] = [];
  parts.forEach((p, i) => {
    if (i > 0) {
      out.push(
        <span key={`e${i}`} style={{ fontFamily: FONTS.sans, color: COLORS.textPrimary }}>{email}</span>,
      );
    }
    out.push(p);
  });
  return out;
}

// Hover-aware auth button. `primary` (teal fill) darkens to --teal-hover on
// hover; `secondary` (white fill) and `accentOutline` (white fill + teal
// outline + teal label) pick up the --btn-hover grey on hover. Inline styles
// can't express :hover, so hover is tracked in state.
type BtnVariant = "primary" | "secondary" | "accentOutline";
function AuthButton({
  children, variant = "primary", type = "button", disabled, onClick, style,
}: {
  children: ReactNode;
  variant?: BtnVariant;
  type?: "button" | "submit";
  disabled?: boolean;
  onClick?: () => void;
  style?: CSSProperties;
}) {
  const [hover, setHover] = useState(false);
  const h = hover && !disabled;
  const base: CSSProperties = {
    fontFamily: FONTS.sans, fontSize: 14, fontWeight: 600,
    padding: "12px 16px", borderRadius: "var(--radius-ctl)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
    display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
    boxSizing: "border-box",
    transition: "background 0.12s, border-color 0.12s, color 0.12s",
  };
  const variants: Record<BtnVariant, CSSProperties> = {
    primary: {
      background: h ? "var(--teal-hover)" : "var(--teal)",
      border: `1px solid ${h ? "var(--teal-hover)" : "var(--teal)"}`,
      color: "#fff",
    },
    secondary: {
      background: h ? "var(--btn-hover)" : COLORS.card,
      border: `1px solid ${COLORS.borderInactive2}`,
      color: COLORS.textPrimary,
    },
    accentOutline: {
      background: h ? "var(--btn-hover)" : COLORS.card,
      border: "1px solid var(--teal)",
      color: "var(--teal-deep)",
    },
  };
  return (
    <button type={type} disabled={disabled} onClick={onClick}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{ ...base, ...variants[variant], ...style }}>
      {children}
    </button>
  );
}

function Field({
  label, type = "text", value, onChange, placeholder, required, autoFocus,
  autoComplete, onInput, children, name,
}: {
  label?: ReactNode;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  required?: boolean;
  autoFocus?: boolean;
  autoComplete?: string;
  onInput?: (v: string) => void;
  children?: ReactNode;
  name?: string;
}) {
  return (
    <label style={{ display: "block", marginBottom: 16 }}>
      {label && <span style={labelStyle}>{label}</span>}
      <input
        type={type}
        name={name}
        value={value}
        placeholder={placeholder}
        required={required}
        autoFocus={autoFocus}
        autoComplete={autoComplete}
        onChange={(e) => { onChange(e.target.value); onInput?.(e.target.value); }}
        style={inputStyle}
      />
      {children}
    </label>
  );
}

function FormError({ children }: { children: ReactNode }) {
  if (!children) return null;
  return <div style={{ fontSize: 13, color: COLORS.fail, marginTop: 12 }}>{children}</div>;
}

// Step dots (signup → verify): 2 segments.
function Steps({ on }: { on: 1 | 2 }) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
      {[1, 2].map((i) => (
        <i key={i} style={{
          flex: 1, height: 3, borderRadius: 0,
          background: i <= on ? "var(--teal)" : COLORS.borderInactive,
        }} />
      ))}
    </div>
  );
}

// Password-strength meter (cosmetic; scores 0–4 like the mockup).
function scorePw(v: string): number {
  let s = 0;
  if (v.length >= 8) s++;
  if (/[A-Z]/.test(v) && /[a-z]/.test(v)) s++;
  if (/[0-9]/.test(v)) s++;
  if (/[^A-Za-z0-9]/.test(v)) s++;
  return s;
}
function StrengthMeter({ score }: { score: number }) {
  const colorFor = (i: number) => {
    if (i >= score) return COLORS.borderInactive;
    if (score === 1) return COLORS.fail;
    if (score === 2) return "var(--refer-b)";
    if (score === 3) return "var(--teal-mid)";
    return "var(--teal)";
  };
  return (
    <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
      {[0, 1, 2, 3].map((i) => (
        <i key={i} style={{ flex: 1, height: 3, background: colorFor(i), transition: "background 0.2s" }} />
      ))}
    </div>
  );
}

// 6 single-char numeric inputs with auto-advance, backspace-to-prev,
// arrow-navigation, and paste-distributes.
function CodeInputs({
  digits, setDigits, ariaPrefix,
}: {
  digits: string[];
  setDigits: (d: string[]) => void;
  ariaPrefix: string;
}) {
  const { t } = useI18n();
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const onChange = (idx: number, raw: string) => {
    const v = raw.replace(/[^0-9]/g, "").slice(0, 1);
    const next = [...digits];
    next[idx] = v;
    setDigits(next);
    if (v && idx < 5) refs.current[idx + 1]?.focus();
  };
  const onKeyDown = (idx: number, e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" && !digits[idx] && idx > 0) refs.current[idx - 1]?.focus();
    if (e.key === "ArrowLeft" && idx > 0) { e.preventDefault(); refs.current[idx - 1]?.focus(); }
    if (e.key === "ArrowRight" && idx < 5) { e.preventDefault(); refs.current[idx + 1]?.focus(); }
  };
  const onPaste = (e: ClipboardEvent<HTMLInputElement>) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/[^0-9]/g, "").slice(0, 6);
    const next = Array.from({ length: 6 }, (_, i) => pasted[i] || "");
    setDigits(next);
    refs.current[Math.min(pasted.length, 5)]?.focus();
  };

  return (
    <div style={{ display: "flex", gap: 8, justifyContent: "center", margin: "8px 0 4px" }}>
      {digits.map((d, i) => (
        <input
          key={i}
          ref={(el) => { refs.current[i] = el; }}
          inputMode="numeric"
          maxLength={1}
          aria-label={t("auth.codeDigitAria", { prefix: ariaPrefix, n: i + 1 })}
          value={d}
          autoFocus={i === 0}
          onChange={(e) => onChange(i, e.target.value)}
          onKeyDown={(e) => onKeyDown(i, e)}
          onPaste={onPaste}
          style={{
            width: 48, height: 56, textAlign: "center",
            fontFamily: FONTS.sans, fontVariantNumeric: "tabular-nums",
            fontSize: 24, fontWeight: 700, color: COLORS.textPrimary,
            border: `1px solid ${d ? "var(--teal-mid)" : COLORS.borderInactive2}`,
            borderRadius: "var(--radius-ctl)", background: COLORS.cardAlt,
            boxSizing: "border-box",
          }}
        />
      ))}
    </div>
  );
}

// Resend control with a 30s cooldown after each send.
function Resend({ cooldown, onResend }: { cooldown: number; onResend: () => void }) {
  const { t } = useI18n();
  return (
    <div style={{ textAlign: "center", fontSize: 12, color: COLORS.textBody, marginTop: 16 }}>
      {t("auth.resend.prompt")}{" "}
      <button type="button" style={{ ...linkBtnStyle, fontSize: 12, opacity: cooldown > 0 ? 0.45 : 1,
        cursor: cooldown > 0 ? "not-allowed" : "pointer", color: cooldown > 0 ? COLORS.textFaint : "var(--teal-deep)" }}
        disabled={cooldown > 0} onClick={onResend}>
        {t("auth.resend.action")}
      </button>
      {cooldown > 0 && <span> {t("auth.resend.cooldown", { n: cooldown })}</span>}
    </div>
  );
}

// Map an ApiError to a friendly (localized) message.
function apiMessage(t: TFn, e: unknown, fallback: string): string {
  if (e instanceof ApiError) {
    if (e.status === 429) return t("auth.apiError.tooMany");
    if (e.status === 0) return t("auth.apiError.unreachable");
    return e.message || fallback;
  }
  return e instanceof Error ? e.message : fallback;
}

// Global auth footer bar (shown on every auth screen): legal links, a language
// selector, and a GitHub icon. The language selector + GitHub icon are
// non-functional PLACEHOLDERS for now (Stage 1) — Stage 2 wires the selector to
// real language switching; later, wrap the GitHub icon in an <a href> and add a
// /terms link. Full width; the grey top border matches the sign-in divider.
function AuthFooter() {
  const { t, lang, setLang } = useI18n();
  const link: CSSProperties = {
    fontFamily: FONTS.sans, fontSize: 12, fontWeight: 500,
    color: COLORS.textBody, textDecoration: "none", cursor: "pointer",
  };
  const placeholder: CSSProperties = { ...link, cursor: "default" };
  const dot: CSSProperties = { color: COLORS.borderInactive2, fontSize: 12 };
  const group: CSSProperties = { display: "flex", alignItems: "center", gap: 16 };
  return (
    <footer style={{
      flex: "0 0 auto", width: "100%", height: 48,
      borderTop: `1px solid ${COLORS.borderInactive2}`,
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 24px", boxSizing: "border-box", fontFamily: FONTS.sans,
    }}>
      <div style={group}>
        <a href="/privacy" style={link}>{t("footer.privacy")}</a>
        <span aria-hidden="true" style={dot}>·</span>
        <a href="/terms" style={link}>{t("footer.terms")}</a>
        <span aria-hidden="true" style={dot}>·</span>
        <a href="/report" style={link}>{t("footer.report")}</a>
      </div>
      <div style={group}>
        {/* Language selector — switches the active catalog (persisted to localStorage). */}
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6,
          color: COLORS.textBody, cursor: "pointer" }} title={t("footer.languageLabel")}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10" /><line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <select value={lang} onChange={(e) => setLang(e.target.value as Lang)}
            aria-label={t("footer.languageLabel")}
            style={{
              appearance: "none", WebkitAppearance: "none", MozAppearance: "none",
              background: "transparent", border: "none", color: COLORS.textBody,
              fontFamily: FONTS.sans, fontSize: 12, fontWeight: 500, cursor: "pointer",
              padding: 0, lineHeight: 1.3,
            }}>
            {LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </label>
        {/* TODO: wrap in <a href="<repo URL>"> when the GitHub link is ready */}
        <span aria-label={t("footer.github")} title={t("footer.githubSoon")}
          style={{ ...placeholder, display: "inline-flex", color: COLORS.textBody }}>
          <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
          </svg>
        </span>
      </div>
    </footer>
  );
}

export function AuthFlow({ onAuthed }: { onAuthed: () => void }) {
  const { t } = useI18n();
  const [screen, setScreen] = useState<Screen>("signin");

  // sign in
  const [siEmail, setSiEmail] = useState("");
  const [siPw, setSiPw] = useState("");

  // sign up
  const [suEmail, setSuEmail] = useState("");
  const [suPw, setSuPw] = useState("");
  const [suPw2, setSuPw2] = useState("");
  const [suName, setSuName] = useState("");
  const [suRole, setSuRole] = useState("");
  // Honeypot: hidden field; humans don't fill it, naive bots do. Sent to the
  // backend regardless; the server silently 200s a bot without revealing it.
  const [honeypot, setHoneypot] = useState("");

  // verify
  const [verifyEmail, setVerifyEmail] = useState("");
  const [verifyDigits, setVerifyDigits] = useState<string[]>(["", "", "", "", "", ""]);

  // forgot / reset
  const [fpEmail, setFpEmail] = useState("");
  const [resetEmail, setResetEmail] = useState("");
  const [resetDigits, setResetDigits] = useState<string[]>(["", "", "", "", "", ""]);
  const [rsPw, setRsPw] = useState("");
  const [rsPw2, setRsPw2] = useState("");

  // success — store the kind so the message stays language-reactive.
  const [successKind, setSuccessKind] = useState<"emailVerified" | "passwordUpdated">("emailVerified");

  // shared
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // resend cooldowns (seconds remaining)
  const [verifyCooldown, setVerifyCooldown] = useState(0);
  const [resetCooldown, setResetCooldown] = useState(0);
  const cdTimer = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  function startCooldown(which: "verify" | "reset") {
    const set = which === "verify" ? setVerifyCooldown : setResetCooldown;
    set(RESEND_COOLDOWN_S);
    if (cdTimer.current[which]) clearInterval(cdTimer.current[which]);
    cdTimer.current[which] = setInterval(() => {
      set((tt) => {
        if (tt <= 1) { clearInterval(cdTimer.current[which]); return 0; }
        return tt - 1;
      });
    }, 1000);
  }

  function go(next: Screen) {
    setErr(null);
    setScreen(next);
  }

  // ── handlers ───────────────────────────────────────────────────────────
  async function doSignIn(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await login(siEmail.trim().toLowerCase(), siPw);
      onAuthed();
    } catch (ex) {
      if (ex instanceof EmailNotVerifiedError) {
        // Account exists but isn't verified: route to the verify screen.
        setVerifyEmail(ex.email);
        setVerifyDigits(["", "", "", "", "", ""]);
        startCooldown("verify");
        go("verify");
      } else if (ex instanceof ApiError && ex.status === 401) {
        setErr(t("auth.signin.errInvalid"));
      } else {
        setErr(apiMessage(t, ex, t("auth.signin.errGeneric")));
      }
    } finally {
      setBusy(false);
    }
  }

  async function doSignUp(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (suPw.length < 8) { setErr(t("auth.signup.errShort")); return; }
    if (suPw !== suPw2) { setErr(t("auth.signup.errMismatch")); return; }
    if (!suName.trim()) { setErr(t("auth.signup.errName")); return; }
    setBusy(true);
    try {
      const email = suEmail.trim().toLowerCase();
      const r = await register(email, suPw, suName.trim(), suRole, honeypot);
      // Honeypot bot path: server 200s without needsVerification. Bounce back
      // to sign in silently rather than progress.
      if (!r.needsVerification) { go("signin"); return; }
      setVerifyEmail(email);
      setVerifyDigits(["", "", "", "", "", ""]);
      startCooldown("verify");
      go("verify");
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 409) {
        setErr(t("auth.signup.errExists"));
      } else {
        setErr(apiMessage(t, ex, t("auth.signup.errForm")));
      }
    } finally {
      setBusy(false);
    }
  }

  async function doVerify(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const code = verifyDigits.join("");
    if (code.length !== 6) { setErr(t("auth.verify.errAllDigits")); return; }
    setBusy(true);
    try {
      await verifyCode(verifyEmail, code);
      setSuccessKind("emailVerified");
      go("success");
    } catch (ex) {
      setErr(apiMessage(t, ex, t("auth.verify.errBadCode")));
    } finally {
      setBusy(false);
    }
  }

  async function doResendVerify() {
    setErr(null);
    try {
      await resendCode(verifyEmail);
      setVerifyDigits(["", "", "", "", "", ""]);
      startCooldown("verify");
    } catch (ex) {
      setErr(apiMessage(t, ex, t("auth.verify.errResend")));
    }
  }

  async function doForgot(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      const email = fpEmail.trim().toLowerCase();
      await requestReset(email);
      setResetEmail(email);
      setResetDigits(["", "", "", "", "", ""]);
      setRsPw("");
      setRsPw2("");
      startCooldown("reset");
      go("reset");
    } catch (ex) {
      setErr(apiMessage(t, ex, t("auth.forgot.errSend")));
    } finally {
      setBusy(false);
    }
  }

  async function doReset(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const code = resetDigits.join("");
    if (code.length !== 6) { setErr(t("auth.reset.errEnterCode")); return; }
    if (rsPw.length < 8) { setErr(t("auth.signup.errShort")); return; }
    if (rsPw !== rsPw2) { setErr(t("auth.signup.errMismatch")); return; }
    setBusy(true);
    try {
      await resetPassword(resetEmail, code, rsPw);
      setSuccessKind("passwordUpdated");
      go("success");
    } catch (ex) {
      setErr(apiMessage(t, ex, t("auth.verify.errBadCode")));
    } finally {
      setBusy(false);
    }
  }

  async function doResendReset() {
    setErr(null);
    try {
      await requestReset(resetEmail);
      setResetDigits(["", "", "", "", "", ""]);
      startCooldown("reset");
    } catch (ex) {
      setErr(apiMessage(t, ex, t("auth.verify.errResend")));
    }
  }

  // ── render ───────────────────────────────────────────────────────────
  // Login = collage hero (2/3 + 1/3 split); all other screens = simple centered box.
  const hero = screen === "signin";
  const reqMark = <span style={{ color: "var(--teal)" }}>*</span>;
  return (
    <div style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
      boxSizing: "border-box",
    }}>
      {/* content region — holds the collage/divider + the auth column; the
          footer bar sits below it, full width. */}
      <div style={{
        flex: 1, minHeight: 0, display: "flex",
        justifyContent: hero ? "flex-end" : "center",
        paddingRight: hero ? 36 : 0, position: "relative",
        boxSizing: "border-box",
      }}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />

      {hero && (
        <>
          <style>{`@media (max-width: 900px){ .auth-collage{ display: none !important; } }`}</style>
          <img className="auth-collage auth-collage-img"
            src="/web_collage_prod@2x.png"
            srcSet="/web_collage_prod@2x.png 2x, /web_collage_prod@3x.png 3x"
            alt="" aria-hidden="true"
            style={{
              position: "absolute", left: 36, top: 36, width: "calc(64vw - 72px)",
              height: "calc(100% - 84px)",
              objectFit: "contain", objectPosition: "center",
              userSelect: "none",
            }}
          />
          <div className="auth-collage" aria-hidden="true" style={{
            position: "absolute", top: 0, bottom: 0,
            left: "calc(65.333vw - 36px)", width: 1,
            background: COLORS.borderInactive2,
          }} />
        </>
      )}

      <div style={{
        width: hero ? "33.333vw" : "100%",
        maxWidth: hero ? undefined : 488, minWidth: hero ? 360 : undefined,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: hero ? "flex-start" : "center",
        padding: hero ? "36px 24px 48px" : "24px 24px 48px", boxSizing: "border-box",
      }}>

      <div style={{ width: "100%", maxWidth: 440, marginBottom: 24 }}>
        <img
          className="cortex-logo"
          src={hero ? "/cortex_logo_only@2x.png" : "/cortex_logo_word_horizontal@3x.png"}
          srcSet={hero
            ? "/cortex_logo_only@2x.png 2x, /cortex_logo_only@3x.png 3x"
            : "/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"}
          alt={t("common.logoAlt")}
          style={{ width: hero ? "75%" : "95%", height: "auto", display: "block", margin: "0 auto" }}
        />
      </div>

      {screen === "signin" && (
        <div style={cardStyle}>
          <h1 style={{ ...h1Style, marginBottom: 24 }}>{t("auth.signin.title")}</h1>
          <form onSubmit={doSignIn}>
            <Field type="email" value={siEmail} onChange={setSiEmail}
              placeholder={t("common.email")} autoComplete="email" required autoFocus />
            <Field type="password" value={siPw} onChange={setSiPw}
              placeholder={t("common.password")} autoComplete="current-password" required>
              <div style={{ textAlign: "right", marginTop: 4 }}>
                <button type="button" style={{ ...linkBtnStyle, fontSize: 12 }}
                  onClick={() => { setFpEmail(siEmail); go("forgot"); }}>
                  {t("auth.signin.forgot")}
                </button>
              </div>
            </Field>
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <AuthButton variant="primary" type="submit" disabled={busy} style={{ flex: 1 }}>
                {busy ? t("auth.signin.submitting") : t("auth.signin.submit")}
              </AuthButton>
            </div>
          </form>
          <div style={switchStyle}>
            <AuthButton variant="accentOutline" onClick={() => go("signup")} style={{ width: "100%" }}>
              {t("auth.signin.create")}
            </AuthButton>
          </div>
        </div>
      )}

      {screen === "signup" && (
        <div style={cardStyle}>
          <Steps on={1} />
          <h1 style={h1Style}>{t("auth.signup.title")}</h1>
          <p style={ledeStyle}>{t("auth.signup.lede")}</p>
          <form onSubmit={doSignUp}>
            <Field label={<>{t("common.email")} {reqMark}</>}
              type="email" value={suEmail} onChange={setSuEmail}
              placeholder={t("auth.signup.emailPh")} autoComplete="email" required autoFocus />
            <Field label={<>{t("common.password")} {reqMark}</>}
              type="password" value={suPw} onChange={setSuPw}
              autoComplete="new-password" required>
              <StrengthMeter score={scorePw(suPw)} />
              <div style={hintStyle}>{t("auth.signup.passwordHint")}</div>
            </Field>
            <Field label={<>{t("auth.signup.confirmPassword")} {reqMark}</>}
              type="password" value={suPw2} onChange={setSuPw2} autoComplete="new-password" required />
            <Field label={<>{t("auth.signup.displayName")} {reqMark}</>}
              value={suName} onChange={setSuName} placeholder={t("auth.signup.displayNamePh")} required />
            <label style={{ display: "block", marginBottom: 16 }}>
              <span style={labelStyle}>{t("auth.signup.role")}</span>
              <select value={suRole} onChange={(e) => setSuRole(e.target.value)} style={inputStyle}>
                <option value="">{t("auth.signup.select")}</option>
                {EXPERTISE.map(([k, v]) => <option key={k} value={v}>{t(k)}</option>)}
              </select>
            </label>
            {/* honeypot — visually + a11y hidden; naive bots fill every input */}
            <input type="text" name="company" tabIndex={-1} autoComplete="off"
              value={honeypot} onChange={(e) => setHoneypot(e.target.value)}
              aria-hidden="true"
              style={{ position: "absolute", left: "-9999px", width: 1, height: 1,
                opacity: 0, pointerEvents: "none" }} />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <AuthButton variant="secondary" onClick={() => go("signin")} style={{ flex: 1 }}>
                {t("common.back")}
              </AuthButton>
              <AuthButton variant="primary" type="submit" disabled={busy} style={{ flex: 1 }}>
                {busy ? t("auth.signup.submitting") : t("auth.signup.submit")}
              </AuthButton>
            </div>
          </form>
          <div style={metaStyle}>{t("auth.signup.consent")}</div>
        </div>
      )}

      {screen === "verify" && (
        <div style={cardStyle}>
          <Steps on={2} />
          <h1 style={h1Style}>{t("auth.verify.title")}</h1>
          <p style={ledeStyle}>{ledeWithEmail(t("auth.verify.lede", { email: verifyEmail }), verifyEmail)}</p>
          <form onSubmit={doVerify}>
            <CodeInputs digits={verifyDigits} setDigits={setVerifyDigits} ariaPrefix={t("auth.verify.codeAria")} />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <AuthButton variant="primary" type="submit" disabled={busy} style={{ flex: 1 }}>
                {busy ? t("auth.verify.submitting") : t("auth.verify.submit")}
              </AuthButton>
            </div>
          </form>
          <Resend cooldown={verifyCooldown} onResend={doResendVerify} />
          <div style={{ ...switchStyle, marginTop: 8, paddingTop: 0, borderTop: "none" }}>
            {t("auth.verify.wrongAddress")}{" "}
            <button type="button" style={linkBtnStyle} onClick={() => go("signup")}>{t("auth.verify.editEmail")}</button>
          </div>
        </div>
      )}

      {screen === "forgot" && (
        <div style={cardStyle}>
          <h1 style={h1Style}>{t("auth.forgot.title")}</h1>
          <p style={ledeStyle}>{t("auth.forgot.lede")}</p>
          <form onSubmit={doForgot}>
            <Field label={t("common.email")} type="email" value={fpEmail} onChange={setFpEmail}
              placeholder={t("auth.signup.emailPh")} autoComplete="email" required autoFocus />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <AuthButton variant="secondary" onClick={() => go("signin")} style={{ flex: 1 }}>
                {t("common.back")}
              </AuthButton>
              <AuthButton variant="primary" type="submit" disabled={busy} style={{ flex: 1 }}>
                {busy ? t("auth.forgot.submitting") : t("auth.forgot.submit")}
              </AuthButton>
            </div>
          </form>
        </div>
      )}

      {screen === "reset" && (
        <div style={cardStyle}>
          <h1 style={h1Style}>{t("auth.reset.title")}</h1>
          <p style={ledeStyle}>{ledeWithEmail(t("auth.reset.lede", { email: resetEmail }), resetEmail)}</p>
          <form onSubmit={doReset}>
            <div style={{ display: "block", marginBottom: 16 }}>
              <span style={labelStyle}>{t("auth.reset.codeLabel")}</span>
              <CodeInputs digits={resetDigits} setDigits={setResetDigits} ariaPrefix={t("auth.reset.codeAria")} />
            </div>
            <Field label={t("auth.reset.newPassword")} type="password" value={rsPw} onChange={setRsPw}
              autoComplete="new-password" required>
              <StrengthMeter score={scorePw(rsPw)} />
              <div style={hintStyle}>{t("auth.signup.passwordHint")}</div>
            </Field>
            <Field label={t("auth.reset.confirmNewPassword")} type="password" value={rsPw2} onChange={setRsPw2}
              autoComplete="new-password" required />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <AuthButton variant="primary" type="submit" disabled={busy} style={{ flex: 1 }}>
                {busy ? t("auth.reset.submitting") : t("auth.reset.submit")}
              </AuthButton>
            </div>
          </form>
          <Resend cooldown={resetCooldown} onResend={doResendReset} />
        </div>
      )}

      {screen === "success" && (
        <div style={{ ...cardStyle, textAlign: "center" }}>
          <div style={{
            width: 56, height: 56, borderRadius: 0,
            background: "var(--teal-weak)", border: `1px solid var(--teal-mid)`,
            display: "flex", alignItems: "center", justifyContent: "center",
            margin: "0 auto 16px",
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--teal)"
              strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h1 style={h1Style}>
            {t(successKind === "emailVerified" ? "auth.success.emailVerifiedTitle" : "auth.success.passwordUpdatedTitle")}
          </h1>
          <p style={{ ...ledeStyle, marginBottom: 24 }}>
            {t(successKind === "emailVerified" ? "auth.success.emailVerifiedMsg" : "auth.success.passwordUpdatedMsg")}
          </p>
          <div style={{ display: "flex", gap: 12 }}>
            <AuthButton variant="primary" onClick={() => {
              setSiPw("");
              go("signin");
            }} style={{ flex: 1 }}>
              {t("auth.success.continue")}
            </AuthButton>
          </div>
        </div>
      )}

        <div style={{
          marginTop: hero ? "auto" : 40, paddingTop: 48,
          textAlign: "center", color: COLORS.textFaint, fontSize: 12,
          letterSpacing: "0.04em", lineHeight: 1.5,
        }}>
          <div>{t("auth.credit.developedBy")}</div>
          <div style={{ marginTop: 6 }}>{t("auth.credit.sponsored")}</div>
        </div>
      </div>
      </div>
      <AuthFooter />
    </div>
  );
}
