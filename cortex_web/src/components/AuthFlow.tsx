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
// Visuals match cortex_web_design/mockups/locked-v1-auth: serif "CORTEX."
// wordmark, teal accent, SHARP edges, inline-SVG line icons, no em dashes.

import {
  CSSProperties, KeyboardEvent, ClipboardEvent, ReactNode, useRef, useState,
} from "react";
import {
  ApiError, EmailNotVerifiedError, login, register, verifyCode, resendCode,
  requestReset, resetPassword,
} from "../api";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";

type Screen = "signin" | "signup" | "verify" | "forgot" | "reset" | "success";

// Self-reported expertise dropdown shown at signup. Captured as an early
// data-quality signal even if the participant abandons mid-flow.
const EXPERTISE = [
  "Attending epileptologist",
  "Attending neurologist (non-epilepsy)",
  "Clinical neurophysiology fellow",
  "Neurology resident",
  "EEG technologist",
  "Researcher / scientist",
  "Other",
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

function btnStyle(primary: boolean, disabled?: boolean): CSSProperties {
  return {
    fontFamily: FONTS.sans,
    fontSize: 14,
    fontWeight: 600,
    padding: "12px 16px",
    borderRadius: "var(--radius-ctl)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    border: `1px solid ${primary ? "var(--teal)" : COLORS.borderInactive2}`,
    background: primary ? "var(--teal)" : COLORS.card,
    color: primary ? "#fff" : COLORS.textPrimary,
  };
}
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

function Field({
  label, type = "text", value, onChange, placeholder, required, autoFocus,
  autoComplete, onInput, children, name,
}: {
  label: ReactNode;
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
      <span style={labelStyle}>{label}</span>
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
          aria-label={`${ariaPrefix} digit ${i + 1}`}
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
  return (
    <div style={{ textAlign: "center", fontSize: 12, color: COLORS.textBody, marginTop: 16 }}>
      Didn't get it?{" "}
      <button type="button" style={{ ...linkBtnStyle, fontSize: 12, opacity: cooldown > 0 ? 0.45 : 1,
        cursor: cooldown > 0 ? "not-allowed" : "pointer", color: cooldown > 0 ? COLORS.textFaint : "var(--teal-deep)" }}
        disabled={cooldown > 0} onClick={onResend}>
        Resend code
      </button>
      {cooldown > 0 && <span> · resend in {cooldown}s</span>}
    </div>
  );
}

// Map an ApiError to a friendly message.
function apiMessage(e: unknown, fallback: string): string {
  if (e instanceof ApiError) {
    if (e.status === 429) return "Too many attempts. Please wait a minute and try again.";
    if (e.status === 0) return "Cannot reach the server.";
    return e.message || fallback;
  }
  return e instanceof Error ? e.message : fallback;
}

export function AuthFlow({ onAuthed }: { onAuthed: () => void }) {
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

  // success
  const [successTitle, setSuccessTitle] = useState("You're all set");
  const [successMsg, setSuccessMsg] = useState(
    "Your email is verified. Sign in to reach the assessment.");

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
      set((t) => {
        if (t <= 1) { clearInterval(cdTimer.current[which]); return 0; }
        return t - 1;
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
        setErr("Email or password is incorrect.");
      } else {
        setErr(apiMessage(ex, "Something went wrong."));
      }
    } finally {
      setBusy(false);
    }
  }

  async function doSignUp(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (suPw.length < 8) { setErr("Password must be at least 8 characters."); return; }
    if (suPw !== suPw2) { setErr("Passwords don't match."); return; }
    if (!suName.trim()) { setErr("Please enter a display name."); return; }
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
        setErr("An account with this email already exists. Try signing in instead.");
      } else {
        setErr(apiMessage(ex, "Please check the form."));
      }
    } finally {
      setBusy(false);
    }
  }

  async function doVerify(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const code = verifyDigits.join("");
    if (code.length !== 6) { setErr("Enter all 6 digits."); return; }
    setBusy(true);
    try {
      await verifyCode(verifyEmail, code);
      setSuccessTitle("Email verified");
      setSuccessMsg("Your email is verified. Sign in to reach the assessment.");
      go("success");
    } catch (ex) {
      setErr(apiMessage(ex, "That code isn't right. Check the digits and try again."));
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
      setErr(apiMessage(ex, "Could not resend the code."));
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
      setErr(apiMessage(ex, "Could not send a reset code."));
    } finally {
      setBusy(false);
    }
  }

  async function doReset(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const code = resetDigits.join("");
    if (code.length !== 6) { setErr("Enter the 6-digit reset code."); return; }
    if (rsPw.length < 8) { setErr("Password must be at least 8 characters."); return; }
    if (rsPw !== rsPw2) { setErr("Passwords don't match."); return; }
    setBusy(true);
    try {
      await resetPassword(resetEmail, code, rsPw);
      setSuccessTitle("Password updated");
      setSuccessMsg("Your password has been reset. Sign in with your new password.");
      go("success");
    } catch (ex) {
      setErr(apiMessage(ex, "That code isn't right. Check the digits and try again."));
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
      setErr(apiMessage(ex, "Could not resend the code."));
    }
  }

  // ── render ───────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "32px 24px 64px", position: "relative",
      background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
      boxSizing: "border-box",
    }}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />

      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{
          fontFamily: FONTS.serif, fontWeight: 500, fontSize: 42,
          letterSpacing: "0.14em", color: COLORS.textPrimary, lineHeight: 1,
        }}>
          CORTEX<span style={{ color: "var(--teal)" }}>.</span>
        </div>
        <div style={{
          marginTop: 8, fontFamily: FONTS.serif, fontSize: 14,
          letterSpacing: "0.14em", color: COLORS.textBody,
        }}>
          EEG Skill Certification
        </div>
      </div>

      {screen === "signin" && (
        <div style={cardStyle}>
          <h1 style={h1Style}>Sign in</h1>
          <p style={ledeStyle}>Sign in to take the assessment and review your certification history.</p>
          <form onSubmit={doSignIn}>
            <Field label="Email" type="email" value={siEmail} onChange={setSiEmail}
              placeholder="you@example.org" autoComplete="email" required autoFocus />
            <Field label="Password" type="password" value={siPw} onChange={setSiPw}
              autoComplete="current-password" required>
              <div style={{ textAlign: "right", marginTop: 4 }}>
                <button type="button" style={{ ...linkBtnStyle, fontSize: 12 }}
                  onClick={() => { setFpEmail(siEmail); go("forgot"); }}>
                  Forgot password?
                </button>
              </div>
            </Field>
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <button type="submit" disabled={busy} style={{ ...btnStyle(true, busy), flex: 1 }}>
                {busy ? "Signing in…" : <>Sign in <span style={{ fontFamily: FONTS.sans }}>→</span></>}
              </button>
            </div>
          </form>
          <div style={switchStyle}>
            New to CORTEX?{" "}
            <button type="button" style={linkBtnStyle} onClick={() => go("signup")}>
              Create an account
            </button>
          </div>
        </div>
      )}

      {screen === "signup" && (
        <div style={cardStyle}>
          <Steps on={1} />
          <h1 style={h1Style}>Create your account</h1>
          <p style={ledeStyle}>
            Your results are saved to this account. We'll email a 6-digit code to confirm your address.
          </p>
          <form onSubmit={doSignUp}>
            <Field label={<>Email <span style={{ color: "var(--teal)" }}>*</span></>}
              type="email" value={suEmail} onChange={setSuEmail}
              placeholder="you@example.org" autoComplete="email" required autoFocus />
            <Field label={<>Password <span style={{ color: "var(--teal)" }}>*</span></>}
              type="password" value={suPw} onChange={setSuPw}
              autoComplete="new-password" required>
              <StrengthMeter score={scorePw(suPw)} />
              <div style={hintStyle}>At least 8 characters.</div>
            </Field>
            <Field label={<>Confirm password <span style={{ color: "var(--teal)" }}>*</span></>}
              type="password" value={suPw2} onChange={setSuPw2} autoComplete="new-password" required />
            <Field label={<>Display name <span style={{ color: "var(--teal)" }}>*</span></>}
              value={suName} onChange={setSuName} placeholder="e.g. J. Doe, MD" required />
            <label style={{ display: "block", marginBottom: 16 }}>
              <span style={labelStyle}>Primary role / expertise</span>
              <select value={suRole} onChange={(e) => setSuRole(e.target.value)} style={inputStyle}>
                <option value="">Select…</option>
                {EXPERTISE.map((o) => <option key={o} value={o}>{o}</option>)}
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
              <button type="button" style={{ ...btnStyle(false), flex: 1 }} onClick={() => go("signin")}>
                Back
              </button>
              <button type="submit" disabled={busy} style={{ ...btnStyle(true, busy), flex: 1 }}>
                {busy ? "Creating account…" : "Create account"}
              </button>
            </div>
          </form>
          <div style={metaStyle}>
            By creating an account you agree to take part in this research study and to the consent
            terms shown before your first test.
          </div>
        </div>
      )}

      {screen === "verify" && (
        <div style={cardStyle}>
          <Steps on={2} />
          <h1 style={h1Style}>Verify your email</h1>
          <p style={ledeStyle}>
            Enter the 6-digit code we sent to{" "}
            <span style={{ fontFamily: FONTS.sans, color: COLORS.textPrimary }}>{verifyEmail}</span>.
            The code expires in 15 minutes.
          </p>
          <form onSubmit={doVerify}>
            <CodeInputs digits={verifyDigits} setDigits={setVerifyDigits} ariaPrefix="Verification code" />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <button type="submit" disabled={busy} style={{ ...btnStyle(true, busy), flex: 1 }}>
                {busy ? "Verifying…" : "Verify & continue"}
              </button>
            </div>
          </form>
          <Resend cooldown={verifyCooldown} onResend={doResendVerify} />
          <div style={{ ...switchStyle, marginTop: 8, paddingTop: 0, borderTop: "none" }}>
            Wrong address?{" "}
            <button type="button" style={linkBtnStyle} onClick={() => go("signup")}>Edit email</button>
          </div>
        </div>
      )}

      {screen === "forgot" && (
        <div style={cardStyle}>
          <h1 style={h1Style}>Reset your password</h1>
          <p style={ledeStyle}>Enter your account email. If it's registered, we'll send a 6-digit reset code.</p>
          <form onSubmit={doForgot}>
            <Field label="Email" type="email" value={fpEmail} onChange={setFpEmail}
              placeholder="you@example.org" autoComplete="email" required autoFocus />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <button type="button" style={{ ...btnStyle(false), flex: 1 }} onClick={() => go("signin")}>
                Back
              </button>
              <button type="submit" disabled={busy} style={{ ...btnStyle(true, busy), flex: 1 }}>
                {busy ? "Sending…" : "Send reset code"}
              </button>
            </div>
          </form>
        </div>
      )}

      {screen === "reset" && (
        <div style={cardStyle}>
          <h1 style={h1Style}>Set a new password</h1>
          <p style={ledeStyle}>
            Enter the code sent to{" "}
            <span style={{ fontFamily: FONTS.sans, color: COLORS.textPrimary }}>{resetEmail}</span>,
            then choose a new password.
          </p>
          <form onSubmit={doReset}>
            <div style={{ display: "block", marginBottom: 16 }}>
              <span style={labelStyle}>Reset code</span>
              <CodeInputs digits={resetDigits} setDigits={setResetDigits} ariaPrefix="Reset code" />
            </div>
            <Field label="New password" type="password" value={rsPw} onChange={setRsPw}
              autoComplete="new-password" required>
              <StrengthMeter score={scorePw(rsPw)} />
              <div style={hintStyle}>At least 8 characters.</div>
            </Field>
            <Field label="Confirm new password" type="password" value={rsPw2} onChange={setRsPw2}
              autoComplete="new-password" required />
            <FormError>{err}</FormError>
            <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
              <button type="submit" disabled={busy} style={{ ...btnStyle(true, busy), flex: 1 }}>
                {busy ? "Resetting…" : "Reset password"}
              </button>
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
          <h1 style={h1Style}>{successTitle}</h1>
          <p style={{ ...ledeStyle, marginBottom: 24 }}>{successMsg}</p>
          <div style={{ display: "flex", gap: 12 }}>
            <button type="button" style={{ ...btnStyle(true), flex: 1 }} onClick={() => {
              setSiPw("");
              go("signin");
            }}>
              Continue to sign in
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
