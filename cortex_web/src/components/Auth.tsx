// Public-signup auth screen. Two tabs — "Sign up" (the default; for the
// LinkedIn-link case where most visitors are new) and "Sign in" (for
// returning participants). Both call api.register / api.login which store
// the JWT on success and surface clean error messages on failure.

import { useState } from "react";
import { ApiError, login, register } from "../api";
import { COLORS, FONTS } from "../../ui/theme";
import { Button, Card, ErrorText, Field, Heading, SelectField, Stage } from "./ui";

// Self-reported expertise dropdown shown at signup. Captured as an early
// data-quality signal even if the participant abandons mid-flow before the
// full Registration form.
const EXPERTISE = [
  "Attending epileptologist",
  "Attending neurologist (non-epilepsy)",
  "Clinical neurophysiology fellow",
  "Neurology resident",
  "EEG technologist",
  "Researcher / scientist",
  "Other",
];

export function Auth({ onAuthed, onBack }: { onAuthed: () => void; onBack: () => void }) {
  const [mode, setMode] = useState<"signup" | "signin">("signup");

  // shared
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // signup-only
  const [displayName, setDisplayName] = useState("");
  const [expertise, setExpertise] = useState("");
  const [confirm, setConfirm] = useState("");
  // Honeypot: hidden form field; humans don't fill it, naive scrapers do.
  // We send its value to the backend regardless and let it silently 200 a
  // bot without revealing the trap.
  const [honeypot, setHoneypot] = useState("");

  async function submit() {
    setErr(null);
    if (!email || !password) {
      setErr("Email and password are required.");
      return;
    }
    if (mode === "signup") {
      if (!displayName) {
        setErr("Please enter a display name.");
        return;
      }
      if (password.length < 8) {
        setErr("Password must be at least 8 characters.");
        return;
      }
      if (password !== confirm) {
        setErr("Passwords don't match.");
        return;
      }
    }
    setBusy(true);
    try {
      if (mode === "signup") {
        await register(email.trim().toLowerCase(), password,
                       displayName.trim(), expertise, honeypot);
      } else {
        await login(email.trim().toLowerCase(), password);
      }
      onAuthed();
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 401) setErr("Email or password is incorrect.");
        else if (e.status === 409) setErr("An account with this email already exists. Try signing in instead.");
        else if (e.status === 429) setErr("Too many attempts. Please wait a minute and try again.");
        else if (e.status === 400) setErr(e.message || "Please check the form.");
        else if (e.status === 0) setErr("Cannot reach the server.");
        else setErr(e.message || "Something went wrong.");
      } else {
        setErr(e instanceof Error ? e.message : "Something went wrong.");
      }
    } finally {
      setBusy(false);
    }
  }

  const tabStyle = (active: boolean) => ({
    flex: 1, padding: "10px 12px", textAlign: "center" as const,
    cursor: "pointer", fontWeight: 600, fontSize: 14, userSelect: "none" as const,
    color: active ? COLORS.textPrimary : COLORS.textTertiary,
    borderBottom: `2px solid ${active ? COLORS.accent : "transparent"}`,
  });

  return (
    <Stage maxW={460}>
      <Card>
        <Heading>{mode === "signup" ? "Create an account" : "Sign in"}</Heading>

        <div style={{ display: "flex", marginBottom: 18,
                       borderBottom: `1px solid ${COLORS.borderInactive}` }}>
          <div style={tabStyle(mode === "signup")} onClick={() => { setMode("signup"); setErr(null); }}>
            New here
          </div>
          <div style={tabStyle(mode === "signin")} onClick={() => { setMode("signin"); setErr(null); }}>
            I have an account
          </div>
        </div>

        <p style={{ color: COLORS.textBody, fontSize: 14, marginTop: 0, marginBottom: 20 }}>
          {mode === "signup"
            ? "Enter your email and choose a password. Your results are stored under this account; you can return later to review them."
            : "Sign in with the email and password you used when you registered."}
        </p>

        <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <Field label="Email" value={email} onChange={setEmail}
                 type="email" placeholder="you@example.org" autoFocus required />
          <Field label="Password" value={password} onChange={setPassword}
                 type="password" required />

          {mode === "signup" && (
            <>
              <Field label="Confirm password" value={confirm} onChange={setConfirm}
                     type="password" required />
              <Field label="Display name" value={displayName} onChange={setDisplayName}
                     placeholder="e.g. J. Doe, MD" required />
              <SelectField label="Primary role / expertise" value={expertise}
                           onChange={setExpertise} options={EXPERTISE} />
              {/* honeypot — hidden visually + from accessibility, but a naive
                  bot that fills every input will fill it and trip the trap. */}
              <input type="text" name="company" tabIndex={-1} autoComplete="off"
                     value={honeypot} onChange={(e) => setHoneypot(e.target.value)}
                     style={{ position: "absolute", left: "-9999px", width: 1, height: 1,
                              opacity: 0, pointerEvents: "none" }}
                     aria-hidden="true" />
            </>
          )}

          <ErrorText>{err}</ErrorText>

          <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
            <Button kind="ghost" onClick={onBack}>Back</Button>
            <Button type="submit" disabled={busy} style={{ flex: 1 }}>
              {busy
                ? (mode === "signup" ? "Creating account…" : "Signing in…")
                : (mode === "signup" ? "Create account" : "Sign in")}
            </Button>
          </div>
        </form>

        <div style={{ marginTop: 16, fontFamily: FONTS.sans,
                       color: COLORS.textTertiary, fontSize: 12, textAlign: "center" }}>
          {mode === "signup"
            ? "By creating an account you agree to take part in this research study and to the consent terms on the next page."
            : "Forgot your password? Contact the study coordinator."}
        </div>
      </Card>
    </Stage>
  );
}
