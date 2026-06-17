// Login screen — participant access code + password. POSTs /api/auth; on
// success the JWT is stored and the flow advances to consent. The test is
// password-protected (PLAN §1, §8): no code/password, no access.

import { useState } from "react";
import { ApiError, login } from "../api";
import { COLORS } from "../../ui/theme";
import { Button, Card, ErrorText, Field, Heading, Stage } from "./ui";

export function Login({ onAuthed, onBack }: { onAuthed: () => void; onBack: () => void }) {
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!code || !password) {
      setErr("Enter your access code and password.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await login(code.trim(), password);
      onAuthed();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setErr("Invalid access code or password.");
      else if (e instanceof ApiError && e.status === 0) setErr("Cannot reach the server.");
      else setErr(e instanceof Error ? e.message : "Login failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Stage maxW={460}>
      <Card>
        <Heading>Participant Access</Heading>
        <p style={{ color: COLORS.textBody, fontSize: 14, marginTop: 0, marginBottom: 24 }}>
          Enter the access code and password you were issued for this study.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <Field label="Access code" value={code} onChange={setCode}
                 placeholder="cortex-xxxxxxxx" autoFocus required />
          <Field label="Password" value={password} onChange={setPassword}
                 type="password" required />
          <ErrorText>{err}</ErrorText>
          <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
            <Button kind="ghost" onClick={onBack}>Back</Button>
            <Button type="submit" disabled={busy} style={{ flex: 1 }}>
              {busy ? "Signing in…" : "Continue"}
            </Button>
          </div>
        </form>
      </Card>
    </Stage>
  );
}
