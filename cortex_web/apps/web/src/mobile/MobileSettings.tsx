// Phone settings: edit profile (display name, expertise, clinical/demographic
// fields), change password, change email. Same authed endpoints the desktop
// Settings uses (PUT /api/profile, POST /api/account/{password,email}) — only
// the presentation lives here, per the mobile/desktop boundary.

import { FormEvent, useEffect, useState } from "react";
import * as api from "../api";
import { COLORS } from "../../ui/theme";
import { EXPERTISE } from "../components/AuthFlow";
import { PROFILE_SECTIONS } from "../profileFields";
import * as S from "./styles";

function Status({ msg }: { msg: string | null }) {
  if (!msg) return null;
  const failed = msg.toLowerCase() !== "saved";
  return (
    <div style={{ fontSize: 13, marginTop: 10,
      color: failed ? COLORS.fail : COLORS.pass }}>
      {msg}
    </div>
  );
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function MobileSettings({ onBack }: { onBack: () => void }) {
  const [acct, setAcct] = useState<api.AccountProfile | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  // profile form
  const [name, setName] = useState("");
  const [expertise, setExpertise] = useState("");
  const [profile, setProfile] = useState<Record<string, string>>({});
  const [profileMsg, setProfileMsg] = useState<string | null>(null);
  // password form
  const [curPw, setCurPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [newPw2, setNewPw2] = useState("");
  const [pwMsg, setPwMsg] = useState<string | null>(null);
  // email form
  const [newEmail, setNewEmail] = useState("");
  const [emailPw, setEmailPw] = useState("");
  const [emailMsg, setEmailMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let gone = false;
    api.getProfile().then((a) => {
      if (gone) return;
      setAcct(a);
      setName(a.displayName);
      setExpertise(a.expertise);
      setProfile(a.profile ?? {});
    }).catch((e) => { if (!gone) setLoadErr(errText(e)); });
    return () => { gone = true; };
  }, []);

  async function saveProfile(e: FormEvent) {
    e.preventDefault();
    setProfileMsg(null); setBusy(true);
    try {
      await api.updateProfile(name.trim(), expertise, profile);
      setProfileMsg("Saved");
    } catch (ex) { setProfileMsg(errText(ex)); } finally { setBusy(false); }
  }

  async function savePassword(e: FormEvent) {
    e.preventDefault();
    setPwMsg(null);
    if (newPw.length < 8) { setPwMsg("New password must be at least 8 characters."); return; }
    if (newPw !== newPw2) { setPwMsg("New passwords don't match."); return; }
    setBusy(true);
    try {
      await api.changePassword(curPw, newPw);
      setCurPw(""); setNewPw(""); setNewPw2("");
      setPwMsg("Saved");
    } catch (ex) { setPwMsg(errText(ex)); } finally { setBusy(false); }
  }

  async function saveEmail(e: FormEvent) {
    e.preventDefault();
    setEmailMsg(null); setBusy(true);
    try {
      const r = await api.changeEmail(newEmail.trim(), emailPw);
      setAcct((a) => (a ? { ...a, email: r.email } : a));
      setNewEmail(""); setEmailPw("");
      setEmailMsg("Saved");
    } catch (ex) { setEmailMsg(errText(ex)); } finally { setBusy(false); }
  }

  return (
    <div style={S.page}>
      <header style={S.header}>
        <button style={S.ghostBtn} onClick={onBack}>← Back</button>
        <span style={{ fontSize: 15, fontWeight: 700 }}>Settings</span>
        <span style={{ width: 64 }} />{/* balance the back button */}
      </header>

      <main style={S.main}>
        {loadErr && (
          <section style={S.card}>
            <div style={{ color: COLORS.fail, fontSize: 14 }}>{loadErr}</div>
          </section>
        )}
        {!loadErr && acct === null && (
          <section style={S.card}><div style={S.faint}>Loading…</div></section>
        )}

        {acct !== null && (
          <>
            <section style={S.card}>
              <h2 style={S.sectionTitle}>Account</h2>
              <div style={{ fontSize: 14 }}>{acct.email}</div>
              {acct.publicId && (
                <div style={{ ...S.faint, marginTop: 4 }}>Account ID {acct.publicId}</div>
              )}
            </section>

            <section style={S.card}>
              <h2 style={S.sectionTitle}>Profile</h2>
              <form onSubmit={saveProfile}>
                <label style={S.label}>Display name
                  <input style={S.input} value={name}
                    onChange={(e) => setName(e.target.value)} required />
                </label>
                <label style={S.label}>Role
                  <select style={S.input} value={expertise}
                    onChange={(e) => setExpertise(e.target.value)}>
                    <option value="">Select…</option>
                    {EXPERTISE.map(([, v]) => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
                {PROFILE_SECTIONS.flatMap((sec) => sec.fields).map((f) => (
                  <label key={f.key} style={S.label}>{f.label}
                    {f.kind === "text" ? (
                      <input style={S.input} value={profile[f.key] ?? ""}
                        placeholder={f.placeholder}
                        onChange={(e) => setProfile((p) => ({ ...p, [f.key]: e.target.value }))} />
                    ) : (
                      <select style={S.input} value={profile[f.key] ?? ""}
                        onChange={(e) => setProfile((p) => ({ ...p, [f.key]: e.target.value }))}>
                        <option value="">Select…</option>
                        {(f.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    )}
                  </label>
                ))}
                <div style={{ marginTop: 14 }}>
                  <button type="submit" style={S.primaryBtn} disabled={busy}>Save profile</button>
                </div>
                <Status msg={profileMsg} />
              </form>
            </section>

            {acct.authProvider !== "google" && (
              <>
                <section style={S.card}>
                  <h2 style={S.sectionTitle}>Change password</h2>
                  <form onSubmit={savePassword}>
                    <label style={S.label}>Current password
                      <input style={S.input} type="password" value={curPw}
                        onChange={(e) => setCurPw(e.target.value)}
                        autoComplete="current-password" required />
                    </label>
                    <label style={S.label}>New password
                      <input style={S.input} type="password" value={newPw}
                        onChange={(e) => setNewPw(e.target.value)}
                        autoComplete="new-password" required />
                    </label>
                    <label style={S.label}>Confirm new password
                      <input style={S.input} type="password" value={newPw2}
                        onChange={(e) => setNewPw2(e.target.value)}
                        autoComplete="new-password" required />
                    </label>
                    <div style={{ marginTop: 14 }}>
                      <button type="submit" style={S.primaryBtn} disabled={busy}>Update password</button>
                    </div>
                    <Status msg={pwMsg} />
                  </form>
                </section>

                <section style={S.card}>
                  <h2 style={S.sectionTitle}>Change email</h2>
                  <form onSubmit={saveEmail}>
                    <label style={S.label}>New email
                      <input style={S.input} type="email" value={newEmail}
                        onChange={(e) => setNewEmail(e.target.value)}
                        autoComplete="email" required />
                    </label>
                    <label style={S.label}>Password
                      <input style={S.input} type="password" value={emailPw}
                        onChange={(e) => setEmailPw(e.target.value)}
                        autoComplete="current-password" required />
                    </label>
                    <div style={{ marginTop: 14 }}>
                      <button type="submit" style={S.primaryBtn} disabled={busy}>Update email</button>
                    </div>
                    <Status msg={emailMsg} />
                  </form>
                </section>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
