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

  const [reminders, setReminders] = useState(true);
  const [remindersMsg, setRemindersMsg] = useState<string | null>(null);

  useEffect(() => {
    let gone = false;
    api.getProfile().then((a) => {
      if (gone) return;
      setAcct(a);
      setName(a.displayName);
      setExpertise(a.expertise);
      setProfile(a.profile ?? {});
      setReminders(a.trainingReminders !== false);
    }).catch((e) => { if (!gone) setLoadErr(errText(e)); });
    return () => { gone = true; };
  }, []);

  const [collection, setCollection] =
    useState<{ badges: api.Award[]; milestones: api.Award[] } | null>(null);
  useEffect(() => {
    let gone = false;
    api.getAwards().then((a) => { if (!gone) setCollection(a); })
      .catch(() => { /* section renders its empty state */ });
    return () => { gone = true; };
  }, []);

  async function toggleReminders(on: boolean) {
    setReminders(on); setRemindersMsg(null);   // optimistic; rolled back on error
    try {
      await api.setTrainingReminders(on);
      setRemindersMsg("Saved");
    } catch (ex) { setReminders(!on); setRemindersMsg(errText(ex)); }
  }

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
                    <option value="">Not specified</option>
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
                        <option value="">Not specified</option>
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

            <section style={S.card}>
              <h2 style={S.sectionTitle}>Badges &amp; milestones</h2>
              {(() => {
                if (!collection) return <div style={S.faint}>Loading…</div>;
                // newest row per badge key decides current state (held / lost)
                const byKey = new Map<string, api.Award>();
                for (const b of collection.badges) if (!byKey.has(b.key)) byKey.set(b.key, b);
                const current = [...byKey.values()];
                const fmt = (iso: string) => new Date(iso).toLocaleDateString(
                  undefined, { year: "numeric", month: "short", day: "numeric" });
                if (current.length === 0 && collection.milestones.length === 0) {
                  return <div style={S.faint}>No badges yet. Pass a domain on a certification test to earn its badge.</div>;
                }
                return (
                  <>
                    {current.length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                        {current.map((b) => (
                          <span key={b.key} style={{
                            padding: "5px 10px", fontSize: 12.5,
                            border: b.revokedUtc ? "1px solid var(--bd-subtle)" : "1px solid var(--teal)",
                            background: b.revokedUtc ? "transparent" : "var(--teal-weak)",
                            color: b.revokedUtc ? "var(--ink-subtle)" : "var(--teal-deep)",
                          }}>
                            <b>{b.label}</b>{" "}
                            {b.revokedUtc ? `lost ${fmt(b.revokedUtc)}` : `since ${fmt(b.awardedUtc)}`}
                          </span>
                        ))}
                      </div>
                    )}
                    {collection.milestones.map((m) => (
                      <div key={m.awardId} style={{ fontSize: 13, padding: "4px 0" }}>
                        <span style={S.faint}>{fmt(m.awardedUtc)}</span>{" "}
                        <b>{m.label}</b>
                      </div>
                    ))}
                  </>
                );
              })()}
            </section>

            <section style={S.card}>
              <h2 style={S.sectionTitle}>Email reminders</h2>
              <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 14 }}>
                <input type="checkbox" checked={reminders}
                  onChange={(e) => toggleReminders(e.target.checked)} />
                Training reminders
              </label>
              <div style={{ ...S.faint, marginTop: 6 }}>
                A short note when your training deck is due, at most once a day.
              </div>
              <Status msg={remindersMsg} />
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
