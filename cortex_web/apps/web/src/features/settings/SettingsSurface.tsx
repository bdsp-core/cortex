import { type CSSProperties, useEffect, useState } from "react";

import * as api from "../../api";
import { EXPERTISE, PROFILE_SECTIONS } from "../../profileFields";

function BadgeCollection() {
  const [awards, setAwards] = useState<{ badges: api.Award[]; milestones: api.Award[] } | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    api.getAwards()
      .then((result) => { if (live) setAwards(result); })
      .catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  }, []);
  const formatDate = (iso: string) => new Date(iso).toLocaleDateString(undefined,
    { year: "numeric", month: "short", day: "numeric" });
  const byKey = new Map<string, api.Award>();
  for (const badge of awards?.badges ?? []) {
    if (!byKey.has(badge.key)) byKey.set(badge.key, badge);
  }
  const current = [...byKey.values()];
  const held = current.filter((badge) => !badge.revokedUtc);
  const lost = current.filter((badge) => badge.revokedUtc);
  const chip: CSSProperties = {
    display: "inline-flex", alignItems: "baseline", gap: 6,
    padding: "6px 12px", fontSize: 13, whiteSpace: "nowrap",
  };
  return (
    <section className="cx-panel">
      <h2>Badges &amp; milestones</h2>
      <p className="sub">
        Domain badges are earned by passing a domain on a certification test and
        are lost if a later test falls below that domain&apos;s bar. Milestones
        mark moments along your training program.
      </p>
      {failed && <p className="sub">Could not load your collection. Reload to retry.</p>}
      {awards && current.length === 0 && (
        <p className="sub">No badges yet. Pass a domain on a certification test to earn its badge.</p>
      )}
      {awards && current.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: awards.milestones.length ? 18 : 0 }}>
          {held.map((badge) => (
            <span key={badge.key} style={{ ...chip, background: "var(--teal-weak)",
              border: "1px solid var(--teal)", color: "var(--teal-deep)" }}>
              <b>{badge.label}</b>
              <span style={{ fontSize: 11.5, opacity: 0.85 }}>
                {badge.detail ? `${badge.detail} · ` : ""}since {formatDate(badge.awardedUtc)}
              </span>
            </span>
          ))}
          {lost.map((badge) => (
            <span key={badge.key} style={{ ...chip, background: "transparent",
              border: "1px solid var(--bd-subtle)", color: "var(--ink-subtle)" }}>
              <b style={{ fontWeight: 600 }}>{badge.label}</b>
              <span style={{ fontSize: 11.5 }}>lost {formatDate(badge.revokedUtc!)}</span>
            </span>
          ))}
        </div>
      )}
      {awards && awards.milestones.length > 0 && (
        <div>
          {awards.milestones.map((milestone) => (
            <div key={milestone.awardId} style={{ display: "flex", alignItems: "baseline",
              flexWrap: "wrap", gap: 12, padding: "8px 0",
              borderTop: "1px solid var(--bd-subtle)" }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 12,
                color: "var(--ink-subtle)", whiteSpace: "nowrap" }}>
                {formatDate(milestone.awardedUtc)}
              </span>
              <span style={{ fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>{milestone.label}</span>
              {milestone.detail && <span style={{ fontSize: 13, color: "var(--ink-subtle)" }}>{milestone.detail}</span>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/** Account, research-profile, reminder, credential, and recognition settings. */
export function SettingsSurface() {
  const [loaded, setLoaded] = useState(false);
  const [email, setEmail] = useState("");
  const [publicId, setPublicId] = useState("");
  const [authProvider, setAuthProvider] = useState("local");
  const [displayName, setDisplayName] = useState("");
  const [expertise, setExpertise] = useState("");
  const [profile, setProfile] = useState<Record<string, string>>({});
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileMessage, setProfileMessage] = useState<{ ok?: string; err?: string }>({});
  const [newEmail, setNewEmail] = useState("");
  const [emailPassword, setEmailPassword] = useState("");
  const [emailMessage, setEmailMessage] = useState<{ ok?: string; err?: string }>({});
  const [reminders, setReminders] = useState(true);
  const [remindersMessage, setRemindersMessage] = useState<{ ok?: string; err?: string }>({});
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmedPassword, setConfirmedPassword] = useState("");
  const [passwordMessage, setPasswordMessage] = useState<{ ok?: string; err?: string }>({});
  const [savingEmail, setSavingEmail] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  useEffect(() => {
    api.getProfile()
      .then((account) => {
        setEmail(account.email);
        setAuthProvider(account.authProvider || "local");
        setPublicId(account.publicId || "");
        setDisplayName(account.displayName);
        setExpertise(account.expertise);
        setProfile(account.profile || {});
        setReminders(account.trainingReminders !== false);
      })
      .catch(() => { /* show an editable empty form */ })
      .finally(() => setLoaded(true));
  }, []);

  const setField = (key: string) => (value: string) => (
    setProfile((current) => ({ ...current, [key]: value }))
  );

  async function saveProfile() {
    setSavingProfile(true);
    setProfileMessage({});
    try {
      await api.updateProfile(displayName.trim(), expertise, profile);
      setProfileMessage({ ok: "Saved." });
    } catch (error) {
      setProfileMessage({ err: (error as Error)?.message || "Could not save." });
    } finally {
      setSavingProfile(false);
    }
  }

  async function saveEmail() {
    if (savingEmail) return;
    setSavingEmail(true);
    setEmailMessage({});
    try {
      const result = await api.changeEmail(newEmail.trim(), emailPassword);
      setEmail(result.email);
      setNewEmail("");
      setEmailPassword("");
      setEmailMessage({ ok: "Email updated." });
    } catch (error) {
      const status = (error as api.ApiError)?.status;
      setEmailMessage({ err: status === 409
        ? "That email is already in use."
        : status === 403 ? "Password is incorrect." : "Could not update email." });
    } finally {
      setSavingEmail(false);
    }
  }

  async function toggleReminders(on: boolean) {
    setReminders(on);
    setRemindersMessage({});
    try {
      await api.setTrainingReminders(on);
      setRemindersMessage({ ok: on ? "Reminders on." : "Reminders off." });
    } catch {
      setReminders(!on);
      setRemindersMessage({ err: "Could not save. Try again." });
    }
  }

  async function savePassword() {
    if (savingPassword) return;
    setPasswordMessage({});
    if (newPassword.length < 8) {
      setPasswordMessage({ err: "New password must be at least 8 characters." });
      return;
    }
    if (newPassword !== confirmedPassword) {
      setPasswordMessage({ err: "New passwords do not match." });
      return;
    }
    setSavingPassword(true);
    try {
      await api.changePassword(currentPassword, newPassword);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmedPassword("");
      setPasswordMessage({ ok: "Password updated." });
    } catch (error) {
      const status = (error as api.ApiError)?.status;
      setPasswordMessage({ err: status === 403
        ? "Current password is incorrect."
        : "Could not update password." });
    } finally {
      setSavingPassword(false);
    }
  }

  if (!loaded) return <div className="cx-settings"><p className="sub">Loading…</p></div>;
  const isGoogle = authProvider === "google";
  const allFields = PROFILE_SECTIONS.flatMap((section) => section.fields);

  return (
    <div className="cx-settings">
      <h2>Settings</h2>
      <p className="sub">Update your account and profile details. Changes apply to your next test.</p>

      {publicId && (
        <section className="cx-panel">
          <h2>User ID</h2>
          <p className="sub">Your unique 9-digit CORTEX account ID. Include it when contacting support or reporting a problem.</p>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", fontSize: 22, letterSpacing: "0.14em", color: "var(--ink)" }}>
            {publicId}
          </div>
        </section>
      )}

      <BadgeCollection />

      <section className="cx-panel">
        <h2>Profile</h2>
        <p className="sub">Your role and background, used for research analysis.</p>
        <div className="cx-form-grid">
          <div className="cx-field">
            <label>Display name</label>
            <input className="cx-input" value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
          </div>
          <div className="cx-field">
            <label>Primary role / expertise</label>
            <select className="cx-select" value={expertise} onChange={(event) => setExpertise(event.target.value)}>
              <option value="">Not specified</option>
              {EXPERTISE.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </div>
          {allFields.map((field) => (
            <div className="cx-field" key={field.key}>
              <label>{field.label}</label>
              {field.kind === "text" ? (
                <input className="cx-input" value={profile[field.key] ?? ""} placeholder={field.placeholder}
                  onChange={(event) => setField(field.key)(event.target.value)} />
              ) : (
                <select className="cx-select" value={profile[field.key] ?? ""}
                  onChange={(event) => setField(field.key)(event.target.value)}>
                  <option value="">Not specified</option>
                  {field.options!.map((option) => <option key={option} value={option}>{option}</option>)}
                </select>
              )}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16 }}>
          <button type="button" className="cx-btn primary" onClick={saveProfile} disabled={savingProfile}>
            {savingProfile ? "Saving…" : "Save profile"}
          </button>
          {profileMessage.ok && <span className="cx-msg-ok">{profileMessage.ok}</span>}
          {profileMessage.err && <span className="cx-msg-err">{profileMessage.err}</span>}
        </div>
      </section>

      <section className="cx-panel">
        <h2>Email</h2>
        <p className="sub">Current: <b style={{ color: "var(--ink)" }}>{email}</b></p>
        {isGoogle ? (
          <p className="sub">This account signs in with Google; its email is managed there.</p>
        ) : (
          <div className="cx-form-grid">
            <div className="cx-field">
              <label>New email</label>
              <input className="cx-input" type="email" value={newEmail} onChange={(event) => setNewEmail(event.target.value)} />
            </div>
            <div className="cx-field">
              <label>Current password</label>
              <input className="cx-input" type="password" value={emailPassword} onChange={(event) => setEmailPassword(event.target.value)} />
            </div>
            <div className="cx-field full" style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
              <button type="button" className="cx-btn" onClick={saveEmail} disabled={savingEmail}>Update email</button>
              {emailMessage.ok && <span className="cx-msg-ok">{emailMessage.ok}</span>}
              {emailMessage.err && <span className="cx-msg-err">{emailMessage.err}</span>}
            </div>
          </div>
        )}
      </section>

      <section className="cx-panel">
        <h2>Email reminders</h2>
        <p className="sub">A short note when your training deck is due, sent at most once a day on a spaced schedule. Exam invitations are never sent.</p>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: 14, color: "var(--ink)" }}>
          <input type="checkbox" checked={reminders} onChange={(event) => toggleReminders(event.target.checked)} />
          Training reminders
        </label>
        {remindersMessage.ok && <span className="cx-msg-ok" style={{ marginLeft: 12 }}>{remindersMessage.ok}</span>}
        {remindersMessage.err && <span className="cx-msg-err" style={{ marginLeft: 12 }}>{remindersMessage.err}</span>}
      </section>

      <section className="cx-panel">
        <h2>Password</h2>
        {isGoogle ? (
          <p className="sub">This account signs in with Google; no password is set.</p>
        ) : (
          <div className="cx-form-grid">
            <div className="cx-field">
              <label>Current password</label>
              <input className="cx-input" type="password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} />
            </div>
            <div className="cx-field" />
            <div className="cx-field">
              <label>New password</label>
              <input className="cx-input" type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} />
            </div>
            <div className="cx-field">
              <label>Confirm new password</label>
              <input className="cx-input" type="password" value={confirmedPassword} onChange={(event) => setConfirmedPassword(event.target.value)} />
            </div>
            <div className="cx-field full" style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
              <button type="button" className="cx-btn" onClick={savePassword} disabled={savingPassword}>Update password</button>
              {passwordMessage.ok && <span className="cx-msg-ok">{passwordMessage.ok}</span>}
              {passwordMessage.err && <span className="cx-msg-err">{passwordMessage.err}</span>}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
