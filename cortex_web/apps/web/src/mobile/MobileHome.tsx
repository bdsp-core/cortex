// Signed-in phone home: certification status (KPIs + per-task grid), past
// attempts, and the desktop gate for the test itself. Read-only companion by
// design — running a sitting requires a computer (the EEG/spectrogram task
// was calibrated at desktop scale; see the gate card copy).

import { useEffect, useState } from "react";
import * as api from "../api";
import { COLORS, VERDICT_STYLE } from "../../ui/theme";
import * as S from "./styles";

function fmtDay(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "—" : d.toLocaleDateString();
}

function Verdict({ v }: { v: string }) {
  const st = VERDICT_STYLE[v] ?? VERDICT_STYLE.PENDING;
  return (
    <span style={{ color: st.color, fontWeight: 700, fontSize: 13, whiteSpace: "nowrap" }}>
      {st.label}
    </span>
  );
}

export function MobileHome({ onSettings, onSignOut }: {
  onSettings: () => void;
  onSignOut: () => void;
}) {
  const [dash, setDash] = useState<api.DashboardData | null>(null);
  const [history, setHistory] = useState<api.HistorySession[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let gone = false;
    Promise.all([api.getDashboard(), api.getHistory()])
      .then(([d, h]) => { if (!gone) { setDash(d); setHistory(h.sessions); } })
      .catch((e) => { if (!gone) setErr(e instanceof Error ? e.message : String(e)); });
    return () => { gone = true; };
  }, []);

  return (
    <div style={S.page}>
      <header style={S.header}>
        <img
          src="/cortex_logo_word_horizontal@3x.png"
          srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
          alt="CORTEX" style={{ height: 22 }} />
        <span style={{ display: "flex", gap: 8 }}>
          <button style={S.ghostBtn} onClick={onSettings}>Settings</button>
          <button style={S.ghostBtn} onClick={onSignOut}>Sign out</button>
        </span>
      </header>

      <main style={S.main}>
        <section style={{ ...S.card, borderLeft: "3px solid var(--teal)" }}>
          <div style={{ fontSize: 14, lineHeight: 1.5 }}>
            <strong>Tests run on a computer.</strong> EEG needs a big screen,
            so certification and training aren&apos;t available on phones. Sign in
            at <span style={{ whiteSpace: "nowrap", fontWeight: 600 }}>app.cortexeeg.org</span>{" "}
            on a desktop or laptop to take the test; your progress will show
            up here.
          </div>
        </section>

        {err && (
          <section style={S.card}>
            <div style={{ color: COLORS.fail, fontSize: 14 }}>{err}</div>
          </section>
        )}
        {!err && (dash === null || history === null) && (
          <section style={S.card}>
            <div style={S.faint}>Loading your results…</div>
          </section>
        )}

        {dash !== null && (
          <section style={S.card}>
            <h2 style={S.sectionTitle}>Certification status</h2>
            {!dash.hasResult ? (
              <div style={{ fontSize: 14, color: COLORS.textBody }}>
                No certification attempt yet.
              </div>
            ) : (
              <>
                {dash.kpis && (
                  <div style={{ display: "flex", gap: 16, marginBottom: 6, fontSize: 13,
                    color: COLORS.textBody, flexWrap: "wrap" }}>
                    <span><strong>{dash.kpis.tasksCertified}/{dash.kpis.tasksTotal}</strong> tasks certified</span>
                    {dash.kpis.meanAuroc != null && (
                      <span>mean AUROC <strong>{dash.kpis.meanAuroc.toFixed(2)}</strong></span>
                    )}
                    <span>last test {fmtDay(dash.kpis.lastAssessed)}</span>
                  </div>
                )}
                {dash.tasks.map((t) => (
                  <div key={t.taskK} style={S.rowLine}>
                    <span>{t.label}</span>
                    <span style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                      {t.ell != null && (
                        <span style={S.faint}>
                          ℓ {t.ell.toFixed(2)}{t.ellStar != null ? ` / ${t.ellStar.toFixed(2)}` : ""}
                        </span>
                      )}
                      <Verdict v={t.verdict} />
                    </span>
                  </div>
                ))}
              </>
            )}
          </section>
        )}

        {history !== null && history.length > 0 && (
          <section style={S.card}>
            <h2 style={S.sectionTitle}>Past tests</h2>
            {history.map((s) => {
              const verdicts = s.result?.verdicts ?? [];
              const passed = verdicts.filter((v) => v === "PASS").length;
              return (
                <div key={s.session_id} style={S.rowLine}>
                  <span>{fmtDay(s.finished_utc)}</span>
                  <span style={S.faint}>
                    {s.n_questions != null ? `${s.n_questions} questions · ` : ""}
                    {passed}/{verdicts.length || "—"} passed
                  </span>
                </div>
              );
            })}
          </section>
        )}
      </main>

      <footer style={{ padding: "12px 16px", textAlign: "center", ...S.faint }}>
        CORTEX · <a href="/privacy" style={{ color: "inherit" }}>Privacy</a> ·{" "}
        <a href="/report" style={{ color: "inherit" }}>Report an issue</a>
      </footer>
    </div>
  );
}
