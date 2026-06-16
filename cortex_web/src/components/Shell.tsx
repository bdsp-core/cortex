// Dashboard shell — the persistent app chrome after sign-in (locked-v1
// "Mastery board"): a 220px left nav rail (brand · 4 nav surfaces · rail CTAs
// · who/theme/sign-out) and a main content area that renders the active
// surface from internal nav state. The certification test launches FROM the
// shell (rail CTA → onStartTest).
//
// For this phase only the Dashboard surface's KPI strip is wired to real
// /api/dashboard; the mastery grid/charts and the other three surfaces are
// titled placeholders (Phase 4).
//
// Conventions ported from the mockup: theme tokens (var(--*)), SHARP edges
// (var(--radius-*)), inline-SVG line icons (never emoji), canonical "protocol",
// no em dashes. Responsive breakpoints (<=1000px icon-only rail, <=640px top
// bar) mirror the mockup media queries; because those need @media rules that
// inline styles cannot express, the rail/layout class CSS is injected once.

import { useEffect, useState } from "react";
import * as api from "../api";
import { FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";

type Surface = "dashboard" | "training" | "protocol" | "history";

// Scoped CSS for the shell chrome. Inline styles can't carry @media queries,
// so the rail / layout / KPI structural rules (and only those) live here,
// keyed to .cx-* classes. All colors/spacing reference the theme tokens.
const SHELL_CSS = `
.cx-app{display:grid;grid-template-columns:220px 1fr;min-height:100vh;
  background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.45;}
.cx-rail{background:var(--rail-bg);border-right:1px solid var(--bd-subtle);
  padding:var(--s24) 0 var(--s16);display:flex;flex-direction:column;
  position:sticky;top:0;align-self:start;height:100vh;}
.cx-brand{display:flex;align-items:baseline;gap:6px;padding:0 var(--s24) var(--s24);}
.cx-logo{font-weight:700;letter-spacing:.14em;font-size:26px;color:var(--ink);}
.cx-logo .dot{color:var(--teal);}
.cx-nav{display:flex;flex-direction:column;gap:2px;padding:0 var(--s12);}
.cx-nav button{display:flex;align-items:center;gap:var(--s12);
  padding:var(--s8) var(--s12);border-radius:var(--radius-ctl);
  color:var(--ink-subtle);font-size:14px;font-weight:500;font-family:inherit;
  border:none;border-left:3px solid transparent;background:none;
  width:100%;text-align:left;cursor:pointer;
  transition:background .18s,color .18s;}
.cx-nav button:hover{background:var(--panel-hover);color:var(--ink);}
.cx-nav button.active{background:var(--teal-weak);color:var(--teal-deep);
  font-weight:600;border-left-color:var(--teal);}
.cx-nav .ic{width:17px;height:17px;flex:none;stroke:currentColor;fill:none;
  stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;}
.cx-spacer{flex:1;}
.cx-cta{display:flex;flex-direction:column;gap:var(--s8);padding:var(--s16) var(--s12) 0;}
.cx-cta .cx-btn{width:100%;justify-content:center;text-align:center;line-height:1.25;}
.cx-btn{font-family:inherit;font-size:14px;font-weight:600;
  padding:var(--s12) var(--s16);border-radius:var(--radius-ctl);
  border:1px solid var(--bd);background:var(--panel);color:var(--ink);
  cursor:pointer;transition:background .18s,border-color .18s;
  display:inline-flex;align-items:center;gap:var(--s8);}
.cx-btn:hover{background:var(--panel-hover);border-color:var(--bd-strong);}
.cx-btn.primary{background:var(--teal);border-color:var(--teal);color:#fff;}
.cx-btn.primary:hover{background:var(--teal-hover);border-color:var(--teal-hover);}
.cx-btn .arrow{font-family:var(--mono);}
.cx-foot{padding:var(--s16) var(--s24) 0;border-top:1px solid var(--bd-subtle);margin:0 var(--s12);}
.cx-who .name{font-weight:600;font-size:14px;color:var(--ink);}
.cx-who .sub{display:block;font-weight:400;font-size:12px;color:var(--ink-subtle);margin-top:1px;}
.cx-footrow{display:flex;align-items:center;gap:var(--s8);margin-top:var(--s12);}
.cx-signout{flex:1;display:inline-flex;align-items:center;gap:var(--s8);
  padding:var(--s8) var(--s12);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-ctl);background:var(--panel);color:var(--ink-subtle);
  font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;
  transition:background .18s,border-color .18s;}
.cx-signout:hover{background:var(--panel-hover);border-color:var(--bd);}
.cx-signout .ic{width:15px;height:15px;flex:none;stroke:currentColor;fill:none;
  stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;}
.cx-wrap{max-width:1320px;padding:var(--s24) var(--s32) var(--s64);}

.cx-kpi-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--s16);
  margin-bottom:var(--s24);}
.cx-kpi{background:var(--panel);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-panel);padding:var(--s16) var(--s24);
  display:flex;flex-direction:column;gap:var(--s4);}
.cx-kpi .v{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:30px;font-weight:700;line-height:1;color:var(--ink);}
.cx-kpi .v .u{font-size:15px;font-weight:600;color:var(--ink-subtle);margin-left:4px;}
.cx-kpi .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-subtle);}
.cx-kpi .note{font-size:11px;color:var(--ink-faint);font-family:var(--mono);}
.cx-kpi-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--s8);}
.cx-samplemark{font-size:11px;color:var(--ink-faint);letter-spacing:.04em;
  border:1px dashed var(--bd);border-radius:var(--radius-ctl);
  padding:2px 8px;background:var(--panel-hover);margin-left:auto;}
@media (max-width:720px){ .cx-kpi-strip{grid-template-columns:1fr;} }

.cx-panel{background:var(--panel);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-panel);padding:var(--s24);}
.cx-panel + .cx-panel{margin-top:var(--s24);}
.cx-phead{display:flex;align-items:baseline;justify-content:space-between;
  margin-bottom:var(--s16);gap:var(--s12);}
.cx-phead h2{font-size:20px;font-weight:600;margin:0;letter-spacing:-.01em;}
.cx-phead .sub{font-size:12px;color:var(--ink-subtle);}
.cx-placeholder{color:var(--ink-subtle);font-size:14px;}

.cx-foot-credit{margin-top:var(--s32);font-size:11px;color:var(--ink-faint);text-align:center;}
.cx-foot-credit .sep{margin:0 var(--s8);opacity:.6;}

/* icon-only rail on medium widths */
@media (max-width:1000px){
  .cx-app{grid-template-columns:60px 1fr;}
  .cx-rail{padding:var(--s16) 0;}
  .cx-brand{padding:0 0 var(--s16);justify-content:center;}
  .cx-logo{font-size:0;letter-spacing:0;}
  .cx-logo::before{content:"C";font-size:19px;letter-spacing:0;color:var(--teal);}
  .cx-nav{padding:0 8px;}
  .cx-nav button{justify-content:center;padding:var(--s12) 0;border-left:none;border-right:3px solid transparent;}
  .cx-nav button.active{border-left:none;border-right-color:var(--teal);}
  .cx-nav .lbl-text{display:none;}
  .cx-foot{padding:var(--s12) 0 0;margin:0 8px;text-align:center;}
  .cx-who{display:none;}
  .cx-signout .lbl-text{display:none;}
  .cx-cta{display:none;}
  .cx-wrap{padding:var(--s16) var(--s16) var(--s48);}
}
/* top-bar rail on narrow widths */
@media (max-width:640px){
  .cx-app{grid-template-columns:1fr;}
  .cx-rail{flex-direction:row;align-items:center;height:auto;position:static;
    padding:var(--s8) var(--s16);gap:var(--s8);
    border-right:none;border-bottom:1px solid var(--bd-subtle);}
  .cx-brand{padding:0;}
  .cx-logo{font-size:17px;letter-spacing:.1em;}
  .cx-logo::before{content:none;}
  .cx-nav{flex-direction:row;padding:0;gap:2px;margin-left:var(--s8);}
  .cx-nav button{padding:var(--s8);border-right:none;}
  .cx-nav button.active{border-right:none;background:var(--teal-weak);}
  .cx-nav .lbl-text{display:none;}
  .cx-foot{border-top:none;padding:0;margin:0;}
  .cx-who{display:none;}
  .cx-footrow{margin-top:0;}
  .cx-spacer{flex:1;}
}
`;

// Inline-SVG line icons for the four nav surfaces (ported verbatim from the
// mockup's nav, NOT emoji).
const ICONS: Record<Surface, JSX.Element> = {
  dashboard: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="3" width="7" height="9" /><rect x="14" y="3" width="7" height="5" />
      <rect x="14" y="12" width="7" height="9" /><rect x="3" y="16" width="7" height="5" />
    </svg>
  ),
  training: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" /><path d="M12 8v4l3 2" />
    </svg>
  ),
  protocol: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 6h18M3 12h18M3 18h12" />
    </svg>
  ),
  history: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 4h11l3 3v13H5z" /><path d="M8 9h8M8 13h8M8 17h5" />
    </svg>
  ),
};

const NAV: Array<{ id: Surface; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "training", label: "Daily training" },
  { id: "protocol", label: "My protocol" },
  { id: "history", label: "Certification history" },
];

// A titled placeholder panel for the not-yet-built surfaces (Phase 4).
function Placeholder({ title, note }: { title: string; note: string }) {
  return (
    <section className="cx-panel" aria-label={title}>
      <div className="cx-phead"><h2>{title}</h2></div>
      <div className="cx-placeholder">{note}</div>
    </section>
  );
}

// The Dashboard surface: the KPI strip wired to /api/dashboard plus a
// mastery-board placeholder (Phase 4).
function DashboardSurface() {
  const [kpis, setKpis] = useState<api.DashboardKpis | null>(null);
  const [sample, setSample] = useState(false);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let live = true;
    api.getDashboard()
      .then((d) => { if (live) { setKpis(d.kpis); setSample(d.sample); } })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, []);

  const due = kpis?.dueToday;
  const dueTotal = due ? due.new + due.learning + due.review : null;

  return (
    <>
      <section className="cx-kpi-strip" aria-label="Key training metrics">
        <div className="cx-kpi">
          <div className="cx-kpi-top">
            <span className="k">Current streak</span>
            {sample && <span className="cx-samplemark">sample data</span>}
          </div>
          <span className="v">{kpis ? kpis.streak : "–"}<span className="u">days</span></span>
        </div>
        <div className="cx-kpi">
          <span className="k">Due today</span>
          <span className="v">{dueTotal ?? "–"}<span className="u">items</span></span>
          {due && (
            <span className="note">{due.new} new · {due.learning} learning · {due.review} review</span>
          )}
        </div>
        <div className="cx-kpi">
          <span className="k">Next re-certification</span>
          <span className="v">{kpis ? kpis.nextRecertDays : "–"}<span className="u">days</span></span>
        </div>
      </section>

      <section className="cx-panel" aria-label="Mastery board">
        <div className="cx-phead"><h2>Mastery board</h2></div>
        <div className="cx-placeholder">
          {err
            ? "Could not load your dashboard. Try refreshing."
            : "Mastery board: coming in the next build."}
        </div>
      </section>
    </>
  );
}

export function Shell({
  onStartTest, onStartTraining, onSignOut,
}: {
  onStartTest: () => void;
  onStartTraining: () => void;
  onSignOut: () => void;
}) {
  const [surface, setSurface] = useState<Surface>("dashboard");
  const name = api.getDisplayName() ?? "Clinician";

  return (
    <div className="cx-app">
      <style>{SHELL_CSS}</style>

      <aside className="cx-rail">
        <div className="cx-brand">
          <span className="cx-logo" style={{ fontFamily: FONTS.serif }}>
            CORTEX<span className="dot">.</span>
          </span>
        </div>

        <nav className="cx-nav" aria-label="Primary">
          {NAV.map((n) => (
            <button
              key={n.id}
              type="button"
              className={surface === n.id ? "active" : ""}
              aria-current={surface === n.id ? "page" : undefined}
              onClick={() => setSurface(n.id)}
            >
              {ICONS[n.id]}
              <span className="lbl-text">{n.label}</span>
            </button>
          ))}
        </nav>

        <div className="cx-cta">
          <button
            type="button"
            className="cx-btn primary"
            onClick={() => { setSurface("training"); onStartTraining(); }}
          >
            Resume training <span className="arrow">→</span>
          </button>
          <button type="button" className="cx-btn" onClick={onStartTest}>
            Re-take certification test
          </button>
        </div>

        <div className="cx-spacer" />

        <div className="cx-foot">
          <div className="cx-who">
            <span className="name">{name}</span>
            <span className="sub">Epilepsy & Clinical Neurophysiology</span>
          </div>
          <div className="cx-footrow">
            <ThemeToggle />
            <button type="button" className="cx-signout" onClick={onSignOut} aria-label="Sign out">
              <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <path d="M16 17l5-5-5-5M21 12H9" />
              </svg>
              <span className="lbl-text">Sign out</span>
            </button>
          </div>
        </div>
      </aside>

      <main className="cx-wrap">
        {surface === "dashboard" && <DashboardSurface />}
        {surface === "training" && (
          <Placeholder
            title="Daily training"
            note="Spaced-repetition training decks: coming in the next build."
          />
        )}
        {surface === "protocol" && (
          <Placeholder
            title="My protocol"
            note="Your active learning protocol: coming in the next build."
          />
        )}
        {surface === "history" && (
          <Placeholder
            title="Certification history"
            note="Your past certification results: coming in the next build."
          />
        )}

        <footer className="cx-foot-credit">
          Developed by Elijah W. Keldsen and M. Brandon Westover
          <span className="sep">·</span>
          Supported by the Clinical Data Animation Center (CDAC).
        </footer>
      </main>
    </div>
  );
}
