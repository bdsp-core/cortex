// Dashboard shell — the persistent app chrome after sign-in (locked-v1
// "Mastery board"): a 220px left nav rail (brand · 4 nav surfaces · rail CTAs
// · who/theme/sign-out) and a main content area that renders the active
// surface from internal nav state. The certification test launches FROM the
// shell (rail CTA → onStartTest).
//
// The Dashboard surface is wired to /api/dashboard, /api/trajectories and
// /api/regimen. All values are real cert-result data — per-task ℓ/θ/ℓ*/AUROC +
// verdict and cert-summary KPIs come from the latest result. Learning-protocol
// surfaces (streak, deck, ℓ/θ/RT training trajectories) carry NO data until the
// L1 trainer is ported; they render honest "available after training"
// placeholders rather than sample data.
//
// Conventions ported from the mockup: theme tokens (var(--*)), SHARP edges
// (var(--radius-*)), inline-SVG line icons (never emoji), canonical "protocol",
// no em dashes. Responsive breakpoints (<=1000px icon-only rail, <=640px top
// bar) mirror the mockup media queries; because those need @media rules that
// inline styles cannot express, the rail/layout class CSS is injected once.

import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api";
import { bootstrapOnce } from "../bootstrapStore";
import { ThemeToggle } from "../theme/ThemeProvider";
import { Ring, Sparkline, MiniChart, Heatmap, HeatLegend, StreakBar } from "./charts";
import { useI18n, LANGS, Lang } from "../i18n/LanguageProvider";

// Cohorts is a 660-line surface with its own chart engine, reached only via the
// "Cohorts" rail tab — lazy-load it so it isn't in the dashboard first paint.
const CohortSurface = lazy(() => import("./Cohorts").then((m) => ({ default: m.CohortSurface })));

type Surface = "dashboard" | "training" | "protocol" | "history" | "cohorts" | "settings";
// "drilldown" is a routed sub-view of the shell (a focused per-task page),
// NOT a top-level nav surface — it has no rail entry. The shell tracks it in
// its own view state alongside the selected task.
type View = Surface | "drilldown";

import { SHELL_CSS } from "./shell/shellCss";
import { reopenLabel, washoutActive } from "../washout";
import { CohortInviteBanner } from "./CohortInviteBanner";
import { MilestoneBanner } from "./MilestoneBanner";
import {
  FULL_NAMES,
  TASK_LABELS,
  TASK_ORDER,
  VerdictChip as Chip,
  aurocOf,
  formatAssessmentDate as fmtDate,
  verdictPresentation as chipFor,
  verdictsOf,
} from "../features/dashboard/verdict";
import { HistorySurface } from "../features/history/HistorySurface";
import { ProtocolSurface } from "../features/protocol/ProtocolSurface";
import { SettingsSurface } from "../features/settings/SettingsSurface";

// Compatibility export for existing callers; new code imports the feature module.
export { QuestionTable } from "../features/history/QuestionTable";

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
  cohorts: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="9" cy="8" r="3.2" />
      <path d="M3.5 19c.6-3 2.6-4.7 5.5-4.7s4.9 1.7 5.5 4.7" />
      <circle cx="17" cy="9" r="2.6" />
      <path d="M15.6 14.9c2.5.2 4.2 1.6 4.8 4.1" />
    </svg>
  ),
  settings: (
    <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
};

const NAV: Array<{ id: Surface; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "training", label: "Daily training" },
  { id: "protocol", label: "My protocol" },
  { id: "history", label: "Certification history" },
  { id: "cohorts", label: "Cohorts" },
  { id: "settings", label: "Settings" },
];


// Per-task view model from the real dashboard tasks. ℓ/ℓ*/AUROC are null for a
// legacy (verdicts-only) result or before any assessment.
interface TaskVM {
  code: string;
  label: string;
  taskK: number;
  ell: number | null;
  ellStar: number | null;
  theta: number | null;
  auroc: number | null;
  verdict: string;
}

// Per-task trajectory series, grouped from the flat trajectory point list.
interface TrajVM {
  ell: number[];
  theta: number[];
  sd: number[];
  rt: number[];   // ms
  phase: string[];   // per-point: "eval" | "train" | "recert" (drives marker weight)
}

function buildTasks(d: api.DashboardData): TaskVM[] {
  const byCode = new Map(d.tasks.map((t) => [t.code, t]));
  // The backend already returns real per-task ℓ/ℓ*/AUROC + verdict (or an empty
  // list before any assessment). Render all 7 in canonical order; tasks absent
  // from the response are "not yet assessed".
  return TASK_ORDER.map((code, k) => {
    const t = byCode.get(code);
    if (!t) {
      return { code, label: TASK_LABELS[code] ?? code, taskK: k,
               ell: null, ellStar: null, theta: null, auroc: null, verdict: "NOT_ASSESSED" };
    }
    return {
      code,
      label: t.label ?? TASK_LABELS[code] ?? code,
      taskK: t.taskK ?? k,
      ell: t.ell,
      ellStar: t.ellStar,
      theta: t.theta,
      auroc: t.auroc,
      verdict: t.verdict ?? "PENDING",
    };
  });
}

function buildTraj(points: api.TrajectoryPoint[]): Map<number, TrajVM> {
  const m = new Map<number, TrajVM>();
  for (const p of points) {
    let v = m.get(p.taskK);
    if (!v) { v = { ell: [], theta: [], sd: [], rt: [], phase: [] }; m.set(p.taskK, v); }
    v.ell.push(p.ell);
    v.theta.push(p.theta);
    v.sd.push(p.sd);
    v.rt.push(p.rt);
    v.phase.push(p.phase);
  }
  return m;
}

// Three stacked detail mini-charts for the selected task (ℓ / θ / RT).
function DetailCharts({ task, traj }: { task: TaskVM; traj?: TrajVM }) {
  if (!traj || traj.ell.length < 1) {
    return <div className="cx-placeholder">No trajectory yet for this task.</div>;
  }
  const n = traj.ell.length;
  // A trajectory only exists alongside a real result, so ℓ/ℓ* are non-null here;
  // coalesce for the chart math to satisfy the nullable types.
  const ellNow = task.ell ?? 0;
  const ellStar = task.ellStar ?? 0;

  // (a) ℓ — skill toward ℓ*, ±sd band, dashed cut. ℓ is the ONLY param with ℓ*.
  const sd = (i: number) => traj.sd[i] ?? 0;
  const lows = traj.ell.map((v, i) => v - sd(i));
  const highs = traj.ell.map((v, i) => v + sd(i));
  let aMin = Math.min(...lows, ellStar);
  let aMax = Math.max(...highs, ellStar);
  const aPad = (aMax - aMin) * 0.14 || 0.1;
  aMin -= aPad; aMax += aPad;

  // (b) θ — bias, neutral 0 reference, NO cut. Symmetric domain around 0.
  const tAbs = Math.max(0.2, ...traj.theta.map(Math.abs)) * 1.25;

  // (c) RT — median reaction time (s), NO target line. Reaction time can be
  // absent (e.g. a seeded/imported result with no per-trial timing); only chart
  // it when every point has a finite value.
  const rtFinite = traj.rt.every((v) => typeof v === "number" && isFinite(v));
  const rtS = traj.rt.map((v) => (v ?? 0) / 1000);
  let rMin = Math.min(...rtS);
  let rMax = Math.max(...rtS);
  const rPad = (rMax - rMin) * 0.18 || 0.3;
  // Clamp the padded floor at 0: a reaction time can't be negative, so the
  // axis must never show sub-zero seconds (flat/low series used to).
  rMin = Math.max(0, rMin - rPad); rMax += rPad;

  const thNow = traj.theta[n - 1];
  const fmtTheta = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  return (
    <div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">ℓ(t) <span className="k">skill · 1/σ</span></span>
          <span className="cur">now <b>{ellNow.toFixed(2)}</b> · ℓ* {ellStar.toFixed(2)}</span>
        </div>
        <MiniChart
          series={traj.ell} yMin={aMin} yMax={aMax} fmt={(v) => v.toFixed(2)}
          label={`${task.label} skill ℓ trajectory toward ℓ*`} phases={traj.phase}
          band={[lows, highs]} rule={{ v: ellStar, label: "ℓ* = " + ellStar.toFixed(2) }}
        />
      </div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">θ(t) <span className="k">decision bias</span></span>
          <span className="cur">now <b>{fmtTheta(thNow)}</b> · neutral 0</span>
        </div>
        <MiniChart
          series={traj.theta} yMin={-tAbs} yMax={tAbs} fmt={fmtTheta} phases={traj.phase}
          label={`${task.label} decision bias θ trajectory toward neutral`} zeroRule
        />
      </div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">RT(t) <span className="k">median reaction time</span></span>
          {rtFinite && <span className="cur">now <b>{(traj.rt[n - 1] / 1000).toFixed(1)}s</b></span>}
        </div>
        {rtFinite ? (
          <MiniChart
            series={rtS} yMin={rMin} yMax={rMax} fmt={(v) => v.toFixed(1) + "s"}
            label={`${task.label} median reaction-time trend`} phases={traj.phase}
          />
        ) : (
          <div className="cx-placeholder">Reaction-time data not recorded for this attempt.</div>
        )}
      </div>
    </div>
  );
}

// First-login welcome (shown while the participant has no cert result yet).
function WelcomeModal({ onStart, onClose }: { onStart: () => void; onClose: () => void }) {
  return (
    <div className="cx-welcome-backdrop" role="dialog" aria-modal="true"
         aria-label="Welcome to CORTEX" onClick={onClose}>
      <div className="cx-welcome-card" onClick={(e) => e.stopPropagation()}>
        <img
          className="cx-welcome-logo"
          src="/cortex_logo_word_horizontal@3x.png"
          srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
          alt="CORTEX"
        />
        <h2>Welcome to the CORTEX protocol</h2>
        <p>Thank you for joining us.</p>
        <p>By taking part and engaging with the assessments, you're directly helping{" "}
          <strong>train physicians and EEG specialists around the world</strong> to read
          EEG more accurately and consistently.</p>
        <p>To begin, please take the <strong>certification test</strong>. It tunes the
          adaptive learning algorithm to your current skill, so everything that follows
          is tailored to you.</p>
        <p>Suggestions and problem reports are welcome through the{" "}
          <strong>Report a problem</strong> tab in the left navigation.</p>
        <p className="cx-welcome-sign">
          Sincerely,<br />Elijah W. Keldsen and M. Brandon Westover
        </p>
        <div className="cx-welcome-actions">
          <button type="button" className="cx-btn" onClick={onClose}>Not now</button>
          <button type="button" className="cx-btn primary" onClick={onStart}>
            Take the certification test
          </button>
        </div>
      </div>
    </div>
  );
}

// The Dashboard surface: cert-summary KPI strip + the mastery board (mastery
// grid, detail panel, active-protocol table, consistency sidebar). Per-task
// ℓ/θ/ℓ*/AUROC + verdict and the KPIs are real (from the latest result); the
// training-trajectory / protocol / streak surfaces show "available after
// training" placeholders until the trainer ships.
// KPI-tile caption that rides under the bar, anchored at the fill's leading
// edge: `value` right-aligns into the anchor, `sep` (the "·") sits ON it, and
// `rest` flows right. The row mirrors the bar row's [lo] [bar] [hi] geometry
// with hidden endpoint copies so percentages line up with the bar itself. The
// two flex-grow cells (pct : 100−pct) can't shrink below their text, so the
// anchor clamps inside the tile at the extremes — it never spills out.
function KpiCaption({ pct, lo, hi, value, sep, rest }: {
  pct: number; lo: string; hi: string;
  value?: string;   // left of the anchor (e.g. "0.86"); omit for single-part captions
  sep?: string;     // rendered on the anchor itself (e.g. "·")
  rest: string;     // right of the anchor (e.g. "GPD", "of 7 domains", "no test yet")
}) {
  const p = Math.max(0, Math.min(100, pct));
  return (
    <div style={{ display: "flex", gap: 6 }}>
      <span className="note" style={{ visibility: "hidden" }}>{lo}</span>
      <div style={{ flex: 1, minWidth: 0, display: "flex", overflow: "hidden" }}>
        <span className="note" style={{ flexGrow: p, flexBasis: 0, whiteSpace: "nowrap", textAlign: "right" }}>
          {value ? `${value}\u00A0` : ""}
        </span>
        {sep != null && <span className="note">{sep}</span>}
        <span className="note" style={{ flexGrow: 100 - p, flexBasis: 0, whiteSpace: "nowrap" }}>
          {sep != null ? `\u00A0${rest}` : rest}
        </span>
      </div>
      <span className="note" style={{ visibility: "hidden" }}>{hi}</span>
    </div>
  );
}

function DashboardSurface({ onDrilldown, onStartTest }: { onDrilldown: (taskK: number) => void; onStartTest: () => void }) {
  const [dash, setDash] = useState<api.DashboardData | null>(null);
  const [trajPts, setTrajPts] = useState<api.TrajectoryPoint[]>([]);
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [activity, setActivity] = useState<Record<string, number>>({});
  const [err, setErr] = useState(false);
  const [selected, setSelected] = useState<string>("gpd");
  const [showWelcome, setShowWelcome] = useState(false);

  useEffect(() => {
    let live = true;
    // One shared /api/bootstrap round-trip for the whole dashboard entry
    // (App's washout status and the invite banner join the same promise).
    // Sections stay failure-isolated server-side: a null section leaves that
    // surface in its empty state — same behavior as when these were four
    // independent calls and one failed.
    bootstrapOnce()
      .then((b) => {
        if (!live) return;
        if (b.dashboard) setDash(b.dashboard); else setErr(true);
        if (b.trajectories) setTrajPts(b.trajectories.trajectories);
        if (b.regimen) setRegimen(b.regimen.regimen);
        if (b.activity) setActivity(b.activity.days || {});
      })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, []);

  // Welcome the participant on each login until they complete the certification
  // test (the once-per-session flag is cleared on sign-out, see api.logout).
  useEffect(() => {
    if (dash && !dash.hasResult && !sessionStorage.getItem("cortex-welcome-seen")) {
      setShowWelcome(true);
    }
  }, [dash]);
  const dismissWelcome = () => {
    try { sessionStorage.setItem("cortex-welcome-seen", "1"); } catch { /* private mode */ }
    setShowWelcome(false);
  };

  const tasks = useMemo(() => (dash ? buildTasks(dash) : []), [dash]);
  const trajByK = useMemo(() => buildTraj(trajPts), [trajPts]);

  const hasData = !!dash?.hasResult;
  // Real cert-summary KPIs (tasks certified / worst + best domain AUROC /
  // last assessed). Each tile's fill % is computed once and shared by the
  // bar and its caption (which rides under the fill's leading edge).
  const kpis = dash?.kpis ?? null;
  const certPct = kpis ? Math.round(100 * kpis.tasksCertified / Math.max(1, kpis.tasksTotal)) : 0;
  const aurocPct = (d: { auroc: number } | null | undefined) =>
    d ? Math.round(100 * Math.max(0, Math.min(1, (d.auroc - 0.5) / 0.5))) : 0;
  const worstPct = aurocPct(kpis?.worstDomain);
  const bestPct = aurocPct(kpis?.bestDomain);

  const sel = tasks.find((t) => t.code === selected) ?? tasks[0];
  const nCertified = tasks.filter((t) => chipFor(t.verdict).cls === "pass").length;

  // Active protocol rows: deck entries that are still being trained (any
  // new/learning/due) carry through their per-task verdict chip.
  const verdictByK = useMemo(
    () => new Map(tasks.map((t) => [t.taskK, t.verdict])),
    [tasks],
  );

  return (
    <>
      {showWelcome && (
        <WelcomeModal
          onClose={dismissWelcome}
          onStart={() => { dismissWelcome(); onStartTest(); }}
        />
      )}
      <section className="cx-kpi-strip" aria-label="Certification summary">
        {/* Bars, not big numerals. AUROC bars are anchored at chance:
            fill = (AUROC - 0.5) / 0.5, so an empty bar means guessing and a
            full bar means perfect discrimination; the exact value rides in
            the caption. The certified bar is the plain fraction of domains. */}
        <div className="cx-kpi">
          <span className="k">Domains certified</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span className="note">0</span>
            <div className="cx-kpi-bar" aria-hidden="true" style={{ flex: 1 }}>
              <i style={{ width: `${certPct}%` }} />
            </div>
            <span className="note">{kpis ? kpis.tasksTotal : 7}</span>
          </div>
          <KpiCaption pct={certPct} lo="0" hi={String(kpis ? kpis.tasksTotal : 7)}
            value={kpis ? String(kpis.tasksCertified) : undefined}
            rest={kpis ? `of ${kpis.tasksTotal} domains` : "no test yet"} />
        </div>
        <div className="cx-kpi">
          <span className="k">Weakest domain AUROC</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span className="note">0.5</span>
            <div className="cx-kpi-bar" aria-hidden="true" style={{ flex: 1 }}>
              <i style={{ width: `${worstPct}%` }} />
            </div>
            <span className="note">1.0</span>
          </div>
          <KpiCaption pct={worstPct} lo="0.5" hi="1.0"
            value={kpis?.worstDomain ? kpis.worstDomain.auroc.toFixed(2) : undefined}
            sep={kpis?.worstDomain ? "·" : undefined}
            rest={kpis?.worstDomain ? kpis.worstDomain.label : "no test yet"} />
        </div>
        <div className="cx-kpi">
          <span className="k">Strongest domain AUROC</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span className="note">0.5</span>
            <div className="cx-kpi-bar" aria-hidden="true" style={{ flex: 1 }}>
              <i style={{ width: `${bestPct}%` }} />
            </div>
            <span className="note">1.0</span>
          </div>
          <KpiCaption pct={bestPct} lo="0.5" hi="1.0"
            value={kpis?.bestDomain ? kpis.bestDomain.auroc.toFixed(2) : undefined}
            sep={kpis?.bestDomain ? "·" : undefined}
            rest={kpis?.bestDomain ? kpis.bestDomain.label : "no test yet"} />
        </div>
      </section>

      {err && !dash && (
        <section className="cx-panel" aria-label="Mastery board">
          <div className="cx-phead"><h2>Mastery board</h2></div>
          <div className="cx-placeholder">Could not load your dashboard. Try refreshing.</div>
        </section>
      )}

      {dash && (
        <div className="cx-board">
          <div>
            {!hasData && (
              <div className="cx-empty-cta">
                <div className="txt">
                  <h3>You haven't been assessed yet</h3>
                  <p>Take the certification test to tune the learning algorithm to your skill and unlock your mastery board.</p>
                </div>
                <button type="button" className="cx-btn primary" onClick={onStartTest}>
                  Take the certification test
                </button>
              </div>
            )}
            {/* ── mastery grid ── */}
            <section className="cx-panel" aria-label="Per-task mastery grid">
              <div className="cx-phead">
                <h2>
                  Mastery{" "}
                  <span className="sub" style={{ fontWeight: 400 }}>
                    · {nCertified} / 7 tasks certified
                  </span>
                </h2>
                <div className="cx-legend">
                  <span><i style={{ background: "var(--pass)" }} />pass</span>
                  <span><i style={{ background: "var(--refer-b)" }} />refer</span>
                  <span><i style={{ background: "var(--fail)" }} />fail</span>
                  <span><i style={{ background: "var(--teal)" }} />progress to ℓ*</span>
                </div>
              </div>
              <div className="cx-mastery-grid">
                {tasks.map((t) => {
                  const tr = trajByK.get(t.taskK);
                  return (
                    <div
                      key={t.code}
                      role="button"
                      tabIndex={0}
                      className={`cx-tile${t.code === selected ? " sel" : ""}`}
                      aria-label={`${t.label} mastery tile`}
                      aria-pressed={t.code === selected}
                      onClick={() => setSelected(t.code)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelected(t.code); }
                      }}
                    >
                      <div className="ringcell">
                        <Ring ell={t.ell} ellStar={t.ellStar} />
                      </div>
                      <div className="head">
                        <div className="task">{t.label}</div>
                        <Chip verdict={t.verdict} />
                      </div>
                      <div className="foot">
                        <span className="auroc">
                          {t.auroc != null
                            ? <>AUROC <b>{t.auroc.toFixed(2)}</b></>
                            : "not yet assessed"}
                        </span>
                        {tr && tr.ell.length >= 2 && <Sparkline series={tr.ell} />}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* ── detail trajectory ── */}
            {hasData && sel && (
              <section className="cx-panel" aria-label="Skill trajectory detail">
                <div className="cx-phead" style={{ alignItems: "flex-start" }}>
                  <div style={{ width: "100%" }}>
                    <div className="cx-detail-head">
                      <span className="tname">{sel.label}</span>
                      <Chip verdict={sel.verdict} />
                      <span className="meta">
                        {FULL_NAMES[sel.code] ?? sel.label} · eval → daily training → re-cert
                      </span>
                      <button type="button" className="cx-linkbtn"
                        onClick={() => onDrilldown(sel.taskK)}>
                        Open full view
                      </button>
                    </div>
                    <div className="cx-detail-meta-row">
                      <div className="cx-metric"><span className="v">{sel.ell != null ? sel.ell.toFixed(2) : "–"}</span><span className="k">current ℓ</span></div>
                      <div className="cx-metric"><span className="v">{sel.ellStar != null ? sel.ellStar.toFixed(2) : "–"}</span><span className="k">target ℓ*</span></div>
                      <div className="cx-metric">
                        <span className="v">{sel.ell != null && sel.ellStar != null
                          ? (sel.ell - sel.ellStar >= 0 ? "+" : "−") + Math.abs(sel.ell - sel.ellStar).toFixed(2)
                          : "–"}</span>
                        <span className="k">gap to ℓ*</span>
                      </div>
                      <div className="cx-metric">
                        <span className="v">{sel.theta != null ? (sel.theta >= 0 ? "+" : "−") + Math.abs(sel.theta).toFixed(2) : "–"}</span>
                        <span className="k">bias θ</span>
                      </div>
                      <div className="cx-metric"><span className="v">{sel.auroc != null ? sel.auroc.toFixed(2) : "–"}</span><span className="k">AUROC</span></div>
                    </div>
                  </div>
                </div>
                {(() => {
                  const tr = trajByK.get(sel.taskK);
                  if (tr && tr.ell.length >= 1) {
                    return (
                      <>
                        <DetailCharts task={sel} traj={tr} />
                        <div className="cx-phase-axis">
                          <span style={{ width: "18%" }}>EVAL</span>
                          <span style={{ width: "64%" }}>DAILY TRAINING</span>
                          <span style={{ width: "18%" }}>RE-CERT (proj.)</span>
                        </div>
                      </>
                    );
                  }
                  return (
                    <div className="cx-placeholder">
                      Complete a certification test to chart your skill (ℓ), decision bias (θ), and response time for this domain.
                    </div>
                  );
                })()}
              </section>
            )}

            {/* ── active protocol table ── */}
            {hasData && (
            <section className="cx-panel" aria-label="Active learning protocol">
              <div className="cx-phead">
                <h2>Active protocol</h2>
              </div>
              {!regimen && (
                <div className="cx-placeholder">
                  No protocol yet. Your first training session builds a spaced-repetition plan from your certification result, weighted toward the domains still below their ℓ* bar.
                </div>
              )}
              {regimen && (
                <table className="cx-deck">
                  <thead>
                    <tr>
                      <th className="task">Targeted task</th>
                      <th className="r">ℓ → ℓ*</th>
                      <th className="cnt">New</th>
                      <th className="cnt">Learning</th>
                      <th className="cnt">Due</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(regimen.deck ?? []).map((row) => (
                      <tr key={row.code}>
                        <td className="task">{row.label}</td>
                        <td className="ellcell r">
                          {row.ell.toFixed(2)} <span className="of">→ {row.ellStar.toFixed(2)}</span>
                        </td>
                        <td className={`cnt new${row.new ? "" : " zero"}`}>{row.new}</td>
                        <td className={`cnt learn${row.learning ? "" : " zero"}`}>{row.learning}</td>
                        <td className={`cnt due${row.due ? "" : " zero"}`}>{row.due}</td>
                        <td><Chip verdict={verdictByK.get(row.taskK) ?? "IN_TRAINING"} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
            )}
          </div>

          {/* ── consistency sidebar: real activity heatmap (sign-in / cert /
                training), shown for everyone from day one ── */}
          <aside>
            <section className="cx-panel plain" aria-label="Recent activity">
              <div className="cx-phead"><h2>Consistency</h2></div>
              <div className="cx-heat-wrap">
                <Heatmap activity={activity} />
                <HeatLegend lastEval={kpis?.lastAssessed} />
                <StreakBar activity={activity} />
              </div>
            </section>

            <section className="cx-panel plain" aria-label="Today's deck"
              style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--bd-subtle)" }}>
              <div className="cx-phead"><h2>Today's deck</h2><span className="sub">due now</span></div>
              {hasData && regimen ? (
                <div className="cx-deck-mini">
                  <table className="cx-deck">
                    <thead>
                      <tr>
                        <th className="task">Task</th>
                        <th className="cnt">New</th>
                        <th className="cnt">Lrn</th>
                        <th className="cnt">Due</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(regimen.deck ?? []).map((row) => (
                        <tr key={row.code}>
                          <td className="task">{row.label}</td>
                          <td className={`cnt new${row.new ? "" : " zero"}`}>{row.new}</td>
                          <td className={`cnt learn${row.learning ? "" : " zero"}`}>{row.learning}</td>
                          <td className={`cnt due${row.due ? "" : " zero"}`}>{row.due}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="cx-placeholder">No deck yet. Your first training session builds it.</div>
              )}
              <div className="cx-deckhint">
                <span className="k"><i style={{ background: "var(--c-new)" }} />New</span>
                <span className="k"><i style={{ background: "var(--c-learn)" }} />Learning</span>
                <span className="k"><i style={{ background: "var(--c-due)" }} />Due</span>
              </div>
            </section>
          </aside>
        </div>
      )}
    </>
  );
}

// ── Daily training: Anki-style deck + a non-scored PRACTICE run ──────────────
// The real adaptive trainer is not ported yet (project decision), so "Start
// today's session" records an honest training_session start+finalize behind a
// clear "Practice — not scored" banner. No verdicts, no engine — it's deck
// review + bookkeeping. The deck comes from a real regimen; until the trainer
// generates one there is no deck (no sample data).
function TrainingSurface() {
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [practicing, setPracticing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [doneNote, setDoneNote] = useState<string | null>(null);
  // Bridges practice start→finish. A useRef (not a window global) so a second
  // tab or overlapping session can't clobber it.
  const trainingIdRef = useRef<string | null>(null);

  useEffect(() => {
    let live = true;
    api.getRegimen()
      .then((r) => { if (live) { setRegimen(r.regimen); setLoaded(true); } })
      .catch(() => { /* leave deck empty */ });
    return () => { live = false; };
  }, []);

  const totals = useMemo(() => {
    const d = regimen?.deck ?? [];
    return {
      neu: d.reduce((a, r) => a + r.new, 0),
      learn: d.reduce((a, r) => a + r.learning, 0),
      due: d.reduce((a, r) => a + r.due, 0),
    };
  }, [regimen]);
  const totalItems = totals.neu + totals.learn + totals.due;

  // Start the practice session: record a training_session, then enter the
  // practice panel. The session is finalized (with a few sample trajectory
  // points appended) on finish/exit.
  const start = useCallback(async () => {
    setBusy(true);
    setDoneNote(null);
    try {
      const { trainingId } = await api.startTrainingSession();
      trainingIdRef.current = trainingId;
      setPracticing(true);
    } catch {
      setDoneNote("Could not start a practice session. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  const finish = useCallback(async (nItems: number) => {
    setBusy(true);
    const tid = trainingIdRef.current;
    try {
      if (tid) {
        await api.finalizeTrainingSession(tid, nItems, { mode: "practice", scored: false });
        // H0 (2026-06-19): previously wrote placeholder trajectory points with
        // constant theta/sd/rt into the REAL param_trajectories table so the
        // charts had movement. That contaminated the learning dataset and was
        // then served as real (sample:false). Removed. Real trajectory points
        // will come from the ported trainer (Phase L1); until then the
        // dashboard falls back to clearly-labelled sample data.
      }
      setDoneNote(`Practice session recorded (${nItems} items reviewed). Not scored.`);
    } catch {
      setDoneNote("Practice finished, but recording it failed.");
    } finally {
      trainingIdRef.current = null;
      setPracticing(false);
      setBusy(false);
    }
  }, []);

  if (practicing) {
    return <PracticePanel deck={regimen?.deck ?? []} onFinish={finish} busy={busy} />;
  }

  return (
    <section className="cx-panel" aria-label="Daily training">
      <div className="cx-phead">
        <h2>Daily training</h2>
      </div>
      {loaded && !regimen && (
        <div className="cx-placeholder">
          No deck yet. A session gives you focused one-vs-rest reads on your weakest domains, with the true label revealed after every answer. Your first session builds the deck.
        </div>
      )}
      {regimen && (
        <table className="cx-deck">
          <thead>
            <tr>
              <th className="task">Task</th>
              <th className="cnt">New</th>
              <th className="cnt">Learning</th>
              <th className="cnt">Due</th>
            </tr>
          </thead>
          <tbody>
            {(regimen.deck ?? []).map((row) => (
              <tr key={row.code}>
                <td className="task">{row.label}</td>
                <td className={`cnt new${row.new ? "" : " zero"}`}>{row.new}</td>
                <td className={`cnt learn${row.learning ? "" : " zero"}`}>{row.learning}</td>
                <td className={`cnt due${row.due ? "" : " zero"}`}>{row.due}</td>
              </tr>
            ))}
            <tr className="cx-totals">
              <td className="task">All tasks</td>
              <td className="cnt new">{totals.neu}</td>
              <td className="cnt learn">{totals.learn}</td>
              <td className="cnt due">{totals.due}</td>
            </tr>
          </tbody>
        </table>
      )}
      <div className="cx-cta-row">
        <button type="button" className="cx-btn primary" onClick={start} disabled={busy || !regimen}>
          Start today's session <span className="arrow">→</span>
        </button>
        <span className="cx-cta-note">
          {regimen
            ? `${totalItems} ${totalItems === 1 ? "item" : "items"} due. Practice mode is not scored.`
            : "Available after your first training session."}
        </span>
      </div>
      {doneNote && <div className="cx-cta-note" style={{ marginTop: "var(--s12)" }}>{doneNote}</div>}
    </section>
  );
}

// A minimal, honest practice panel. The real adaptive trainer + live Viewer are
// not wired in (would risk the build / cert-test smoke); this steps through the
// due deck as flashcard-style prompts with immediate advance and NO scoring.
function PracticePanel({
  deck, onFinish, busy,
}: {
  deck: api.RegimenDeckEntry[]; onFinish: (nItems: number) => void; busy: boolean;
}) {
  // Build a flat queue of (task) prompts, capped to a short practice set.
  const queue = useMemo(() => {
    const items: api.RegimenDeckEntry[] = [];
    for (const r of deck) {
      const n = Math.min(3, r.new + r.learning + r.due); // a few per task
      for (let i = 0; i < n; i++) items.push(r);
    }
    return items.slice(0, 12);
  }, [deck]);

  const [i, setI] = useState(0);
  const total = queue.length;
  const cur = queue[i];
  const advance = () => {
    if (i + 1 >= total) onFinish(total);
    else setI(i + 1);
  };

  return (
    <section className="cx-panel" aria-label="Practice session">
      <div className="cx-practice-banner" role="status">
        <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="9" /><path d="M12 8v4M12 16h.01" />
        </svg>
        Practice mode, not scored. This is review only; it does not affect your certification.
      </div>
      <div className="cx-phead">
        <h2>Today's practice</h2>
        <span className="sub">{total ? `item ${Math.min(i + 1, total)} of ${total}` : "no items due"}</span>
      </div>
      {total === 0 ? (
        <>
          <div className="cx-placeholder">Nothing due to review right now.</div>
          <div className="cx-cta-row">
            <button type="button" className="cx-btn primary" onClick={() => onFinish(0)} disabled={busy}>
              Finish
            </button>
          </div>
        </>
      ) : (
        <>
          <div style={{
            padding: "var(--s32) var(--s24)", textAlign: "center",
            border: "1px solid var(--bd-subtle)", borderRadius: "var(--radius-panel)",
            background: "var(--page)",
          }}>
            <div style={{ fontSize: 13, color: "var(--ink-subtle)", textTransform: "uppercase", letterSpacing: ".06em" }}>
              Review prompt
            </div>
            <div style={{ fontSize: 22, fontWeight: 600, margin: "var(--s12) 0 var(--s8)" }}>
              {cur.label}
            </div>
            <div style={{ fontSize: 13, color: "var(--ink-subtle)" }}>
              {FULL_NAMES[cur.code] ?? cur.code}
            </div>
          </div>
          <div className="cx-cta-row">
            <button type="button" className="cx-btn primary" onClick={advance} disabled={busy}>
              {i + 1 >= total ? "Finish session" : "Next"} <span className="arrow">→</span>
            </button>
            <button type="button" className="cx-btn" onClick={() => onFinish(i)} disabled={busy}>
              Exit
            </button>
          </div>
        </>
      )}
    </section>
  );
}

// ── Per-task drill-down: a focused page for one task (larger ℓ/θ/RT charts,
// its metrics, and its verdict over past attempts from getHistory). ──────────
function DrilldownSurface({ taskK, onBack }: { taskK: number; onBack: () => void }) {
  const [dash, setDash] = useState<api.DashboardData | null>(null);
  const [trajPts, setTrajPts] = useState<api.TrajectoryPoint[]>([]);
  const [sessions, setSessions] = useState<api.HistorySession[]>([]);

  useEffect(() => {
    let live = true;
    // The drill-down is a sub-view of the dashboard, so its dash + trajectory
    // data is already in the shared bootstrap payload (the cache is
    // invalidated on dashboard exit, so this is the same data the two
    // standalone GETs returned). History is not a bootstrap section and is
    // still fetched here.
    bootstrapOnce().then((b) => {
      if (!live) return;
      if (b.dashboard) setDash(b.dashboard);
      if (b.trajectories) setTrajPts(b.trajectories.trajectories);
    }).catch(() => {});
    api.getHistory().then((h) => { if (live) setSessions(h.sessions); }).catch(() => {});
    return () => { live = false; };
  }, []);

  const tasks = useMemo(() => (dash ? buildTasks(dash) : []), [dash]);
  const trajByK = useMemo(() => buildTraj(trajPts), [trajPts]);
  const task = tasks.find((t) => t.taskK === taskK);
  const traj = trajByK.get(taskK);

  return (
    <>
      <button type="button" className="cx-back" onClick={onBack}>
        <svg className="ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6" /></svg>
        Back to dashboard
      </button>
      {!task ? (
        <section className="cx-panel"><div className="cx-placeholder">Loading task…</div></section>
      ) : (
        <>
          <section className="cx-panel" aria-label={`${task.label} detail`}>
            <div className="cx-phead" style={{ alignItems: "flex-start" }}>
              <div style={{ width: "100%" }}>
                <div className="cx-detail-head">
                  <span className="tname">{task.label}</span>
                  <Chip verdict={task.verdict} />
                  <span className="meta">{FULL_NAMES[task.code] ?? task.label}</span>
                </div>
                <div className="cx-detail-meta-row">
                  <div className="cx-metric"><span className="v">{task.ell != null ? task.ell.toFixed(2) : "–"}</span><span className="k">current ℓ</span></div>
                  <div className="cx-metric"><span className="v">{task.ellStar != null ? task.ellStar.toFixed(2) : "–"}</span><span className="k">target ℓ*</span></div>
                  <div className="cx-metric">
                    <span className="v">{task.ell != null && task.ellStar != null
                      ? (task.ell - task.ellStar >= 0 ? "+" : "−") + Math.abs(task.ell - task.ellStar).toFixed(2)
                      : "–"}</span>
                    <span className="k">gap to ℓ*</span>
                  </div>
                  <div className="cx-metric">
                    <span className="v">{task.theta != null ? (task.theta >= 0 ? "+" : "−") + Math.abs(task.theta).toFixed(2) : "–"}</span>
                    <span className="k">bias θ</span>
                  </div>
                  <div className="cx-metric"><span className="v">{task.auroc != null ? task.auroc.toFixed(2) : "–"}</span><span className="k">AUROC</span></div>
                </div>
              </div>
            </div>
            {traj && traj.ell.length >= 1 ? (
              <>
                <DetailCharts task={task} traj={traj} />
                <div className="cx-phase-axis">
                  <span style={{ width: "18%" }}>EVAL</span>
                  <span style={{ width: "64%" }}>DAILY TRAINING</span>
                  <span style={{ width: "18%" }}>RE-CERT (proj.)</span>
                </div>
              </>
            ) : (
              <div className="cx-placeholder">
                Complete a certification test to chart your skill (ℓ), decision bias (θ), and response time for this domain.
              </div>
            )}
          </section>

          <section className="cx-panel" aria-label={`${task.label} verdict history`}>
            <div className="cx-phead"><h2>Verdict over past attempts</h2></div>
            {sessions.length === 0 ? (
              <div className="cx-placeholder">No certification attempts yet for this task.</div>
            ) : (
              <div className="cx-hist">
                {sessions.map((s) => {
                  const v = verdictsOf(s.result)[taskK] ?? "PENDING";
                  const au = aurocOf(s.result, taskK);
                  return (
                    <div key={s.session_id} className="cx-hist-row">
                      <div className="cx-hist-head" style={{ cursor: "default" }}>
                        <span className="date">{fmtDate(s.finished_utc)}</span>
                        <span className="qn">{au !== null ? `AUROC ${au.toFixed(2)}` : "—"}</span>
                        <span />
                        <Chip verdict={v} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </>
      )}
    </>
  );
}

// Dashboard footer — mirrors the auth-page footer (legal links + language
// selector) but uses theme tokens so it adapts to light/dark, plus the credit.
function DashboardFooter() {
  const { t, lang, setLang } = useI18n();
  return (
    <footer className="cx-foot-bar">
        <div className="links">
          <a href="/privacy">{t("footer.privacy")}</a>
          <span className="sep" aria-hidden="true">·</span>
          <a href="/terms">{t("footer.terms")}</a>
          <span className="sep" aria-hidden="true">·</span>
          <a href="/citation">{t("footer.citation")}</a>
          <span className="sep" aria-hidden="true">·</span>
          <a href="/report">{t("footer.report")}</a>
        </div>
        <label className="cx-foot-lang" title={t("footer.languageLabel")}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10" /><line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <select value={lang} onChange={(e) => setLang(e.target.value as Lang)}
            aria-label={t("footer.languageLabel")}>
            {LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
        </label>
    </footer>
  );
}

export function Shell({
  onStartTest, onStartTraining, onSignOut, inviteHighlightId, testDisabledUntil,
  examResumable,
}: {
  onStartTest: () => void;
  onStartTraining: () => void;
  onSignOut: () => void;
  // Cohort id from an invite email's /?cohort=... deep link: the floating
  // invite banner pulses the matching invitation.
  inviteHighlightId?: string | null;
  // Post-training exam washout (server-enforced): while in force, the test
  // CTAs grey out with a "reopens at" note. Refreshed by App on every
  // dashboard entry; the server 409 is the actual enforcement.
  testDisabledUntil?: string | null;
  // A resumable exam sitting exists (CTA label: Resume vs Start).
  examResumable?: boolean;
}) {
  const [view, setView] = useState<View>("dashboard");
  const [drillTask, setDrillTask] = useState<number>(0);
  const name = api.getDisplayName() ?? "Clinician";

  // Has this participant completed the certification test? Adjusts the rail
  // CTAs — a brand-new user gets a single "Take the certification test" instead
  // of Resume training / Re-take (neither applies before the first test).
  const [hasResult, setHasResult] = useState<boolean | null>(null);
  // Training-exposure flag (server-driven: all/cohort/off). Default hidden until
  // the dashboard confirms it, so a gated build never flashes the entry.
  const [trainingEnabled, setTrainingEnabled] = useState(false);
  const [trainingResumable, setTrainingResumable] = useState(false);
  useEffect(() => {
    let live = true;
    // These three flags live in the dashboard section of the shared
    // /api/bootstrap payload the DashboardSurface already requests, so a
    // second standalone /api/dashboard round-trip was pure duplication. A
    // null section leaves the CTAs in their default state, exactly as a
    // failed standalone fetch did.
    bootstrapOnce().then((b) => {
      if (!live || !b.dashboard) return;
      setHasResult(b.dashboard.hasResult);
      setTrainingEnabled(b.dashboard.trainingEnabled ?? false);
      setTrainingResumable(b.dashboard.trainingResumable ?? false);
    }).catch(() => {});
    return () => { live = false; };
  }, []);

  // A nav surface is "active" when the current view matches it; the drilldown
  // sub-view keeps the dashboard tab highlighted (it's a routed sub-page of it).
  // Post-training washout: grey the test CTAs while it is in force, and
  // schedule a re-render for the exact expiry moment so the button flips
  // the second the window ends (no interaction needed).
  const [, setWashoutTick] = useState(0);
  useEffect(() => {
    if (!testDisabledUntil) return;
    const ms = new Date(testDisabledUntil).getTime() - Date.now();
    if (isNaN(ms) || ms <= 0) return;
    const id = window.setTimeout(() => setWashoutTick((n) => n + 1), ms + 250);
    return () => window.clearTimeout(id);
  }, [testDisabledUntil]);
  const testLocked = washoutActive(testDisabledUntil);

  const activeNav: Surface = view === "drilldown" ? "dashboard" : view;

  return (
    <div className="cx-app">
      <style>{SHELL_CSS}</style>
      <CohortInviteBanner variant="desktop" highlightCohortId={inviteHighlightId} />
      <MilestoneBanner />

      <aside className="cx-rail">
        <div className="cx-brand">
          <img
            className="cx-logo-img"
            src="/cortex_logo_top_words@2x.png"
            srcSet="/cortex_logo_top_words@2x.png 2x, /cortex_logo_top_words@3x.png 3x"
            alt="CORTEX"
          />
        </div>

        <nav className="cx-nav" aria-label="Primary">
          {NAV.map((n) => (
            <button
              key={n.id}
              type="button"
              className={activeNav === n.id ? "active" : ""}
              aria-current={activeNav === n.id ? "page" : undefined}
              onClick={() => setView(n.id)}
            >
              {ICONS[n.id]}
              <span className="lbl-text">{n.label}</span>
            </button>
          ))}
        </nav>

        <div className="cx-cta">
          {hasResult === false ? (
            <button type="button" className="cx-btn primary" onClick={onStartTest}
              disabled={testLocked}>
              Take the certification test
            </button>
          ) : (
            <>
              {trainingEnabled && (
                <button
                  type="button"
                  className="cx-btn primary"
                  onClick={() => { setView("training"); onStartTraining(); }}
                >
                  {trainingResumable ? "Resume training" : "Start training"}
                </button>
              )}
              <button type="button" className="cx-btn" onClick={onStartTest}
                disabled={testLocked}>
                {examResumable ? "Resume certification test" : "Start certification test"}
              </button>
            </>
          )}
          {testLocked && testDisabledUntil && (
            <div className="cx-cta-note">
              Testing reopens at {reopenLabel(testDisabledUntil)}
            </div>
          )}
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

        <nav className="cx-nav cx-nav-aux" aria-label="Support">
          <button type="button" onClick={() => { window.location.href = "/report"; }}>
            <svg className="ic" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 21V5M4 5h13l-3 4 3 4H4" />
            </svg>
            <span className="lbl-text">Report a problem</span>
          </button>
        </nav>
      </aside>

      <main className="cx-wrap">
        <div className="cx-content">
          {view === "dashboard" && (
            <DashboardSurface
              onDrilldown={(k) => { setDrillTask(k); setView("drilldown"); }}
              onStartTest={onStartTest}
            />
          )}
          {view === "drilldown" && (
            <DrilldownSurface taskK={drillTask} onBack={() => setView("dashboard")} />
          )}
          {view === "training" && <TrainingSurface />}
          {view === "protocol" && <ProtocolSurface />}
          {view === "history" && <HistorySurface />}
          {view === "cohorts" && (
            <Suspense fallback={<div className="cx-settings"><p className="sub">Loading…</p></div>}>
              <CohortSurface />
            </Suspense>
          )}
          {view === "settings" && <SettingsSurface />}
        </div>

        <DashboardFooter />
      </main>
    </div>
  );
}
