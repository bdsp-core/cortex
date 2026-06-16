// Dashboard shell — the persistent app chrome after sign-in (locked-v1
// "Mastery board"): a 220px left nav rail (brand · 4 nav surfaces · rail CTAs
// · who/theme/sign-out) and a main content area that renders the active
// surface from internal nav state. The certification test launches FROM the
// shell (rail CTA → onStartTest).
//
// The Dashboard surface is fully wired (KPI strip + mastery grid + ℓ/θ/RT
// detail charts + active-protocol table + consistency sidebar) to /api/dashboard,
// /api/trajectories and /api/regimen. Real cert verdicts/AUROC come from the
// latest result; ℓ/θ/RT training trajectories + deck are sample (chipped). The
// other three surfaces (Daily training / My protocol / Certification history)
// remain titled placeholders (Phase 4b).
//
// Conventions ported from the mockup: theme tokens (var(--*)), SHARP edges
// (var(--radius-*)), inline-SVG line icons (never emoji), canonical "protocol",
// no em dashes. Responsive breakpoints (<=1000px icon-only rail, <=640px top
// bar) mirror the mockup media queries; because those need @media rules that
// inline styles cannot express, the rail/layout class CSS is injected once.

import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";
import { Ring, Sparkline, MiniChart, Heatmap, HeatLegend } from "./charts";

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

/* ---------- board layout (main | sidebar) ---------- */
.cx-board{display:grid;grid-template-columns:1fr 312px;gap:var(--s24);align-items:start;}
@media (max-width:1080px){ .cx-board{grid-template-columns:1fr;} }

.cx-legend{display:flex;gap:var(--s16);font-size:11px;color:var(--ink-subtle);}
.cx-legend i{display:inline-block;width:10px;height:10px;border-radius:0;
  margin-right:5px;vertical-align:-1px;}

/* ---------- mastery grid + tiles ---------- */
.cx-mastery-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:var(--s16);}
@media (max-width:640px){ .cx-mastery-grid{grid-template-columns:1fr;} }
.cx-tile{border:1px solid var(--bd-subtle);border-radius:0;padding:var(--s16);
  background:var(--panel);cursor:pointer;transition:border-color .18s,box-shadow .18s;
  display:grid;grid-template-columns:56px 1fr;grid-template-rows:auto auto;
  column-gap:var(--s16);row-gap:var(--s12);text-align:left;font-family:inherit;
  color:var(--ink);}
.cx-tile:hover{border-color:var(--bd);}
.cx-tile.sel{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal);}
.cx-tile .ringcell{grid-row:1 / span 2;align-self:center;}
.cx-tile .head{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--s8);}
.cx-tile .task{font-weight:600;font-size:15px;line-height:1.2;}
.cx-tile .task .code{display:block;font-size:11px;color:var(--ink-subtle);
  font-weight:400;text-transform:uppercase;letter-spacing:.05em;margin-top:1px;}
.cx-tile .foot{display:flex;align-items:flex-end;justify-content:flex-end;gap:var(--s8);
  grid-column:2;}

/* ---------- verdict chip (ALWAYS carries a text label) ---------- */
.cx-chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
  letter-spacing:.03em;text-transform:uppercase;padding:2px 7px;
  border-radius:var(--radius-ctl);border:1px solid transparent;white-space:nowrap;}
.cx-chip i{width:7px;height:7px;border-radius:0;flex:none;}
.cx-chip.pass{color:var(--chip-pass-fg);background:var(--chip-pass-bg);border-color:var(--chip-pass-bd);}
.cx-chip.pass i{background:var(--pass);}
.cx-chip.fail{color:var(--chip-fail-fg);background:var(--chip-fail-bg);border-color:var(--chip-fail-bd);}
.cx-chip.fail i{background:var(--fail);}
.cx-chip.referb{color:var(--chip-referb-fg);background:var(--chip-referb-bg);border-color:var(--chip-referb-bd);}
.cx-chip.referb i{background:var(--refer-b);}
.cx-chip.referu{color:var(--chip-referu-fg);background:var(--chip-referu-bg);border-color:var(--chip-referu-bd);}
.cx-chip.referu i{background:var(--refer-u);}
.cx-chip.train{color:var(--teal-deep);background:var(--teal-weak);border-color:var(--teal-mid);}
.cx-chip.train i{background:var(--teal);}

/* ---------- detail trajectory ---------- */
.cx-detail-head{display:flex;align-items:center;gap:var(--s12);flex-wrap:wrap;}
.cx-detail-head .tname{font-size:20px;font-weight:600;}
.cx-detail-head .meta{font-size:12px;color:var(--ink-subtle);}
.cx-detail-meta-row{display:flex;gap:var(--s32);margin-top:var(--s12);flex-wrap:wrap;}
.cx-metric{display:flex;flex-direction:column;gap:var(--s2);}
.cx-metric .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:18px;font-weight:700;}
.cx-metric .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-subtle);}
.cx-phase-axis{display:flex;font-size:11px;color:var(--ink-subtle);margin-top:var(--s8);
  text-transform:uppercase;letter-spacing:.05em;}
.cx-phase-axis span{text-align:center;}
.cx-tri-chart + .cx-tri-chart{margin-top:var(--s16);}
.cx-tri-head{display:flex;align-items:baseline;justify-content:space-between;gap:var(--s12);margin-bottom:2px;}
.cx-tri-head .name{font-size:13px;font-weight:600;color:var(--ink);}
.cx-tri-head .name .k{color:var(--ink-subtle);font-weight:400;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;margin-left:6px;}
.cx-tri-head .cur{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12px;color:var(--ink-subtle);}
.cx-tri-head .cur b{color:var(--ink);}

/* ---------- protocol / deck tables ---------- */
table.cx-deck{width:100%;border-collapse:collapse;font-size:13px;margin-top:var(--s8);}
table.cx-deck th{text-align:left;font-size:11px;font-weight:600;color:var(--ink-subtle);
  text-transform:uppercase;letter-spacing:.04em;padding:var(--s8) var(--s12);
  border-bottom:1px solid var(--bd-subtle);}
table.cx-deck th.r,table.cx-deck td.r{text-align:right;}
table.cx-deck td{padding:var(--s2) var(--s12);border-bottom:1px solid var(--bd-subtle);}
table.cx-deck tr:last-child td{border-bottom:none;}
table.cx-deck tr:hover td{background:var(--panel-hover);}
table.cx-deck td.task{min-width:11em;}
.cx-deck td.cnt,.cx-deck th.cnt{font-family:var(--mono);font-variant-numeric:tabular-nums;
  text-align:right;min-width:3.5em;}
td.cnt.new{color:var(--c-new);}
td.cnt.learn{color:var(--c-learn);}
td.cnt.due{color:var(--c-due);}
td.cnt.zero{color:var(--ink-faint);}
td.ellcell{font-family:var(--mono);font-variant-numeric:tabular-nums;}
td.ellcell .of{color:var(--ink-faint);}
.cx-task-sub{color:var(--ink-faint);font-size:11px;margin-left:4px;}

/* ---------- sidebar ---------- */
.cx-side-streak{display:flex;align-items:center;gap:var(--s16);margin-bottom:var(--s16);}
.cx-side-streak .figure{display:flex;flex-direction:column;}
.cx-side-streak .figure .n{font-family:var(--mono);font-size:34px;font-weight:700;line-height:1;color:var(--ink);}
.cx-side-streak .figure .u{font-size:12px;color:var(--ink-subtle);text-transform:uppercase;letter-spacing:.05em;}
.cx-weekstrip{display:flex;gap:6px;margin-left:auto;}
.cx-weekstrip .d{display:flex;flex-direction:column;align-items:center;gap:4px;}
.cx-weekstrip .d .cell{width:22px;height:22px;border-radius:0;border:1px solid var(--bd-subtle);background:var(--panel-hover);}
.cx-weekstrip .d.done .cell{background:var(--teal);border-color:var(--teal);}
.cx-weekstrip .d.today .cell{background:var(--panel);border:2px solid var(--teal);}
.cx-weekstrip .d .lab{font-size:10px;color:var(--ink-faint);}
.cx-heat-wrap{margin-top:var(--s8);}
.cx-heat-legend{display:flex;align-items:center;gap:6px;justify-content:flex-end;
  font-size:11px;color:var(--ink-subtle);margin-top:var(--s8);}
.cx-heat-sw{width:11px;height:11px;border-radius:0;border:1px solid rgba(0,0,0,.05);}
.cx-deck-mini{margin-top:var(--s8);}
.cx-deck-mini table.cx-deck td.task{min-width:auto;}
.cx-deckhint{font-size:11px;color:var(--ink-subtle);margin-top:var(--s12);}
.cx-deckhint .k{display:inline-flex;align-items:center;gap:4px;margin-right:var(--s12);}
.cx-deckhint .k i{width:8px;height:8px;border-radius:0;}

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

// The 7 fixed tasks in engine-index order. The dashboard always renders all 7,
// even before any real data lands.
const TASK_ORDER = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"] as const;

// Full task names for the detail subtitle (mirrors the mockup fullnames map).
const FULL_NAMES: Record<string, string> = {
  spike: "Epileptiform spikes / sharp waves",
  sz: "Electrographic seizure",
  lpd: "Lateralized periodic discharges",
  gpd: "Generalized periodic discharges",
  lrda: "Lateralized rhythmic delta activity",
  grda: "Generalized rhythmic delta activity",
  iic: "Ictal-interictal continuum / other",
};

// Map a verdict string (engine: PASS/FAIL/REFER_BORDERLINE/REFER_UNINFORMATIVE
// /PENDING; sample also IN_TRAINING) to a chip class + always-present label.
// The verdict ramp comes from theme tokens; chips ALWAYS carry text.
function chipFor(verdict: string): { cls: string; text: string } {
  switch ((verdict || "").toUpperCase()) {
    case "PASS": return { cls: "pass", text: "Pass" };
    case "FAIL": return { cls: "fail", text: "Fail" };
    case "REFER_BORDERLINE": return { cls: "referb", text: "Refer · borderline" };
    case "REFER_UNINFORMATIVE": return { cls: "referu", text: "Refer · uninformative" };
    case "IN_TRAINING": return { cls: "train", text: "In training" };
    default: return { cls: "train", text: "In training" };
  }
}

// A verdict chip — inline-SVG-free dot + text label.
function Chip({ verdict }: { verdict: string }) {
  const { cls, text } = chipFor(verdict);
  return <span className={`cx-chip ${cls}`}><i />{text}</span>;
}

// Per-task view model assembled from dashboard tasks + (real) result overrides.
interface TaskVM {
  code: string;
  label: string;
  taskK: number;
  ell: number;
  ellStar: number;
  auroc: number;
  verdict: string;
}

// Per-task trajectory series, grouped from the flat trajectory point list.
interface TrajVM {
  ell: number[];
  theta: number[];
  sd: number[];
  rt: number[];   // ms
}

function buildTasks(d: api.DashboardData): TaskVM[] {
  const byCode = new Map(d.tasks.map((t) => [t.code, t]));
  return TASK_ORDER.map((code, k) => {
    const t = byCode.get(code);
    const base: TaskVM = {
      code,
      label: t?.label ?? code,
      taskK: t?.taskK ?? k,
      ell: t?.ell ?? 0,
      ellStar: t?.ellStar ?? 1,
      auroc: t?.auroc ?? 0,
      verdict: t?.verdict ?? "PENDING",
    };
    // REAL override: when a cert result exists, take the verdict + AUROC from it
    // (the actual certification outcome), keyed by engine task index.
    if (d.hasResult && d.result) {
      const rv = d.result.verdicts?.[base.taskK];
      if (rv) base.verdict = rv;
      const ra = d.result.roc?.[base.taskK]?.auroc;
      if (typeof ra === "number") base.auroc = ra;
    }
    return base;
  });
}

function buildTraj(points: api.TrajectoryPoint[]): Map<number, TrajVM> {
  const m = new Map<number, TrajVM>();
  for (const p of points) {
    let v = m.get(p.taskK);
    if (!v) { v = { ell: [], theta: [], sd: [], rt: [] }; m.set(p.taskK, v); }
    v.ell.push(p.ell);
    v.theta.push(p.theta);
    v.sd.push(p.sd);
    v.rt.push(p.rt);
  }
  return m;
}

// Three stacked detail mini-charts for the selected task (ℓ / θ / RT).
function DetailCharts({ task, traj }: { task: TaskVM; traj?: TrajVM }) {
  if (!traj || traj.ell.length < 2) {
    return <div className="cx-placeholder">No trajectory yet for this task.</div>;
  }
  const n = traj.ell.length;

  // (a) ℓ — skill toward ℓ*, ±sd band, dashed cut. ℓ is the ONLY param with ℓ*.
  const lows = traj.ell.map((v, i) => v - traj.sd[i]);
  const highs = traj.ell.map((v, i) => v + traj.sd[i]);
  let aMin = Math.min(...lows, task.ellStar);
  let aMax = Math.max(...highs, task.ellStar);
  const aPad = (aMax - aMin) * 0.14 || 0.1;
  aMin -= aPad; aMax += aPad;

  // (b) θ — bias, neutral 0 reference, NO cut. Symmetric domain around 0.
  const tAbs = Math.max(0.2, ...traj.theta.map(Math.abs)) * 1.25;

  // (c) RT — median reaction time (s), NO target line.
  const rtS = traj.rt.map((v) => v / 1000);
  let rMin = Math.min(...rtS);
  let rMax = Math.max(...rtS);
  const rPad = (rMax - rMin) * 0.18 || 0.3;
  rMin -= rPad; rMax += rPad;

  const thNow = traj.theta[n - 1];
  const fmtTheta = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  return (
    <div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">ℓ(t) <span className="k">skill · 1/σ</span></span>
          <span className="cur">now <b>{task.ell.toFixed(2)}</b> · ℓ* {task.ellStar.toFixed(2)}</span>
        </div>
        <MiniChart
          series={traj.ell} yMin={aMin} yMax={aMax} fmt={(v) => v.toFixed(2)}
          label={`${task.label} skill ℓ trajectory toward ℓ*`}
          band={[lows, highs]} rule={{ v: task.ellStar, label: "ℓ* = " + task.ellStar.toFixed(2) }}
        />
      </div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">θ(t) <span className="k">decision bias</span></span>
          <span className="cur">now <b>{fmtTheta(thNow)}</b> · neutral 0</span>
        </div>
        <MiniChart
          series={traj.theta} yMin={-tAbs} yMax={tAbs} fmt={fmtTheta}
          label={`${task.label} decision bias θ trajectory toward neutral`} zeroRule
        />
      </div>
      <div className="cx-tri-chart">
        <div className="cx-tri-head">
          <span className="name">RT(t) <span className="k">median reaction time</span></span>
          <span className="cur">now <b>{(traj.rt[n - 1] / 1000).toFixed(1)}s</b></span>
        </div>
        <MiniChart
          series={rtS} yMin={rMin} yMax={rMax} fmt={(v) => v.toFixed(1) + "s"}
          label={`${task.label} median reaction-time trend`}
        />
      </div>
    </div>
  );
}

const WEEKDAYS = ["M", "T", "W", "T", "F", "S", "S"];

// The Dashboard surface: KPI strip + the full mastery board (mastery grid,
// detail trajectory, active-protocol table, consistency sidebar). KPIs/ℓ/θ/RT
// + deck are sample for now (flagged where sample:true); verdict + AUROC come
// from the real cert result when one exists.
function DashboardSurface() {
  const [dash, setDash] = useState<api.DashboardData | null>(null);
  const [trajPts, setTrajPts] = useState<api.TrajectoryPoint[]>([]);
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [sample, setSample] = useState(false);
  const [err, setErr] = useState(false);
  const [selected, setSelected] = useState<string>("gpd");

  useEffect(() => {
    let live = true;
    // Each call is independent; a 401/failure on one must not blank the others.
    api.getDashboard()
      .then((d) => { if (live) { setDash(d); setSample(d.sample); } })
      .catch(() => { if (live) setErr(true); });
    api.getTrajectories()
      .then((t) => { if (live) setTrajPts(t.trajectories); })
      .catch(() => { /* leave detail charts empty */ });
    api.getRegimen()
      .then((r) => { if (live) setRegimen(r.regimen); })
      .catch(() => { /* leave protocol table empty */ });
    return () => { live = false; };
  }, []);

  const tasks = useMemo(() => (dash ? buildTasks(dash) : []), [dash]);
  const trajByK = useMemo(() => buildTraj(trajPts), [trajPts]);

  const kpis = dash?.kpis ?? null;
  const due = kpis?.dueToday;
  const dueTotal = due ? due.new + due.learning + due.review : null;

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

      {err && !dash && (
        <section className="cx-panel" aria-label="Mastery board">
          <div className="cx-phead"><h2>Mastery board</h2></div>
          <div className="cx-placeholder">Could not load your dashboard. Try refreshing.</div>
        </section>
      )}

      {dash && (
        <div className="cx-board">
          <div>
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
                    <button
                      key={t.code}
                      type="button"
                      className={`cx-tile${t.code === selected ? " sel" : ""}`}
                      aria-label={`${t.label} mastery tile`}
                      aria-pressed={t.code === selected}
                      onClick={() => setSelected(t.code)}
                    >
                      <div className="ringcell">
                        <Ring ell={t.ell} ellStar={t.ellStar} chipClass={chipFor(t.verdict).cls} />
                      </div>
                      <div className="head">
                        <div className="task">
                          {t.label}
                          <span className="code">{t.code}</span>
                        </div>
                        <Chip verdict={t.verdict} />
                      </div>
                      <div className="foot">
                        {tr && tr.ell.length >= 2 && <Sparkline series={tr.ell} />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* ── detail trajectory ── */}
            {sel && (
              <section className="cx-panel" aria-label="Skill trajectory detail">
                <div className="cx-phead" style={{ alignItems: "flex-start" }}>
                  <div style={{ width: "100%" }}>
                    <div className="cx-detail-head">
                      <span className="tname">{sel.label}</span>
                      <Chip verdict={sel.verdict} />
                      <span className="meta">
                        {FULL_NAMES[sel.code] ?? sel.label} · eval → daily training → re-cert
                      </span>
                    </div>
                    <div className="cx-detail-meta-row">
                      <div className="cx-metric"><span className="v">{sel.ell.toFixed(2)}</span><span className="k">current ℓ</span></div>
                      <div className="cx-metric"><span className="v">{sel.ellStar.toFixed(2)}</span><span className="k">target ℓ*</span></div>
                      <div className="cx-metric">
                        <span className="v">{(sel.ell - sel.ellStar >= 0 ? "+" : "−") + Math.abs(sel.ell - sel.ellStar).toFixed(2)}</span>
                        <span className="k">gap to ℓ*</span>
                      </div>
                      {(() => {
                        const tr = trajByK.get(sel.taskK);
                        const th = tr && tr.theta.length ? tr.theta[tr.theta.length - 1] : null;
                        const rt = tr && tr.rt.length ? tr.rt[tr.rt.length - 1] : null;
                        return (
                          <>
                            <div className="cx-metric">
                              <span className="v">{th === null ? "–" : (th >= 0 ? "+" : "−") + Math.abs(th).toFixed(2)}</span>
                              <span className="k">bias θ</span>
                            </div>
                            <div className="cx-metric">
                              <span className="v">{rt === null ? "–" : (rt / 1000).toFixed(1) + "s"}</span>
                              <span className="k">median RT</span>
                            </div>
                          </>
                        );
                      })()}
                      <div className="cx-metric"><span className="v">{sel.auroc ? sel.auroc.toFixed(2) : "–"}</span><span className="k">AUROC</span></div>
                      <div className="cx-metric">
                        <span className="v">{sel.ell >= sel.ellStar ? "0" : "~" + Math.max(1, Math.round((sel.ellStar - sel.ell) * 80))}</span>
                        <span className="k">items to resolve</span>
                      </div>
                    </div>
                  </div>
                </div>
                <DetailCharts task={sel} traj={trajByK.get(sel.taskK)} />
                <div className="cx-phase-axis">
                  <span style={{ width: "18%" }}>EVAL</span>
                  <span style={{ width: "64%" }}>DAILY TRAINING</span>
                  <span style={{ width: "18%" }}>RE-CERT (proj.)</span>
                </div>
              </section>
            )}

            {/* ── active protocol table ── */}
            <section className="cx-panel" aria-label="Active learning protocol">
              <div className="cx-phead">
                <h2>Active protocol</h2>
                {regimen ? null : <span className="sub">loading…</span>}
              </div>
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
                    {regimen.deck.map((row) => (
                      <tr key={row.code}>
                        <td className="task">
                          {row.label}
                          <span className="cx-task-sub">{row.code}</span>
                        </td>
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
          </div>

          {/* ── consistency sidebar ── */}
          <aside>
            <section className="cx-panel" aria-label="Streak and recent activity">
              <div className="cx-phead"><h2>Consistency</h2></div>
              <div className="cx-side-streak">
                <div className="figure">
                  <span className="n">{kpis ? kpis.streak : "–"}</span>
                  <span className="u">day streak</span>
                </div>
                <div className="cx-weekstrip" aria-label="Last 7 days">
                  {WEEKDAYS.map((lab, i) => (
                    <div key={i} className={`d ${i === WEEKDAYS.length - 1 ? "today" : "done"}`}>
                      <span className="cell" />
                      <span className="lab">{lab}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="cx-heat-wrap">
                <Heatmap />
                <HeatLegend />
              </div>
            </section>

            <section className="cx-panel" aria-label="Today's deck">
              <div className="cx-phead"><h2>Today's deck</h2><span className="sub">due now</span></div>
              {regimen && (
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
                      {regimen.deck.map((row) => (
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
