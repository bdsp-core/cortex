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

import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { ThemeToggle } from "../theme/ThemeProvider";
import { Ring, Sparkline, MiniChart, Heatmap, HeatLegend } from "./charts";
import { useI18n, LANGS, Lang } from "../i18n/LanguageProvider";

type Surface = "dashboard" | "training" | "protocol" | "history";
// "drilldown" is a routed sub-view of the shell (a focused per-task page),
// NOT a top-level nav surface — it has no rail entry. The shell tracks it in
// its own view state alongside the selected task.
type View = Surface | "drilldown";

// Scoped CSS for the shell chrome. Inline styles can't carry @media queries,
// so the rail / layout / KPI structural rules (and only those) live here,
// keyed to .cx-* classes. All colors/spacing reference the theme tokens.
const SHELL_CSS = `
.cx-app{display:grid;grid-template-columns:220px 1fr;min-height:100vh;
  background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.45;}
.cx-rail{background:var(--rail-bg);border-right:1px solid var(--bd-subtle);
  padding:var(--s24) 0 var(--s8);display:flex;flex-direction:column;
  position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;}
.cx-brand{display:flex;align-items:center;padding:0 var(--s12) var(--s24);}
.cx-logo-img{width:100%;height:auto;display:block;}
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
.cx-nav-aux{margin-top:var(--s8);padding-top:6px;border-top:1px solid var(--bd-subtle);}
.cx-nav-aux button{padding-top:4px;padding-bottom:4px;}
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
.cx-wrap{display:flex;flex-direction:column;min-height:100vh;}
.cx-content{flex:1 0 auto;width:100%;box-sizing:border-box;padding:var(--s24) var(--s32) var(--s48);}

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
.cx-tile .foot{display:flex;align-items:flex-end;justify-content:space-between;gap:var(--s8);
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
.cx-chip.none{color:var(--ink-subtle);background:var(--panel-hover);border-color:var(--bd-subtle);}
.cx-chip.none i{background:var(--ink-faint);}
/* empty-state CTA (shown before the first cert result) */
.cx-empty-cta{display:flex;align-items:center;justify-content:space-between;gap:var(--s16);
  background:var(--teal-weak);border:1px solid var(--teal);border-radius:12px;
  padding:var(--s16) var(--s24);margin-bottom:var(--s24);}
.cx-empty-cta .txt h3{margin:0 0 4px;font-size:15px;color:var(--ink);}
.cx-empty-cta .txt p{margin:0;font-size:13px;color:var(--ink-subtle);}
.cx-empty-cta .cx-btn{flex:none;white-space:nowrap;}
/* welcome modal (first login, no result yet) */
.cx-welcome-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);
  display:flex;align-items:center;justify-content:center;padding:var(--s24);z-index:50;}
.cx-welcome-card{background:var(--panel);border:1px solid var(--bd-subtle);border-radius:14px;
  max-width:520px;width:100%;padding:28px 28px var(--s24);box-shadow:0 20px 60px rgba(0,0,0,.3);}
.cx-welcome-card h2{margin:0 0 var(--s12);font-size:22px;color:var(--ink);}
.cx-welcome-card p{margin:0 0 var(--s12);font-size:14px;line-height:1.55;color:var(--ink-subtle);}
.cx-welcome-card p strong{color:var(--ink);font-weight:600;}
.cx-welcome-logo{display:block;width:95%;height:auto;margin:0 auto var(--s24);}
.cx-welcome-sign{margin-top:var(--s16);color:var(--ink);}
.cx-welcome-actions{display:flex;gap:var(--s12);justify-content:flex-end;margin-top:var(--s24);flex-wrap:wrap;}
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

.cx-foot-bar{padding:calc(var(--s8) + 1px) var(--s32) var(--s8);border-top:1px solid var(--bd-subtle);
  display:flex;align-items:center;justify-content:space-between;gap:var(--s16);flex-wrap:wrap;}
.cx-foot-bar a{font-size:12px;font-weight:500;color:var(--ink-subtle);text-decoration:none;}
.cx-foot-bar a:hover{color:var(--ink);text-decoration:underline;}
.cx-foot-bar .sep{margin:0 var(--s8);color:var(--ink-faint);}
.cx-foot-lang{display:inline-flex;align-items:center;gap:6px;color:var(--ink-subtle);}
.cx-foot-lang select{appearance:none;-webkit-appearance:none;background:transparent;border:none;
  color:var(--ink-subtle);font-family:inherit;font-size:12px;font-weight:500;cursor:pointer;padding:0;}
.cx-foot-lang select:hover{color:var(--ink);}

/* ---------- protocol: multi-week plan header ---------- */
.cx-weekhdr{display:flex;align-items:baseline;gap:var(--s12);margin-bottom:var(--s16);}
.cx-weekhdr .big{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:26px;font-weight:700;color:var(--ink);}
.cx-weekhdr .sub{font-size:13px;color:var(--ink-subtle);}
.cx-weekbar{display:flex;gap:6px;margin:var(--s8) 0 var(--s16);flex-wrap:wrap;}
.cx-weekbar .wk{flex:1;min-width:28px;height:8px;border-radius:0;
  background:var(--panel-hover);border:1px solid var(--bd-subtle);}
.cx-weekbar .wk.done{background:var(--teal);border-color:var(--teal);}
.cx-weekbar .wk.now{background:var(--panel);border:2px solid var(--teal);}

/* ---------- history rows + expand ---------- */
.cx-hist{display:flex;flex-direction:column;gap:var(--s8);}
.cx-hist-row{border:1px solid var(--bd-subtle);border-radius:var(--radius-ctl);
  background:var(--panel);overflow:hidden;}
.cx-hist-head{display:grid;grid-template-columns:auto auto 1fr auto;align-items:center;
  gap:var(--s16);padding:var(--s12) var(--s16);width:100%;text-align:left;
  background:none;border:none;font-family:inherit;color:var(--ink);cursor:pointer;
  transition:background .18s;}
.cx-hist-head:hover{background:var(--panel-hover);}
.cx-hist-head .date{font-weight:600;font-size:14px;min-width:9.5em;}
.cx-hist-head .qn{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:13px;color:var(--ink-subtle);min-width:6em;}
.cx-hist-strip{display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;}
.cx-hist-strip i{width:14px;height:14px;border-radius:0;border:1px solid transparent;}
.cx-hist-strip i.pass{background:var(--pass);}
.cx-hist-strip i.fail{background:var(--fail);}
.cx-hist-strip i.referb{background:var(--refer-b);}
.cx-hist-strip i.referu{background:var(--refer-u);}
.cx-hist-strip i.train{background:var(--teal);}
.cx-hist-caret{font-family:var(--mono);color:var(--ink-faint);font-size:12px;}
.cx-hist-body{padding:var(--s16);border-top:1px solid var(--bd-subtle);
  background:var(--page);}
.cx-hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
  gap:var(--s8);}
.cx-hist-cell{display:flex;align-items:center;justify-content:space-between;
  gap:var(--s8);padding:var(--s8) var(--s12);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-ctl);background:var(--panel);}
.cx-hist-cell .tk{font-size:13px;font-weight:600;}
.cx-hist-cell .tk .au{display:block;font-family:var(--mono);font-size:11px;
  color:var(--ink-subtle);font-weight:400;}

/* ---------- training: practice banner + CTA ---------- */
.cx-practice-banner{display:flex;align-items:center;gap:var(--s12);
  padding:var(--s12) var(--s16);border-radius:var(--radius-ctl);
  background:var(--teal-weak);border:1px solid var(--teal-mid);
  color:var(--teal-deep);font-size:13px;font-weight:600;margin-bottom:var(--s16);}
.cx-practice-banner .ic{width:18px;height:18px;flex:none;stroke:currentColor;fill:none;
  stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;}
.cx-totals td{font-weight:700;border-top:1px solid var(--bd);}
.cx-cta-row{margin-top:var(--s16);display:flex;gap:var(--s12);align-items:center;flex-wrap:wrap;}
.cx-cta-note{font-size:12px;color:var(--ink-subtle);}

/* ---------- drilldown back link ---------- */
.cx-back{display:inline-flex;align-items:center;gap:var(--s8);
  font-family:inherit;font-size:13px;font-weight:600;color:var(--ink-subtle);
  background:none;border:none;cursor:pointer;padding:0;margin-bottom:var(--s16);}
.cx-back:hover{color:var(--ink);}
.cx-back .ic{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:1.7;
  stroke-linecap:round;stroke-linejoin:round;}
.cx-tile .viewdet{font-size:11px;font-weight:600;color:var(--teal-deep);
  background:none;border:none;font-family:inherit;cursor:pointer;padding:0;
  text-decoration:underline;text-underline-offset:2px;}
.cx-tile .viewdet:hover{color:var(--teal);}

/* Resize smoothness: confine each card's reflow to itself (layout
   containment only, so the tile selection ring + card shadows are NOT
   clipped) and localize the scaling chart SVGs' re-rasterization (paint
   containment is safe — these SVGs cast no outside shadow). Measured: cuts
   p95 resize frame time ~33ms -> ~17ms and janky frames 8 -> 1 under 6x CPU. */
.cx-kpi,.cx-panel,.cx-tile{contain:layout;}
.cx-kpi svg,.cx-panel svg,.cx-tile svg,.cx-deck svg{contain:layout paint;}

/* icon-only rail on medium widths */
@media (max-width:1000px){
  .cx-app{grid-template-columns:60px 1fr;}
  .cx-rail{padding:var(--s16) 0;}
  .cx-brand{padding:0 0 var(--s16);justify-content:center;}
  .cx-logo-img{width:auto;height:36px;}
  .cx-nav{padding:0 8px;}
  .cx-nav button{justify-content:center;padding:var(--s12) 0;border-left:none;border-right:3px solid transparent;}
  .cx-nav button.active{border-left:none;border-right-color:var(--teal);}
  .cx-nav .lbl-text{display:none;}
  .cx-foot{padding:var(--s12) 0 0;margin:0 8px;text-align:center;}
  .cx-who{display:none;}
  .cx-signout .lbl-text{display:none;}
  .cx-cta{display:none;}
  .cx-content{padding:var(--s16) var(--s16) var(--s48);}
  .cx-foot-bar{padding-left:var(--s16);padding-right:var(--s16);}
}
/* top-bar rail on narrow widths */
@media (max-width:640px){
  .cx-app{grid-template-columns:1fr;}
  .cx-rail{flex-direction:row;align-items:center;height:auto;position:static;
    padding:var(--s8) var(--s16);gap:var(--s8);
    border-right:none;border-bottom:1px solid var(--bd-subtle);}
  .cx-brand{padding:0;}
  .cx-logo-img{width:auto;height:28px;}
  .cx-nav{flex-direction:row;padding:0;gap:2px;margin-left:var(--s8);}
  .cx-nav button{padding:var(--s8);border-right:none;}
  .cx-nav button.active{border-right:none;background:var(--teal-weak);}
  .cx-nav .lbl-text{display:none;}
  .cx-nav-aux{border-top:none;padding-top:0;margin-top:0;}
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
    case "NOT_ASSESSED": return { cls: "none", text: "Not yet assessed" };
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
      // Before any real cert result, the dashboard tasks are illustrative
      // sample data — blank them so a new user sees empty ("Not yet assessed")
      // tiles, not prototype values.
      ell: d.hasResult ? (t?.ell ?? 0) : 0,
      ellStar: t?.ellStar ?? 1,
      auroc: d.hasResult ? (t?.auroc ?? 0) : 0,
      verdict: d.hasResult ? (t?.verdict ?? "PENDING") : "NOT_ASSESSED",
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
        <p>Thank you for joining us. We're excited you're here.</p>
        <p>By taking part and engaging with the assessments, you're directly helping{" "}
          <strong>train physicians and EEG specialists around the world</strong> to read
          EEG more accurately and consistently.</p>
        <p>To begin, please take the <strong>certification test</strong>. It tunes the
          adaptive learning algorithm to your current skill, so everything that follows
          is tailored to you.</p>
        <p>Have an idea to make CORTEX better? We'd love to hear it. Use the{" "}
          <strong>Report a problem</strong> tab in the left navigation any time.</p>
        <p className="cx-welcome-sign">
          Sincerely,<br />Elijah W. Keldsen and M. Brandon Westover
        </p>
        <div className="cx-welcome-actions">
          <button type="button" className="cx-btn" onClick={onClose}>Maybe later</button>
          <button type="button" className="cx-btn primary" onClick={onStart}>
            Take the certification test
          </button>
        </div>
      </div>
    </div>
  );
}

// The Dashboard surface: KPI strip + the full mastery board (mastery grid,
// detail trajectory, active-protocol table, consistency sidebar). KPIs/ℓ/θ/RT
// + deck are sample for now (flagged where sample:true); verdict + AUROC come
// from the real cert result when one exists.
function DashboardSurface({ onDrilldown, onStartTest }: { onDrilldown: (taskK: number) => void; onStartTest: () => void }) {
  const [dash, setDash] = useState<api.DashboardData | null>(null);
  const [trajPts, setTrajPts] = useState<api.TrajectoryPoint[]>([]);
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [sample, setSample] = useState(false);
  const [err, setErr] = useState(false);
  const [selected, setSelected] = useState<string>("gpd");
  const [showWelcome, setShowWelcome] = useState(false);

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
  // Sample KPIs/charts/protocol are blanked until a real cert result exists.
  const kpis = hasData ? (dash?.kpis ?? null) : null;
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
      {showWelcome && (
        <WelcomeModal
          onClose={dismissWelcome}
          onStart={() => { dismissWelcome(); onStartTest(); }}
        />
      )}
      <section className="cx-kpi-strip" aria-label="Key training metrics">
        <div className="cx-kpi">
          <div className="cx-kpi-top">
            <span className="k">Current streak</span>
            {hasData && sample && <span className="cx-samplemark">sample data</span>}
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
                        <button
                          type="button"
                          className="viewdet"
                          aria-label={`View details for ${t.label}`}
                          onClick={(e) => { e.stopPropagation(); onDrilldown(t.taskK); }}
                        >
                          View details
                        </button>
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
            {hasData && (
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
            )}
          </div>

          {/* ── consistency sidebar (heatmap + deck; empty until first result) ── */}
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
                    <div key={i} className={`d ${i === WEEKDAYS.length - 1 ? "today" : hasData ? "done" : ""}`}>
                      <span className="cell" />
                      <span className="lab">{lab}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="cx-heat-wrap">
                <Heatmap empty={!hasData} />
                <HeatLegend />
              </div>
            </section>

            <section className="cx-panel" aria-label="Today's deck">
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
              ) : (
                <div className="cx-placeholder">No items due yet.</div>
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

// Pull a per-task verdict array from a real cert result, indexed by engine
// task index. Falls back to the canonical 7-task order length.
function verdictsOf(result: api.CertResult): string[] {
  const v = result.verdicts;
  return Array.isArray(v) ? (v as string[]) : [];
}
function aurocOf(result: api.CertResult, k: number): number | null {
  const a = result.roc?.[k]?.auroc;
  return typeof a === "number" ? a : null;
}

// Format an ISO-Z timestamp as a short readable date. Falls back to the raw
// string if it isn't parseable.
function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// ── My protocol: read-only view of the active protocol (getRegimen) ──────────
function ProtocolSurface() {
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [sample, setSample] = useState(false);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let live = true;
    api.getRegimen()
      .then((r) => { if (live) { setRegimen(r.regimen); setSample(r.sample); } })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="cx-panel" aria-label="My protocol">
      <div className="cx-phead">
        <h2>My protocol</h2>
        {sample && <span className="cx-samplemark">sample data</span>}
      </div>
      {err && !regimen && (
        <div className="cx-placeholder">Could not load your protocol. Try refreshing.</div>
      )}
      {regimen && (
        <>
          <div className="cx-weekhdr">
            <span className="big">Week {regimen.weekOf} of {regimen.weeks}</span>
            <span className="sub">spaced-repetition plan toward each task's target ℓ*</span>
          </div>
          <div className="cx-weekbar" aria-label={`Week ${regimen.weekOf} of ${regimen.weeks}`}>
            {Array.from({ length: regimen.weeks }, (_v, i) => {
              const wk = i + 1;
              const cls = wk < regimen.weekOf ? "done" : wk === regimen.weekOf ? "now" : "";
              return <span key={wk} className={`wk ${cls}`} />;
            })}
          </div>
          <table className="cx-deck">
            <thead>
              <tr>
                <th className="task">Task</th>
                <th className="r">ℓ → ℓ*</th>
                <th className="cnt">New</th>
                <th className="cnt">Learning</th>
                <th className="cnt">Due</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {regimen.deck.map((row) => {
                const reached = row.ell >= row.ellStar;
                return (
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
                    <td><Chip verdict={reached ? "PASS" : "IN_TRAINING"} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

// ── Certification history: past attempts (getHistory) — REAL data ────────────
function HistorySurface() {
  const [sessions, setSessions] = useState<api.HistorySession[] | null>(null);
  const [err, setErr] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.getHistory()
      .then((h) => { if (live) setSessions(h.sessions); })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="cx-panel" aria-label="Certification history">
      <div className="cx-phead"><h2>Certification history</h2></div>
      {err && !sessions && (
        <div className="cx-placeholder">Could not load your history. Try refreshing.</div>
      )}
      {sessions && sessions.length === 0 && (
        <div className="cx-placeholder">
          No certification attempts yet. Take the certification test to see your results here.
        </div>
      )}
      {sessions && sessions.length > 0 && (
        <div className="cx-hist">
          {sessions.map((s) => {
            const verdicts = verdictsOf(s.result);
            const isOpen = open === s.session_id;
            return (
              <div key={s.session_id} className="cx-hist-row">
                <button
                  type="button"
                  className="cx-hist-head"
                  aria-expanded={isOpen}
                  onClick={() => setOpen(isOpen ? null : s.session_id)}
                >
                  <span className="date">{fmtDate(s.finished_utc)}</span>
                  <span className="qn">{s.n_questions ?? "—"} questions</span>
                  <span className="cx-hist-strip" aria-label="Per-task verdicts">
                    {verdicts.map((v, k) => (
                      <i key={k} className={chipFor(v).cls} title={`${TASK_ORDER[k] ?? k}: ${chipFor(v).text}`} />
                    ))}
                  </span>
                  <span className="cx-hist-caret">{isOpen ? "▾" : "▸"}</span>
                </button>
                {isOpen && (
                  <div className="cx-hist-body">
                    <div className="cx-hist-grid">
                      {verdicts.map((v, k) => {
                        const au = aurocOf(s.result, k);
                        return (
                          <div key={k} className="cx-hist-cell">
                            <span className="tk">
                              {FULL_NAMES[TASK_ORDER[k]] ? (TASK_ORDER[k]) : `task ${k}`}
                              {au !== null && <span className="au">AUROC {au.toFixed(2)}</span>}
                            </span>
                            <Chip verdict={v} />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ── Daily training: Anki-style deck + a non-scored PRACTICE run ──────────────
// The real adaptive trainer is not ported yet (project decision), so "Start
// today's session" records an honest training_session start+finalize and a
// few sample param_trajectory points, behind a clear "Practice — not scored"
// banner. No verdicts, no engine — it's deck review + bookkeeping.
function TrainingSurface() {
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [sample, setSample] = useState(false);
  const [practicing, setPracticing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [doneNote, setDoneNote] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.getRegimen()
      .then((r) => { if (live) { setRegimen(r.regimen); setSample(r.sample); } })
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
      setPracticing(true);
      // stash on a ref-less closure via the finish handler below
      (window as unknown as { __cortexTrainingId?: string }).__cortexTrainingId = trainingId;
    } catch {
      setDoneNote("Could not start a practice session. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  const finish = useCallback(async (nItems: number) => {
    setBusy(true);
    const tid = (window as unknown as { __cortexTrainingId?: string }).__cortexTrainingId;
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
      setPracticing(false);
      setBusy(false);
    }
  }, [regimen]);

  if (practicing) {
    return <PracticePanel deck={regimen?.deck ?? []} onFinish={finish} busy={busy} />;
  }

  return (
    <section className="cx-panel" aria-label="Daily training">
      <div className="cx-phead">
        <h2>Daily training</h2>
        {sample && <span className="cx-samplemark">sample data</span>}
      </div>
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
            {regimen.deck.map((row) => (
              <tr key={row.code}>
                <td className="task">
                  {row.label}<span className="cx-task-sub">{row.code}</span>
                </td>
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
          {totalItems} items due. Practice mode is not scored.
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
    api.getDashboard().then((d) => { if (live) setDash(d); }).catch(() => {});
    api.getTrajectories().then((t) => { if (live) setTrajPts(t.trajectories); }).catch(() => {});
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
                  <div className="cx-metric"><span className="v">{task.ell.toFixed(2)}</span><span className="k">current ℓ</span></div>
                  <div className="cx-metric"><span className="v">{task.ellStar.toFixed(2)}</span><span className="k">target ℓ*</span></div>
                  <div className="cx-metric">
                    <span className="v">{(task.ell - task.ellStar >= 0 ? "+" : "−") + Math.abs(task.ell - task.ellStar).toFixed(2)}</span>
                    <span className="k">gap to ℓ*</span>
                  </div>
                  <div className="cx-metric"><span className="v">{task.auroc ? task.auroc.toFixed(2) : "–"}</span><span className="k">AUROC</span></div>
                </div>
              </div>
            </div>
            <DetailCharts task={task} traj={traj} />
            <div className="cx-phase-axis">
              <span style={{ width: "18%" }}>EVAL</span>
              <span style={{ width: "64%" }}>DAILY TRAINING</span>
              <span style={{ width: "18%" }}>RE-CERT (proj.)</span>
            </div>
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
  onStartTest, onStartTraining, onSignOut,
}: {
  onStartTest: () => void;
  onStartTraining: () => void;
  onSignOut: () => void;
}) {
  const [view, setView] = useState<View>("dashboard");
  const [drillTask, setDrillTask] = useState<number>(0);
  const name = api.getDisplayName() ?? "Clinician";

  // Has this participant completed the certification test? Adjusts the rail
  // CTAs — a brand-new user gets a single "Take the certification test" instead
  // of Resume training / Re-take (neither applies before the first test).
  const [hasResult, setHasResult] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    api.getDashboard().then((d) => { if (live) setHasResult(d.hasResult); }).catch(() => {});
    return () => { live = false; };
  }, []);

  // A nav surface is "active" when the current view matches it; the drilldown
  // sub-view keeps the dashboard tab highlighted (it's a routed sub-page of it).
  const activeNav: Surface = view === "drilldown" ? "dashboard" : view;

  return (
    <div className="cx-app">
      <style>{SHELL_CSS}</style>

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
            <button type="button" className="cx-btn primary" onClick={onStartTest}>
              Take the certification test
            </button>
          ) : (
            <>
              <button
                type="button"
                className="cx-btn primary"
                onClick={() => { setView("training"); onStartTraining(); }}
              >
                Resume training <span className="arrow">→</span>
              </button>
              <button type="button" className="cx-btn" onClick={onStartTest}>
                Re-take certification test
              </button>
            </>
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
        </div>

        <DashboardFooter />
      </main>
    </div>
  );
}
