// Dashboard shell styles, extracted verbatim from Shell.tsx (was a ~370-line
// inline CSS-in-JS string). Injected once via <style> at the Shell root; the
// .cx-* classes are global for the whole authenticated app while Shell is
// mounted. No logic — pure presentation.
export const SHELL_CSS = `
.cx-app{display:grid;grid-template-columns:220px 1fr;min-height:100vh;
  background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.45;}
.cx-rail{background:var(--rail-bg);border-right:1px solid var(--bd-subtle);
  padding:var(--s24) 0 var(--s8);display:flex;flex-direction:column;}
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
.cx-btn:disabled{opacity:.45;cursor:not-allowed;}
.cx-btn:disabled:hover{background:var(--panel);border-color:var(--bd);}
.cx-btn.primary:disabled:hover{background:var(--teal);border-color:var(--teal);}
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
/* min-width:0 lets the content column shrink below a wide child's min-content
   (e.g. the per-question table), so the table scrolls inside its own container
   instead of pushing the page wider. */
.cx-wrap{display:flex;flex-direction:column;min-height:100vh;min-width:0;}
.cx-content{flex:1 0 auto;width:100%;box-sizing:border-box;padding:var(--s24) var(--s32) var(--s48);}

/* One stat strip, not three sibling cards: a single bordered container with
   hairline dividers between the cells. */
.cx-kpi-strip{display:grid;grid-template-columns:repeat(3,1fr);
  background:var(--panel);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-panel);margin-bottom:var(--s24);}
.cx-kpi{padding:var(--s16) var(--s24);
  display:flex;flex-direction:column;gap:var(--s4);}
.cx-kpi+.cx-kpi{border-left:1px solid var(--bd-subtle);}
.cx-kpi .v{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:30px;font-weight:700;line-height:1;color:var(--ink);}
.cx-kpi .v .u{font-size:15px;font-weight:600;color:var(--ink-subtle);margin-left:4px;}
.cx-kpi .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-subtle);}
.cx-kpi .note{font-size:11px;color:var(--ink-faint);font-family:var(--mono);}
.cx-kpi-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--s8);}
.cx-kpi-bar{height:8px;background:var(--panel-hover);border:1px solid var(--bd-subtle);
  margin:var(--s8) 0 6px;position:relative;overflow:hidden;}
.cx-kpi-bar i{position:absolute;top:0;left:0;bottom:0;background:var(--teal);}
@media (max-width:720px){ .cx-kpi-strip{grid-template-columns:1fr;}
  .cx-kpi+.cx-kpi{border-left:none;border-top:1px solid var(--bd-subtle);} }

.cx-panel{background:var(--panel);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-panel);padding:var(--s24);}
/* Sidebar variant: heading + content directly on the page, no card chrome,
   so the main Mastery panel is the visually dominant surface. */
.cx-panel.plain{background:transparent;border:none;padding:0;}
.cx-panel + .cx-panel{margin-top:var(--s24);}
.cx-phead{display:flex;align-items:baseline;justify-content:space-between;
  margin-bottom:var(--s16);gap:var(--s12);}
.cx-phead h2{font-family:var(--serif);font-size:21px;font-weight:600;margin:0;letter-spacing:0;}
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
  column-gap:var(--s16);row-gap:var(--s4);text-align:left;font-family:inherit;
  color:var(--ink);}
.cx-tile:hover{border-color:var(--bd);}
.cx-tile.sel{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal);}
.cx-tile .ringcell{grid-row:1 / span 2;align-self:center;}
.cx-tile .head{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--s8);}
.cx-tile .task{font-weight:600;font-size:20px;line-height:1.2;}
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
.cx-welcome-card h2{font-family:var(--serif);margin:0 0 var(--s12);font-size:22px;color:var(--ink);}
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
.cx-heat{position:relative;}
.cx-heat-tip{position:absolute;transform:translate(-50%,calc(-100% - 7px));
  background:var(--ink);color:var(--panel);font-size:11px;line-height:1.25;font-weight:500;
  padding:4px 8px;border-radius:5px;white-space:nowrap;pointer-events:none;z-index:6;
  box-shadow:0 2px 10px rgba(0,0,0,.22);}
.cx-heat-tip::after{content:"";position:absolute;left:50%;top:100%;transform:translateX(-50%);
  border:5px solid transparent;border-top-color:var(--ink);}
.cx-heat-legend{display:flex;align-items:center;gap:var(--s12);flex-wrap:wrap;
  font-size:11px;color:var(--ink-subtle);margin-top:var(--s8);}
.cx-heat-item{display:inline-flex;align-items:center;gap:5px;}
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

/* ---------- settings page ---------- */
.cx-settings{max-width:760px;}
/* Cohorts reuses the settings typography but fills the whole content area
   (centered by the symmetric .cx-content padding). */
.cx-cohorts{max-width:none;width:100%;margin:0 auto;}
.cx-settings>h2{font-family:var(--serif);font-size:21px;font-weight:700;margin:0 0 var(--s4);color:var(--ink);}
.cx-settings>.sub{color:var(--ink-subtle);font-size:13px;margin:0 0 var(--s24);}
.cx-settings section{margin-bottom:var(--s24);}
.cx-settings section h2{font-size:16px;font-weight:700;margin:0 0 var(--s4);color:var(--ink);}
.cx-settings section .sub{color:var(--ink-subtle);font-size:13px;margin:0 0 var(--s16);}
.cx-form-grid{display:grid;grid-template-columns:1fr 1fr;gap:var(--s12) var(--s16);}
.cx-field{display:flex;flex-direction:column;gap:5px;}
.cx-field.full{grid-column:1 / -1;}
.cx-field label{font-size:12px;font-weight:600;color:var(--ink-subtle);}
.cx-input,.cx-select{font-family:inherit;font-size:14px;color:var(--ink);
  background:var(--field-bg);border:1px solid var(--bd);border-radius:var(--radius-ctl);
  padding:8px 10px;width:100%;box-sizing:border-box;}
.cx-input:focus,.cx-select:focus{outline:none;border-color:var(--teal);}
.cx-msg-ok{color:var(--teal-deep);font-size:13px;font-weight:600;}
.cx-msg-err{color:var(--fail,#c0392b);font-size:13px;font-weight:600;}
@media (max-width:640px){ .cx-form-grid{grid-template-columns:1fr;} }

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
/* wrap so a long verdict chip (e.g. "Refer · borderline") drops to its own line
   inside the cell instead of overflowing; margin-left:auto keeps it right-aligned
   whether it sits inline or wraps below the task name. */
.cx-hist-cell{display:flex;align-items:center;flex-wrap:wrap;
  gap:4px var(--s8);padding:var(--s8) var(--s12);border:1px solid var(--bd-subtle);
  border-radius:var(--radius-ctl);background:var(--panel);}
.cx-hist-cell .tk{font-size:13px;font-weight:600;min-width:0;}
/* the longest chip ("Refer · uninformative") exceeds the cell even on its own
   line, so bound it to the cell and let its text wrap rather than overflow. */
.cx-hist-cell .cx-chip{margin-left:auto;max-width:100%;white-space:normal;text-align:right;}
.cx-hist-cell .tk .au{display:block;font-family:var(--mono);font-size:11px;
  color:var(--ink-subtle);font-weight:400;}
/* per-question breakdown table (scrolls; adapts to width without page overflow) */
.cx-q-section{margin-top:var(--s16);}
.cx-q-title{font-size:11px;font-weight:600;color:var(--ink-subtle);
  text-transform:uppercase;letter-spacing:.04em;margin-bottom:var(--s8);}
.cx-q-scroll{max-height:min(60vh,520px);overflow:auto;
  border:1px solid var(--bd-subtle);border-radius:var(--radius-ctl);background:var(--panel);
  -webkit-overflow-scrolling:touch;}
.cx-q-table{width:100%;min-width:660px;border-collapse:collapse;font-size:12.5px;}
.cx-q-table th,.cx-q-table td{padding:6px 10px;text-align:left;white-space:nowrap;
  border-bottom:1px solid var(--bd-subtle);}
.cx-q-table th{position:sticky;top:0;z-index:1;background:var(--panel);font-weight:600;
  font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--ink-subtle);}
.cx-q-table th.r,.cx-q-table td.r{text-align:right;}
.cx-q-table td.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
.cx-q-table td.ok{color:var(--pass);font-weight:600;}
.cx-q-table td.no{color:var(--fail);font-weight:600;}
.cx-q-table tbody tr:last-child td{border-bottom:none;}
.cx-q-table tbody tr:hover{background:var(--panel-hover);}

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
