// Cohorts surface: manager-run peer groups ("pods"). Members follow each
// other's per-domain skill (ℓ) / bias (t) evolution over a 6-month window,
// identified by 9-digit public User IDs — display names are present only in
// manager API responses (the server enforces this; the UI just renders what
// it gets). Aesthetics mirror the dashboard (cx-* panels, theme tokens,
// inline-SVG charts in the charts/ idiom).
//
// Chart color system (validated with the dataviz palette checker against the
// app's light/dark panel surfaces): 8 fixed categorical slots assigned by the
// server's stable member order (invite time), so a member keeps their color
// as the pod grows; members past 8 render in the muted overflow gray and
// keep full identity via legend + tooltip. Identity is never color-alone:
// the legend always shows text labels, and ≤4-member charts add direct
// end-of-line labels.

import {
  useEffect, useMemo, useRef, useState,
  type MouseEvent as RMouseEvent, type ReactNode,
} from "react";
import * as api from "../api";
import { useTheme } from "../theme/ThemeProvider";

// ── constants ────────────────────────────────────────────────────
const TASKS = [
  { label: "Spike", full: "Epileptiform spikes / sharp waves" },
  { label: "Seizure", full: "Electrographic seizure" },
  { label: "LPD", full: "Lateralized periodic discharges" },
  { label: "GPD", full: "Generalized periodic discharges" },
  { label: "LRDA", full: "Lateralized rhythmic delta activity" },
  { label: "GRDA", full: "Generalized rhythmic delta activity" },
  { label: "Other", full: "Ictal-interictal continuum / other" },
];
type Param = "skill" | "bias";

// Categorical series palettes — one per theme, validated (lightness band,
// chroma floor, adjacent-pair CVD, surface contrast) against --panel.
const SERIES_LIGHT = ["#2a78d6", "#0d9488", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"];
const SERIES_DARK = ["#3987e5", "#0fa593", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
const OVERFLOW_LIGHT = "#8a919b";
const OVERFLOW_DARK = "#5f6772";

function isDark(): boolean {
  return typeof document !== "undefined"
    && document.documentElement.getAttribute("data-theme") === "dark";
}
function seriesColor(slot: number): string {
  const pal = isDark() ? SERIES_DARK : SERIES_LIGHT;
  return slot < pal.length ? pal[slot] : (isDark() ? OVERFLOW_DARK : OVERFLOW_LIGHT);
}

// Member label. The server includes displayName only for managers; everyone
// else identifies peers by public User ID alone.
function memberLabel(m: { publicId: string; displayName?: string; isYou?: boolean }): string {
  const base = m.displayName ? `${m.displayName} · ${m.publicId}` : m.publicId;
  return m.isYou ? `${base} (you)` : base;
}

// ── chart geometry ───────────────────────────────────────────────
// The viewBox matches the full-width panel at typical desktop widths so the
// chart renders near 1:1 (labels stay their designed size) instead of
// upscaling a narrow drawing.
const W = 1120, H = 330, mL = 52, mR = 20, mT = 16, mB = 28;
const IW = W - mL - mR, IH = H - mT - mB;

interface FlatPoint {
  memberIdx: number;
  t: number;          // epoch ms
  v: number;          // plotted value
  sd: number | null;
  phase: string;
  x: number;          // viewBox coords
  y: number;
}

function monthTicks(t0: number, t1: number): Array<{ t: number; label: string }> {
  const out: Array<{ t: number; label: string }> = [];
  const d = new Date(t0);
  d.setUTCDate(1); d.setUTCHours(0, 0, 0, 0);
  d.setUTCMonth(d.getUTCMonth() + 1);
  while (d.getTime() <= t1) {
    out.push({ t: d.getTime(), label: d.toLocaleDateString(undefined, { month: "short" }) });
    d.setUTCMonth(d.getUTCMonth() + 1);
  }
  return out;
}

interface ChartProps {
  members: api.CohortMemberSeries[];
  taskK: number;
  param: Param;
  from: string;
  to: string;
  emphasized: string | null;               // publicId from legend hover
  onEmphasize: (pid: string | null) => void;
}

function CohortChart({ members, taskK, param, from, to, emphasized, onEmphasize }: ChartProps) {
  useTheme();   // re-render (re-read palette + tokens) when the theme flips
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<FlatPoint | null>(null);

  const t0 = Date.parse(from), t1 = Date.parse(to);
  const X = (t: number) => mL + ((t - t0) / Math.max(1, t1 - t0)) * IW;

  // Per-member point lists for the selected (task, param).
  const lines = useMemo(() => {
    return members.map((m, idx) => {
      const task = m.series.find((s) => s.taskK === taskK);
      const pts = (task?.points ?? [])
        .map((p) => ({ t: Date.parse(p.ts), v: param === "skill" ? p.skill : p.bias, sd: p.sd, phase: p.phase }))
        .filter((p): p is { t: number; v: number; sd: number | null; phase: string } =>
          Number.isFinite(p.t) && typeof p.v === "number" && Number.isFinite(p.v))
        .sort((a, b) => a.t - b.t);
      return { m, idx, pts };
    });
  }, [members, taskK, param]);

  const allVals = lines.flatMap((l) => l.pts.map((p) => p.v));
  const hasData = allVals.length > 0;
  // y-range: data extent padded 12%; bias always includes the neutral 0 line.
  let yMin = hasData ? Math.min(...allVals) : (param === "skill" ? 0 : -1);
  let yMax = hasData ? Math.max(...allVals) : (param === "skill" ? 2 : 1);
  if (param === "bias") { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
  const span = (yMax - yMin) || 1;
  yMin -= span * 0.12; yMax += span * 0.12;
  const Y = (v: number) => mT + (1 - (v - yMin) / (yMax - yMin)) * IH;

  const flat: FlatPoint[] = lines.flatMap((l) =>
    l.pts.map((p) => ({
      memberIdx: l.idx, t: p.t, v: p.v, sd: p.sd, phase: p.phase,
      x: X(p.t), y: Y(p.v),
    })));

  // grid + axis labels (theme ink via CSS vars directly in SVG attrs)
  const yTicks = [0, 1, 2, 3].map((k) => yMin + (k / 3) * (yMax - yMin));
  const xTicks = Number.isFinite(t0) && Number.isFinite(t1) ? monthTicks(t0, t1) : [];

  // direct end-of-line labels when few members have data (collision-eased)
  const withData = lines.filter((l) => l.pts.length > 0);
  let endLabels: Array<{ x: number; y: number; text: string; color: string }> = [];
  if (withData.length > 0 && withData.length <= 4) {
    endLabels = withData.map((l) => {
      const last = l.pts[l.pts.length - 1];
      return { x: X(last.t) + 7, y: Y(last.v) + 3, text: l.m.publicId, color: seriesColor(l.idx) };
    }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < endLabels.length; i++) {
      if (endLabels[i].y - endLabels[i - 1].y < 12) endLabels[i].y = endLabels[i - 1].y + 12;
    }
  }

  function onMove(e: RMouseEvent<HTMLDivElement>) {
    const box = boxRef.current?.getBoundingClientRect();
    if (!box || flat.length === 0) { setHover(null); return; }
    const sx = W / box.width;
    const px = (e.clientX - box.left) * sx;
    const py = (e.clientY - box.top) * (H / box.height);
    let best: FlatPoint | null = null;
    let bestD = 24 * 24;
    for (const p of flat) {
      const d = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py);
      if (d < bestD) { bestD = d; best = p; }
    }
    setHover(best);
    onEmphasize(best ? members[best.memberIdx].publicId : null);
  }

  const emphIdx = emphasized == null ? null : members.findIndex((m) => m.publicId === emphasized);
  const dimmed = (idx: number) => emphIdx != null && emphIdx >= 0 && emphIdx !== idx;

  const paramLabel = param === "skill" ? "Skill (ℓ)" : "Bias (t)";
  const hoverMember = hover ? members[hover.memberIdx] : null;

  return (
    <div
      ref={boxRef}
      style={{ position: "relative" }}
      onMouseMove={onMove}
      onMouseLeave={() => { setHover(null); onEmphasize(null); }}
    >
      <svg
        width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
        role="img" aria-label={`${paramLabel} over the last 6 months, ${TASKS[taskK]?.label ?? taskK}`}
      >
        {yTicks.map((v, k) => (
          <g key={`y${k}`}>
            <line x1={mL} y1={Y(v).toFixed(1)} x2={W - mR} y2={Y(v).toFixed(1)} stroke="var(--grid-ink)" />
            <text x={mL - 8} y={(Y(v) + 3).toFixed(1)} textAnchor="end"
              fontFamily="ui-monospace,Menlo,monospace" fontSize={10} fill="var(--ink-faint)">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((tk, k) => (
          <g key={`x${k}`}>
            <line x1={X(tk.t).toFixed(1)} y1={mT} x2={X(tk.t).toFixed(1)} y2={mT + IH} stroke="var(--grid-ink)" />
            <text x={X(tk.t).toFixed(1)} y={H - 8} textAnchor="middle"
              fontFamily="ui-monospace,Menlo,monospace" fontSize={10} fill="var(--ink-faint)">
              {tk.label}
            </text>
          </g>
        ))}
        {param === "bias" && (
          <line x1={mL} y1={Y(0).toFixed(1)} x2={W - mR} y2={Y(0).toFixed(1)}
            stroke="var(--neutral-rule)" strokeWidth={1} />
        )}
        {lines.map((l) => {
          if (l.pts.length === 0) return null;
          const color = seriesColor(l.idx);
          const d = l.pts.map((p, i) => (i ? "L" : "M") + X(p.t).toFixed(1) + " " + Y(p.v).toFixed(1)).join(" ");
          const you = l.m.isYou;
          return (
            <g key={l.m.publicId} opacity={dimmed(l.idx) ? 0.22 : 1}>
              {l.pts.length > 1 && (
                <path d={d} fill="none" stroke={color} strokeWidth={you ? 2.6 : 1.8}
                  strokeLinejoin="round" strokeLinecap="round" />
              )}
              {l.pts.map((p, i) => (
                <circle key={i} cx={X(p.t).toFixed(1)} cy={Y(p.v).toFixed(1)}
                  r={p.phase === "train" ? 2.2 : 3.2}
                  fill={p.phase === "train" ? "var(--panel)" : color}
                  stroke={color} strokeWidth={1.4} />
              ))}
            </g>
          );
        })}
        {endLabels.map((el, i) => (
          <text key={`el${i}`} x={Math.min(el.x, W - 4).toFixed(1)} y={el.y.toFixed(1)}
            textAnchor="start" fontFamily="ui-monospace,Menlo,monospace" fontSize={10}
            fill="var(--ink-subtle)">
            {el.text}
          </text>
        ))}
        {!hasData && (
          <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={13} fill="var(--ink-subtle)">
            No evaluations in the last 6 months yet.
          </text>
        )}
      </svg>
      {hover && hoverMember && (
        <div style={{
          position: "absolute",
          left: `${(Math.min(hover.x, W - 190) / W) * 100}%`,
          top: `${(Math.max(hover.y - 8, 4) / H) * 100}%`,
          transform: "translateY(-100%)",
          background: "var(--panel)", color: "var(--ink)",
          border: "1px solid var(--bd-subtle)", borderRadius: 6,
          padding: "6px 9px", fontSize: 12, pointerEvents: "none",
          boxShadow: "0 4px 14px rgba(0,0,0,0.18)", whiteSpace: "nowrap", zIndex: 5,
        }}>
          <div style={{ fontWeight: 600 }}>{memberLabel(hoverMember)}</div>
          <div className="sub" style={{ fontSize: 11 }}>
            {new Date(hover.t).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })}
            {" · "}{hover.phase === "train" ? "Training" : "Certification test"}
          </div>
          <div style={{ fontFamily: "ui-monospace,Menlo,monospace" }}>
            {paramLabel} {hover.v.toFixed(2)}{hover.sd != null ? ` ± ${hover.sd.toFixed(2)}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}

// ── small controls (dashboard-flat, teal-accented like the nav) ─────
function ToggleBtn({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: ReactNode;
}) {
  return (
    <button
      type="button"
      className="cx-btn"
      aria-pressed={active}
      onClick={onClick}
      style={active ? {
        background: "var(--teal-weak)", color: "var(--teal-deep)",
        borderColor: "var(--teal-mid)",
      } : undefined}
    >
      {children}
    </button>
  );
}

// ── the main cohort view (presentational; also renders the demo) ───
interface ViewProps {
  detail: api.CohortDetail;
  perf: api.CohortPerformance;
  onInvite?: (pid: string) => Promise<void>;
  onRemove?: (pid: string) => Promise<void>;
  onLeave?: () => Promise<void>;
  onDelete?: () => Promise<void>;
}

function CohortView({ detail, perf, onInvite, onRemove, onLeave, onDelete }: ViewProps) {
  const [param, setParam] = useState<Param>("skill");
  const [taskK, setTaskK] = useState(0);
  const [emphasized, setEmphasized] = useState<string | null>(null);
  const [invitePid, setInvitePid] = useState("");
  const [inviteMsg, setInviteMsg] = useState<{ ok?: string; err?: string }>({});
  const [confirmDelete, setConfirmDelete] = useState(false);
  const isManager = detail.role === "manager";
  const members = perf.members;

  async function doInvite() {
    setInviteMsg({});
    const pid = invitePid.replace(/\s+/g, "");
    if (!/^[1-9]\d{8}$/.test(pid)) { setInviteMsg({ err: "A User ID is 9 digits." }); return; }
    try {
      await onInvite?.(pid);
      setInvitePid(""); setInviteMsg({ ok: "Invitation sent." });
    } catch (e) {
      const s = (e as api.ApiError)?.status;
      setInviteMsg({
        err: s === 404 ? "No account with that User ID." :
          s === 409 ? "Already in this cohort or has a pending invitation." :
          s === 429 ? "Too many invitations right now. Try again later." :
          "Could not send the invitation.",
      });
    }
  }

  return (
    <>
      <section className="cx-panel" aria-label="Cohort performance">
        <div className="cx-phead" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12 }}>
          <div style={{ marginRight: "auto" }}>
            <h2>{detail.name}</h2>
            <p className="sub" style={{ margin: 0 }}>
              {TASKS[taskK].full} · {param === "skill" ? "skill (ℓ) estimate" : "bias (t) estimate"} per
              evaluation, last 6 months
            </p>
          </div>
          <div style={{ display: "flex", gap: 6 }} role="group" aria-label="Parameter">
            <ToggleBtn active={param === "skill"} onClick={() => setParam("skill")}>Skill</ToggleBtn>
            <ToggleBtn active={param === "bias"} onClick={() => setParam("bias")}>Bias</ToggleBtn>
          </div>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "10px 0 14px" }} role="group" aria-label="Domain">
          {TASKS.map((t, k) => (
            <ToggleBtn key={t.label} active={taskK === k} onClick={() => setTaskK(k)}>{t.label}</ToggleBtn>
          ))}
        </div>
        <CohortChart
          members={members} taskK={taskK} param={param}
          from={perf.from} to={perf.to}
          emphasized={emphasized} onEmphasize={setEmphasized}
        />
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginTop: 10 }} aria-label="Members legend">
          {members.map((m, i) => (
            <span
              key={m.publicId}
              onMouseEnter={() => setEmphasized(m.publicId)}
              onMouseLeave={() => setEmphasized(null)}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12,
                color: "var(--ink-subtle)", cursor: "default",
                opacity: emphasized && emphasized !== m.publicId ? 0.4 : 1,
              }}
            >
              <i style={{
                width: 10, height: 10, borderRadius: 5, background: seriesColor(i),
                display: "inline-block", flex: "none",
              }} />
              <span style={{ fontFamily: "ui-monospace,Menlo,monospace" }}>{memberLabel(m)}</span>
            </span>
          ))}
        </div>
      </section>

      <section className="cx-panel" aria-label="Cohort members">
        <div className="cx-phead"><h2>Members</h2></div>
        <p className="sub">
          {isManager
            ? "You manage this cohort. Add teammates by their 9-digit User ID (they will be asked to accept)."
            : "Members are shown by User ID. Your line is marked (you)."}
        </p>
        <div>
          {(detail.members ?? []).map((m) => (
            <div key={m.publicId} style={{
              display: "flex", alignItems: "center", gap: 10, padding: "8px 0",
              borderTop: "1px solid var(--bd-subtle)",
            }}>
              <span style={{ fontFamily: "ui-monospace,Menlo,monospace", letterSpacing: "0.06em" }}>
                {m.publicId}
              </span>
              {m.displayName != null && <span>{m.displayName}</span>}
              {m.isYou && <span className="sub">(you)</span>}
              {m.isManager && <span className="cx-chip train"><i />Manager</span>}
              {m.status === "invited" && <span className="cx-chip none"><i />Invited</span>}
              <span style={{ marginLeft: "auto" }} />
              {isManager && !m.isManager && (
                <button type="button" className="cx-btn" onClick={() => onRemove?.(m.publicId)}>
                  {m.status === "invited" ? "Revoke invite" : "Remove"}
                </button>
              )}
              {!isManager && m.isYou && (
                <button type="button" className="cx-btn" onClick={() => onLeave?.()}>Leave cohort</button>
              )}
            </div>
          ))}
        </div>

        {isManager && (
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10, marginTop: 14 }}>
            <input
              className="cx-input" style={{ maxWidth: 220 }}
              placeholder="9-digit User ID" inputMode="numeric"
              value={invitePid} onChange={(e) => setInvitePid(e.target.value)}
            />
            <button type="button" className="cx-btn primary" onClick={doInvite}>Add member</button>
            {inviteMsg.ok && <span className="cx-msg-ok">{inviteMsg.ok}</span>}
            {inviteMsg.err && <span className="cx-msg-err">{inviteMsg.err}</span>}
          </div>
        )}

        {isManager && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 18, paddingTop: 12, borderTop: "1px solid var(--bd-subtle)" }}>
            <span className="sub">Deleting the cohort removes every member and cannot be undone.</span>
            <span style={{ marginLeft: "auto" }} />
            {!confirmDelete ? (
              <button type="button" className="cx-btn" onClick={() => setConfirmDelete(true)}>Delete cohort</button>
            ) : (
              <>
                <button type="button" className="cx-btn" onClick={() => setConfirmDelete(false)}>Keep it</button>
                <button type="button" className="cx-btn primary" onClick={() => onDelete?.()}>Confirm delete</button>
              </>
            )}
          </div>
        )}
      </section>
    </>
  );
}

// ── live loader around CohortView ───────────────────────────────────
function LiveCohort({ cohortId, onChanged }: { cohortId: string; onChanged: () => void }) {
  const [detail, setDetail] = useState<api.CohortDetail | null>(null);
  const [perf, setPerf] = useState<api.CohortPerformance | null>(null);
  const [err, setErr] = useState("");
  const [bump, setBump] = useState(0);

  useEffect(() => {
    let dead = false;
    setErr("");
    Promise.all([api.getCohort(cohortId), api.getCohortPerformance(cohortId)])
      .then(([d, p]) => { if (!dead) { setDetail(d); setPerf(p); } })
      .catch(() => { if (!dead) setErr("Could not load this cohort. Please try again."); });
    return () => { dead = true; };
  }, [cohortId, bump]);

  if (err) return <section className="cx-panel"><p className="cx-msg-err">{err}</p></section>;
  if (!detail || !perf) return <section className="cx-panel"><p className="sub">Loading…</p></section>;

  const refreshLocal = () => setBump((b) => b + 1);
  return (
    <CohortView
      detail={detail} perf={perf}
      onInvite={async (pid) => { await api.inviteToCohort(cohortId, pid); refreshLocal(); }}
      onRemove={async (pid) => { await api.removeCohortMember(cohortId, pid); refreshLocal(); }}
      onLeave={async () => { await api.leaveCohort(cohortId); onChanged(); }}
      onDelete={async () => { await api.deleteCohort(cohortId); onChanged(); }}
    />
  );
}

// ── demo data for the not-in-a-cohort state (clearly-example values) ─
function demoPerf(): api.CohortPerformance {
  const now = Date.now();
  const day = 86400_000;
  const mk = (pid: string, startDays: number, vals: Array<[number, number]>): api.CohortMemberSeries => ({
    publicId: pid, isManager: pid === "204481375", isYou: pid === "581236490",
    joinedUtc: null,
    series: [{
      taskK: 0,
      points: vals.map(([off, v], i) => ({
        ts: new Date(now - (startDays - off) * day).toISOString(),
        skill: v, bias: Math.round((0.5 - v / 3) * 100) / 100, sd: 0.3 - i * 0.04,
        phase: "eval",
      })),
    }],
  });
  return {
    cohortId: "demo", name: "Example cohort", // never rendered as a real pod
    from: new Date(now - 183 * day).toISOString(),
    to: new Date(now).toISOString(),
    members: [
      mk("204481375", 160, [[0, 0.52], [45, 0.78], [95, 1.05], [140, 1.22]]),
      mk("581236490", 150, [[0, 0.95], [60, 1.02], [120, 1.31]]),
      mk("937154028", 120, [[0, 0.38], [55, 0.61], [105, 0.72]]),
    ],
  };
}
const DEMO_DETAIL: api.CohortDetail = {
  cohortId: "demo", name: "Example cohort", role: "member", status: "active",
  createdUtc: "",
  members: [
    { publicId: "204481375", status: "active", isManager: true, isYou: false, joinedUtc: null },
    { publicId: "581236490", status: "active", isManager: false, isYou: true, joinedUtc: null },
    { publicId: "937154028", status: "active", isManager: false, isYou: false, joinedUtc: null },
  ],
};

// ── surfaces ────────────────────────────────────────────────────────
function CreateCohortForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [name, setName] = useState("");
  const [msg, setMsg] = useState<{ err?: string }>({});
  const [busy, setBusy] = useState(false);
  async function create() {
    const n = name.trim();
    if (!n) { setMsg({ err: "Give the cohort a name." }); return; }
    setBusy(true); setMsg({});
    try {
      const r = await api.createCohort(n);
      onCreated(r.cohortId);
    } catch (e) {
      const s = (e as api.ApiError)?.status;
      setMsg({ err: s === 409 ? "You already manage the maximum number of cohorts." : "Could not create the cohort." });
    } finally { setBusy(false); }
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10 }}>
      <input
        className="cx-input" style={{ maxWidth: 240 }} placeholder="Cohort name"
        value={name} onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") create(); }}
      />
      <button type="button" className="cx-btn primary" onClick={create} disabled={busy}>
        {busy ? "Creating…" : "Create a cohort"}
      </button>
      {msg.err && <span className="cx-msg-err">{msg.err}</span>}
    </div>
  );
}

export function CohortSurface() {
  const [loaded, setLoaded] = useState(false);
  const [cohorts, setCohorts] = useState<api.CohortSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState("");
  const [bump, setBump] = useState(0);
  const refresh = () => setBump((b) => b + 1);

  useEffect(() => {
    let dead = false;
    api.listCohorts()
      .then((r) => {
        if (dead) return;
        setCohorts(r.cohorts);
        setSelected((prev) => {
          if (prev && r.cohorts.some((c) => c.cohortId === prev)) return prev;
          const firstActive = r.cohorts.find((c) => c.status === "active");
          return (firstActive ?? r.cohorts[0])?.cohortId ?? null;
        });
        setLoadErr("");
      })
      .catch(() => { if (!dead) setLoadErr("Could not load your cohorts. Please try again."); })
      .finally(() => { if (!dead) setLoaded(true); });
    return () => { dead = true; };
  }, [bump]);

  if (!loaded) return <div className="cx-settings cx-cohorts"><p className="sub">Loading…</p></div>;
  if (loadErr) {
    return (
      <div className="cx-settings cx-cohorts">
        <h2>Cohorts</h2>
        <section className="cx-panel">
          <p className="cx-msg-err">{loadErr}</p>
          <button type="button" className="cx-btn" onClick={refresh}>Retry</button>
        </section>
      </div>
    );
  }

  // Not in any cohort: the real page, greyed out under an explainer card.
  if (cohorts.length === 0) {
    return (
      <div className="cx-settings cx-cohorts" style={{ position: "relative" }}>
        <div aria-hidden style={{ filter: "grayscale(1) opacity(0.45)", pointerEvents: "none", userSelect: "none" }}>
          <h2>Cohorts</h2>
          <p className="sub">Follow your team's skill and bias evolution.</p>
          <CohortView detail={DEMO_DETAIL} perf={demoPerf()} />
        </div>
        <div style={{
          position: "absolute", inset: 0, display: "flex",
          alignItems: "flex-start", justifyContent: "center", paddingTop: 90,
        }}>
          <section className="cx-panel" style={{ maxWidth: 480, boxShadow: "0 12px 40px rgba(0,0,0,0.22)" }}>
            <h2>Join a cohort to see live results</h2>
            <p className="sub">
              Cohorts let a team follow each other's skill and bias evolution across the 7
              domains, identified only by User ID. Ask your cohort manager to add your
              9-digit User ID (it is on the Settings page) and accept the invitation here.
            </p>
            <p className="sub">Or start a cohort for your team — you become its manager:</p>
            <CreateCohortForm onCreated={() => refresh()} />
          </section>
        </div>
      </div>
    );
  }

  const sel = cohorts.find((c) => c.cohortId === selected) ?? cohorts[0];
  return (
    <div className="cx-settings cx-cohorts">
      <h2>Cohorts</h2>
      <p className="sub">Follow your team's skill and bias evolution. Peers are identified by User ID.</p>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, margin: "10px 0 16px" }}>
        {cohorts.map((c) => (
          <ToggleBtn key={c.cohortId} active={c.cohortId === sel.cohortId} onClick={() => setSelected(c.cohortId)}>
            {c.name}{c.role === "manager" ? " · manager" : ""}{c.status === "invited" ? " · invitation" : ""}
          </ToggleBtn>
        ))}
        <span style={{ marginLeft: "auto" }} />
        <NewCohortButton onCreated={(id) => { setSelected(id); refresh(); }} />
      </div>

      {sel.status === "invited" ? (
        <InvitationCard cohort={sel} onChanged={refresh} />
      ) : (
        <LiveCohort cohortId={sel.cohortId} onChanged={refresh} />
      )}
    </div>
  );
}

function NewCohortButton({ onCreated }: { onCreated: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  if (!open) {
    return <button type="button" className="cx-btn" onClick={() => setOpen(true)}>New cohort</button>;
  }
  return <CreateCohortForm onCreated={(id) => { setOpen(false); onCreated(id); }} />;
}

function InvitationCard({ cohort, onChanged }: { cohort: api.CohortSummary; onChanged: () => void }) {
  const [err, setErr] = useState("");
  async function act(fn: (id: string) => Promise<{ ok: boolean }>) {
    setErr("");
    try { await fn(cohort.cohortId); onChanged(); }
    catch { setErr("Something went wrong. Please try again."); }
  }
  return (
    <section className="cx-panel" aria-label="Cohort invitation">
      <h2>Invitation: {cohort.name}</h2>
      <p className="sub">
        You have been invited to join this cohort. Members of a cohort can see each
        other's skill and bias evolution, identified by 9-digit User ID (the cohort
        manager can also see names). You can leave at any time.
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button type="button" className="cx-btn primary" onClick={() => act(api.acceptCohortInvite)}>
          Accept invitation
        </button>
        <button type="button" className="cx-btn" onClick={() => act(api.declineCohortInvite)}>
          Decline
        </button>
        {err && <span className="cx-msg-err">{err}</span>}
      </div>
    </section>
  );
}
