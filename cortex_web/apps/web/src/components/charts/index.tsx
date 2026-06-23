// Theme-aware inline-SVG chart primitives for the Dashboard mastery board.
// Ported geometry + semantics 1:1 from the locked-v1 mockup
// (cortex_web_design/mockups/locked-v1/index.html) render functions:
// ring(), sparkline(), miniChart(), renderHeatmap()/heatFill().
//
// All colors are read at render time from the active theme's CSS custom
// properties via cssVar('--token'); each component subscribes to useTheme() so
// the SVGs re-render (with the new inks) when the theme flips, mirroring the
// mockup's renderAll() on toggle.
//
// GROUNDING rules (do NOT violate): ℓ is the only parameter with a cut ℓ*
// (dashed target rule). θ charts against a neutral 0 line and NEVER gates a
// verdict. RT has no target. Verdict colors come from the theme ramp tokens,
// never invented.

import { useRef, useState, type MouseEvent as RMouseEvent } from "react";
import { cssVar } from "../../../ui/theme";
import { useTheme } from "../../theme/ThemeProvider";

// Verdict -> theme ramp token. Mirrors VERDICT_VAR in the mockup. Accepts both
// the chip-class shorthands the grid uses and the raw engine verdict strings,
// so callers can pass whichever they have.
const VERDICT_VAR: Record<string, string> = {
  pass: "--pass",
  fail: "--fail",
  referb: "--refer-b",
  referu: "--refer-u",
  train: "--teal",
};

export function verdictVar(chipClass: string): string {
  return VERDICT_VAR[chipClass] ?? "--teal";
}

// Diverging "margin to ℓ*" color: green shades when ℓ is at/above ℓ* (positive
// margin), red shades when below — paler near the threshold, deeper further
// away. |ℓ − ℓ*| ≥ MARGIN_SAT is fully saturated.
const MARGIN_SAT = 0.75;
function marginColor(d: number): string {
  const t = Math.min(1, Math.abs(d) / MARGIN_SAT);
  const dark = typeof document !== "undefined"
    && document.documentElement.getAttribute("data-theme") === "dark";
  const base = dark ? [42, 46, 54] : [233, 236, 238];   // pale start
  const green = [38, 152, 90], red = [206, 66, 62];
  const target = d >= 0 ? green : red;
  const a = 0.28 + 0.72 * t;   // keep a visible tint even at the threshold
  const m = (i: number) => Math.round(base[i] + (target[i] - base[i]) * a);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}

// ── Ring: signed margin to ℓ* (ℓ − ℓ*), colored red→green by that margin ─────
export function Ring({
  ell, ellStar, size = 56,
}: {
  ell: number | null; ellStar: number | null; size?: number;
}) {
  useTheme(); // re-render on theme flip
  const r = size / 2 - 5;
  const cx = size / 2;
  const cy = size / 2;
  // No measured ℓ (legacy verdicts-only result) → indeterminate track + "—".
  const d = (typeof ell === "number" && typeof ellStar === "number") ? ell - ellStar : null;
  const track = cssVar("--bd-subtle");
  const ink = cssVar("--ink");
  const inkSub = cssVar("--ink-subtle");
  const margin = d === null ? "—" : (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(2);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={track} strokeWidth={5} />
      {d !== null && (
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={marginColor(d)} strokeWidth={5} />
      )}
      <text
        x={cx} y={cy - 1} textAnchor="middle" dominantBaseline="middle"
        fontFamily="ui-monospace,Menlo,monospace" fontSize={d === null ? 13 : 11} fontWeight={700} fill={ink}
      >
        {margin}
      </text>
      <text
        x={cx} y={cy + 11} textAnchor="middle" dominantBaseline="middle"
        fontFamily="system-ui,sans-serif" fontSize={7} fill={inkSub} letterSpacing="0.5"
      >
        vs ℓ*
      </text>
    </svg>
  );
}

// ── Sparkline: ℓ trend (teal line + endpoint dot) ───────────────────────────
export function Sparkline({
  series, w = 72, h = 22,
}: {
  series: number[]; w?: number; h?: number;
}) {
  useTheme();
  if (series.length < 2) return null;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const rng = max - min || 1;
  const pad = 2;
  const pts = series.map((v, i) => {
    const x = pad + (i / (series.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((v - min) / rng) * (h - 2 * pad);
    return [x, y] as const;
  });
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const last = pts[pts.length - 1];
  const teal = cssVar("--teal");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <path
        d={d} fill="none" stroke={teal} strokeWidth={1.4}
        strokeLinejoin="round" strokeLinecap="round" opacity={0.85}
      />
      <circle cx={last[0].toFixed(1)} cy={last[1].toFixed(1)} r={1.8} fill={teal} />
    </svg>
  );
}

// ── MiniChart: generic single-hue phase-shaded line chart ───────────────────
// opts mirror the mockup miniChart():
//   band     -> ±sd shaded band ([lows, highs]); ℓ only
//   rule     -> dashed certification cut {v, label}; ℓ* only
//   zeroRule -> neutral θ=0 reference (solid faint, NOT a cut); θ only
// Phase shading: eval (first segment) -> training (middle) -> re-cert (last).
export interface MiniChartProps {
  series: number[];
  yMin: number;
  yMax: number;
  fmt: (v: number) => string;
  label: string;
  band?: [number[], number[]] | null;
  rule?: { v: number; label: string } | null;
  zeroRule?: boolean;
  // Per-point phase ("eval"|"recert"=certification anchor, "train"=training).
  // Anchors render as solid, larger, darker dots; training points lighter.
  phases?: string[];
}

export function MiniChart(o: MiniChartProps) {
  useTheme();
  const W = 720, H = 150, mL = 46, mR = 54, mT = 12, mB = 20;
  const iw = W - mL - mR, ih = H - mT - mB;
  const s = o.series;
  const n = s.length;
  const { yMin, yMax } = o;
  const teal = cssVar("--teal");
  const gridInk = cssVar("--grid-ink");
  const inkFaint = cssVar("--ink-faint");
  const inkSub = cssVar("--ink-subtle");
  const phaseEval = cssVar("--phase-eval");
  const phaseRecert = cssVar("--phase-recert");
  const phaseRule = cssVar("--phase-rule");
  const panel = cssVar("--panel");
  const neutralRule = cssVar("--neutral-rule");
  // A single eval point (n===1) sits in the eval zone at the left; multi-point
  // series spread eval→training→re-cert across the width.
  const X = (i: number) => (n <= 1 ? mL + iw * 0.08 : mL + (i / (n - 1)) * iw);
  const Y = (v: number) => mT + (1 - (v - yMin) / (yMax - yMin)) * ih;

  // y gridlines + labels
  const ticks = 3;
  const grid: JSX.Element[] = [];
  for (let k = 0; k <= ticks; k++) {
    const val = yMin + (k / ticks) * (yMax - yMin);
    const y = Y(val);
    grid.push(
      <g key={`g${k}`}>
        <line x1={mL} y1={y.toFixed(1)} x2={W - mR} y2={y.toFixed(1)} stroke={gridInk} />
        <text
          x={mL - 8} y={(y + 3).toFixed(1)} textAnchor="end"
          fontFamily="ui-monospace,Menlo,monospace" fontSize={10} fill={inkFaint}
        >
          {o.fmt(val)}
        </text>
      </g>,
    );
  }

  // phase shading: eval (0..1), training (1..n-2), recert (n-2..last). With a
  // single eval point there's no training/re-cert span yet — shade only a thin
  // eval zone on the left.
  const xEval = n >= 2 ? X(1) : mL + iw * 0.16;
  const xRecert = n >= 2 ? X(n - 2) : W - mR;
  const phases = (
    <>
      <rect x={mL} y={mT} width={(xEval - mL).toFixed(1)} height={ih} fill={phaseEval} />
      <rect x={xRecert.toFixed(1)} y={mT} width={(W - mR - xRecert).toFixed(1)} height={ih} fill={phaseRecert} />
      <line x1={xEval.toFixed(1)} y1={mT} x2={xEval.toFixed(1)} y2={mT + ih} stroke={phaseRule} strokeDasharray="2 3" />
      <line x1={xRecert.toFixed(1)} y1={mT} x2={xRecert.toFixed(1)} y2={mT + ih} stroke={phaseRule} strokeDasharray="2 3" />
    </>
  );

  // optional ±sd band — a filled ribbon for a series, or a vertical ±σ error
  // bar at the single eval point.
  let band: JSX.Element | null = null;
  if (o.band && n >= 2) {
    const top = s.map((_v, i) => `${X(i).toFixed(1)},${Y(o.band![1][i]).toFixed(1)}`);
    const bot = s.map((_v, i) => `${X(i).toFixed(1)},${Y(o.band![0][i]).toFixed(1)}`).reverse();
    band = <polygon points={top.concat(bot).join(" ")} fill={teal} fillOpacity={0.14} />;
  } else if (o.band && n === 1) {
    const x0 = X(0);
    band = (
      <line
        x1={x0.toFixed(1)} y1={Y(o.band[1][0]).toFixed(1)}
        x2={x0.toFixed(1)} y2={Y(o.band[0][0]).toFixed(1)}
        stroke={teal} strokeWidth={6} strokeOpacity={0.18} strokeLinecap="round"
      />
    );
  }

  // optional neutral θ=0 reference (solid faint, NOT a cut)
  let zero: JSX.Element | null = null;
  if (o.zeroRule) {
    const y0 = Y(0);
    zero = (
      <>
        <line x1={mL} y1={y0.toFixed(1)} x2={W - mR} y2={y0.toFixed(1)} stroke={neutralRule} strokeWidth={1} />
        <text
          x={W - mR + 4} y={(y0 + 3).toFixed(1)} textAnchor="start"
          fontFamily="ui-monospace,Menlo,monospace" fontSize={9} fill={inkSub}
        >
          0
        </text>
      </>
    );
  }

  // optional dashed certification cut (ℓ* only)
  let rule: JSX.Element | null = null;
  if (o.rule) {
    const yR = Y(o.rule.v);
    rule = (
      <>
        <line
          x1={mL} y1={yR.toFixed(1)} x2={W - mR} y2={yR.toFixed(1)}
          stroke={inkSub} strokeWidth={1.2} strokeDasharray="5 4"
        />
        <text
          x={W - mR} y={(yR - 5).toFixed(1)} textAnchor="end"
          fontFamily="ui-monospace,Menlo,monospace" fontSize={10} fill={inkSub}
        >
          {o.rule.label}
        </text>
      </>
    );
  }

  const d = s.map((v, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
  // Certification anchors (eval/recert) read darker + larger than training
  // points; without phase info, fall back to emphasizing the latest point.
  const inkDark = cssVar("--ink");
  const isCert = (i: number) =>
    o.phases ? (o.phases[i] === "eval" || o.phases[i] === "recert") : i === n - 1;
  const dots = s.map((v, i) => {
    const cert = isCert(i);
    return (
      <circle
        key={`d${i}`} cx={X(i).toFixed(1)} cy={Y(v).toFixed(1)} r={cert ? 3.3 : 2.2}
        fill={cert ? (o.phases ? inkDark : teal) : panel}
        stroke={cert ? (o.phases ? inkDark : teal) : teal} strokeWidth={1.4}
        opacity={cert ? 1 : 0.8}
      />
    );
  });

  return (
    <svg
      width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
      role="img" aria-label={o.label}
    >
      {phases}
      {grid}
      {band}
      {zero}
      {rule}
      <path d={d} fill="none" stroke={teal} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {dots}
    </svg>
  );
}

// ── Heatmap: 16-week activity calendar, shaded by activity TYPE ─────────────
// 1 = signed in (light teal), 2 = certification test (medium), 3 = training
// completed (darkest). The day's highest activity wins.
const ACTIVITY_LABEL = ["", "Signed in", "Certification test", "Training completed"];

// Discrete teal shade per activity level (shares the trajectory-chart hue).
function heatFillLevel(level: number): string {
  if (level <= 0) return cssVar("--zero-fill");
  const teal = [47, 143, 131];
  const dark = typeof document !== "undefined"
    && document.documentElement.getAttribute("data-theme") === "dark";
  const base = dark ? [20, 22, 28] : [255, 255, 255];
  const a = level >= 3 ? 1.0 : level === 2 ? 0.6 : 0.3;
  const r = Math.round(base[0] + (teal[0] - base[0]) * a);
  const g = Math.round(base[1] + (teal[1] - base[1]) * a);
  const b = Math.round(base[2] + (teal[2] - base[2]) * a);
  return `rgb(${r},${g},${b})`;
}

export function HeatLegend() {
  useTheme();
  const items: Array<[number, string]> = [[1, "Sign-in"], [2, "Certification"], [3, "Training"]];
  return (
    <div className="cx-heat-legend">
      {items.map(([lvl, label]) => (
        <span key={lvl} className="cx-heat-item">
          <span className="cx-heat-sw" style={{ background: heatFillLevel(lvl) }} />{label}
        </span>
      ))}
    </div>
  );
}

// `activity` maps a UTC date (YYYY-MM-DD) → highest activity level that day.
// Each cell shows its date on hover; future days in the current week are blank.
export function Heatmap({ activity = {} }: { activity?: Record<string, number> } = {}) {
  useTheme();
  // GitHub-style hover tooltip: a custom popover that appears instantly on enter
  // and sits centered ABOVE the hovered cell (the native <title> is slow and
  // anchors at the cursor). Positioned from the cell's rendered box so it tracks
  // the scaled SVG correctly.
  const wrapRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const showTip = (e: RMouseEvent<SVGRectElement>, text: string) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const cr = e.currentTarget.getBoundingClientRect();
    const wr = wrap.getBoundingClientRect();
    setTip({ x: cr.left - wr.left + cr.width / 2, y: cr.top - wr.top, text });
  };
  const weeks = 16, days = 7, cell = 13, gap = 3, padL = 30, padT = 22;
  const W = padL + weeks * (cell + gap);
  const H = padT + days * (cell + gap) + 4;
  const gridInk = cssVar("--grid-ink");
  const inkFaint = cssVar("--ink-faint");
  // Align the grid to today in UTC (matches the server's UTC activity days).
  // Rows are Sunday-first (Sun at the top), so the row index = getUTCDay (Sun=0).
  const MS = 86400000;
  const now = new Date();
  const todayMs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const todayRow = new Date(todayMs).getUTCDay();
  const keyOf = (ms: number) => new Date(ms).toISOString().slice(0, 10);
  const niceOf = (ms: number) =>
    new Date(ms).toLocaleDateString(undefined,
      { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
  const cells: JSX.Element[] = [];
  for (let w = 0; w < weeks; w++) {
    for (let d = 0; d < days; d++) {
      const daysAgo = (weeks - 1 - w) * 7 + (todayRow - d);
      if (daysAgo < 0) continue;   // future days in the current week
      const ms = todayMs - daysAgo * MS;
      const level = activity[keyOf(ms)] ?? 0;
      const x = padL + w * (cell + gap);
      const y = padT + d * (cell + gap);
      const text = level > 0 ? `${niceOf(ms)}: ${ACTIVITY_LABEL[level]}` : niceOf(ms);
      cells.push(
        <rect key={`${w}-${d}`} x={x} y={y} width={cell} height={cell} rx={2}
          fill={heatFillLevel(level)} stroke={gridInk}
          onMouseEnter={(e) => showTip(e, text)} onMouseLeave={() => setTip(null)} />,
      );
    }
  }
  // Day-of-week labels (Sunday-first): Mon / Wed / Fri at rows 1 / 3 / 5.
  const lbls = ([["Mon", 1], ["Wed", 3], ["Fri", 5]] as const).map(([t, d]) => (
    <text key={t} x={padL - 5} y={padT + d * (cell + gap) + cell - 3} textAnchor="end"
      fontFamily="system-ui,sans-serif" fontSize={9} fill={inkFaint}>{t}</text>
  ));
  // Month labels across the top, centered over each contiguous run of weeks
  // whose Sunday falls in that month. 1-week edge runs are too narrow to label.
  const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const colMonth: number[] = [];
  for (let w = 0; w < weeks; w++) {
    const sundayMs = todayMs - ((weeks - 1 - w) * 7 + todayRow) * MS;
    colMonth[w] = new Date(sundayMs).getUTCMonth();
  }
  const monthLbls: JSX.Element[] = [];
  for (let w = 0, start = 0; w <= weeks; w++) {
    if (w === weeks || colMonth[w] !== colMonth[start]) {
      const end = w - 1;
      if (end > start) {   // run spans >= 2 weeks
        const cx = padL + ((start + end) / 2) * (cell + gap) + cell / 2;
        monthLbls.push(
          <text key={`m${start}`} x={cx} y={padT - 7} textAnchor="middle"
            fontFamily="system-ui,sans-serif" fontSize={9} fill={inkFaint}>{MONTH[colMonth[start]]}</text>,
        );
      }
      start = w;
    }
  }
  return (
    <div className="cx-heat" ref={wrapRef}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Activity heatmap, last 16 weeks">
        {monthLbls}
        {lbls}
        {cells}
      </svg>
      {tip && (
        <div className="cx-heat-tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>
      )}
    </div>
  );
}
