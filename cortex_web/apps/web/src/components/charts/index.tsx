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

// ── Ring: progress toward ℓ* (verdict-colored arc + center "% ℓ*") ──────────
export function Ring({
  ell, ellStar, chipClass, size = 56,
}: {
  ell: number | null; ellStar: number | null; chipClass: string; size?: number;
}) {
  useTheme(); // re-render on theme flip
  const r = size / 2 - 5;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  // No measured ℓ (legacy verdicts-only result) → indeterminate track + "—".
  const known = typeof ell === "number" && typeof ellStar === "number";
  const star = (ellStar ?? 0) || 1;
  const frac = known ? Math.max(0, Math.min(1, (ell as number) / star)) : 0;
  const over = known && (ell as number) >= star;
  const stroke = over ? cssVar(verdictVar(chipClass)) : cssVar("--teal");
  const track = cssVar("--bd-subtle");
  const ink = cssVar("--ink");
  const inkSub = cssVar("--ink-subtle");
  const off = circ * (1 - frac);
  const pct = known ? Math.round(((ell as number) / star) * 100) : null;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={track} strokeWidth={5} />
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke={stroke} strokeWidth={5}
        strokeLinecap="round" strokeDasharray={circ.toFixed(1)}
        strokeDashoffset={off.toFixed(1)} transform={`rotate(-90 ${cx} ${cy})`}
      />
      <text
        x={cx} y={cy - 1} textAnchor="middle" dominantBaseline="middle"
        fontFamily="ui-monospace,Menlo,monospace" fontSize={13} fontWeight={700} fill={ink}
      >
        {pct === null ? "—" : pct}
      </text>
      <text
        x={cx} y={cy + 11} textAnchor="middle" dominantBaseline="middle"
        fontFamily="system-ui,sans-serif" fontSize={7} fill={inkSub} letterSpacing="0.5"
      >
        % ℓ*
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
  const X = (i: number) => mL + (i / (n - 1)) * iw;
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

  // phase shading: eval (0..1), training (1..n-2), recert (n-2..last)
  const xEval = X(1);
  const xRecert = X(n - 2);
  const phases = (
    <>
      <rect x={mL} y={mT} width={(xEval - mL).toFixed(1)} height={ih} fill={phaseEval} />
      <rect x={xRecert.toFixed(1)} y={mT} width={(W - mR - xRecert).toFixed(1)} height={ih} fill={phaseRecert} />
      <line x1={xEval.toFixed(1)} y1={mT} x2={xEval.toFixed(1)} y2={mT + ih} stroke={phaseRule} strokeDasharray="2 3" />
      <line x1={xRecert.toFixed(1)} y1={mT} x2={xRecert.toFixed(1)} y2={mT + ih} stroke={phaseRule} strokeDasharray="2 3" />
    </>
  );

  // optional ±sd band
  let band: JSX.Element | null = null;
  if (o.band) {
    const top = s.map((_v, i) => `${X(i).toFixed(1)},${Y(o.band![1][i]).toFixed(1)}`);
    const bot = s.map((_v, i) => `${X(i).toFixed(1)},${Y(o.band![0][i]).toFixed(1)}`).reverse();
    band = <polygon points={top.concat(bot).join(" ")} fill={teal} fillOpacity={0.14} />;
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
  const dots = s.map((v, i) => (
    <circle
      key={`d${i}`} cx={X(i).toFixed(1)} cy={Y(v).toFixed(1)} r={2.2}
      fill={i === n - 1 ? teal : panel} stroke={teal} strokeWidth={1.4}
    />
  ));

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

// ── Heatmap: 16-week contribution calendar, teal sqrt ramp ──────────────────
const HEAT_MAX = 42;

// Theme-aware single-hue teal sqrt ramp; shared by heatmap + legend. sqrt so
// light days stay visible. Mirrors heatFill() in the mockup.
function heatFill(c: number): string {
  const teal = [47, 143, 131];
  const dark = typeof document !== "undefined"
    && document.documentElement.getAttribute("data-theme") === "dark";
  const base = dark ? [20, 22, 28] : [255, 255, 255];
  if (c <= 0) return cssVar("--zero-fill");
  const k = Math.sqrt(c / HEAT_MAX);
  const a = 0.15 + 0.85 * k;
  const r = Math.round(base[0] + (teal[0] - base[0]) * a);
  const g = Math.round(base[1] + (teal[1] - base[1]) * a);
  const b = Math.round(base[2] + (teal[2] - base[2]) * a);
  return `rgb(${r},${g},${b})`;
}

export function HeatLegend() {
  useTheme();
  const stops = [0, 0.18, 0.42, 0.68, 1.0].map((f) => heatFill(f * HEAT_MAX));
  return (
    <div className="cx-heat-legend">
      <span>less</span>
      {stops.map((c, i) => (
        <span key={i} className="cx-heat-sw" style={{ background: c }} />
      ))}
      <span>more</span>
    </div>
  );
}

export function Heatmap({ empty = false }: { empty?: boolean } = {}) {
  useTheme();
  const weeks = 16, days = 7, cell = 13, gap = 3, padL = 14, padT = 14;
  // deterministic pseudo-random sample counts (matches the mockup's LCG seed)
  let seed = 20260615;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const max = HEAT_MAX;
  const W = padL + weeks * (cell + gap);
  const H = padT + days * (cell + gap) + 10;
  const gridInk = cssVar("--grid-ink");
  const cells: JSX.Element[] = [];
  for (let w = 0; w < weeks; w++) {
    for (let d = 0; d < days; d++) {
      const r0 = rnd();
      const recent = w >= weeks - 3 ? 0.4 : 0;
      const weekend = d === 5 || d === 6 ? -0.18 : 0;
      let c = empty ? 0 : Math.max(0, Math.round((r0 + recent + weekend) * max));
      if (!empty && rnd() < 0.12) c = 0;
      const x = padL + w * (cell + gap);
      const y = padT + d * (cell + gap);
      cells.push(
        <rect key={`${w}-${d}`} x={x} y={y} width={cell} height={cell} rx={2} fill={heatFill(c)} stroke={gridInk}>
          <title>{c} items</title>
        </rect>,
      );
    }
  }
  const inkFaint = cssVar("--ink-faint");
  const lbls = ([["M", 0], ["W", 2], ["F", 4]] as const).map(([t, d]) => (
    <text
      key={t} x={2} y={padT + d * (cell + gap) + cell - 2}
      fontFamily="system-ui,sans-serif" fontSize={8} fill={inkFaint}
    >
      {t}
    </text>
  ));
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Activity heatmap, last 16 weeks">
      {lbls}
      {cells}
    </svg>
  );
}
