// Visual constants for the React components. Colors are theme-scoped CSS
// variables (defined in index.html: LIGHT default + DARK toggle on
// <html data-theme>); the token names below are kept stable so component
// inline styles need no churn. Canvas-only colors (EEG paper / landing pen)
// stay literal because a 2d-context fillStyle cannot resolve `var()`; canvas
// code that needs a themed color resolves it at draw time via cssVar().

import type { CSSProperties } from "react";

export const COLORS = {
  bg: "var(--page)",
  card: "var(--panel)",
  cardAlt: "var(--field-bg)",
  borderInactive: "var(--bd-subtle)",
  borderInactive2: "var(--bd)",
  borderFocus: "var(--bd-strong)",
  textPrimary: "var(--ink)",
  textSecondary: "var(--ink)",
  textBody: "var(--ink-subtle)",
  textTertiary: "var(--ink-subtle)",
  textMuted: "var(--ink-faint)",
  textFaint: "var(--ink-faint)",
  accent: "var(--accent)", // answer-selection + exam-button accent = site teal (theme-constant)
  pass: "var(--pass)",
  fail: "var(--fail)",
  referBorderline: "var(--refer-b)",
  referUninformative: "var(--refer-u)",
  pending: "var(--pending)",
  eegBackdrop: "#2e425e", // landing static-EEG pen (canvas literal)
  eegTrace: "#000000", // clinical EEG paper trace (canvas literal)
  ekgTrace: "#c80000", // (200,0,0) (canvas literal)
} as const;

// Resolve a themed CSS variable to a concrete color string for use in a
// canvas 2d context (which cannot consume `var()`). SSR/test-safe.
export function cssVar(name: string, fallback = "#000000"): string {
  if (typeof window === "undefined" || typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// Branding (Landing / Consent / Registration) uses Palatino; the Viewer +
// Results use the system sans. Ship a metrically-close free face for non-Mac.
export const FONTS = {
  serif: '"Palatino Linotype", "Book Antiqua", Palatino, "URW Palladio L", serif',
  sans: 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
} as const;

// Compact display-control button — the EEG pan controls. Matched to the
// "Save & finish later" button (lighter --bd-subtle border, rounded corners,
// 12px font / 8px-12px padding) so the viewer's bottom controls row shares ONE
// button height + border weight instead of the taller, darker, square-cornered
// default `.cx-test button`. Shared by Viewer (exam) + TrainingRunner (learning)
// so the two pan controls can't drift apart.
export const PAN_BTN_STYLE: CSSProperties = {
  background: "none",
  border: `1px solid ${COLORS.borderInactive}`,
  borderRadius: 4,
  fontSize: 12,
  padding: "8px 12px",
};

// Staged fade-in for a results reveal: each `.cx-reveal-in` layer starts hidden
// (opacity 0, nudged down 8px) and animates up in place; parents stagger the
// cascade via inline `animationDelay`. Static under prefers-reduced-motion.
// Shared by the training session-end reveal (TrainingRunner) and the exam
// Results page so the two "results" reveals stay identical.
export const REVEAL_CSS = `
@keyframes cxRevealIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.cx-reveal-in { opacity: 0; animation: cxRevealIn .55s ease forwards; }
@media (prefers-reduced-motion: reduce) { .cx-reveal-in { animation: none; opacity: 1; transform: none; } }
`;

export const GEOMETRY = {
  landing: { w: 980, h: 660 },
  consent: { w: 980, h: 660 },
  registration: { w: 980, h: 720 },
  viewer: { w: 1500, h: 950 },
  spectrogramWidth: { min: 250, max: 300 },
} as const;

// IIIC answer options — index == engine task k. Button text "N  ·  Label".
export const IIIC_OPTIONS = [
  { code: "sz", label: "Seizure" },
  { code: "lpd", label: "LPD" },
  { code: "gpd", label: "GPD" },
  { code: "lrda", label: "LRDA" },
  { code: "grda", label: "GRDA" },
  { code: "iic", label: "Other" },
] as const;

// Display controls (defaults bolded in comments).
export const GAIN_LADDER = [1, 2, 7, 10, 15, 20, 30, 50, 70, 100, 200, 300]; // default 100
export const MONTAGES = ["bipolar", "average", "laplacian"] as const; // default bipolar
export const BANDPASS_OPTIONS = [
  "0.5-70 Hz", // default
  "0.5-40 Hz",
  "0.5-30 Hz",
  "0.5-20 Hz",
  "1-70 Hz",
  "off",
];
export const NOTCH_OPTIONS = ["60 Hz", "50 Hz", "off"]; // default 60 Hz
export const WINDOW_OPTIONS = [5, 10, 15, 20, 30]; // seconds, default 10

// Bipolar montage — exact channel pairs (LL, RL, LP, RP, central) + the
// NaN-separator rows the desktop inserts after indices 4, 8, 12, 16, then EKG.
export const BIPOLAR_PAIRS: [string, string][] = [
  ["Fp1", "F7"], ["F7", "T3"], ["T3", "T5"], ["T5", "O1"], // LL
  ["Fp2", "F8"], ["F8", "T4"], ["T4", "T6"], ["T6", "O2"], // RL
  ["Fp1", "F3"], ["F3", "C3"], ["C3", "P3"], ["P3", "O1"], // LP
  ["Fp2", "F4"], ["F4", "C4"], ["C4", "P4"], ["P4", "O2"], // RP
  ["Fz", "Cz"], ["Cz", "Pz"], // central
];
export const BIPOLAR_SEPARATOR_AFTER = [4, 8, 12, 16];

// Spectrogram: jet 9-stop LUT, fixed dB levels, frequency band.
export const JET_LUT: [number, number, number][] = [
  [0, 0, 127], [0, 0, 255], [0, 127, 255], [0, 255, 255], [127, 255, 127],
  [255, 255, 0], [255, 127, 0], [255, 0, 0], [127, 0, 0],
];
export const SPEC_DB_LEVELS: [number, number] = [-10.0, 25.0];
export const SPEC_FREQ_RANGE: [number, number] = [0.5, 25.0]; // Hz
export const SPEC_REGIONS = ["LL", "RL", "LP", "RP"] as const;

// EEG display: clip at ±EEG_CLIP_MULT × gain before scaling; 1 plot unit =
// gain_uv µV.
export const EEG_CLIP_MULT = 3;

// IIIC clip geometry (desktop v1.3.8 alignment, scripts/eeg_bank_viewer.py).
// A 30-s clip whose central 10 s is the panel-labeled epoch; the test only
// scores the labeled portion, panning reveals context on either side.
export const IIIC_CLIP_S = 30;
export const IIIC_LABEL_START_S = 10; // start of labeled epoch in clip coords
export const IIIC_LABEL_LEN_S = 10;   // length of labeled epoch
export const IIIC_LABEL_END_S = IIIC_LABEL_START_S + IIIC_LABEL_LEN_S;

// Spectrogram clip-marker geometry. The 30-s EEG clip sits at the centre of a
// 10-minute spectrogram, so the clip bounds are 285 s and 315 s in spec time
// (fractions 0.475 and 0.525). Used for the dotted-white verticals marking
// "this is where the EEG you're seeing lives in the 10-min context."
export const SPEC_TOTAL_S = 600;
export const SPEC_CLIP_START_FRAC = (SPEC_TOTAL_S / 2 - IIIC_CLIP_S / 2) / SPEC_TOTAL_S;
export const SPEC_CLIP_END_FRAC = (SPEC_TOTAL_S / 2 + IIIC_CLIP_S / 2) / SPEC_TOTAL_S;

// Verdict → color + display label (Results screen).
export const VERDICT_STYLE: Record<string, { color: string; label: string }> = {
  PASS: { color: COLORS.pass, label: "PASS" },
  FAIL: { color: COLORS.fail, label: "FAIL" },
  REFER_BORDERLINE: { color: COLORS.referBorderline, label: "REFER (borderline)" },
  REFER_UNINFORMATIVE: { color: COLORS.referUninformative, label: "REFER (uninformative)" },
  PENDING: { color: COLORS.pending, label: "—" },
};

// Keyboard shortcuts (app-level; intercept before form controls).
export const KEYS = {
  selectAnswer: ["1", "2", "3", "4", "5", "6"],
  confirm: ["Enter"],
  panLeft: ["ArrowLeft"],
  panRight: ["ArrowRight"],
  gainUp: ["ArrowUp"],
  gainDown: ["ArrowDown"],
  cycleMontage: ["Control"],
} as const;
