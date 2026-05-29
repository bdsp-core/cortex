// Exact visual constants extracted from scripts/eeg_bank_viewer.py so the
// web app is pixel-identical to the desktop. Single source of truth for the
// React components.

export const COLORS = {
  bg: "#0b0d12",
  card: "#14161c",
  cardAlt: "#15171c",
  borderInactive: "#3a3d45",
  borderInactive2: "#454a55",
  borderFocus: "#8a8f9b",
  textPrimary: "#eef1f5",
  textSecondary: "#dde0e6",
  textBody: "#aab0ba",
  textTertiary: "#767b87",
  textMuted: "#6b7280",
  textFaint: "#5c606a",
  accent: "#f5a623", // answer-selection outline
  pass: "#7ed391",
  fail: "#d8806a",
  referBorderline: "#d4b169",
  referUninformative: "#9aa0ab",
  pending: "#5c606a",
  eegBackdrop: "#2e425e", // landing static-EEG pen
  eegTrace: "#000000",
  ekgTrace: "#c80000", // (200,0,0)
} as const;

// Branding (Landing / Consent / Registration) uses Palatino; the Viewer +
// Results use the system sans. Ship a metrically-close free face for non-Mac.
export const FONTS = {
  serif: '"Palatino Linotype", "Book Antiqua", Palatino, "URW Palladio L", serif',
  sans: 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
} as const;

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
