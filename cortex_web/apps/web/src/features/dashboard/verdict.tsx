import type { CertResult } from "../../api";
import type { PercentileDomainScore } from "../../percentile/types";

export const TASK_ORDER = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"] as const;

export const FULL_NAMES: Readonly<Record<string, string>> = {
  spike: "Epileptiform spikes / sharp waves",
  sz: "Electrographic seizure",
  lpd: "Lateralized periodic discharges",
  gpd: "Generalized periodic discharges",
  lrda: "Lateralized rhythmic delta activity",
  grda: "Generalized rhythmic delta activity",
  iic: "Ictal-interictal continuum / other",
};

export const TASK_LABELS: Readonly<Record<string, string>> = {
  spike: "Spike",
  sz: "Seizure",
  lpd: "LPD",
  gpd: "GPD",
  lrda: "LRDA",
  grda: "GRDA",
  iic: "Other",
};

export interface VerdictPresentation {
  cls: string;
  text: string;
}

export function verdictPresentation(verdict: string): VerdictPresentation {
  switch ((verdict || "").toUpperCase()) {
    case "PASS": return { cls: "pass", text: "Pass" };
    case "FAIL": return { cls: "fail", text: "Fail" };
    case "REFER_BORDERLINE": return { cls: "referb", text: "Refer · borderline" };
    case "REFER_UNINFORMATIVE": return { cls: "referu", text: "Refer · uninformative" };
    case "ABOVE_CUT": return { cls: "pass", text: "At Expert" };
    case "BELOW_CUT": return { cls: "fail", text: "Below Expert" };
    case "INDETERMINATE_AT_CUT": return { cls: "referb", text: "Indeterminate" };
    case "UNDETERMINABLE_CAP": return { cls: "referu", text: "Insufficient precision · cap" };
    case "UNDETERMINABLE_BANK": return { cls: "referu", text: "Insufficient precision · bank" };
    case "IN_TRAINING": return { cls: "train", text: "In training" };
    case "NOT_ASSESSED": return { cls: "none", text: "Not yet assessed" };
    default: return { cls: "train", text: "In training" };
  }
}

export function VerdictChip({ verdict }: { verdict: string }) {
  const { cls, text } = verdictPresentation(verdict);
  return <span className={`cx-chip ${cls}`}><i />{text}</span>;
}

export function verdictsOf(result: CertResult): string[] {
  return Array.isArray(result.verdicts) ? result.verdicts as string[] : [];
}

export function aurocOf(result: CertResult, taskK: number): number | null {
  const value = result.roc?.[taskK]?.auroc;
  return typeof value === "number" ? value : null;
}

export function percentileOf(
  result: CertResult, taskCode: string,
): PercentileDomainScore | null {
  const report = result.percentile;
  if (report?.status !== "available" || !report.profile.display) return null;
  return report.domains?.[taskCode as keyof typeof report.domains] ?? null;
}

export function formatPercentile(value: number): string {
  if (value < 5) return "<5th";
  if (value > 95) return ">95th";
  const n = Math.round(value), mod100 = n % 100;
  const suffix = mod100 >= 11 && mod100 <= 13 ? "th"
    : n % 10 === 1 ? "st" : n % 10 === 2 ? "nd"
      : n % 10 === 3 ? "rd" : "th";
  return `${n}${suffix}`;
}

export function formatPercentileRange(lower: number, upper: number): string {
  return `${formatPercentile(lower)}–${formatPercentile(upper)}`;
}

export function formatAssessmentDate(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
