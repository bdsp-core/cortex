import type { CertResult } from "../../api";

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
    case "ABOVE_CUT": return { cls: "pass", text: "Above cut" };
    case "BELOW_CUT": return { cls: "fail", text: "Below cut" };
    case "INDETERMINATE_AT_CUT": return { cls: "referb", text: "Indeterminate at cut" };
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
