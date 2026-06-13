// Results screen — per-task verdict table (PASS / FAIL / REFER, colored per
// theme) plus an optional technical-details grid. The aggregate roll-up is
// deliberately NOT a single pass/fail (PLAN / OPEN_DECISIONS: per-task
// certificates are the v1.0 policy).

import { useState } from "react";
import { COLORS, FONTS, VERDICT_STYLE } from "../../ui/theme";
import { Button, Card, Heading, Stage } from "./ui";

export interface ResultSummary {
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  pi?: number[];
  R?: number[];
  nPerTask?: number[];
  // K=7 manifests pass these so the verdict table reads the right labels.
  // Optional for back-compat with the older K=6 hard-coded path.
  taskCodes?: string[];
  taskLabels?: string[];
  taskClasses?: ("iiic" | "spike")[];
}

const DEFAULT_IIIC = [
  { code: "sz",   label: "Seizure" },
  { code: "lpd",  label: "LPD" },
  { code: "gpd",  label: "GPD" },
  { code: "lrda", label: "LRDA" },
  { code: "grda", label: "GRDA" },
  { code: "iic",  label: "Other" },
];

/** Row plan for the results table: every task in bundle order, with its real
 *  engine index, code, label, and which screen handled it. Falls back to the
 *  K=6 IIIC list when the older manifest didn't carry these fields. */
function rowPlan(s: ResultSummary):
  Array<{ idx: number; code: string; label: string; cls: "iiic" | "spike" }> {
  if (s.taskCodes && s.taskLabels) {
    return s.taskCodes.map((code, idx) => ({
      idx, code, label: s.taskLabels![idx],
      cls: (s.taskClasses?.[idx] ?? "iiic"),
    }));
  }
  return DEFAULT_IIIC.map((o, idx) => ({ ...o, idx, cls: "iiic" as const }));
}

const STOP_REASON_LABEL: Record<string, string> = {
  all_resolved: "All patterns resolved",
  ALL_RESOLVED: "All patterns resolved",
  max_questions: "Maximum questions reached",
  MAX_QUESTIONS: "Maximum questions reached",
  REFER_BORDERLINE: "Referred — borderline",
  REFER_UNINFORMATIVE: "Referred — uninformative",
};

export function Results({ summary, onFinish }: { summary: ResultSummary; onFinish?: () => void }) {
  const [showTech, setShowTech] = useState(false);
  const rows = rowPlan(summary);
  return (
    <Stage maxW={720}>
      <Card>
        <Heading>Assessment Complete</Heading>
        <div style={{ color: COLORS.textBody, marginBottom: 8 }}>
          {summary.nQuestions} recordings reviewed ·{" "}
          {STOP_REASON_LABEL[summary.stopReason] || summary.stopReason}
        </div>
        <div style={{ marginTop: 16 }}>
          {rows.map((o) => {
            const v = summary.verdicts[o.idx] ?? "PENDING";
            const st = VERDICT_STYLE[v] ?? VERDICT_STYLE.PENDING;
            return (
              <div key={o.code} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "12px 0", borderBottom: `1px solid ${COLORS.borderInactive}`,
              }}>
                <span style={{ color: COLORS.textSecondary, fontWeight: 600, fontSize: 15 }}>
                  {o.label}
                  {o.cls === "spike" && (
                    <span style={{ color: COLORS.textTertiary, fontSize: 12, marginLeft: 6 }}>
                      (spike)
                    </span>
                  )}
                </span>
                <span style={{ color: st.color, fontWeight: 700, fontSize: 15 }}>{st.label}</span>
              </div>
            );
          })}
        </div>

        <button
          onClick={() => setShowTech((s) => !s)}
          style={{
            background: "none", border: "none", color: COLORS.textTertiary,
            cursor: "pointer", fontSize: 13, marginTop: 16, padding: 0,
            fontFamily: FONTS.sans,
          }}
        >
          {showTech ? "▾ Hide technical details" : "▸ Show technical details"}
        </button>
        {showTech && (
          <div style={{ marginTop: 12, fontFamily: "monospace", fontSize: 12,
                        color: COLORS.textBody }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr repeat(3, 90px)", gap: 6 }}>
              <span style={{ color: COLORS.textTertiary }}>task</span>
              <span style={{ color: COLORS.textTertiary }}>π (pass)</span>
              <span style={{ color: COLORS.textTertiary }}>info R</span>
              <span style={{ color: COLORS.textTertiary }}>n</span>
              {rows.map((o) => (
                <Row key={o.code} label={o.label}
                     pi={summary.pi?.[o.idx]} R={summary.R?.[o.idx]} n={summary.nPerTask?.[o.idx]} />
              ))}
            </div>
          </div>
        )}

        {onFinish && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 24 }}>
            <Button onClick={onFinish}>Done</Button>
          </div>
        )}
      </Card>
    </Stage>
  );
}

function Row({ label, pi, R, n }: { label: string; pi?: number; R?: number; n?: number }) {
  const f = (x?: number, d = 2) => (x == null ? "—" : x.toFixed(d));
  return (
    <>
      <span style={{ color: COLORS.textSecondary }}>{label}</span>
      <span>{f(pi)}</span>
      <span>{f(R)}</span>
      <span>{n ?? "—"}</span>
    </>
  );
}
