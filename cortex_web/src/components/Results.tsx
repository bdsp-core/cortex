// Results screen — per-task verdict + AUROC table (PASS / FAIL / REFER,
// colored per theme) with a 95% CrI bar drawn against the per-task pass
// cut, plus a "Show technical details" panel for π/info-R/n. The aggregate
// roll-up is deliberately NOT a single pass/fail (PLAN / OPEN_DECISIONS:
// per-task certificates are the v1.0 policy).

import { useState } from "react";
import { aurocFromL, aurocSummary } from "../../engine/auroc";
import { COLORS, FONTS, VERDICT_STYLE } from "../../ui/theme";
import { Button, Card, Heading, Stage } from "./ui";

export interface ResultSummary {
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  pi?: number[];
  R?: number[];
  nPerTask?: number[];
  lMean?: number[];
  lSd?: number[];
  ellStar?: number[];      // per-task ℓ* cut → AUROC cut via aurocFromL
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

interface Row {
  idx: number;
  code: string;
  label: string;
  cls: "iiic" | "spike";
}

function rowPlan(s: ResultSummary): Row[] {
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
  bank_exhausted: "Bank exhausted",
  REFER_BORDERLINE: "Referred — borderline",
  REFER_UNINFORMATIVE: "Referred — uninformative",
};

export function Results({ summary, onFinish }: { summary: ResultSummary; onFinish?: () => void }) {
  const [showTech, setShowTech] = useState(false);
  const rows = rowPlan(summary);
  return (
    <Stage maxW={820}>
      {/* maxHeight + overflowY makes the whole results page scroll on
          short viewports (desktop v1.4.2 parity). */}
      <div style={{ maxHeight: "calc(100vh - 48px)", overflowY: "auto",
                    width: "100%" }}>
        <Card>
          <Heading>Assessment Complete</Heading>
          <div style={{ color: COLORS.textBody, marginBottom: 16, fontSize: 14 }}>
            {summary.nQuestions} recordings reviewed ·{" "}
            {STOP_REASON_LABEL[summary.stopReason] || summary.stopReason}
          </div>

          {/* AUROC + verdict per-task table */}
          <div style={{ marginTop: 8 }}>
            <div style={{ display: "grid",
                          gridTemplateColumns: "150px 1fr 100px 110px",
                          gap: "8px 12px",
                          fontSize: 12, color: COLORS.textTertiary,
                          paddingBottom: 6,
                          borderBottom: `1px solid ${COLORS.borderInactive}` }}>
              <span>Task</span>
              <span>AUROC (95% CrI vs cut)</span>
              <span style={{ textAlign: "right" }}>AUROC</span>
              <span style={{ textAlign: "right" }}>Verdict</span>
            </div>
            {rows.map((o) => (
              <TaskRow key={o.code} row={o} summary={summary} />
            ))}
          </div>

          <button
            onClick={() => setShowTech((s) => !s)}
            style={{
              background: "none", border: "none", color: COLORS.textTertiary,
              cursor: "pointer", fontSize: 13, marginTop: 20, padding: 0,
              fontFamily: FONTS.sans,
            }}
          >
            {showTech ? "▾ Hide technical details" : "▸ Show technical details"}
          </button>
          {showTech && (
            <div style={{ marginTop: 12, fontFamily: "monospace", fontSize: 12,
                          color: COLORS.textBody }}>
              <div style={{ display: "grid",
                            gridTemplateColumns: "1fr repeat(5, 70px)",
                            gap: 6 }}>
                <span style={{ color: COLORS.textTertiary }}>task</span>
                <span style={{ color: COLORS.textTertiary }}>ℓ̂</span>
                <span style={{ color: COLORS.textTertiary }}>SD ℓ</span>
                <span style={{ color: COLORS.textTertiary }}>π (pass)</span>
                <span style={{ color: COLORS.textTertiary }}>info R</span>
                <span style={{ color: COLORS.textTertiary }}>n</span>
                {rows.map((o) => (
                  <TechRow key={o.code} label={o.label}
                           l={summary.lMean?.[o.idx]} sd={summary.lSd?.[o.idx]}
                           pi={summary.pi?.[o.idx]} R={summary.R?.[o.idx]}
                           n={summary.nPerTask?.[o.idx]} />
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
      </div>
    </Stage>
  );
}

function TaskRow({ row, summary }: { row: Row; summary: ResultSummary }) {
  const v = summary.verdicts[row.idx] ?? "PENDING";
  const st = VERDICT_STYLE[v] ?? VERDICT_STYLE.PENDING;

  // AUROC summary: point + CrI from posterior ℓ. Pass cut via ℓ*.
  const lMean = summary.lMean?.[row.idx];
  const lSd = summary.lSd?.[row.idx];
  const ellStar = summary.ellStar?.[row.idx];
  const haveAuroc = lMean != null && lSd != null;
  const summ = haveAuroc ? aurocSummary(lMean!, lSd!) : null;
  const cut = ellStar != null ? aurocFromL(ellStar) : null;

  return (
    <div style={{ display: "grid",
                  gridTemplateColumns: "150px 1fr 100px 110px",
                  gap: "8px 12px",
                  alignItems: "center",
                  padding: "10px 0",
                  borderBottom: `1px solid ${COLORS.borderInactive}` }}>
      <span style={{ color: COLORS.textSecondary, fontWeight: 600, fontSize: 14 }}>
        {row.label}
        {row.cls === "spike" && (
          <span style={{ color: COLORS.textTertiary, fontSize: 11, marginLeft: 6 }}>
            (spike)
          </span>
        )}
      </span>

      {/* CrI bar against the AUROC cut */}
      <CriBar lo={summ?.lo95} hi={summ?.hi95} point={summ?.point} cut={cut} />

      <span style={{ color: COLORS.textPrimary, fontWeight: 600, fontSize: 14,
                     textAlign: "right" }}>
        {summ ? `${(summ.point * 100).toFixed(1)}%` : "—"}
      </span>
      <span style={{ color: st.color, fontWeight: 700, fontSize: 14,
                     textAlign: "right" }}>
        {st.label}
      </span>
    </div>
  );
}

/** Horizontal AUROC scale (0.5–1.0) with a translucent band for the CrI, a
 *  dot for the point estimate, and a dashed line marking the per-task pass
 *  cut. Gives a glance-readable "how far above/below the bar am I?" */
function CriBar({ lo, hi, point, cut }: {
  lo?: number; hi?: number; point?: number; cut?: number | null;
}) {
  // We scale the bar over [0.5, 1.0]: AUROC < 0.5 is at-chance, so the
  // useful range starts there.
  const SCALE_LO = 0.5;
  const SCALE_HI = 1.0;
  const frac = (v?: number) => v == null ? null :
    Math.max(0, Math.min(1, (v - SCALE_LO) / (SCALE_HI - SCALE_LO)));
  const fLo = frac(lo);
  const fHi = frac(hi);
  const fPt = frac(point);
  const fCut = frac(cut ?? undefined);
  return (
    <div style={{ position: "relative", height: 16,
                  background: COLORS.cardAlt, borderRadius: 3,
                  border: `1px solid ${COLORS.borderInactive2}` }}>
      {/* tick at 0.75 (midpoint of useful range) for orientation */}
      <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0,
                    width: 1, background: COLORS.borderInactive }} />
      {fLo != null && fHi != null && (
        <div style={{ position: "absolute",
                      left: `${fLo * 100}%`,
                      width: `${(fHi - fLo) * 100}%`,
                      top: 2, bottom: 2,
                      background: COLORS.accent, opacity: 0.32,
                      borderRadius: 2 }} />
      )}
      {fPt != null && (
        <div style={{ position: "absolute",
                      left: `calc(${fPt * 100}% - 3px)`,
                      top: 2, bottom: 2, width: 6, borderRadius: 3,
                      background: COLORS.accent }} />
      )}
      {fCut != null && (
        <div title="pass cut"
             style={{ position: "absolute",
                      left: `calc(${fCut * 100}% - 1px)`,
                      top: -3, bottom: -3, width: 2,
                      background: COLORS.textPrimary }} />
      )}
      {/* axis labels */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 16,
                    display: "flex", justifyContent: "space-between",
                    fontSize: 9, color: COLORS.textTertiary }}>
        <span>0.5</span><span>0.75</span><span>1.0</span>
      </div>
    </div>
  );
}

function TechRow({ label, l, sd, pi, R, n }: {
  label: string; l?: number; sd?: number; pi?: number; R?: number; n?: number;
}) {
  const f = (x?: number, d = 2) => (x == null ? "—" : x.toFixed(d));
  return (
    <>
      <span style={{ color: COLORS.textSecondary }}>{label}</span>
      <span>{f(l)}</span>
      <span>{f(sd)}</span>
      <span>{f(pi)}</span>
      <span>{f(R)}</span>
      <span>{n ?? "—"}</span>
    </>
  );
}
