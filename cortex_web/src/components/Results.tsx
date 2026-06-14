// Results screen — per-task verdict table (PASS / FAIL / REFER, colored per
// theme) plus an optional technical-details grid. The aggregate roll-up is
// deliberately NOT a single pass/fail (PLAN / OPEN_DECISIONS: per-task
// certificates are the v1.0 policy).

import { useEffect, useRef, useState } from "react";
import { COLORS, FONTS, VERDICT_STYLE } from "../../ui/theme";
import { binormalSteps } from "../roc";
import { Button, Card, Heading, Stage } from "./ui";

export interface RocDatum {
  auroc: number;
  hw: number;
  opFar: number | null;
  opHr: number | null;
}

export interface ResultSummary {
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  pi?: number[];
  R?: number[];
  nPerTask?: number[];
  roc?: RocDatum[]; // per-task (engine-index-aligned) binormal ROC + operating point
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

export function Results({ summary, onFinish, onDownloadVideos }: {
  summary: ResultSummary;
  onFinish?: () => void;
  onDownloadVideos?: () => Promise<void>;
}) {
  const [showTech, setShowTech] = useState(false);
  const [openRoc, setOpenRoc] = useState<number | null>(null);
  const [vidState, setVidState] = useState<"idle" | "rendering" | "error">("idle");
  const [vidErr, setVidErr] = useState("");
  const rows = rowPlan(summary);
  return (
    <Stage maxW={820}>
      {/* maxHeight + overflowY makes the whole results page scroll on
          short viewports (desktop v1.4.2 parity). */}
      <div style={{ maxHeight: "calc(100vh - 48px)", overflowY: "auto",
                    width: "100%" }}>
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
            const roc = summary.roc?.[o.idx];
            const open = openRoc === o.idx;
            return (
              <div key={o.code} style={{ borderBottom: `1px solid ${COLORS.borderInactive}` }}>
                <div style={{ display: "flex", justifyContent: "space-between",
                              alignItems: "center", padding: "12px 0" }}>
                  <span style={{ color: COLORS.textSecondary, fontWeight: 600, fontSize: 15 }}>
                    {o.label}
                    {o.cls === "spike" && (
                      <span style={{ color: COLORS.textTertiary, fontSize: 12, marginLeft: 6 }}>(spike)</span>
                    )}
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    {roc && (
                      <button onClick={() => setOpenRoc(open ? null : o.idx)}
                        style={{ background: "none", border: "none", color: COLORS.textTertiary,
                          cursor: "pointer", fontSize: 12, padding: 0, fontFamily: FONTS.sans }}>
                        {open ? "▾ Hide ROC" : "▸ Show ROC"}
                      </button>
                    )}
                    <span style={{ color: st.color, fontWeight: 700, fontSize: 15 }}>{st.label}</span>
                  </span>
                </div>
                {open && roc && (
                  <div style={{ display: "flex", gap: 16, alignItems: "center", padding: "4px 0 16px" }}>
                    <RocCanvas auroc={roc.auroc} opFar={roc.opFar} opHr={roc.opHr} size={220} />
                    <div style={{ fontSize: 13, color: COLORS.textBody }}>
                      <div>AUROC <b style={{ color: COLORS.textPrimary }}>{roc.auroc.toFixed(3)}</b></div>
                      <div style={{ color: COLORS.textTertiary, fontSize: 12 }}>
                        95% CI [{Math.max(0, roc.auroc - roc.hw).toFixed(3)}–{Math.min(1, roc.auroc + roc.hw).toFixed(3)}]
                      </div>
                      <div style={{ marginTop: 8, color: "#9671bd" }}>● your operating point</div>
                      {roc.opFar == null && (
                        <div style={{ color: COLORS.textTertiary, fontSize: 12 }}>
                          (not enough trials to plot a point)
                        </div>
                      )}
                    </div>
                  </div>
                )}
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

        {onDownloadVideos && (
          <div style={{ marginTop: 24, paddingTop: 16, borderTop: `1px solid ${COLORS.borderInactive}` }}>
            <button
              disabled={vidState === "rendering"}
              onClick={async () => {
                setVidState("rendering"); setVidErr("");
                try { await onDownloadVideos(); setVidState("idle"); }
                catch (e) { setVidErr(e instanceof Error ? e.message : String(e)); setVidState("error"); }
              }}
              style={{
                background: vidState === "rendering" ? COLORS.cardAlt : COLORS.accent,
                color: vidState === "rendering" ? COLORS.textTertiary : "#1a1205",
                border: "none", borderRadius: 4, padding: "10px 16px", fontWeight: 700, fontSize: 13,
                cursor: vidState === "rendering" ? "default" : "pointer", fontFamily: FONTS.sans,
              }}
            >
              {vidState === "rendering" ? "Rendering videos…" : "Download visualization videos (.zip)"}
            </button>
            <div style={{ color: COLORS.textTertiary, fontSize: 12, marginTop: 8 }}>
              Per-session MP4s: engine explainer, particle collapse, and pass/fail.
              {vidState === "rendering" && " This takes ~30–60 s."}
            </div>
            {vidState === "error" && (
              <div style={{ color: COLORS.fail, fontSize: 12, marginTop: 6 }}>{vidErr}</div>
            )}
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

// Square ROC plot: chance diagonal, binormal curve at AUROC, examinee dot.
function RocCanvas({ auroc, opFar, opHr, size }: {
  auroc: number; opFar: number | null; opHr: number | null; size: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr; cv.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.card;
    ctx.fillRect(0, 0, size, size);
    const pad = 34;
    const w = size - pad - 8, h = size - pad - 8;
    const X = (x: number) => pad + x * w;
    const Y = (y: number) => size - pad - y * h;
    ctx.strokeStyle = COLORS.borderInactive; ctx.lineWidth = 1;
    ctx.strokeRect(pad, size - pad - h, w, h);
    ctx.fillStyle = COLORS.textTertiary; ctx.font = "9px system-ui"; ctx.textAlign = "center";
    for (const t of [0, 0.5, 1]) {
      ctx.fillText(t.toFixed(1), X(t), size - pad + 12);
      ctx.textAlign = "right"; ctx.fillText(t.toFixed(1), pad - 5, Y(t) + 3); ctx.textAlign = "center";
    }
    ctx.fillStyle = COLORS.textBody;
    ctx.fillText("False-positive rate", pad + w / 2, size - 4);
    ctx.save(); ctx.translate(9, size - pad - h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("True-positive rate", 0, 0); ctx.restore();
    ctx.strokeStyle = "#8a8a8a"; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke();
    ctx.setLineDash([]);
    const { xs, ys } = binormalSteps(auroc);
    ctx.strokeStyle = "#77b5b6"; ctx.lineWidth = 2.2; ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = X(xs[i]), py = Y(ys[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (opFar != null && opHr != null) {
      ctx.fillStyle = "#9671bd"; ctx.strokeStyle = "#6a408d"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X(opFar), Y(opHr), 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
  }, [auroc, opFar, opHr, size]);
  return <canvas ref={ref} style={{ width: size, height: size }} />;
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
