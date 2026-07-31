// The immersive training runner — a full-screen learning-protocol session that
// mirrors the exam Viewer's look (cx-test, EEG + spectrogram canvases, amber
// answer-selection, sharp edges, teal actions) but is FEEDBACK-DRIVEN: each trial
// is a binary "is this <pattern>?" one-vs-rest decision, answering reveals a
// distinct RESULT step (✓/✗ + the true label), then the next item. Daily-bounded
// to a fixed item count. Belief/telemetry stays under the hood (minimal on-screen
// progress); the ported trainer runs synchronously via TrainingController.
import { useCallback, useEffect, useRef, useState } from "react";
import { Bundle, SegmentData } from "../bundle";
import { useEegDisplay } from "../features/eeg/useEegDisplay";
import { EegCanvas } from "./EegCanvas";
import { SpecCanvas } from "./SpecCanvas";
import {
  COLORS, FONTS, GAIN_LADDER, MONTAGES, WINDOW_OPTIONS, PAN_BTN_STYLE,
  REVEAL_CSS, IIIC_LABEL_START_S, IIIC_LABEL_END_S,
  SPEC_CLIP_START_FRAC, SPEC_CLIP_END_FRAC,
} from "../../ui/theme";
import * as api from "../api";
import { TrainingController, DEFAULT_SESSION_ITEMS,
         type TrainerSessionLike } from "../trainingController";
import { buildRevealRows, type RevealRow } from "../trainingReveal";
import type { PercentileProfile } from "../percentile/types";

// Teal ring flashed around the "Is this X?" prompt when the learning policy
// switches task domains mid-session — beta testers missed the silent label
// swap between questions. Two smooth pulses over 2s, then fades clean.
// rgba(47,143,131,…) is --teal (#2f8f83) with alpha; the base teal is
// theme-constant so the ring reads correctly in light and dark.
const TASK_FLASH_CSS = `
.cx-task-flash { animation: cxTaskFlash 2s ease-in-out; }
@keyframes cxTaskFlash {
  0%   { box-shadow: 0 0 0 0 rgba(47,143,131,0); }
  20%  { box-shadow: 0 0 0 3px var(--teal), 0 0 14px 2px rgba(47,143,131,0.35); }
  50%  { box-shadow: 0 0 0 1px rgba(47,143,131,0.15); }
  70%  { box-shadow: 0 0 0 3px var(--teal), 0 0 14px 2px rgba(47,143,131,0.35); }
  100% { box-shadow: 0 0 0 0 rgba(47,143,131,0); }
}
@media (prefers-reduced-motion: reduce) { .cx-task-flash { animation: none; } }
`;

export function TrainingRunner({
  bundle, session, trainingId, labels, total = DEFAULT_SESSION_ITEMS, onExit,
  attainability,
  percentileProfile,
}: {
  bundle: Bundle;
  session: TrainerSessionLike;    // production: server-driven session adapter
  trainingId: string;
  labels: string[];                 // per-task display label, e.g. "GPD"
  total?: number;
  onExit: (summary: { allMastered: boolean; nItems: number }) => void;
  // R5: seed-time "where you stand" report (engine mode only) — shown as a
  // slim strip on the first question, gone after the first answer.
  attainability?: { label: string; tier: "far" | "mid" | "near" }[];
  percentileProfile?: PercentileProfile | null;
}) {
  const ctrlRef = useRef<TrainingController | null>(null);
  if (ctrlRef.current === null) {
    ctrlRef.current = new TrainingController(session, labels, { total });
  }
  const ctrl = ctrlRef.current;

  // Spike questions have no spectrogram — mirror the exam's SpikeViewer: EEG
  // full-width over the whole clip, no labeled-epoch box, no window/pan. IIIC
  // keeps the spectrogram + windowed/panned view. Branch on the current item's
  // task class (same carrier the exam routes on: bundle.inputs.taskClasses).
  const isSpike = ctrl.item != null && bundle.inputs?.taskClasses?.[ctrl.item.task] === "spike";

  const [, force] = useState(0);
  const rerender = useCallback(() => force((n) => n + 1), []);
  const [seg, setSeg] = useState<SegmentData | null>(null);
  const [currentSegId, setCurrentSegId] = useState<number | null>(ctrl.item?.segId ?? null);
  const [correctCount, setCorrectCount] = useState(0);

  // flash the prompt on EVERY new question (user request 2026-07-17; it
  // originally fired only on domain switches). Keyed by segId so the CSS
  // animation restarts per question; no flash on the session's very first
  // question or on resume (the ref starts on the first segment).
  const flashSeg = ctrl.item?.segId ?? null;
  const prevSegRef = useRef<number | null>(flashSeg);
  const [taskFlashKey, setTaskFlashKey] = useState(0);
  useEffect(() => {
    if (flashSeg != null && prevSegRef.current != null
        && flashSeg !== prevSegRef.current) {
      setTaskFlashKey((k) => k + 1);
    }
    if (flashSeg != null) prevSegRef.current = flashSeg;
  }, [flashSeg]);

  // display controls (same defaults as the exam Viewer — including the
  // 0.5-70 Hz bandpass + 60 Hz notch, which the trainer previously requested
  // with an en-dash string that parsed to "no filter")
  const { montage, setMontage, gain, setGain, rows, cycleMontage, stepGain } =
    useEegDisplay(seg);
  const [windowS, setWindowS] = useState(10);
  const [panStart, setPanStart] = useState(IIIC_LABEL_START_S);

  const rootRef = useRef<HTMLDivElement>(null);
  const specBoxRef = useRef<HTMLDivElement>(null);
  const eegBoxRef = useRef<HTMLDivElement>(null);
  const [specSize, setSpecSize] = useState({ w: 280, h: 720 });
  const [eegSize, setEegSize] = useState({ w: 800, h: 720 });

  useEffect(() => {
    const obs = new ResizeObserver(() => {
      if (specBoxRef.current) setSpecSize({ w: specBoxRef.current.clientWidth, h: specBoxRef.current.clientHeight });
      if (eegBoxRef.current) setEegSize({ w: eegBoxRef.current.clientWidth, h: eegBoxRef.current.clientHeight });
    });
    if (specBoxRef.current) obs.observe(specBoxRef.current);
    if (eegBoxRef.current) obs.observe(eegBoxRef.current);
    return () => obs.disconnect();
    // re-observe when the spike/IIIC layout flips (the spec box mounts/unmounts)
  }, [isSpike]);

  // load the chosen segment's media whenever the question item changes
  useEffect(() => {
    if (currentSegId === null) return;
    let alive = true;
    setSeg(null);
    setPanStart(IIIC_LABEL_START_S);
    bundle.segment(currentSegId).then((s) => {
      if (!alive) return;
      setSeg(s);
      ctrl.markShown(performance.now());     // RT clock starts when the trace is visible
    }).catch((e) => {
      // A media fetch failure must not hang the runner or bubble up as an
      // uncaught rejection; log and leave the EEG box in its loading state.
      if (alive) console.error("[training] segment media load failed", currentSegId, e);
    });
    return () => { alive = false; };
  }, [currentSegId, bundle, ctrl]);

  const flush = useCallback(() => {
    const pts = ctrl.drainTrajectory();
    if (pts.length) api.postTrainingProgress(trainingId, pts);
  }, [ctrl, trainingId]);

  // Flush + finalize exactly once (the reveal screen and the exit button can
  // both reach it); the server call is fire-and-forget, as before.
  const finalizedRef = useRef(false);
  const persistFinal = useCallback(() => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    flush();
    api.finalizeTrainingSession(trainingId, ctrl.progress().count, { mode: "training", scored: true }).catch(() => {});
  }, [ctrl, flush, trainingId]);

  // "Save & finish later" — persists and exits immediately (no reveal).
  const finish = useCallback(() => {
    persistFinal();
    onExit({ allMastered: ctrl.allMastered(), nItems: ctrl.progress().count });
  }, [ctrl, persistFinal, onExit]);

  const answer = useCallback((yes: boolean) => {
    if (ctrl.phase !== "question" || !seg) return;
    ctrl.answer(yes, performance.now());
    if (ctrl.lastResult?.correct) setCorrectCount((n) => n + 1);
    // per-answer checkpoint (crash-safety): every answer lands server-side
    // immediately via the api-level outbox; a blip re-flushes on the next one
    flush();
    // keep the trace on screen (dimmed) behind the result reveal; it clears when
    // the next question's media loads (proceed → setCurrentSegId).
    rerender();
  }, [ctrl, seg, flush, rerender]);

  // Full-identification answer (Phase L4 native n-way): pickTask is the
  // 0-based task-axis index of the chosen class.
  const answerPick = useCallback((pickTask: number) => {
    if (ctrl.phase !== "question" || !seg) return;
    ctrl.answerPick(pickTask, performance.now());
    if (ctrl.lastResult?.correct) setCorrectCount((n) => n + 1);
    flush();
    rerender();
  }, [ctrl, seg, flush, rerender]);
  const isNway =
    (ctrl.item as { link?: string } | null)?.link === "nway";

  const proceed = useCallback(() => {
    void (async () => {
      // The next item arrives from the server. The round trip normally lands
      // while the participant reads the reveal.
      await ctrl.waitForNext();
      ctrl.continue();                     // no-op unless on the result step (controller guards)
      if (ctrl.phase === "done") {
        // Completing the session lands on the reveal screen (today's movement),
        // not straight back on the dashboard; the state is persisted right away.
        persistFinal();
        rerender();
        return;
      }
      setCurrentSegId(ctrl.item?.segId ?? null);
      rerender();
    })();
  }, [ctrl, persistFinal, rerender]);

  // live refs so the global key handler reads current window/duration without
  // re-binding the listener on every gain/pan change.
  const windowSRef = useRef(windowS); windowSRef.current = windowS;
  const durRef = useRef(0); durRef.current = seg ? seg.nSamp / seg.fsHz : 0;

  // keyboard — mirrors the exam Viewer's viewer controls plus the trainer's
  // binary answer: Y/1 = yes, N/2 = no; ↑/↓ = gain ladder, ←/→ = pan,
  // Ctrl = cycle montage; Enter/Space = continue (on the result step).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (ctrl.phase === "question") {
        // y/n (or 1/2) answer binary items; on n-way items the digits 1-6
        // answer the six classes in button order (Seizure … Other)
        const nway = (ctrl.item as { link?: string } | null)?.link === "nway";
        if (nway && e.key >= "1" && e.key <= "6") { e.preventDefault(); answerPick(Number(e.key)); }
        else if (!nway && (e.key === "y" || e.key === "Y" || e.key === "1")) { e.preventDefault(); answer(true); }
        else if (!nway && (e.key === "n" || e.key === "N" || e.key === "2")) { e.preventDefault(); answer(false); }
        else if (e.key === "ArrowUp") { e.preventDefault(); stepGain(-1); }
        else if (e.key === "ArrowDown") { e.preventDefault(); stepGain(1); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); setPanStart((p) => Math.max(0, p - windowSRef.current)); }
        else if (e.key === "ArrowRight") { e.preventDefault(); setPanStart((p) => Math.min(Math.max(0, durRef.current - windowSRef.current), p + windowSRef.current)); }
        else if (e.key === "Control") { e.preventDefault(); cycleMontage(); }
      } else if (ctrl.phase === "result" && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault(); proceed();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [ctrl, answer, answerPick, proceed, cycleMontage, stepGain]);

  const { count } = ctrl.progress();
  const label = ctrl.item ? labels[ctrl.item.task] : (ctrl.lastResult?.patternLabel ?? "");
  const dur = seg ? seg.nSamp / seg.fsHz : 0;   // spike view shows the whole clip

  const shell: React.CSSProperties = {
    background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
    height: "100vh", display: "flex", flexDirection: "column", padding: 12, boxSizing: "border-box",
  };

  // ── DONE — the session-end reveal ─────────────────────────────────────
  // Belief stays under the hood during the sitting; this screen is where the
  // day's real posterior movement lands, staged row by row. Numbers come
  // straight from the trainer filters (trainingReveal.ts), nothing invented.
  if (ctrl.phase === "done") {
    const acc = count ? Math.round((100 * correctCount) / count) : 0;
    const allMastered = ctrl.allMastered();
    const empty = count === 0;   // the weak-domain pools were fully spaced-out
    const rows = empty ? [] : buildRevealRows(
      ctrl.startSnapshot, ctrl.snapshot(), labels, ctrl.itemsPerTask(),
      session.policy.ellStars);
    const footDelay = 0.4 + rows.length * 0.45;
    return (
      <div className="cx-test" style={{ ...shell, alignItems: "center", justifyContent: "center", overflowY: "auto" }}>
        <style>{REVEAL_CSS}</style>
        <div style={{ textAlign: "center", maxWidth: 580, width: "100%" }}>
          <div style={{ fontFamily: FONTS.serif, fontSize: 26, color: COLORS.textPrimary, marginBottom: 8 }}>
            {empty ? "No new segments right now" : "Session complete"}
          </div>
          <div style={{ color: COLORS.textBody, marginBottom: 24 }}>
            {empty
              ? "You've recently seen the available segments for your training domains. Check back later, or take a new certification test to refresh the pool."
              : `${count} items · ${acc}% correct`}
          </div>
          {rows.length > 0 && (
            <div style={{ textAlign: "left", marginBottom: 24 }}>
              <div style={{ fontFamily: FONTS.serif, fontSize: 18, color: COLORS.textPrimary, marginBottom: 2 }}>
                Today's movement
              </div>
              <div style={{ fontSize: 12, color: COLORS.textBody, opacity: 0.8, marginBottom: 10 }}>
                Readiness is the probability that your estimated skill clears the domain's ℓ* certification bar.
              </div>
              {rows.map((r, i) => (
                <RevealRowView key={r.taskK} r={r} delayS={0.4 + i * 0.45}
                  showPercentile={!!percentileProfile?.display} />
              ))}
            </div>
          )}
          <div className="cx-reveal-in" style={{ animationDelay: `${empty ? 0 : footDelay}s` }}>
            {allMastered && (
              <div style={{
                background: "var(--teal-weak)", border: "1px solid var(--teal)", color: "var(--teal-deep)",
                padding: "12px 16px", marginBottom: 24, fontSize: 14,
              }}>
                You're ready to re-certify — a new certification test is available from the dashboard.
              </div>
            )}
            <button className="cx-btn primary"
              onClick={() => { if (count > 0) persistFinal(); onExit({ allMastered, nItems: count }); }}
              style={{ fontSize: 15, padding: "12px 24px" }}>
              Return to dashboard <span className="arrow">→</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── QUESTION + RESULT ─────────────────────────────────────────────────
  // One screen. On the result step the question (EEG/spec/controls) stays put
  // but dimmed + inert, and the verdict + progress + Continue appear in a modal
  // card on top; Continue advances to the next question.
  const r = ctrl.lastResult;
  const inResult = ctrl.phase === "result" && r != null;
  const behind: React.CSSProperties = inResult
    ? { opacity: 0.4, filter: "grayscale(0.5)", pointerEvents: "none", userSelect: "none" }
    : {};
  return (
    <div ref={rootRef} tabIndex={0} className="cx-test" style={{ ...shell, outline: "none", position: "relative" }}>
      <style>{TASK_FLASH_CSS}</style>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, transition: "opacity .12s", ...behind }}>
        {/* header: chip (left) + centered prompt + exit (right). The flex:1
            side zones keep the prompt truly centered regardless of widths. */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
          <div style={{ flex: 1, display: "flex", alignItems: "center" }}>
            <span className="cx-chip train"><i />In training</span>
          </div>
          {/* key remount restarts the flash on every subsequent domain switch */}
          <span key={taskFlashKey} className={taskFlashKey > 0 ? "cx-task-flash" : undefined}
            style={{ fontWeight: 600, fontSize: 18, color: COLORS.textPrimary, padding: "4px 14px", whiteSpace: "nowrap" }}>
            {isNway ? <>Which pattern is this?</>
              : <>Is this <span style={{ color: "var(--teal-deep)" }}>{label}</span>?</>}
          </span>
          <div style={{ flex: 1, display: "flex", justifyContent: "flex-end" }}>
            {/* graceful mid-sitting exit: trajectories flush + the sitting
                finalizes, so the regimen picks up exactly here next time */}
            <button className="cx-btn" onClick={finish}>Save &amp; finish later</button>
          </div>
        </div>

        {/* media: spectrogram (left, IIIC only) + EEG (right). For spike there is
            no spectrogram — the EEG fills the row and shows the whole clip. */}
        <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0, marginTop: 8, overflow: "hidden" }}>
          {!isSpike && (
            <div ref={specBoxRef} style={{ flex: "0 0 280px", minHeight: 0, background: "#fff", border: `1px solid ${COLORS.borderInactive}` }}>
              <SpecCanvas spec={seg?.spec ?? null} width={specSize.w || 280} height={specSize.h || 720}
                clipBoundsFrac={[SPEC_CLIP_START_FRAC, SPEC_CLIP_END_FRAC]} />
            </div>
          )}
          <div ref={eegBoxRef} style={{ flex: 1, minHeight: 0, background: "#fff", border: `1px solid ${COLORS.borderInactive}` }}>
            {seg ? (
              <EegCanvas rows={rows} fsHz={seg.fsHz} gainUv={gain}
                windowS={isSpike ? (dur || 10) : windowS} panStartS={isSpike ? 0 : panStart}
                width={eegSize.w} height={eegSize.h} clipTraces
                labeledEpoch={isSpike ? undefined : { startS: IIIC_LABEL_START_S, endS: IIIC_LABEL_END_S }} />
            ) : <div style={{ color: "#888", padding: 20 }}>loading EEG…</div>}
          </div>
        </div>

        {/* controls + the binary Yes/No answer */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginTop: 8 }}>
          <Ctl label="Montage" value={montage} opts={MONTAGES as unknown as string[]} onChange={setMontage} />
          <Ctl label="Gain" value={String(gain)} opts={GAIN_LADDER.map(String)} onChange={(v) => setGain(Number(v))} />
          {!isSpike && (
            <>
              <Ctl label="Window" value={String(windowS)} opts={WINDOW_OPTIONS.map(String)} onChange={(v) => setWindowS(Number(v))} />
              <button style={PAN_BTN_STYLE} onClick={() => setPanStart((p) => Math.max(0, p - windowS))} title="Pan left (←)">◀</button>
              <button style={PAN_BTN_STYLE} onClick={() => setPanStart((p) => p + windowS)} title="Pan right (→)">▶</button>
            </>
          )}
          <span style={{ fontSize: 12, color: COLORS.textBody, opacity: 0.75 }}>
            {isSpike ? "↑/↓ gain · Ctrl montage" : "↑/↓ gain · ←/→ pan · Ctrl montage"}
          </span>
          {/* session progress fills the space between the hints and Yes/No —
              it doubles as the spacer keeping Yes/No pushed right */}
          <div style={{ flex: 1, minWidth: 150, padding: "0 10px" }}>
            <ProgressBar count={count} total={total} inline />
          </div>
          {isNway ? labels.slice(1).map((lb, i) => (
            <button key={lb} disabled={!seg} onClick={() => answerPick(i + 1)}
              style={{ minWidth: 84, padding: "12px 10px", fontWeight: 600, cursor: "pointer",
                background: "var(--teal-weak)", color: "var(--teal-deep)", border: "1px solid var(--teal-mid)" }}>
              {i + 1} · {lb}
            </button>
          )) : (<>
            <button disabled={!seg} onClick={() => answer(true)}
              style={{ minWidth: 130, padding: "12px 16px", fontWeight: 600, cursor: "pointer",
                background: "var(--teal-weak)", color: "var(--teal-deep)", border: "1px solid var(--teal-mid)" }}>
              1 · Yes
            </button>
            <button disabled={!seg} onClick={() => answer(false)}
              style={{ minWidth: 130, padding: "12px 16px", fontWeight: 600, cursor: "pointer",
                background: COLORS.card, color: COLORS.textPrimary, border: `1px solid ${COLORS.borderInactive2}` }}>
              2 · No
            </button>
          </>)}
        </div>
        {/* R5: "where you stand" strip — first question only */}
        {count === 0 && attainability && attainability.length > 0 && (
          <div style={{ marginTop: 6, fontSize: 13, color: COLORS.textBody }}>
            Today&apos;s focus — {" "}
            {(["far", "mid", "near"] as const).map((tier) => {
              const ls = attainability.filter((a) => a.tier === tier)
                .map((a) => a.label);
              if (!ls.length) return null;
              const word = tier === "far" ? "building foundations"
                : tier === "mid" ? "within reach" : "near the bar";
              return <span key={tier} style={{ marginRight: 12 }}>
                <strong>{word}:</strong> {ls.join(", ")}</span>;
            })}
          </div>
        )}
      </div>

      {/* result reveal — a modal card over the dimmed question */}
      {inResult && r && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(18,20,26,0.30)" }}>
          <div style={{ background: COLORS.card, border: `1px solid ${COLORS.borderInactive}`,
            boxShadow: "0 14px 44px rgba(0,0,0,0.30)", padding: "24px 28px", minWidth: 320, maxWidth: 420, textAlign: "center" }}>
            <div style={{ fontSize: 46, lineHeight: 1, color: r.correct ? COLORS.pass : COLORS.fail, fontWeight: 700 }}>
              {r.correct ? "✓" : "✗"}
            </div>
            <div style={{ fontSize: 20, fontWeight: 600, color: COLORS.textPrimary, margin: "10px 0 4px" }}>
              {r.correct ? "Correct" : "Incorrect"}
            </div>
            <div style={{ color: COLORS.textBody, marginBottom: 18 }}>
              {r.nway
                ? <>This is <strong style={{ color: COLORS.textPrimary }}>{r.patternLabel}</strong>{r.correct ? "" : `, not ${r.pickedLabel}`}.</>
                : <>This <strong style={{ color: COLORS.textPrimary }}>{r.isTarget ? "is" : "is not"}</strong> {r.patternLabel}.</>}
            </div>
            <ProgressBar count={count} total={total} />
            <button className="cx-btn primary" onClick={proceed} autoFocus
              style={{ marginTop: 18, fontSize: 15, padding: "12px 24px" }}>
              Continue <span className="arrow">→</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function RevealRowView({ r, delayS, showPercentile }: {
  r: RevealRow; delayS: number; showPercentile: boolean;
}) {
  const dSkill = r.after.skill - r.before.skill;
  const passB = Math.round(100 * r.before.pass);
  const passA = Math.round(100 * r.after.pass);
  const up = dSkill >= 0.005;
  const num: React.CSSProperties = {
    fontFamily: "var(--mono)", fontSize: 12.5, color: COLORS.textBody,
    whiteSpace: "nowrap",
  };
  const chip: React.CSSProperties = {
    fontSize: 11.5, padding: "2px 8px", whiteSpace: "nowrap",
    border: "1px solid var(--teal)", color: "var(--teal-deep)",
  };
  return (
    <div className="cx-reveal-in" style={{
      animationDelay: `${delayS}s`,
      display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap",
      padding: "9px 12px", borderTop: `1px solid ${COLORS.borderInactive}`,
      background: r.newlyMastered ? "var(--teal-weak)" : "transparent",
    }}>
      <span style={{ fontWeight: 600, color: COLORS.textPrimary, flex: "0 0 88px" }}>{r.label}</span>
      <span style={num}>
        ℓ̂ {r.before.skill.toFixed(2)} ±{r.before.sd.toFixed(2)} → {r.after.skill.toFixed(2)} ±{r.after.sd.toFixed(2)}{" "}
        <b style={{ color: up ? "var(--teal-deep)" : COLORS.textBody, fontWeight: up ? 700 : 400 }}>
          ({dSkill >= 0 ? "+" : ""}{dSkill.toFixed(2)})
        </b>
      </span>
      <span style={num}>readiness {passB}% → {passA}%</span>
      {showPercentile && r.beforePercentile && r.afterPercentile && (
        <span style={num}>
          historical percentile{" "}
          {formatPercentile(r.beforePercentile.estimate)}{" "}
          [{formatRange(r.beforePercentile.lower, r.beforePercentile.upper)}]
          {" → "}
          {formatPercentile(r.afterPercentile.estimate)}{" "}
          [{formatRange(r.afterPercentile.lower, r.afterPercentile.upper)}]
        </span>
      )}
      <span style={{ flex: 1 }} />
      {r.newlyMastered
        ? <span style={chip}>✓ mastered today</span>
        : r.mastered
          ? <span style={{ ...chip, border: `1px solid ${COLORS.borderInactive}`, color: COLORS.textBody }}>✓ mastered</span>
          : r.nearBar
            ? <span style={chip}>● near the ℓ* bar</span>
            : null}
    </div>
  );
}

function ordinal(value: number): string {
  const n = Math.round(value);
  const mod100 = n % 100;
  const suffix = mod100 >= 11 && mod100 <= 13
    ? "th" : n % 10 === 1 ? "st" : n % 10 === 2 ? "nd"
      : n % 10 === 3 ? "rd" : "th";
  return `${n}${suffix}`;
}

function formatPercentile(value: number): string {
  if (value < 5) return "<5th";
  if (value > 95) return ">95th";
  return ordinal(value);
}

function formatRange(lower: number, upper: number): string {
  return `${lower < 5 ? "<5th" : ordinal(lower)}–${
    upper > 95 ? ">95th" : ordinal(upper)}`;
}

function ProgressBar({ count, total, inline }: { count: number; total: number; inline?: boolean }) {
  const pct = Math.min(100, (count / total) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, ...(inline ? {} : { justifyContent: "center", marginTop: 6 }) }}>
      <div style={{ flex: inline ? 1 : "0 0 220px", height: 8, background: "var(--bd-subtle)" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "var(--teal)", transition: "width .2s" }} />
      </div>
      <span style={{ fontFamily: "var(--mono)", fontSize: 12, color: COLORS.textBody, fontVariantNumeric: "tabular-nums" }}>
        {count} / {total}
      </span>
    </div>
  );
}

function Ctl({ label, value, opts, onChange }: { label: string; value: string; opts: string[]; onChange: (v: string) => void }) {
  return (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: COLORS.textBody }}>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {opts.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );
}
