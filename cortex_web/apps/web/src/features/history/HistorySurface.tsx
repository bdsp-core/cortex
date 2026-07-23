import { useEffect, useState } from "react";

import * as api from "../../api";
import {
  FULL_NAMES,
  TASK_ORDER,
  VerdictChip,
  aurocOf,
  formatPercentile,
  formatPercentileRange,
  formatAssessmentDate,
  percentileOf,
  verdictPresentation,
  verdictsOf,
} from "../dashboard/verdict";
import { QuestionTable } from "./QuestionTable";

type QuestionState = api.QuestionRow[] | "loading" | "error" | undefined;

/** Certification-history feature, including its lazy participant-safe details. */
export function HistorySurface() {
  const [sessions, setSessions] = useState<api.HistorySession[] | null>(null);
  const [error, setError] = useState(false);
  const [openSessionId, setOpenSessionId] = useState<string | null>(null);
  const [questionsBySession, setQuestionsBySession] = useState<Record<string, QuestionState>>({});

  useEffect(() => {
    let live = true;
    api.getHistory()
      .then((history) => { if (live) setSessions(history.sessions); })
      .catch(() => { if (live) setError(true); });
    return () => { live = false; };
  }, []);

  const toggle = (sessionId: string) => {
    const next = openSessionId === sessionId ? null : sessionId;
    setOpenSessionId(next);
    if (next && questionsBySession[next] === undefined) {
      setQuestionsBySession((current) => ({ ...current, [next]: "loading" }));
      api.getQuestions(next)
        .then((result) => setQuestionsBySession((current) => ({
          ...current,
          [next]: result.questions,
        })))
        .catch(() => setQuestionsBySession((current) => ({
          ...current,
          [next]: "error",
        })));
    }
  };

  return (
    <section className="cx-panel" aria-label="Certification history">
      <div className="cx-phead"><h2>Certification history</h2></div>
      {error && !sessions && (
        <div className="cx-placeholder">Could not load your history. Try refreshing.</div>
      )}
      {sessions && sessions.length === 0 && (
        <div className="cx-placeholder">
          No certification attempts yet. Take the certification test to see your results here.
        </div>
      )}
      {sessions && sessions.length > 0 && (
        <div className="cx-hist">
          {sessions.map((session) => {
            const verdicts = verdictsOf(session.result);
            const isOpen = openSessionId === session.session_id;
            return (
              <div key={session.session_id} className="cx-hist-row">
                <button
                  type="button"
                  className="cx-hist-head"
                  aria-expanded={isOpen}
                  onClick={() => toggle(session.session_id)}
                >
                  <span className="date">{formatAssessmentDate(session.finished_utc)}</span>
                  <span className="qn">{session.n_questions ?? "—"} questions</span>
                  <span className="cx-hist-strip" aria-label="Per-task verdicts">
                    {verdicts.map((verdict, taskK) => {
                      const presentation = verdictPresentation(verdict);
                      return (
                        <i
                          key={taskK}
                          className={presentation.cls}
                          title={`${TASK_ORDER[taskK] ?? taskK}: ${presentation.text}`}
                        />
                      );
                    })}
                  </span>
                  <span className="cx-hist-caret">{isOpen ? "▾" : "▸"}</span>
                </button>
                {isOpen && (
                  <div className="cx-hist-body">
                    <div className="cx-hist-grid">
                      {verdicts.map((verdict, taskK) => {
                        const auroc = aurocOf(session.result, taskK);
                        const percentile = percentileOf(
                          session.result, TASK_ORDER[taskK] ?? "",
                        );
                        return (
                          <div key={taskK} className="cx-hist-cell">
                            <span className="tk">
                              {FULL_NAMES[TASK_ORDER[taskK]] ? TASK_ORDER[taskK] : `task ${taskK}`}
                              {auroc !== null && <span className="au">AUROC {auroc.toFixed(2)}</span>}
                              {percentile !== null && (
                                <span className="au">
                                  historical percentile{" "}
                                  {formatPercentile(percentile.estimate)} · 95%{" "}
                                  {formatPercentileRange(
                                    percentile.lower, percentile.upper,
                                  )}
                                </span>
                              )}
                            </span>
                            <VerdictChip verdict={verdict} />
                          </div>
                        );
                      })}
                    </div>
                    {session.result.percentile?.profile.display && (
                      <div style={{ fontSize: 12, opacity: 0.78,
                                    lineHeight: 1.5, marginTop: 10 }}>
                        <b>
                          {session.result.percentile.profile.displayCopy.label}.
                        </b>{" "}
                        {session.result.percentile.profile.displayCopy.disclosure}
                      </div>
                    )}
                    <div className="cx-q-section">
                      <div className="cx-q-title">Per-question breakdown</div>
                      {questionsBySession[session.session_id] === "loading" && (
                        <div className="cx-placeholder">Loading questions…</div>
                      )}
                      {questionsBySession[session.session_id] === "error" && (
                        <div className="cx-placeholder">Could not load the question breakdown.</div>
                      )}
                      {Array.isArray(questionsBySession[session.session_id]) && (
                        <QuestionTable rows={questionsBySession[session.session_id] as api.QuestionRow[]} />
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
