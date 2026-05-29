// Participant-facing progress: question counter + an ESTIMATED probability of
// the adaptive test reaching a conclusion (all 6 tasks resolved → it stops
// early) rather than running out the question budget.
//
// Deliberately AGGREGATE-ONLY and symmetric in π (uses |π−0.5|), so it never
// reveals whether any task is heading toward PASS vs FAIL — only how likely
// the test is to terminate with a definitive result.
//
// This is a heuristic snapshot estimate, not a calibrated forecast: it
// compares the remaining question budget to a rough "questions still needed
// to resolve each pending task." The rigorous version is a forward
// Monte-Carlo rollout in the worker (STATUS.md roadmap) — a clean upgrade
// since the engine already supports simulated sessions.

import { TrialDiag } from "../engine/types";
import { DEFAULT_N_MIN, DEFAULT_R_STAR, DEFAULT_ALPHA, VERDICT } from "../engine/policy";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

// ~questions for an undecided task to become decisive / open its info gate.
const Q_TO_DECIDE = 14;

export interface Progress {
  answered: number;
  maxQ: number;
  finishProb: number | null; // null until the first trial diagnostic
}

export function estimateFinishProbability(diag: TrialDiag, maxQ: number): number {
  const K = diag.verdicts.length;
  const answered = diag.nPerTask.reduce((a, b) => a + b, 0);
  const budget = Math.max(0, maxQ - answered);

  let totalNeed = 0;
  for (let k = 0; k < K; k++) {
    if (diag.verdicts[k] !== VERDICT.PENDING) continue; // resolved ⇒ no need
    // (a) reach the minimum question count on this task
    const countNeed = Math.max(0, DEFAULT_N_MIN - diag.nPerTask[k]);
    // (b) open the information gate (R grows with questions)
    const infoNeed =
      diag.R[k] >= DEFAULT_R_STAR
        ? 0
        : Q_TO_DECIDE * (1 - clamp01(diag.R[k] / DEFAULT_R_STAR));
    // (c) become decisive — π far enough from the undecided band edge
    const decisive = clamp01(Math.abs(diag.pi[k] - 0.5) / (0.5 - DEFAULT_ALPHA));
    const decisionNeed = Q_TO_DECIDE * (1 - decisive);
    totalNeed += Math.max(countNeed, infoNeed, decisionNeed);
  }
  if (totalNeed === 0) return 1;

  // Logistic in (budget − need): ≈0.5 when budget just covers the need,
  // → 1 with plenty to spare, → 0 when short.
  const x = (budget - totalNeed) / Math.max(10, 0.6 * totalNeed);
  return clamp01(1 / (1 + Math.exp(-x)));
}
