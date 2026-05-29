// Participant-facing progress: question counter + a confidence readout for
// how close the adaptive test is to a definite result.
//
// π_k = P(ℓ_k > ℓ*_k | data) is the engine's MAINTAINED posterior probability
// that the rater passes task k. So max(π_k, 1−π_k) is our current confidence
// in that task's *eventual* direction (toward PASS or toward FAIL) — and it's
// symmetric in π, so it never reveals WHICH way, only how decided we are.
//
// The test stops only when EVERY task has resolved (π crossed 1−α or α), so
// the binding quantity is the least-confident / furthest-from-resolved task:
//
//     resolutionConfidence = min_k max(π_k, 1−π_k)
//
// This is read straight off the posterior (NOT a heuristic). It rises toward
// 1 as the hardest task becomes decisive.
//
// Future (per the lab's intent): show the aggregate probability of eventually
// passing-or-failing (whichever is larger) across all tasks, ideally from a
// forward Monte-Carlo rollout that also accounts for the remaining question
// budget. This min-over-tasks readout is the agreed interim.

import { TrialDiag } from "../engine/types";

export interface Progress {
  answered: number;
  maxQ: number;
  resolveConf: number | null; // null until the first trial diagnostic
}

export function resolutionConfidence(diag: TrialDiag): number {
  let minConf = 1;
  for (let k = 0; k < diag.pi.length; k++) {
    const conf = Math.max(diag.pi[k], 1 - diag.pi[k]);
    if (conf < minConf) minConf = conf;
  }
  return minConf;
}
