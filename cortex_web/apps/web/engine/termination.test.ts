// Adaptive-termination guard (v1.6). The per-domain cap must bound a session
// that AD6 cannot resolve — REFERring stuck domains and ending well before the
// pool is swept, instead of the old fixed 500-question exhaustion.
//
// We make capping DETERMINISTIC (not dependent on real verdict tuning, which the
// CI suite deliberately avoids on synthetic banks) by setting perDomainCap BELOW
// the policy's N_MIN(20): a task can never reach resolution-eligibility, so every
// task is guaranteed to cap → REFER.

import { describe, it, expect } from "vitest";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";

const WORDS = ["seizure", "lpd", "gpd"];
const CODES = ["sz", "lpd", "gpd"];

// A small K=3 IIIC-style bank (no spike phase): `perClass` segs per task, each
// informative on its OWN task. Exact verdicts are irrelevant here.
function makeBank(perClass: number, perDomainCap: number): EngineInputs {
  const segments: EngineInputs["segments"] = [];
  let id = 0;
  WORDS.forEach((word, k) => {
    for (let i = 0; i < perClass; i++) {
      const sMean = [0, 0, 0];
      sMean[k] = 1.0 + 0.1 * i; // informative on the true task
      segments.push({
        segId: id++,
        patternClass: word,
        sMean,
        sSd: [0.3, 0.3, 0.3],
        fsHz: 200,
        nCh: 1,
        nSamp: 1,
        channelNames: ["Fp1"],
        eeg: "",
        spec: "",
      });
    }
  });
  return {
    taskCodes: CODES,
    taskLabels: CODES,
    taskPatternWords: WORDS,
    corrL: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ellStar: [0, 0, 0],
    segments,
    perDomainCap,
  };
}

// Drive a full session to completion, answering every item (the answer value is
// irrelevant to the capping mechanism since cap < N_MIN blocks all resolution).
function runToCompletion(
  inputs: EngineInputs,
  pick: (taskK: number) => number,
): Promise<{ nQuestions: number; stopReason: string; verdicts: string[]; nPerTask: number[] }> {
  return new Promise((resolve) => {
    const session = new WebCortexSession(inputs, "term-test", 7, {
      onItem: (it) => queueMicrotask(() => session.submitAnswer(pick(it.taskK))),
      onDone: (r) => {
        const last = r.trials[r.trials.length - 1];
        resolve({
          nQuestions: r.nQuestions,
          stopReason: r.stopReason,
          verdicts: r.verdicts,
          nPerTask: last?.nPerTask ?? [],
        });
      },
    });
    void session.run();
  });
}

describe("adaptive termination — per-domain cap", () => {
  it("caps every unresolvable domain and ends well before sweeping the pool", async () => {
    const K = 3;
    const cap = 5; // < N_MIN(20) → no task can resolve → all cap
    const perClass = 12; // pool (36) ≫ K×cap (15), so termination is the cap, not exhaustion
    const inputs = makeBank(perClass, cap);

    // Always answer 'wrong' (pick a different task than asked) so nothing resolves.
    const r = await runToCompletion(inputs, (taskK) => (taskK + 1) % K);

    // Ended on the adaptive resolved-or-capped condition, NOT a fixed sweep or
    // pool exhaustion.
    expect(r.stopReason).toBe("resolved_or_referred");
    // Bounded by K×cap = 15, far below the 36-seg pool (and the old 500 sweep).
    expect(r.nQuestions).toBeLessThanOrEqual(K * cap);
    expect(r.nQuestions).toBeGreaterThan(0);
    // No task was asked beyond its budget.
    for (const n of r.nPerTask) expect(n).toBeLessThanOrEqual(cap);
    // Every task REFERred (no leftover PENDING, no PASS/FAIL since cap < N_MIN).
    for (const v of r.verdicts) expect(v.startsWith("REFER")).toBe(true);
  });

  it("does not run to the legacy fixed-count sweep on a large pool", async () => {
    // With a 300-seg pool a legacy fixed sweep would ask far more; the cap keeps
    // it to K×cap regardless of pool size.
    const K = 3;
    const cap = 4;
    const inputs = makeBank(100, cap); // 300-seg pool
    const r = await runToCompletion(inputs, (taskK) => (taskK + 1) % K);
    expect(r.nQuestions).toBeLessThanOrEqual(K * cap); // ≤ 12, not ~300
  });
});
