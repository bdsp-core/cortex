// Session-resume determinism guard. The SPA's mid-test resume
// (src/resume.ts + GET /api/session/active) rebuilds a crashed sitting by
// re-creating the engine with the SAME inputs (drawn pool, verbatim order),
// the SAME session identity (seed), and feeding the checkpointed picks back
// in. That only works if the engine is bit-deterministic in (inputs, seed,
// answer sequence) — which this test pins directly: an abandoned sitting's
// served-item prefix, and a replayed continuation, must exactly match an
// uninterrupted control run.

import { describe, it, expect } from "vitest";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";

const WORDS = ["seizure", "lpd", "gpd"];
const CODES = ["sz", "lpd", "gpd"];

// Small K=3 IIIC-style bank; perDomainCap below N_MIN(20) bounds the session
// deterministically (same device as termination.test.ts).
function makeBank(perClass: number, perDomainCap: number): EngineInputs {
  const segments: EngineInputs["segments"] = [];
  let id = 0;
  WORDS.forEach((word, k) => {
    for (let i = 0; i < perClass; i++) {
      const sMean = [0, 0, 0];
      sMean[k] = 1.0 + 0.1 * i;
      segments.push({
        segId: id++, patternClass: word, sMean, sSd: [0.3, 0.3, 0.3],
        fsHz: 200, nCh: 1, nSamp: 1, channelNames: ["Fp1"], eeg: "", spec: "",
      });
    }
  });
  return {
    taskCodes: CODES, taskLabels: CODES, taskPatternWords: WORDS,
    corrL: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ellStar: [0, 0, 0], segments, perDomainCap,
  };
}

interface Served { segId: number; pick: number }
interface Outcome {
  served: Served[];
  result: { nQuestions: number; verdicts: string[] } | null;
}

// Drive a session with scripted answers. `script(i, taskK)` picks the answer
// for the i-th served item; `stopAfter` abandons the sitting mid-test (the
// crash) and resolves with what was served so far.
function drive(
  inputs: EngineInputs, id: string, seed: number,
  script: (i: number, taskK: number) => number,
  stopAfter: number | null = null,
): Promise<Outcome> {
  return new Promise((resolve) => {
    const served: Served[] = [];
    const session = new WebCortexSession(inputs, id, seed, {
      onItem: (it) => {
        const i = served.length;
        if (stopAfter !== null && i >= stopAfter) {
          resolve({ served, result: null });
          return;
        }
        const pick = script(i, it.taskK);
        served.push({ segId: it.segId, pick });
        queueMicrotask(() => session.submitAnswer(pick));
      },
      onDone: (r: { nQuestions: number; verdicts: string[] }) =>
        resolve({ served, result: { nQuestions: r.nQuestions, verdicts: r.verdicts } }),
    });
    void session.run();
  });
}

describe("session resume — replay determinism", () => {
  it("an abandoned sitting's replayed picks reproduce the control run exactly", async () => {
    const PREFIX = 8;
    const script = (i: number, k: number) => (i + k) % 3;
    const inputs = makeBank(30, 6);

    // control: one uninterrupted sitting
    const control = await drive(inputs, "resume-drift", 7, script);
    expect(control.result).not.toBeNull();
    expect(control.served.length).toBeGreaterThan(PREFIX);

    // the crash: same sitting abandoned after PREFIX answers — its served
    // prefix must already match the control (what the server checkpointed)
    const crashed = await drive(inputs, "resume-drift", 7, script, PREFIX);
    expect(crashed.served).toEqual(control.served.slice(0, PREFIX));

    // the resume: fresh engine, same identity; feed the crashed sitting's
    // recorded picks first (exactly what App.tsx's ReplayDriver does), then
    // continue live — the full run must be indistinguishable from control.
    const resumed = await drive(inputs, "resume-drift", 7,
      (i, k) => (i < PREFIX ? crashed.served[i].pick : script(i, k)));
    expect(resumed.served).toEqual(control.served);
    expect(resumed.result).toEqual(control.result);
  }, 60000);
});
