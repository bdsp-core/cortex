// Web Worker entry — runs the SMC session off the main thread so the UI
// never blocks on particle-filter compute. The main thread talks to it with
// structured-clone messages.
//
// Protocol (main → worker):
//   { type: "init", inputs, sessionId, seed }      start a session
//   { type: "answer", pick }                       submit a 0-based 6-way pick
//   { type: "abort" }
// Worker → main:
//   { type: "item", trialIndex, taskK, segId }     next question is chosen
//   { type: "trial", diag }                        post-answer telemetry
//   { type: "done", result }                       session finished
//   { type: "error", message }

import { WebCortexSession, seedFromSessionId } from "./session";
import { EngineInputs } from "./types";

let session: WebCortexSession | null = null;

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      const inputs: EngineInputs = msg.inputs;
      const seed: number =
        typeof msg.seed === "number" ? msg.seed : await seedFromSessionId(msg.sessionId);
      session = new WebCortexSession(inputs, msg.sessionId, seed, {
        onItem: (item) => (self as any).postMessage({ type: "item", ...item }),
        onTrial: (diag) => (self as any).postMessage({ type: "trial", diag }),
        onDone: (result) => (self as any).postMessage({ type: "done", result }),
        // Speculative precompute ON by default in production (bit-identical to
        // inline; hides the N=1200 selection in think-time). An init message may
        // set speculative:false to fall back to inline compute.
      }, { speculative: msg.speculative ?? true });
      session.run().catch((e) =>
        (self as any).postMessage({ type: "error", message: String(e?.stack || e) }),
      );
    } else if (msg.type === "answer") {
      session?.submitAnswer(msg.pick);
    } else if (msg.type === "abort") {
      session?.abort();
    }
  } catch (e: any) {
    (self as any).postMessage({ type: "error", message: String(e?.stack || e) });
  }
};
