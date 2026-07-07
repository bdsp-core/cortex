// Web Worker entry for the trainer (G3) — runs the ported filter + policy off the
// main thread so per-response compute never blocks the UI. Mirrors the exam
// engine's worker.ts message pattern.
//
// Protocol (main → worker):
//   { type: "init", clouds, params, sigmaInf, ellStars, sigmaStars, bank,
//     seed?, useMixture?, exactKernel?, minMargin?, maxConsec? }   start a session
//   { type: "next" }                    request the next training item
//   { type: "submit", choice, y }       submit the learner's response
// Worker → main:
//   { type: "item", choice } | { type: "done" }      after "next"
//   { type: "state", snapshot }                      after "submit"
//   { type: "error", message }
import { ArrayBank, TrainerSession, buildFilters } from './session';
import type { FilterParams } from './filter';

let session: TrainerSession | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ctx: any = self as any;

ctx.onmessage = (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const params = msg.params as FilterParams;
      const filters = buildFilters(msg.clouds, params, msg.sigmaInf, msg.ellStars, {
        seed: msg.seed ?? 0, useMixture: msg.useMixture ?? true, exactKernel: msg.exactKernel ?? true,
      });
      const bank = new ArrayBank(msg.bank);
      session = new TrainerSession(filters, msg.ellStars, msg.sigmaStars, bank, {
        seed: msg.seed ?? 0, maxConsec: msg.maxConsec ?? 5, minMargin: msg.minMargin ?? 0.30,
      });
    } else if (msg.type === 'next') {
      const choice = session?.next() ?? null;
      ctx.postMessage(choice ? { type: 'item', choice } : { type: 'done' });
    } else if (msg.type === 'submit') {
      session?.submit(msg.choice, msg.y);
      ctx.postMessage({ type: 'state', snapshot: session?.snapshot() });
    }
  } catch (e) {
    ctx.postMessage({ type: 'error', message: String((e as Error)?.stack || e) });
  }
};
