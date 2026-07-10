// Mid-test session resume (GET /api/session/active): the engine is
// deterministic given the drawn pool (verbatim order), the seed (derived
// from `web-${sampleSeed}`), and the answer sequence — so feeding the logged
// picks back through a fresh engine reconstructs the exact pre-crash state.
// ReplayDriver is the pure decision core App.tsx consults on every engine
// "item" event while a replay is pending.

export interface ReplayTrial {
  trialIndex: number;
  segId: number;
  pick: number;
}

export type ReplayStep =
  | { kind: "answer"; pick: number }  // replaying: feed this recorded pick
  | { kind: "mismatch" }              // engine served a different item than logged
  | { kind: "live" };                 // replay done: this item is for the user

export class ReplayDriver {
  private queue: ReplayTrial[];

  constructor(trials: ReplayTrial[]) {
    this.queue = [...trials];
  }

  /** Trials still to replay (0 = live). */
  get remaining(): number {
    return this.queue.length;
  }

  /** Decide what to do with an engine-served item. A mismatch means the
   *  replay no longer reproduces the recorded sequence (bank or engine
   *  drift) — the caller must abandon the resume and start fresh. */
  next(segId: number): ReplayStep {
    const expected = this.queue.shift();
    if (expected === undefined) return { kind: "live" };
    if (expected.segId !== segId) return { kind: "mismatch" };
    return { kind: "answer", pick: expected.pick };
  }
}
