// Server-driven trainer session (Phase L3): the learning engine on the
// backend picks every item; this class adapts its two endpoints to the
// TrainerSessionLike surface the TrainingController consumes.
//
// Prefetch protocol: submit() fires the record round trip immediately (the
// participant is reading the result reveal while it flies); whenReady()
// resolves when the next item + snapshot land — the runner awaits it before
// continue(), so next() stays synchronous. The per-trial ledger keeps its
// single writer (the runner's checkpoint outbox posting TrajPoints); this
// class never writes.
//
// Known, accepted staleness: snapshot() returns the belief BEFORE the
// answer just submitted (it refreshes when the record response lands), so
// the TrajPoint built at answer time lags one update. The server's
// snapshot in each record response is the authoritative post-answer state.
// A record failure resolves whenReady() with no next item, ending the
// sitting gracefully (progress is preserved; the engine rebuilds from the
// ledger on the next start).
import type { Choice, TaskSnapshot } from "./types";
import * as api from "../src/api";

// Choice plus the engine's link tag (binary one-vs-rest vs native n-way
// identification — Phase L4); the incumbent's local Choice never sets it.
export type EngineChoice = Choice & { link?: "binary" | "nway" };

function toChoice(it: api.EngineItem): EngineChoice {
  return { task: it.task, mode: it.mode, segId: it.segId, s: it.s,
           sSd: it.sSd, yStar: it.yStar, margin: 0, info: {}, now: 0,
           link: (it.link as "binary" | "nway") ?? "binary" };
}

export class ServerTrainerSession {
  // The reveal screen's mastery targets (TrainerSessionLike.policy).
  readonly policy: { ellStars: number[] };
  // Seed-time per-domain trainability report (R5: surfaced in the UI).
  readonly attainability?: Record<string, number>;
  private item: Choice | null;
  private snap: TaskSnapshot[];
  private mastered: boolean;
  private inflight: Promise<void> = Promise.resolve();

  private constructor(private trainingId: string,
                      first: api.EngineStepResponse,
                      ellStars: number[]) {
    this.policy = { ellStars };
    this.attainability = first.attainability;
    this.item = first.item ? toChoice(first.item) : null;
    this.snap = first.snapshot as TaskSnapshot[];
    this.mastered = first.allMastered;
  }

  static async start(trainingId: string, segIds: number[],
                     restrictTaskKs?: number[],
                     ellStars: number[] = []): Promise<ServerTrainerSession> {
    const r = await api.engineStart(trainingId, segIds, restrictTaskKs);
    return new ServerTrainerSession(trainingId, r, ellStars);
  }

  next(): Choice | null {
    return this.item;
  }

  submit(choice: Choice, y: number): void {
    this.item = null;
    this.inflight = api
      .engineRecord(this.trainingId, choice.segId, choice.task, y)
      .then((r) => {
        this.item = r.item ? toChoice(r.item) : null;
        this.snap = r.snapshot as TaskSnapshot[];
        this.mastered = r.allMastered;
      })
      .catch(() => {
        // graceful end: item stays null → the controller finishes the
        // sitting; the engine rebuilds from the ledger next time.
        this.item = null;
      });
  }

  whenReady(): Promise<void> {
    return this.inflight;
  }

  snapshot(): TaskSnapshot[] {
    return this.snap;
  }

  allMastered(): boolean {
    return this.mastered;
  }
}
