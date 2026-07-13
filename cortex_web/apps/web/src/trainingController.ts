// Pure controller for the immersive training runner (no React/DOM/worker) — the
// question → result → done state machine over a synchronous TrainerSession. Each
// trial is a binary one-vs-rest decision ("is this <pattern k>?") that maps
// directly to the trainer's belief update (y = yes/no, y* = the item's label).
// The daily session is bounded to a fixed item count. Kept UI-free so it is
// unit-testable; TrainingRunner.tsx is a thin view over it.
import type { Choice } from "../trainer/policy";
import type { TaskSnapshot, TrainerSession } from "../trainer/session";

export type RunnerPhase = "question" | "result" | "done";

export interface TrialResult {
  correct: boolean;
  isTarget: boolean;      // the item's true one-vs-rest label (y* == 1)
  answeredYes: boolean;
  taskK: number;
  segId: number;
  patternLabel: string;   // the trained pattern's label, e.g. "GPD"
}

export interface TrajPoint {
  taskK: number;
  segId: number;
  ell: number;
  theta: number;
  sd: number;
  rt: number;
  seqInSession: number;
}

export const DEFAULT_SESSION_ITEMS = 120;  // daily-bounded; will tune on real data

export class TrainingController {
  phase: RunnerPhase = "question";
  item: Choice | null;
  lastResult: TrialResult | null = null;
  count = 0;
  readonly total: number;
  // Belief state at session start, for the end-of-session reveal (today's
  // movement = end snapshot minus this). Captured before any answer lands.
  readonly startSnapshot: TaskSnapshot[];
  private shownAt = 0;
  private pending: TrajPoint[] = [];
  private seq = 0;
  private perTaskCounts = new Map<number, number>();

  constructor(
    private session: TrainerSession,
    private labels: string[],
    opts: { total?: number } = {},
  ) {
    this.total = opts.total ?? DEFAULT_SESSION_ITEMS;
    this.startSnapshot = session.snapshot();
    this.item = session.next();
    if (this.item === null) this.phase = "done";
  }

  // Call when a question becomes visible, to start the reaction-time clock.
  markShown(now: number): void {
    this.shownAt = now;
  }

  // The learner's yes/no to "is this <pattern k>?". Advances to the result step.
  answer(yes: boolean, now: number): void {
    if (this.phase !== "question" || !this.item) return;
    const it = this.item;
    const y = yes ? 1 : 0;
    const correct = y === it.yStar;
    const rt = now - this.shownAt;
    this.perTaskCounts.set(it.task, (this.perTaskCounts.get(it.task) ?? 0) + 1);
    this.session.submit(it, y);
    const snap = this.session.snapshot()[it.task];
    this.pending.push({
      taskK: it.task, segId: it.segId, ell: snap.skill, theta: snap.theta,
      sd: snap.sd, rt, seqInSession: this.seq++,
    });
    this.lastResult = {
      correct, isTarget: it.yStar === 1, answeredYes: yes, taskK: it.task,
      segId: it.segId, patternLabel: this.labels[it.task] ?? `task ${it.task}`,
    };
    this.count += 1;
    this.phase = "result";
  }

  // Advance from the result step to the next item, or finish the session.
  continue(): void {
    if (this.phase !== "result") return;
    if (this.count >= this.total || this.allMastered()) {
      this.item = null;
      this.phase = "done";
      return;
    }
    const next = this.session.next();
    this.item = next;
    this.phase = next === null ? "done" : "question";
  }

  // Buffered trajectory points (the view POSTs them to /training-progress).
  drainTrajectory(): TrajPoint[] {
    const p = this.pending;
    this.pending = [];
    return p;
  }

  progress(): { count: number; total: number } {
    return { count: this.count, total: this.total };
  }

  allMastered(): boolean {
    return this.session.allMastered();
  }

  snapshot(): TaskSnapshot[] {
    return this.session.snapshot();
  }

  // Items answered per task this session (only touched tasks appear).
  itemsPerTask(): ReadonlyMap<number, number> {
    return this.perTaskCounts;
  }
}
