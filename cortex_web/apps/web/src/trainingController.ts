// Pure controller for the immersive training runner (no React/DOM/worker) — the
// question → result → done state machine over the trainer session surface.
// Trials may be binary one-vs-rest or native n-way identification; both map
// their raw answer onto the server trainer's declared link and gold label.
// The daily session is bounded to a fixed item count. Kept UI-free so it is
// unit-testable; TrainingRunner.tsx is a thin view over it.
import type { Choice, TaskSnapshot } from "../trainer/types";

export type RunnerPhase = "question" | "result" | "done";

// The session surface the controller actually consumes. Production supplies
// ServerTrainerSession. whenReady is its prefetch barrier: it resolves when
// the next item + snapshot have landed from the server.
export interface TrainerSessionLike {
  next(now?: number): Choice | null;
  submit(choice: Choice, y: number): void;
  snapshot(): TaskSnapshot[];
  allMastered(): boolean;
  whenReady?(): Promise<void>;
  // The reveal screen reads the per-task mastery targets from here — the
  // ServerTrainerSession exposes this structurally.
  policy: { ellStars: number[] };
}

export interface TrialResult {
  correct: boolean;
  isTarget: boolean;      // the item's true one-vs-rest label (y* == 1)
  answeredYes: boolean;
  taskK: number;
  segId: number;
  patternLabel: string;   // binary: the trained pattern; n-way: the GOLD class
  nway?: boolean;         // full-identification item (Phase L4)
  pickedLabel?: string;   // n-way: the class the learner picked
}

export interface TrajPoint {
  taskK: number;
  segId: number;
  ell: number;
  theta: number;
  sd: number;
  rt: number;
  seqInSession: number;
  // Response record (Phase L2): what was answered and what the reveal
  // displays — persisted server-side into training_trials, the
  // longitudinal ledger the learning-engine dynamics refit consumes.
  pick: number;              // binary: 1/0 yes-no; n-way: 0-based TASK-axis index
  yStar: number;             // binary: one-vs-rest label; n-way: gold task index
  isCorrect: boolean;
  feedbackShown: string;     // mirrors TrainingRunner's result reveal text
  shownClientUtc: string;
  answeredClientUtc: string;
  mode: string;              // serving mode (skill/bias/review; L4)
  link: string;              // 'binary' | 'nway' — the refit's dispatch key
}

export const DEFAULT_SESSION_ITEMS = 40;   // daily-bounded; tuned down from
// 120 on first live-pilot feedback (2026-07-17): a 120-question sitting is
// daunting, and shorter sittings sample the between-session structure the
// learning model actually needs.

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
  private shownIso = "";
  private pending: TrajPoint[] = [];
  private seq = 0;
  private perTaskCounts = new Map<number, number>();

  constructor(
    private session: TrainerSessionLike,
    private labels: string[],
    opts: { total?: number } = {},
  ) {
    this.total = opts.total ?? DEFAULT_SESSION_ITEMS;
    this.startSnapshot = session.snapshot();
    this.item = session.next();
    if (this.item === null) this.phase = "done";
  }

  // Call when a question becomes visible, to start the reaction-time clock.
  // `wallIso` defaults to the real wall clock; tests may pin it.
  markShown(now: number, wallIso: string = new Date().toISOString()): void {
    this.shownAt = now;
    this.shownIso = wallIso;
  }

  // The learner's yes/no to "is this <pattern k>?". Advances to the result step.
  answer(yes: boolean, now: number,
         wallIso: string = new Date().toISOString()): void {
    if (this.phase !== "question" || !this.item) return;
    const it = this.item;
    const y = yes ? 1 : 0;
    const correct = y === it.yStar;
    const rt = now - this.shownAt;
    const label = this.labels[it.task] ?? `task ${it.task}`;
    this.perTaskCounts.set(it.task, (this.perTaskCounts.get(it.task) ?? 0) + 1);
    this.session.submit(it, y);
    const snap = this.session.snapshot()[it.task];
    this.pending.push({
      taskK: it.task, segId: it.segId, ell: snap.skill, theta: snap.theta,
      sd: snap.sd, rt, seqInSession: this.seq++,
      // Phase L2 response record. feedbackShown must mirror the reveal
      // TrainingRunner renders for phase "result" (which always follows).
      pick: y, yStar: it.yStar, isCorrect: correct,
      feedbackShown: `${correct ? "Correct" : "Incorrect"} — This ` +
        `${it.yStar === 1 ? "is" : "is not"} ${label}.`,
      shownClientUtc: this.shownIso, answeredClientUtc: wallIso,
      mode: it.mode ?? "", link: "binary",
    });
    this.lastResult = {
      correct, isTarget: it.yStar === 1, answeredYes: yes, taskK: it.task,
      segId: it.segId, patternLabel: label,
    };
    this.count += 1;
    this.phase = "result";
  }

  // Full-identification answer (Phase L4 native n-way): `pickTask` is the
  // 0-based TASK-axis index the learner chose; the item's yStar carries the
  // gold task index. Same coding as the exam's n-way picks.
  answerPick(pickTask: number, now: number,
             wallIso: string = new Date().toISOString()): void {
    if (this.phase !== "question" || !this.item) return;
    const it = this.item;
    const correct = pickTask === it.yStar;
    const rt = now - this.shownAt;
    const gold = this.labels[it.yStar] ?? `task ${it.yStar}`;
    const picked = this.labels[pickTask] ?? `task ${pickTask}`;
    this.perTaskCounts.set(it.task, (this.perTaskCounts.get(it.task) ?? 0) + 1);
    this.session.submit(it, pickTask);
    const snap = this.session.snapshot()[it.task];
    this.pending.push({
      taskK: it.task, segId: it.segId, ell: snap.skill, theta: snap.theta,
      sd: snap.sd, rt, seqInSession: this.seq++,
      pick: pickTask, yStar: it.yStar, isCorrect: correct,
      feedbackShown: correct
        ? `Correct — This is ${gold}.`
        : `Incorrect — This is ${gold}, not ${picked}.`,
      shownClientUtc: this.shownIso, answeredClientUtc: wallIso,
      mode: it.mode ?? "", link: "nway",
    });
    this.lastResult = {
      correct, isTarget: true, answeredYes: false, taskK: it.task,
      segId: it.segId, patternLabel: gold, nway: true, pickedLabel: picked,
    };
    this.count += 1;
    this.phase = "result";
  }

  // Engine mode (Phase L3): resolves when a server-driven session has the
  // next item ready; local synchronous sessions resolve immediately. The
  // runner awaits this before continue().
  waitForNext(): Promise<void> {
    return this.session.whenReady?.() ?? Promise.resolve();
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
