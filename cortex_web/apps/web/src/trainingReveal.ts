// Session-end reveal — pure computation of "today's movement" per trained
// domain from the trainer's start/end belief snapshots. Every number here is
// the filter's real posterior state (nothing is manufactured); the ceremony is
// only in WHEN the numbers are shown. UI-free so it is unit-testable;
// TrainingRunner's done screen renders the rows.
import type { TaskSnapshot } from "../trainer/session";

export interface RevealRow {
  taskK: number;
  label: string;
  items: number;                                    // items answered today
  before: { skill: number; sd: number; pass: number };
  after: { skill: number; sd: number; pass: number };
  ellStar: number;
  mastered: boolean;                                // mastered as of session end
  newlyMastered: boolean;                           // crossed into mastery today
  nearBar: boolean;                                 // 1σ interval touches ℓ*, not yet mastered
}

/** Rows for the end-of-session reveal: one per domain trained this session,
 *  newly-mastered first, then near-bar, then by items answered. */
export function buildRevealRows(
  start: TaskSnapshot[],
  end: TaskSnapshot[],
  labels: string[],
  itemsPerTask: ReadonlyMap<number, number>,
  ellStars: number[],
): RevealRow[] {
  const rows: RevealRow[] = [];
  for (const [k, items] of itemsPerTask) {
    const a = start[k];
    const b = end[k];
    if (!items || !a || !b) continue;
    const ellStar = ellStars[k] ?? 0;
    rows.push({
      taskK: k,
      label: labels[k] ?? `task ${k}`,
      items,
      before: { skill: a.skill, sd: a.sd, pass: a.passMass },
      after: { skill: b.skill, sd: b.sd, pass: b.passMass },
      ellStar,
      mastered: b.mastered,
      newlyMastered: b.mastered && !a.mastered,
      nearBar: !b.mastered && b.skill + b.sd >= ellStar,
    });
  }
  rows.sort((x, y) =>
    Number(y.newlyMastered) - Number(x.newlyMastered) ||
    Number(y.nearBar) - Number(x.nearBar) ||
    y.items - x.items ||
    x.taskK - y.taskK);
  return rows;
}
