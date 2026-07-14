import { describe, expect, it } from "vitest";
import { computeStreak } from "./streak";

// Fixed local "today" for determinism.
const TODAY = new Date(2026, 6, 14);   // 2026-07-14 (month is 0-based)
const pad = (n: number) => (n < 10 ? "0" + n : "" + n);
const key = (ago: number) => {
  const d = new Date(2026, 6, 14 - ago);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};
// activity map from {daysAgo: level}
const mk = (m: Record<number, number>): Record<string, number> => {
  const out: Record<string, number> = {};
  for (const [ago, lvl] of Object.entries(m)) out[key(Number(ago))] = lvl;
  return out;
};

describe("computeStreak", () => {
  it("counts consecutive train/test days ending today", () => {
    const s = computeStreak(mk({ 0: 3, 1: 2, 2: 3 }), TODAY);
    expect(s.current).toBe(3);
    expect(s.active).toBe(true);
    expect(s.todayDone).toBe(true);
  });

  it("does NOT count sign-in-only days (level 1)", () => {
    // today signed in only; yesterday + before trained → streak is alive from
    // yesterday, today does not add to it
    const s = computeStreak(mk({ 0: 1, 1: 3, 2: 3 }), TODAY);
    expect(s.current).toBe(2);
    expect(s.active).toBe(false);   // today itself did not qualify
    expect(s.todayDone).toBe(false);
  });

  it("keeps a streak alive when today is not yet done (counts from yesterday)", () => {
    const s = computeStreak(mk({ 1: 3, 2: 2, 3: 3 }), TODAY);
    expect(s.current).toBe(3);
    expect(s.active).toBe(false);
  });

  it("breaks the streak once a full day is missed", () => {
    // today none, yesterday none, then a run → current is 0
    const s = computeStreak(mk({ 2: 3, 3: 3, 4: 3 }), TODAY);
    expect(s.current).toBe(0);
  });

  it("today-only is a 1-day streak", () => {
    const s = computeStreak(mk({ 0: 2 }), TODAY);
    expect(s.current).toBe(1);
    expect(s.active).toBe(true);
  });

  it("reports best run independent of the current streak", () => {
    // a 4-day run a while back; current run is 1 (today)
    const s = computeStreak(mk({ 0: 3, 5: 3, 6: 3, 7: 2, 8: 3 }), TODAY);
    expect(s.current).toBe(1);
    expect(s.best).toBe(4);
  });

  it("empty activity → no streak", () => {
    const s = computeStreak({}, TODAY);
    expect(s).toEqual({ current: 0, best: 0, active: false, todayDone: false });
  });

  it("a lone sign-in never forms a streak", () => {
    const s = computeStreak(mk({ 0: 1, 1: 1, 2: 1 }), TODAY);
    expect(s.current).toBe(0);
    expect(s.best).toBe(0);
  });
});
