// Training/testing streak from the activity map (same data the Consistency
// heatmap renders). A "streak day" is one the participant TRAINED or TESTED,
// not merely signed in: activity level >= 2 (2 = certification test,
// 3 = training completed; 1 = sign-in only, which does NOT count).
//
// Day keys are local YYYY-MM-DD, matching the heatmap's keyOf() and the
// server, which keys activity by the user's local day via the tz offset.

export const STREAK_MIN_LEVEL = 2;   // >= this level counts (test or training)

export interface StreakInfo {
  current: number;   // consecutive qualifying days ending today or yesterday
  best: number;      // longest qualifying run anywhere in the window
  active: boolean;   // did today itself qualify (vs. still-alive-from-yesterday)
  todayDone: boolean;
}

function pad(n: number): string {
  return n < 10 ? "0" + n : "" + n;
}

function keyOf(dt: Date): string {
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
}

/** Compute the current + best training/testing streak from an activity map.
 *  `today` is injectable for tests; defaults to the local current day.
 *
 *  current: the run of consecutive qualifying days ending at the most recent
 *  qualifying day, but only if that day is today or yesterday (so an untrained
 *  today does not immediately break a live streak — the Duolingo convention);
 *  0 once a full day is missed. best: the longest run seen scanning back over
 *  `windowDays`. */
export function computeStreak(
  activity: Record<string, number>,
  today: Date = new Date(),
  windowDays = 400,
): StreakInfo {
  const base = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const qualifies = (ago: number): boolean => {
    const dt = new Date(base.getFullYear(), base.getMonth(), base.getDate() - ago);
    return (activity[keyOf(dt)] ?? 0) >= STREAK_MIN_LEVEL;
  };

  const todayDone = qualifies(0);
  // The current streak counts back from today if today qualifies, else from
  // yesterday if IT qualifies (streak still alive), else it is broken (0).
  let current = 0;
  const start = todayDone ? 0 : qualifies(1) ? 1 : -1;
  if (start >= 0) {
    for (let ago = start; ago < windowDays; ago++) {
      if (qualifies(ago)) current++;
      else break;
    }
  }

  // Best run anywhere in the window.
  let best = 0, run = 0;
  for (let ago = 0; ago < windowDays; ago++) {
    if (qualifies(ago)) { run++; if (run > best) best = run; }
    else run = 0;
  }

  return { current, best, active: todayDone, todayDone };
}
