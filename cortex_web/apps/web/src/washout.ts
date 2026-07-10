// Human label for when the post-training exam washout ends (see
// routers/testing.py WASHOUT_HOURS): "6:30 PM", "6:30 AM tomorrow", or
// "6:30 PM on 7/14/2026". Shared by the acknowledgment modal and the
// disabled test button's note so the two can never disagree.

export function reopenLabel(iso: string, now: Date = new Date()): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "later";
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (d.toDateString() === now.toDateString()) return time;
  if (d.toDateString() === new Date(now.getTime() + 86_400_000).toDateString()) {
    return `${time} tomorrow`;
  }
  return `${time} on ${d.toLocaleDateString()}`;
}

/** Still in force? (Client-side convenience; the server is the enforcer.) */
export function washoutActive(iso: string | null | undefined,
                              now: Date = new Date()): boolean {
  if (!iso) return false;
  const d = new Date(iso);
  return !isNaN(d.getTime()) && d.getTime() > now.getTime();
}
