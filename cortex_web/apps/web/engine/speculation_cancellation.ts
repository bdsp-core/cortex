export type SpeculationCancellationPhase =
  | "between_phases"
  | "mh_history"
  | "screen"
  | "exact_refinement";

export class SpeculationCancelledError extends Error {
  constructor() {
    super("n-way speculative branch cancelled for observed-answer priority");
    this.name = "SpeculationCancelledError";
  }
}

export function isSpeculationCancelled(error: unknown): boolean {
  return error instanceof SpeculationCancelledError;
}

export function throwIfSpeculationCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw new SpeculationCancelledError();
}

/** Worker messages are tasks, while nested-worker completions resume through
 * microtasks. Yielding here lets an already-submitted answer set the abort
 * signal before the next exact phase begins. */
export async function speculationCancellationCheckpoint(
  signal?: AbortSignal,
): Promise<void> {
  if (!signal) return;
  throwIfSpeculationCancelled(signal);
  await new Promise<void>((resolve) => globalThis.setTimeout(resolve, 0));
  throwIfSpeculationCancelled(signal);
}
