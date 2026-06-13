// Helpers that map between the bundle's K-indexed task list and the UI's
// per-test-class screens (the 6-button IIIC viewer; the binary spike screen).
// The engine speaks task indices 0..K-1; the UI speaks "the IIIC button at
// position N." These helpers bridge the two so K=6 and K=7 bundles both work.

import { EngineInputs } from "../engine/types";

export interface TaskInfo {
  /** Engine task index (0..K-1). The pick we submit must equal the engine's
   * `chosen.k` for y=1, so the button must carry the real index, not the
   * 0-based position in the IIIC subset. */
  idx: number;
  code: string;   // "spike" | "sz" | "lpd" | ...
  label: string;  // "Spike" | "Seizure" | ...
}

/** IIIC tasks in bundle order, with their real engine indices. At K=7 the
 *  spike task at index 0 is skipped (it has its own screen); pre-K=7 bundles
 *  (no taskClasses field) are treated as fully IIIC. */
export function iiicTasks(inputs: EngineInputs): TaskInfo[] {
  const tc = inputs.taskClasses;
  const out: TaskInfo[] = [];
  for (let i = 0; i < inputs.taskCodes.length; i++) {
    if (tc && tc[i] !== "iiic") continue;
    out.push({ idx: i, code: inputs.taskCodes[i], label: inputs.taskLabels[i] });
  }
  return out;
}

/** Spike task descriptor for K=7 bundles, else null. Used by the SpikeViewer
 *  (step 2) to know which engine index to submit Yes/No against. */
export function spikeTask(inputs: EngineInputs): TaskInfo | null {
  const tc = inputs.taskClasses;
  if (!tc) return null;
  for (let i = 0; i < tc.length; i++) {
    if (tc[i] === "spike") {
      return { idx: i, code: inputs.taskCodes[i], label: inputs.taskLabels[i] };
    }
  }
  return null;
}
