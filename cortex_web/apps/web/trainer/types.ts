// Shared trainer-facing types (2026-07-17): retained when the former
// browser-local learning model was removed and the server-side learning engine
// became the sole trainer. The engine adapter
// (serverSession.ts), the controller, and the reveal consume these shapes;
// the values are produced by the engine service.

// One served training item (the engine's next-question decision).
export interface Choice {
  task: number; mode: string; segId: number; s: number; sSd: number;
  yStar: number; margin: number; info: Record<string, unknown>; now: number;
}

// Per-task belief snapshot (engine coordinates).
import type { PercentileDomainScore } from "../src/percentile/types";

export interface TaskSnapshot {
  task: number;
  mastered: boolean;
  skill: number;       // posterior-mean ℓ
  theta: number;       // posterior-mean θ (criterion, engine coords)
  sd: number;          // posterior SD of ℓ
  passMass: number;    // π = P(ℓ > ℓ*)
  trainability: number | null;
  percentile?: PercentileDomainScore | null;
}
