import type { AdvanceParams, AdvanceResult, SessionCore } from "./advance";
import type { Chosen } from "./choose_item";

/** Scheduling boundary used by WebCortexSession; contains no browser globals. */
export interface BranchExecutor {
  /** Maximum simultaneously runnable immutable outcome snapshots. */
  readonly capacity: number;
  advance(
    core: SessionCore,
    chosen: Chosen,
    params: AdvanceParams,
    trialIndex: number,
    pick: number,
  ): Promise<AdvanceResult>;
  dispose(): void;
}
