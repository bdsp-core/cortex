import type { AdvanceParams, AdvanceResult, SessionCore } from "./advance";
import type { Chosen } from "./choose_item";

/** Scheduling boundary used by WebCortexSession; contains no browser globals. */
export interface BranchExecutor {
  advance(
    core: SessionCore,
    chosen: Chosen,
    params: AdvanceParams,
    trialIndex: number,
    y: 0 | 1,
  ): Promise<AdvanceResult>;
  dispose(): void;
}
