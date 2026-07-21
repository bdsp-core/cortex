export const RANKED_SPECULATION_PROFILE = {
  maxRank: 2,
  minWorkerCount: 4,
  minOutcomeProbability: 0.10,
  expansionGraceMs: 200,
  allowMhRejuvenation: false,
} as const;

/** A rank expansion may discover that the frozen ESS threshold requires the
 * full rejuvenation path.  That work is deliberately left for the observed
 * answer: speculating through 30 MH steps can consume the entire think-time
 * and then make a different observed outcome wait for worker-pool recovery. */
export class RankedSpeculationRejuvenationDeferredError extends Error {
  constructor() {
    super("ranked speculation deferred at required rejuvenation");
    this.name = "RankedSpeculationRejuvenationDeferredError";
  }
}

export function isRankedSpeculationRejuvenationDeferred(
  error: unknown,
): error is RankedSpeculationRejuvenationDeferredError {
  return error instanceof RankedSpeculationRejuvenationDeferredError;
}

export function shouldExpandRankedSpeculation(args: {
  rank: number;
  probability: number;
  workerCount: number;
  separateBranchExecutorActive: boolean;
}): boolean {
  return !args.separateBranchExecutorActive
    && Number.isInteger(args.rank)
    && args.rank >= 2
    && args.rank <= RANKED_SPECULATION_PROFILE.maxRank
    && Number.isFinite(args.probability)
    && args.probability >= RANKED_SPECULATION_PROFILE.minOutcomeProbability
    && Number.isInteger(args.workerCount)
    && args.workerCount >= RANKED_SPECULATION_PROFILE.minWorkerCount;
}
