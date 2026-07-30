export interface RankedOutcome {
  outcome: number;
  probability: number;
  rank: number;
}

export interface SpeculationPlan {
  coordinator: RankedOutcome;
  helper: RankedOutcome | null;
  deferred: RankedOutcome[];
  cachedProbabilityMass: number;
}

export function rankOutcomes(
  distribution: readonly { outcome: number; probability: number }[],
): RankedOutcome[] {
  const total = distribution.reduce((sum, entry) => sum + entry.probability, 0);
  if (!(total > 0) || !Number.isFinite(total)) throw new Error("invalid outcome distribution");
  return distribution.map((entry, index) => ({ ...entry, index }))
    .sort((a, b) => (b.probability - a.probability) || (a.index - b.index))
    .map((entry, rank) => ({
      outcome: entry.outcome,
      probability: entry.probability / total,
      rank: rank + 1,
    }));
}

export function planRankedSpeculation(
  distribution: readonly { outcome: number; probability: number }[],
  helperAvailable: boolean,
): SpeculationPlan {
  const ranked = rankOutcomes(distribution);
  const coordinator = ranked[0];
  const helper = helperAvailable && ranked.length > 1 ? ranked[1] : null;
  return {
    coordinator,
    helper,
    deferred: ranked.slice(helper ? 2 : 1),
    cachedProbabilityMass: coordinator.probability + (helper?.probability ?? 0),
  };
}

export async function executeRankedBranches<Core, Result>(args: {
  authoritative: Core;
  distribution: readonly { outcome: number; probability: number }[];
  actualOutcome: Promise<number>;
  cloneCore: (core: Core) => Core;
  advance: (core: Core, outcome: number) => Promise<Result>;
  helperAvailable: boolean;
}): Promise<{ outcome: number; rank: number; cacheHit: boolean; result: Result }> {
  const plan = planRankedSpeculation(args.distribution, args.helperAvailable);
  const branches = new Map<number, Promise<Result>>();
  branches.set(
    plan.coordinator.outcome,
    args.advance(args.cloneCore(args.authoritative), plan.coordinator.outcome),
  );
  if (plan.helper) {
    branches.set(
      plan.helper.outcome,
      args.advance(args.cloneCore(args.authoritative), plan.helper.outcome),
    );
  }
  const actual = await args.actualOutcome;
  const rank = rankOutcomes(args.distribution).find((entry) => entry.outcome === actual)?.rank;
  if (rank === undefined) throw new Error(`actual outcome ${actual} is outside the response model`);
  const cached = branches.get(actual);
  return {
    outcome: actual,
    rank,
    cacheHit: cached !== undefined,
    result: await (cached ?? args.advance(args.cloneCore(args.authoritative), actual)),
  };
}

