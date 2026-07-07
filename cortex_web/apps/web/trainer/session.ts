// Trainer runtime session (G3) — the browser-side wrapper that ties the ported
// filter / trainability / policy / bank together and exposes a pull-based
// next()/submit() API for the UI (via worker.ts). Filters are built from the exam
// posterior's per-task marginals (variance-inflated by the caller, D1).
import { TaskFilter, type FilterParams } from './filter';
import { SigmaInfMixtureFilter } from './trainability';
import {
  TaskCandidates, TrainerPolicy, type Bank, type Choice, type FilterLike,
  type ModeThresholds,
} from './policy';

export interface Cloud { theta: number[]; ell: number[]; w: number[]; }

export interface CandidateArrays {
  seg: number[]; sMean: number[]; sSd: number[];
  yStar: number[]; margin: number[]; coherent: boolean[];
}

// In-memory bank over per-task candidate arrays (feedback-safe + exclusion
// filtering), mirroring the Python bank_adapter.candidates contract. The main
// thread ships these arrays to the worker.
export class ArrayBank implements Bank {
  private pools: TaskCandidates[];
  constructor(perTask: CandidateArrays[]) {
    this.pools = perTask.map((p, k) => new TaskCandidates(
      k, p.seg, p.sMean, p.sSd, p.yStar, p.margin, p.coherent));
  }

  candidates(task: number, opts: { excludeSegIds?: Set<number>; feedbackSafe?: boolean; minMargin?: number }): TaskCandidates {
    const pool = this.pools[task];
    const minMargin = opts.minMargin ?? 0.30;
    const mask = pool.segId.map((sid, i) => {
      let ok = !(opts.excludeSegIds && opts.excludeSegIds.has(sid));
      if (ok && opts.feedbackSafe) ok = pool.coherent[i] && pool.margin[i] >= minMargin;
      return ok;
    });
    return pool.subset(mask);
  }
}

export interface TaskSnapshot {
  task: number;
  mastered: boolean;
  skill: number;       // posterior-mean ℓ
  theta: number;       // posterior-mean θ (criterion, engine coords)
  sd: number;          // posterior SD of ℓ
  passMass: number;    // π = P(ℓ > ℓ*)
  trainability: number | null;
}

// Build one filter per task. `useMixture` gives the σ_∞ BMA (honest trainability
// + the D-INT-3 plateau signal); a plain TaskFilter is lighter.
export function buildFilters(
  clouds: Cloud[], params: FilterParams, sigmaInf: number[],
  ellStars: number[],
  opts: { seed?: number; useMixture?: boolean; exactKernel?: boolean } = {},
): FilterLike[] {
  const seed = opts.seed ?? 0;
  return clouds.map((c, k) => {
    const pk: FilterParams = { ...params, sigmaInf: sigmaInf[k] };
    if (opts.useMixture) {
      return new SigmaInfMixtureFilter(c.theta, c.ell, pk, {
        ellInfMean: -Math.log(sigmaInf[k]), ellStar: ellStars[k],
        seed: seed + 17 * k, w: c.w, exactKernel: opts.exactKernel,
      });
    }
    return new TaskFilter(c.theta, c.ell, pk, { w: c.w, seed: seed + 17 * k, exactKernel: opts.exactKernel });
  });
}

export class TrainerSession {
  readonly policy: TrainerPolicy;

  constructor(filters: FilterLike[], ellStars: number[], sigmaStars: number[],
    bank: Bank, opts: { thresholds?: ModeThresholds; seed?: number; maxConsec?: number;
      excludeSegIds?: Set<number>; minMargin?: number } = {}) {
    this.policy = new TrainerPolicy(filters, ellStars, sigmaStars, bank, opts);
  }

  next(now?: number): Choice | null {
    return this.policy.step(now);
  }

  submit(choice: Choice, y: number, feedback = true): void {
    this.policy.record(choice, y, feedback);
  }

  allMastered(): boolean {
    return this.policy.allMastered();
  }

  snapshot(): TaskSnapshot[] {
    return this.policy.filters.map((f, k) => {
      const [pi] = f.passMass(this.policy.ellStars[k]);
      const [mt, ml] = f.mean();
      const trainability = f instanceof SigmaInfMixtureFilter
        ? f.trainability(this.policy.ellStars[k]) : null;
      return {
        task: k,
        mastered: this.policy.modePolicies[k].isMastered(f),
        skill: ml,
        theta: mt,
        sd: f.sd()[1],
        passMass: pi,
        trainability,
      };
    });
  }
}
