// Per-learner σ_∞ mixture filter — the trainability BMA (G3 TS port of
// trainer/trainability.SigmaInfMixtureFilter). A discrete Bayesian model average
// over a ceiling grid: J parallel TaskFilter strata, prior weights from a
// truncated Gaussian, posterior weights updated by each stratum's prequential
// evidence. `trainability(ℓ*) = Σ ω_j·1[grid_j > ℓ*]` = P(this learner's ceiling
// clears the cut | data), the honest recommendation statistic (D18) that also
// drives the D-INT-3 plateau exit. The grid / weights / trainability / summaries
// are DETERMINISTIC (tolerance-parity gated); `step` inherits the filters'
// RNG-driven propagate (statistical).
import { logSumExp } from '../engine/mathfns';
import { TaskFilter, type FilterParams } from './filter';

export interface MixtureOpts {
  ellInfMean: number;
  tau?: number;
  J?: number;
  span?: number;
  ellStar?: number;
  minBelowCut?: number;
  seed?: number;
  w?: number[];
  forget?: number;
  essFrac?: number;
  exactKernel?: boolean;
}

export class SigmaInfMixtureFilter {
  grid: number[];
  strata: TaskFilter[];
  logW: number[];
  private forget: number;
  readonly params: FilterParams;   // base params, for the pooled policy scorers

  constructor(theta: number[], ell: number[], params: FilterParams, opts: MixtureOpts) {
    this.params = params;
    const tau = opts.tau ?? 0.30;
    const J = opts.J ?? 7;
    const span = opts.span ?? 2.2;
    const minBelowCut = opts.minBelowCut ?? 0.15;
    const grid: number[] = [];
    for (let i = 0; i < J; i++) grid.push(opts.ellInfMean + (-span + (2 * span * i) / (J - 1)) * tau);
    if (opts.ellStar !== undefined && Math.min(...grid) > opts.ellStar - minBelowCut) {
      grid.push(opts.ellStar - minBelowCut);
      grid.sort((a, b) => a - b);
    }
    this.grid = grid;
    // truncated-Gaussian prior weights on the grid
    const lp = grid.map((g) => -0.5 * ((g - opts.ellInfMean) / tau) ** 2);
    const m = Math.max(...lp);
    const pw = lp.map((v) => Math.exp(v - m));
    const s = pw.reduce((a, b) => a + b, 0);
    const pwn = pw.map((v) => v / s);
    this.logW = pwn.map(Math.log);
    this.forget = opts.forget ?? 1.0;
    this.strata = grid.map((g, j) => new TaskFilter(
      theta, ell, { ...params, sigmaInf: Math.exp(-g) },
      { w: opts.w, seed: (opts.seed ?? 0) + 101 * j, essFrac: opts.essFrac, exactKernel: opts.exactKernel }));
  }

  weights(): number[] {
    const lse = logSumExp(this.logW);
    return this.logW.map((v) => Math.exp(v - lse));
  }

  trainability(ellStar: number): number {
    const om = this.weights();
    let acc = 0;
    for (let j = 0; j < this.grid.length; j++) if (this.grid[j] > ellStar) acc += om[j];
    return acc;
  }

  step(s: number, y: number, yStar: number, sSd = 0, feedback = true): void {
    if (this.forget < 1.0) this.logW = this.logW.map((v) => this.forget * v);
    for (let j = 0; j < this.strata.length; j++) {
      const pred = this.strata[j].reweight(s, y, sSd);
      this.strata[j].maybeResample();
      this.strata[j].propagate(s, yStar, feedback, y, sSd);
      this.logW[j] += Math.log(Math.max(pred, 1e-300));
    }
    const mx = Math.max(...this.logW);
    this.logW = this.logW.map((v) => v - mx);
  }

  // pooled cloud (policy scorers read these)
  get theta(): number[] { return this.strata.flatMap((f) => f.theta); }
  get ell(): number[] { return this.strata.flatMap((f) => f.ell); }
  get w(): number[] {
    const om = this.weights();
    return this.strata.flatMap((f, j) => f.w.map((wi) => om[j] * wi));
  }

  ess(): number {
    const w = this.w;
    return 1 / w.reduce((a, wi) => a + wi * wi, 0);
  }

  // law of total expectation / variance over strata
  mean(): [number, number] {
    const om = this.weights();
    let mt = 0;
    let ml = 0;
    this.strata.forEach((f, j) => { const [a, b] = f.mean(); mt += om[j] * a; ml += om[j] * b; });
    return [mt, ml];
  }

  sd(): [number, number] {
    const om = this.weights();
    const [mt, ml] = this.mean();
    let vt = 0;
    let vl = 0;
    this.strata.forEach((f, j) => {
      const [mtj, mlj] = f.mean();
      const [sdtj, sdlj] = f.sd();
      vt += om[j] * (sdtj * sdtj + mtj * mtj);
      vl += om[j] * (sdlj * sdlj + mlj * mlj);
    });
    return [Math.sqrt(Math.max(vt - mt * mt, 0)), Math.sqrt(Math.max(vl - ml * ml, 0))];
  }

  passMass(ellStar: number): [number, number] {
    const om = this.weights();
    let pi = 0;
    let mcse2 = 0;
    this.strata.forEach((f, j) => {
      const [pij, mcsej] = f.passMass(ellStar);
      pi += om[j] * pij;
      mcse2 += (om[j] * mcsej) ** 2;
    });
    return [pi, Math.sqrt(mcse2)];
  }

  isMastered(ellStar: number, sdFloor: number, alpha = 0.05, Z = 2.0): boolean {
    const [pi, mcse] = this.passMass(ellStar);
    const [, sdL] = this.sd();
    return sdL <= sdFloor && pi - Z * mcse >= 1 - alpha;
  }
}
