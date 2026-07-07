// Training filter — bootstrap SMC with propagation (G3 TS port of
// trainer/filter.TaskFilter). Per-task 2-D cloud over (θ, ℓ) in engine coords.
// Unlike the exam engine, the trainer filter does NOT use MH rejuvenation:
// reweight → resample (systematic, when ESS<frac·N) → propagate through the
// learning kernel T (no MH). `propagate` is the only genuinely new numeric.
//
// Parity (see conventions.ts header): the DETERMINISTIC pieces (reweight,
// summaries, the AD6 gate, the simple-branch drift) are tolerance-parity gated
// against Python; the RNG-driven noise/resample are validated statistically (the
// exam engine's Rng is xoshiro256**, not NumPy PCG64).
import { normCdf } from '../engine/mathfns';
import { Rng } from '../engine/rng';
import { LAPSE_RATE, SKILL_MODE_MULTIPLIER } from './conventions';

export type Rule = 'soft' | 'hard' | 'static';

export interface FilterParams {
  alphaT: number;
  alphaSigma: number;
  sigmaInf: number;
  qT: number;
  qSigma: number;
  rho: number;
  rule: Rule;
}

// Gauss–Hermite nodes (x) and normalized weights (w/√π). 21-node: exact
// conditional kernel; 11-node: E[w] smear (opt-in, not on the default path).
const GH21_X = [-5.550351873264678, -4.773992343411219, -4.12199554749184, -3.5319728771376777, -2.979991207704598, -2.453552124512838, -1.9449629491862537, -1.448934250650732, -0.961499634418369, -0.47945070707910753, 0.0, 0.47945070707910753, 0.961499634418369, 1.448934250650732, 1.9449629491862537, 2.453552124512838, 2.979991207704598, 3.5319728771376777, 4.12199554749184, 4.773992343411219, 5.550351873264678];
const GH21_WN = [2.098991219565662e-14, 4.975368604121714e-11, 1.4506612844930877e-08, 1.2253548361482539e-06, 4.2192347425516774e-05, 0.0007080477954815355, 0.0064396970514087855, 0.03395272978654286, 0.10839228562641945, 0.21533371569505977, 0.27026018357287707, 0.21533371569505977, 0.10839228562641945, 0.03395272978654286, 0.0064396970514087855, 0.0007080477954815355, 4.2192347425516774e-05, 1.2253548361482539e-06, 1.4506612844930877e-08, 4.975368604121714e-11, 2.098991219565662e-14];

// Systematic-resample indices for a stratified offset u0 ∈ [0,1) (a pure
// function — the deterministic half of the resample; parity-testable).
export function systematicIndices(cumw: number[], u0: number, n: number): number[] {
  const idx = new Array(n);
  for (let j = 0; j < n; j++) {
    const u = (u0 + j) / n;
    let lo = 0;
    let hi = cumw.length - 1;
    while (lo < hi) {                       // first i with cumw[i] >= u (searchsorted 'left')
      const mid = (lo + hi) >> 1;
      if (cumw[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    idx[j] = Math.min(lo, cumw.length - 1);
  }
  return idx;
}

export class TaskFilter {
  theta: number[];
  ell: number[];
  w: number[];
  readonly N: number;
  private p: FilterParams | null;
  private rng: Rng;
  private essFrac: number;
  private exactKernel: boolean;
  private nAnc: number;

  constructor(theta: number[], ell: number[], params: FilterParams | null,
    opts: { w?: number[]; seed?: number; essFrac?: number; exactKernel?: boolean } = {}) {
    this.theta = theta.slice();
    this.ell = ell.slice();
    this.N = theta.length;
    const w0 = opts.w ? opts.w.slice() : new Array(this.N).fill(1 / this.N);
    const s = w0.reduce((a, b) => a + b, 0);
    this.w = w0.map((v) => v / s);
    this.p = params;
    this.rng = new Rng(opts.seed ?? 0);
    this.essFrac = opts.essFrac ?? 0.5;
    this.exactKernel = opts.exactKernel ?? true;
    this.nAnc = this.N;
  }

  get params(): FilterParams { return this.p!; }

  // P(y=1) in engine coords with s_sd attenuation (the shared λ-lapse probit).
  pYes(s: number, sSd = 0): number[] {
    return this.ell.map((l, i) => {
      const el = Math.exp(l);
      let z = el * (s + this.theta[i]);
      if (sSd) z = z / Math.sqrt(1 + (el * sSd) ** 2);
      return LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf(z);
    });
  }

  reweight(s: number, y: number, sSd = 0): number {
    const p = this.pYes(s, sSd);
    const wNew = this.w.map((wi, i) => wi * (y === 1 ? p[i] : 1 - p[i]));
    const tot = wNew.reduce((a, b) => a + b, 0);
    this.w = tot > 0 ? wNew.map((v) => v / tot) : new Array(this.N).fill(1 / this.N);
    return tot;
  }

  // Transition kernel T per particle (mirrors learner_sim.step, no MH). Reweight
  // must have run first. `y` is the OBSERVED response (used by the exact kernel).
  propagate(s: number, yStar: number, feedback = true, y: number | null = null, sSd = 0): void {
    const p = this.p!;
    if (p.rule === 'static') return;
    const f = feedback ? 1.0 : 0.0;
    const sigma = this.ell.map((l) => Math.exp(-Math.min(Math.max(l, -40), 40)));
    const t = this.theta.map((th) => -th);

    if (this.exactKernel && sSd && y !== null) {
      // exact conditional kernel: node posterior p(s_real | θ, y) on 21 GH nodes
      const nodes = GH21_X.map((x) => s + Math.SQRT2 * sSd * x);
      for (let i = 0; i < this.N; i++) {
        let Z = 0;
        const qn = new Array(GH21_X.length);
        const pn = new Array(GH21_X.length);
        const wn = new Array(GH21_X.length);
        for (let a = 0; a < nodes.length; a++) {
          const z = (nodes[a] - t[i]) / sigma[i];
          const pNode = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf(z);
          const like = y === 1 ? pNode : 1 - pNode;
          const q = GH21_WN[a] * like;
          const dN = Math.abs(nodes[a] - t[i]) / sigma[i];
          pn[a] = pNode;
          wn[a] = Math.exp(-((dN - SKILL_MODE_MULTIPLIER) ** 2) / (2 * p.rho ** 2));
          qn[a] = q;
          Z += q;
        }
        let wWeight = 0;
        let Ew2 = 0;
        let Ep = 0;
        let Ep2 = 0;
        for (let a = 0; a < nodes.length; a++) {
          const q = Z > 0 ? qn[a] / Z : qn[a];
          wWeight += q * wn[a];
          Ew2 += q * wn[a] * wn[a];
          Ep += q * pn[a];
          Ep2 += q * pn[a] * pn[a];
        }
        const Vw = Math.max(Ew2 - wWeight * wWeight, 0);
        let delta: number;
        let Vp: number;
        if (p.rule === 'soft') {
          Vp = Math.max(Ep2 - Ep * Ep, 0);
          delta = Ep - yStar;
        } else {
          delta = (y as number) - yStar;
          Vp = 0;
        }
        const sdT = Math.sqrt(p.qT ** 2 + f * p.alphaT ** 2 * Vp);
        const tNew = t[i] + f * p.alphaT * delta + sdT * this.rng.gaussian();
        const logSig = Math.log(sigma[i]);
        const gap = logSig - Math.log(p.sigmaInf);
        const gSigma = -f * wWeight * gap;
        const sdS = Math.sqrt(p.qSigma ** 2 + f * (p.alphaSigma * gap) ** 2 * Vw);
        const logSigNew = logSig + p.alphaSigma * gSigma + sdS * this.rng.gaussian();
        this.theta[i] = -tNew;
        this.ell[i] = -logSigNew;
      }
      return;
    }

    // simple branch (s_sd=0 or exactKernel off)
    for (let i = 0; i < this.N; i++) {
      let delta: number;
      if (p.rule === 'soft') {
        const el = Math.exp(this.ell[i]);
        let z = el * (s + this.theta[i]);
        if (sSd) z = z / Math.sqrt(1 + (el * sSd) ** 2);
        delta = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf(z) - yStar;
      } else if (y !== null) {
        delta = y - yStar;
      } else {
        const el = Math.exp(this.ell[i]);
        const pi = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf(el * (s + this.theta[i]));
        delta = (this.rng.random() < pi ? 1 : 0) - yStar;
      }
      const noiseT = p.qT * this.rng.gaussian();
      const tNew = t[i] + f * p.alphaT * delta + noiseT;
      const d = Math.abs(s - t[i]) / sigma[i];
      const wWeight = Math.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2) / (2 * p.rho ** 2));
      const logSig = Math.log(sigma[i]);
      const gSigma = -f * wWeight * (logSig - Math.log(p.sigmaInf));
      const noiseS = p.qSigma * this.rng.gaussian();
      this.theta[i] = -tNew;
      this.ell[i] = -(logSig + p.alphaSigma * gSigma + noiseS);
    }
  }

  maybeResample(): boolean {
    if (this.ess() >= this.essFrac * this.N) return false;
    const cumw = new Array(this.N);
    let acc = 0;
    for (let i = 0; i < this.N; i++) { acc += this.w[i]; cumw[i] = acc; }
    const idx = systematicIndices(cumw, this.rng.random(), this.N);
    this.theta = idx.map((j) => this.theta[j]);
    this.ell = idx.map((j) => this.ell[j]);
    this.w = new Array(this.N).fill(1 / this.N);
    this.nAnc = new Set(idx).size;
    return true;
  }

  // One training trial: reweight → resample if degenerate → propagate.
  step(s: number, y: number, yStar: number, sSd = 0, feedback = true): void {
    this.reweight(s, y, sSd);
    this.maybeResample();
    this.propagate(s, yStar, feedback, y, sSd);
  }

  ess(): number {
    return 1 / this.w.reduce((a, wi) => a + wi * wi, 0);
  }

  mean(): [number, number] {
    let mt = 0;
    let ml = 0;
    for (let i = 0; i < this.N; i++) { mt += this.w[i] * this.theta[i]; ml += this.w[i] * this.ell[i]; }
    return [mt, ml];
  }

  sd(): [number, number] {
    const [mt, ml] = this.mean();
    let vt = 0;
    let vl = 0;
    for (let i = 0; i < this.N; i++) {
      vt += this.w[i] * (this.theta[i] - mt) ** 2;
      vl += this.w[i] * (this.ell[i] - ml) ** 2;
    }
    return [Math.sqrt(vt), Math.sqrt(vl)];
  }

  passMass(ellStar: number): [number, number] {
    let pi = 0;
    for (let i = 0; i < this.N; i++) if (this.ell[i] > ellStar) pi += this.w[i];
    const nEff = Math.min(this.ess(), this.nAnc);
    return [pi, Math.sqrt(Math.max(pi * (1 - pi), 0) / Math.max(nEff, 1))];
  }

  isMastered(ellStar: number, sdFloor: number, alpha = 0.05, Z = 2.0): boolean {
    const [pi, mcse] = this.passMass(ellStar);
    const [, sdEll] = this.sd();
    return sdEll <= sdFloor && pi - Z * mcse >= 1 - alpha;
  }
}
