// G3 statistical end-to-end — the full ported TS trainer stack (filter + policy +
// bank + session) runs a closed training loop against a fixed high-skill responder.
// This is not a bitwise/Python-parity check (the RNG differs by design); it proves
// the whole stack runs, picks valid items, and that the belief tracks a learner who
// is more skilled than the initial (below-cut) prior — i.e. the filter/policy work.
import { describe, it, expect } from 'vitest';
import { Rng } from '../engine/rng';
import { LAPSE_RATE } from './conventions';
import { normCdf } from '../engine/mathfns';
import { ArrayBank, TrainerSession, buildFilters, type CandidateArrays } from './session';
import type { FilterParams } from './filter';

const K = 2;
const ELL_STAR = [0.30, 0.30];
const SIGMA_STAR = ELL_STAR.map((e) => Math.exp(-e));

function makeBank(rng: Rng): CandidateArrays[] {
  const per: CandidateArrays[] = [];
  let base = 0;
  for (let k = 0; k < K; k++) {
    const n = 80;
    const seg: number[] = [];
    const sMean: number[] = [];
    const sSd: number[] = [];
    const yStar: number[] = [];
    for (let i = 0; i < n; i++) {
      const s = -1.5 + 3 * rng.random();
      seg.push(base + i);
      sMean.push(s);
      sSd.push(0.4 + 0.4 * rng.random());
      yStar.push(s > 0 ? 1 : 0);
    }
    base += n;
    per.push({ seg, sMean, sSd, yStar, margin: sMean.map(() => 1), coherent: sMean.map(() => true) });
  }
  return per;
}

// SDT responder at a FIXED true skill/criterion (plan coords), no learning — the
// belief should still track UP toward it from the below-cut prior.
function responder(rng: Rng, ellTrue: number, tTrue = 0) {
  const sigma = Math.exp(-ellTrue);
  return (s: number): number => {
    const p = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf((s - tTrue) / sigma);
    return rng.random() < p ? 1 : 0;
  };
}

describe('trainer session end-to-end', () => {
  it('runs a closed loop and the belief tracks a strong learner upward', () => {
    const rng = new Rng(42);
    const bank = new ArrayBank(makeBank(rng));
    const params: FilterParams = { alphaT: 0.1, alphaSigma: 0.15, sigmaInf: 0.4, qT: 0.05, qSigma: 0.02, rho: 0.5, rule: 'soft' };
    const clouds = Array.from({ length: K }, (_, k) => {
      const r = new Rng(100 + k);
      const N = 200;
      return {
        theta: Array.from({ length: N }, () => 0.3 * r.gaussian()),
        ell: Array.from({ length: N }, () => 0.0 + 0.3 * r.gaussian()),   // below cut
        w: new Array(N).fill(1 / N),
      };
    });
    const filters = buildFilters(clouds, params, [0.4, 0.4], ELL_STAR, { seed: 7, useMixture: false });
    const session = new TrainerSession(filters, ELL_STAR, SIGMA_STAR, bank, { seed: 7 });
    const before = session.snapshot().map((s) => s.skill);
    const answer = responder(new Rng(9), 1.2);   // true skill ℓ = 1.2, well above cut

    let n = 0;
    for (let i = 0; i < 200; i++) {
      const choice = session.next();
      if (!choice) break;
      expect(Number.isFinite(choice.s)).toBe(true);
      expect(choice.segId).toBeGreaterThanOrEqual(0);
      session.submit(choice, answer(choice.s));
      n += 1;
    }

    expect(n).toBeGreaterThan(5);                 // the loop ran
    // a strong learner should be GRADUATED (the loop ends when all tasks master).
    expect(session.allMastered()).toBe(true);
    const snap = session.snapshot();
    for (const s of snap) {
      expect(Number.isFinite(s.skill)).toBe(true);
      expect(s.skill).toBeGreaterThan(before[s.task] + 0.1);   // belief tracked up
    }
  });
});
