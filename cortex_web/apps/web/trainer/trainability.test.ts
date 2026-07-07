// G3 tolerance-parity — trainer/trainability.SigmaInfMixtureFilter (grid, prior
// weights, the trainability statistic, and the mixture summaries on the initial
// state) vs the Python reference. The RNG-driven `step` is validated statistically
// (via the filter's exact-kernel test).
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { describe, it, expect } from 'vitest';
import { SigmaInfMixtureFilter } from './trainability';
import type { FilterParams } from './filter';

const dir = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(
  readFileSync(join(dir, '__testdata__/trainer_reference.json'), 'utf8'),
).trainability;

const TOL = 1e-6;
function arrClose(got: number[], want: number[]) {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThan(TOL);
}

function build() {
  return new SigmaInfMixtureFilter(ref.theta0, ref.ell0, ref.params as FilterParams, {
    ellInfMean: ref.ellInfMean, tau: ref.tau, J: ref.J, ellStar: ref.ellStar,
    seed: ref.seed, w: ref.w0,
  });
}

describe('trainability mixture parity', () => {
  it('ceiling grid + prior weights', () => {
    const mf = build();
    arrClose(mf.grid, ref.grid);
    arrClose(mf.weights(), ref.weights);
  });

  it('trainability statistic + mixture summaries', () => {
    const mf = build();
    expect(Math.abs(mf.trainability(ref.ellStar) - ref.trainability)).toBeLessThan(TOL);
    const [mt, ml] = mf.mean();
    const [sdt, sdl] = mf.sd();
    const [pi, mcse] = mf.passMass(ref.ellStar);
    expect(Math.abs(mt - ref.summary.meanT)).toBeLessThan(TOL);
    expect(Math.abs(ml - ref.summary.meanL)).toBeLessThan(TOL);
    expect(Math.abs(sdt - ref.summary.sdT)).toBeLessThan(TOL);
    expect(Math.abs(sdl - ref.summary.sdL)).toBeLessThan(TOL);
    expect(Math.abs(pi - ref.summary.pi)).toBeLessThan(TOL);
    expect(Math.abs(mcse - ref.summary.mcse)).toBeLessThan(TOL);
  });
});
