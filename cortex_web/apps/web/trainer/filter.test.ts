// G3 tolerance-parity — trainer/filter.TaskFilter deterministic pieces (reweight,
// summaries, AD6 graduation gate) vs the Python trainer reference.
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { describe, it, expect } from 'vitest';
import { TaskFilter, systematicIndices, type FilterParams } from './filter';

const dir = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(
  readFileSync(join(dir, '__testdata__/trainer_reference.json'), 'utf8'),
).filter;

const TOL = 1e-6;

describe('trainer filter deterministic parity', () => {
  it('reweight → posterior weights + predictive likelihood', () => {
    for (const c of ref.reweight) {
      const f = new TaskFilter(ref.theta0, ref.ell0, null, { w: ref.w0 });
      const tot = f.reweight(c.s, c.y, c.sSd);
      expect(Math.abs(tot - c.tot)).toBeLessThan(TOL);
      expect(f.w.length).toBe(c.wPost.length);
      for (let i = 0; i < f.w.length; i++) {
        expect(Math.abs(f.w[i] - c.wPost[i])).toBeLessThan(TOL);
      }
    }
  });

  it('mean / sd / passMass / isMastered', () => {
    const f = new TaskFilter(ref.theta0, ref.ell0, null, { w: ref.w0 });
    const [mt, ml] = f.mean();
    const [sdt, sdl] = f.sd();
    const [pi, mcse] = f.passMass(ref.summary.ellStar);
    expect(Math.abs(mt - ref.summary.meanT)).toBeLessThan(TOL);
    expect(Math.abs(ml - ref.summary.meanL)).toBeLessThan(TOL);
    expect(Math.abs(sdt - ref.summary.sdT)).toBeLessThan(TOL);
    expect(Math.abs(sdl - ref.summary.sdL)).toBeLessThan(TOL);
    expect(Math.abs(pi - ref.summary.pi)).toBeLessThan(TOL);
    expect(Math.abs(mcse - ref.summary.mcse)).toBeLessThan(TOL);
    expect(f.isMastered(ref.summary.ellStar, ref.summary.sdFloor)).toBe(
      ref.summary.isMastered,
    );
  });

  it('propagate deterministic drift (simple branch, q=0)', () => {
    const d = ref.propagateDrift;
    const f = new TaskFilter(ref.theta0, ref.ell0, d.params as FilterParams, { w: ref.w0 });
    f.propagate(d.s, d.yStar, true, d.y, 0);       // s_sd=0 → simple branch, no noise
    for (let i = 0; i < f.theta.length; i++) {
      expect(Math.abs(f.theta[i] - d.thetaPost[i])).toBeLessThan(1e-6);
      expect(Math.abs(f.ell[i] - d.ellPost[i])).toBeLessThan(1e-6);
    }
  });

  it('systematic-resample indices (deterministic given the offset)', () => {
    const r = ref.resample;
    expect(systematicIndices(r.cumw, r.u0, r.cumw.length)).toEqual(r.idx);
  });

  it('exact-kernel propagate keeps the cloud finite and shifts skill upward', () => {
    // statistical/sanity: RNG noise differs from NumPy, but the drift is correct.
    const params: FilterParams = { alphaT: 0.1, alphaSigma: 0.2, sigmaInf: 0.4, qT: 0.05, qSigma: 0.02, rho: 0.5, rule: 'soft' };
    const f = new TaskFilter(ref.theta0, ref.ell0, params, { w: ref.w0, seed: 1 });
    const before = f.mean()[1];
    for (let k = 0; k < 30; k++) f.propagate(0.9, 1, true, 1, 0.6);  // s>0, y*=1: learn
    expect(f.ell.every((v) => Number.isFinite(v))).toBe(true);
    expect(f.mean()[1]).toBeGreaterThan(before);    // skill (ℓ) rose toward the ceiling
  });
});
