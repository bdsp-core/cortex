// G3 tolerance-parity — trainer/conventions.ts vs the Python trainer reference
// (regenerate via trainer/__testdata__/gen_trainer_reference.py). Tolerance ~1e-6
// is limited by the shared erf (Cody/NR form) that backs normCdf.
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { describe, it, expect } from 'vitest';
import * as C from './conventions';

const dir = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(
  readFileSync(join(dir, '__testdata__/trainer_reference.json'), 'utf8'),
).conventions;
const IN = ref.in;

const TOL = 1e-6;
function arrClose(got: number[], want: number[], tol = TOL) {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThan(tol);
}

describe('trainer conventions parity', () => {
  it('constants', () => {
    expect(C.LAPSE_RATE).toBe(ref.LAPSE_RATE);
    expect(Math.abs(C.WILSON_OPT_ACC - ref.WILSON_OPT_ACC)).toBeLessThan(TOL);
    expect(Math.abs(C.SKILL_MODE_MULTIPLIER - ref.SKILL_MODE_MULTIPLIER)).toBeLessThan(TOL);
  });

  it('engineToPlan / planToEngine', () => {
    const [sig, t] = C.engineToPlan(IN.theta, IN.ell);
    arrClose(sig, ref.engineToPlan.sigma);
    arrClose(t, ref.engineToPlan.t);
    const [th, el] = C.planToEngine(IN.sigma, IN.t);
    arrClose(th, ref.planToEngine.theta);
    arrClose(el, ref.planToEngine.ell);
  });

  it('observation model', () => {
    arrClose(C.pYesPlan(IN.s, IN.sigma, IN.t), ref.pYesPlan);
    arrClose(C.pYesEngine(IN.s, IN.theta, IN.ell), ref.pYesEngine);
  });

  it('difficulty placement', () => {
    arrClose(IN.accs.map((a: number) => C.difficultyMultiplier(a)), ref.difficultyMultiplier);
    arrClose(IN.ms.map((m: number) => C.accuracyAtMultiplier(m)), ref.accuracyAtMultiplier);
  });

  it('auroc + sigma*', () => {
    arrClose(C.aurocFromEll(IN.ell), ref.aurocFromEll);
    arrClose(C.aurocFromSigma(IN.sigma), ref.aurocFromSigma);
    arrClose(C.sigmaStarFromEllStar(IN.ell_star), ref.sigmaStarFromEllStar);
  });
});
