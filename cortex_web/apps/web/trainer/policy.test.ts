// G3 tolerance-parity — trainer/policy.ts: the deterministic scorers and the
// fixed-cloud decision logic (choose_mode + select, with the policy's own state
// evolving) vs the Python reference. The RNG-driven filter propagate is validated
// separately (filter.test.ts); the policy decisions are deterministic given a cloud.
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { describe, it, expect } from 'vitest';
import { TaskFilter, type FilterParams } from './filter';
import { expectedReward } from './reward';
import {
  TaskCandidates, TaskModePolicy, RetentionScheduler, defaultThresholds,
  expectedSkillWeight, biasProbeScore, certProbeScore, atBarAccuracy,
  biasStationarySd, derivedTStar,
} from './policy';

const dir = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(
  readFileSync(join(dir, '__testdata__/trainer_reference.json'), 'utf8'),
).policy;

const TOL = 1e-6;
function arrClose(got: number[], want: number[]) {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThan(TOL);
}

const c = ref.cand;
function cands() {
  return new TaskCandidates(0, c.seg, c.sMean, c.sSd, c.yStar, c.margin, c.coherent);
}
function filter() {
  return new TaskFilter(ref.theta, ref.ell, ref.params as FilterParams, { w: ref.w });
}

describe('trainer policy scorers parity', () => {
  it('deterministic scorers', () => {
    arrClose(expectedSkillWeight(c.sMean, c.sSd, ref.sigHat, ref.tHat, 1, undefined, 0.5), ref.expectedSkillWeight);
    arrClose(biasProbeScore(c.sMean, c.sSd, ref.sigHat, ref.tHat), ref.biasProbeScore);
    arrClose(certProbeScore(c.sMean, c.sSd, ref.sigmaStar, ref.tHat), ref.certProbeScore);
    arrClose(c.sMean.map((s: number, i: number) => atBarAccuracy(s, c.sSd[i], ref.sigmaStar, ref.tHat, c.yStar[i])), ref.atBarAccuracy);
    expect(Math.abs(biasStationarySd(0.10, 0.05, ref.sigHat) - ref.biasStationarySd)).toBeLessThan(TOL);
    expect(Math.abs(derivedTStar(0.10, 0.05, ref.sigHat) - ref.derivedTStar)).toBeLessThan(TOL);
  });

  it('bias-correction score (= expected reward with betaT=1, betaSigma=0)', () => {
    const f = filter();
    arrClose(
      expectedReward(f, f.params, c.sMean, c.yStar,
        { betaT: 1, betaSigma: 0, betaR: 0 }, { sSd: c.sSd }),
      ref.biasCorrectionScore,
    );
  });

  it('fixed-cloud decision sequence (choose_mode + select)', () => {
    const f = filter();
    const mp = new TaskModePolicy(0, ref.ellStar, ref.sigmaStar, defaultThresholds());
    const ret = new RetentionScheduler();
    for (let i = 0; i < ref.decisionSeq.length; i++) {
      const [mode, est] = mp.chooseMode(f);
      const sel = mp.select(mode, est, cands(), ret, i, f);
      expect(sel).not.toBeNull();
      const [idx] = sel as [number, Record<string, unknown>];
      mp.noteServed(mode, c.yStar[idx]);
      expect(mode).toBe(ref.decisionSeq[i].mode);
      expect(idx).toBe(ref.decisionSeq[i].idx);
    }
  });
});
