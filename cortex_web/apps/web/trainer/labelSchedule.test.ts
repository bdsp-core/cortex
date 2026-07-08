// Label-schedule randomization — BIT-parity vs the Python reference
// (docs/LABEL_SCHEDULE_PECR.md). Unlike the filter RNGs (xoshiro vs PCG64,
// statistically validated), the schedule stream is a splitmix64 counter
// stream and IS byte-matched cross-language: every vector here must agree
// EXACTLY (integers/uniform doubles) with trainer/label_schedule.py via
// __testdata__/trainer_reference.json (gen_trainer_reference.py).
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { describe, it, expect } from 'vitest';
import { TaskFilter, type FilterParams } from './filter';
import {
  defaultScheduleParams, flipParams, pairedBinChoice, ScheduleRng,
  seedFromString, SideScheduler, sm64,
} from './label_schedule';
import {
  TaskCandidates, TaskModePolicy, RetentionScheduler, defaultThresholds,
} from './policy';

const dir = dirname(fileURLToPath(import.meta.url));
const all = JSON.parse(
  readFileSync(join(dir, '__testdata__/trainer_reference.json'), 'utf8'),
);
const ref = all.schedule;

describe('label-schedule cross-language bit parity', () => {
  it('sm64 golden vectors (exact u64)', () => {
    for (const [xin, xout] of ref.sm64 as [string, string][]) {
      expect(sm64(BigInt(xin)).toString()).toBe(xout);
    }
  });

  it('ScheduleRng uniforms (exact doubles)', () => {
    for (const key of Object.keys(ref.uniforms)) {
      const [seed, sid] = key.split(',').map(Number);
      const r = new ScheduleRng(seed, sid);
      for (const want of ref.uniforms[key] as number[]) {
        expect(r.uniform()).toBe(want);
      }
    }
  });

  it('shipped params match the python reference', () => {
    expect(defaultScheduleParams()).toEqual(ref.params);
    expect(flipParams()).toEqual(ref.flipParams);
  });

  it('SideScheduler walk trace (proposals, deviations, state) — exact', () => {
    const sch = new SideScheduler(defaultScheduleParams(), new ScheduleRng(42, 0));
    ref.walkTrace.forEach((step: { prop: number; served: number; d: number; run: number }, i: number) => {
      const side = sch.propose();
      expect(side).toBe(step.prop);
      const served = i % 7 === 0 ? -side : side;
      expect(served).toBe(step.served);
      sch.commit(served);
      expect(sch.d).toBe(step.d);
      expect(sch.run).toBe(step.run);
    });
  });

  it('pairedBinChoice masks — exact index sets', () => {
    const bp = ref.binPool;
    for (const tgt of Object.keys(ref.binMasks)) {
      const t = Number(tgt);
      const m1 = pairedBinChoice(bp.sMean, bp.yStar, 1, t);
      const m2 = pairedBinChoice(bp.sMean, bp.yStar, 1, t, bp.sSd);
      const idx = (m: boolean[] | null) => {
        expect(m).not.toBeNull();
        const out: number[] = [];
        (m as boolean[]).forEach((v, i) => { if (v) out.push(i); });
        return out;
      };
      expect(idx(m1)).toEqual(ref.binMasks[tgt].noSd);
      expect(idx(m2)).toEqual(ref.binMasks[tgt].withSd);
      // side-blindness: want flipped ⇒ identical bin
      expect(pairedBinChoice(bp.sMean, bp.yStar, 0, t, bp.sSd)).toEqual(m2);
    }
  });

  it('randomized-flag fixed-cloud decision sequence — exact', () => {
    const p = all.policy;
    const c = p.cand;
    const cands = new TaskCandidates(0, c.seg, c.sMean, c.sSd, c.yStar,
      c.margin, c.coherent);
    const f = new TaskFilter(p.theta, p.ell, p.params as FilterParams, { w: p.w });
    const mp = new TaskModePolicy(0, p.ellStar, p.sigmaStar, defaultThresholds());
    mp.lsched = new SideScheduler(defaultScheduleParams(), new ScheduleRng(5, 0));
    mp.flip = new SideScheduler(flipParams(), new ScheduleRng(5, 1));
    const ret = new RetentionScheduler();
    ref.decisionSeqRandomized.forEach(
      (step: { mode: string; idx: number; want: number }, i: number) => {
        const [mode, est] = mp.chooseMode(f);
        const sel = mp.select(mode, est, cands, ret, i, f);
        expect(sel).not.toBeNull();
        const [idx, info] = sel as [number, Record<string, unknown>];
        mp.noteServed(mode, c.yStar[idx], info);
        expect(mode).toBe(step.mode);
        expect(idx).toBe(step.idx);
        expect(info.wantLabel).toBe(step.want);
      });
    expect((mp.lsched as SideScheduler).d).toBe(ref.finalWalk.d);
    expect((mp.lsched as SideScheduler).run).toBe(ref.finalWalk.run);
  });
});

describe('per-session seed derivation (PECR §6.4)', () => {
  it('seedFromString: FNV-1a 64 known vectors + session-distinctness', () => {
    expect(seedFromString('')).toBe(0xcbf29ce484222325n);
    expect(seedFromString('a')).toBe(0xaf63dc4c8601ec8cn);
    // distinct trainingIds ⇒ distinct schedule streams
    const a = new ScheduleRng(seedFromString('train-001'), 0);
    const b = new ScheduleRng(seedFromString('train-002'), 0);
    expect(a.uniform()).not.toBe(b.uniform());
    // deterministic: same id ⇒ same stream (auditable replay)
    const c = new ScheduleRng(seedFromString('train-001'), 0);
    expect(c.uniform()).toBe(new ScheduleRng(seedFromString('train-001'), 0).uniform());
  });
});
