// Tier-2 mode-conditional policy + cross-task scheduler (G3 TS port of the
// DEFAULT/shipped path of trainer/policy.py). Three within-task modes
// (bias-correction / skill-building / retention), a deficiency-weighted worst-first
// interleave with a consecutive-same-task cap, and AD6-style graduation on the
// filtered posterior. The Python module's opt-in ablation branches
// (progress_placement, rollout, finish_first, probe_every, terminal_confirmation,
// p_static/jump/smear, skill_sigma_z) are default-off and are NOT ported.
//
// Parity: the decision logic (choose_mode / is_mastered / select item / scheduler
// pick) is DETERMINISTIC given the filter cloud, and is fixed-cloud tolerance-parity
// gated (policy.test.ts). The RNG only enters the filter's propagate (statistical).
import { normCdf } from '../engine/mathfns';
import { Rng } from '../engine/rng';
import {
  LAPSE_RATE, SKILL_MODE_MULTIPLIER, normPdf,
} from './conventions';
import type { FilterParams } from './filter';
import {
  defaultScheduleParams, flipParams, pairedBinChoice, ScheduleRng,
  SideScheduler, type ScheduleParams,
} from './label_schedule';
import { expectedReward, type CloudView } from './reward';

export const BIAS = 'bias';
export const SKILL = 'skill';
export const RETENTION = 'retention';
export const DEFAULT_T_STAR = 0.30;

const GH9_X = [-3.1909932017815277, -2.266580584531843, -1.468553289216668, -0.7235510187528376, 0.0, 0.7235510187528376, 1.468553289216668, 2.266580584531843, 3.1909932017815277];
const GH9_WN = [2.2345844007746576e-05, 0.0027891413212317653, 0.04991640676521791, 0.2440975028949394, 0.4063492063492064, 0.2440975028949394, 0.04991640676521791, 0.0027891413212317653, 2.2345844007746576e-05];

// A filter (TaskFilter or SigmaInfMixtureFilter) as the policy consumes it.
export interface FilterLike extends CloudView {
  params: FilterParams;
  mean(): [number, number];
  sd(): [number, number];
  passMass(ellStar: number): [number, number];
  isMastered(ellStar: number, sdFloor: number, alpha?: number, Z?: number): boolean;
  step(s: number, y: number, yStar: number, sSd?: number, feedback?: boolean): void;
}

// Per-task candidate pool (the bank_adapter TaskCandidates shape).
export class TaskCandidates {
  constructor(
    public task: number,
    public segId: number[],
    public sMean: number[],
    public sSd: number[],
    public yStar: number[],
    public margin: number[],
    public coherent: boolean[],
  ) {}

  get length(): number { return this.segId.length; }

  subset(mask: boolean[]): TaskCandidates {
    const keep: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) keep.push(i);
    const pick = <T>(a: T[]) => keep.map((i) => a[i]);
    return new TaskCandidates(this.task, pick(this.segId), pick(this.sMean),
      pick(this.sSd), pick(this.yStar), pick(this.margin), pick(this.coherent));
  }
}

export interface Bank {
  candidates(task: number, opts: {
    excludeSegIds?: Set<number>; feedbackSafe?: boolean; minMargin?: number;
  }): TaskCandidates;
}

// ── scorers (deterministic) ──
export function expectedSkillWeight(s: number[], sSd: number[], sigHat: number,
  tHat: number, side = 1, m = SKILL_MODE_MULTIPLIER, rho = 0.5): number[] {
  return s.map((sv, i) => {
    const mu = (sv - tHat) / sigHat;
    const v = rho * rho + (sSd[i] / sigHat) ** 2;
    return (rho / Math.sqrt(v)) * Math.exp(-((mu - side * m) ** 2) / (2 * v));
  });
}

export function biasProbeScore(s: number[], sSd: number[], sigHat: number, tHat: number): number[] {
  return s.map((sv, i) => {
    const att = Math.sqrt(1 + (sSd[i] / sigHat) ** 2);
    const z = (sv - tHat) / sigHat / att;
    return Math.exp(-0.5 * z * z) / att;
  });
}

export function certProbeScore(s: number[], sSd: number[], sigmaStar: number,
  tHat: number, lapse = LAPSE_RATE): number[] {
  return s.map((sv, i) => {
    const att2 = 1 + (sSd[i] / sigmaStar) ** 2;
    const z = (sv - tHat) / sigmaStar / Math.sqrt(att2);
    const p = lapse + (1 - 2 * lapse) * normCdf(z);
    const dp = (1 - 2 * lapse) * normPdf(z) * Math.abs(z) / att2;
    return (dp * dp) / (p * (1 - p));
  });
}

export function atBarAccuracy(s: number, sSd: number, sigmaStar: number,
  tHat: number, yStar: number, lapse = LAPSE_RATE): number {
  const att = Math.sqrt(1 + (sSd / sigmaStar) ** 2);
  const p1 = lapse + (1 - 2 * lapse) * normCdf((s - tHat) / sigmaStar / att);
  return yStar === 1 ? p1 : 1 - p1;
}

export function biasStationarySd(alphaT: number, qT: number, sigmaHat: number,
  itemSSd = 0.3, offsetMult = SKILL_MODE_MULTIPLIER, lapse = LAPSE_RATE): number {
  const sigE = Math.sqrt(sigmaHat * sigmaHat + itemSSd * itemSSd);
  const a = offsetMult * sigmaHat;
  const kappa = Math.min(alphaT * (1 - 2 * lapse) * normPdf(a / sigE) / sigE, 0.9);
  let pBar = 0;
  let pBar2 = 0;
  for (let i = 0; i < GH9_X.length; i++) {
    const z = (a + Math.SQRT2 * itemSSd * GH9_X[i]) / sigmaHat;
    const p = lapse + (1 - 2 * lapse) * normCdf(z);
    pBar += GH9_WN[i] * p;
    pBar2 += GH9_WN[i] * p * p;
  }
  const varP = pBar2 - pBar * pBar;
  const deltaBar = 1 - pBar;
  const varStat = (qT * qT + alphaT * alphaT * varP) / (kappa * (2 - kappa))
    + (alphaT * deltaBar) ** 2 / (2 - kappa) ** 2;
  return Math.sqrt(varStat);
}

export function derivedTStar(alphaT: number, qT: number, sigmaHat: number,
  c = 2.0, itemSSd = 0.3, lapse = LAPSE_RATE, lo = 0.08, hi = 0.45): number {
  return Math.min(Math.max(c * biasStationarySd(alphaT, qT, sigmaHat, itemSSd,
    SKILL_MODE_MULTIPLIER, lapse), lo), hi);
}

function biasCorrectionScore(filt: FilterLike, cands: TaskCandidates): number[] {
  return expectedReward(filt, filt.params, cands.sMean, cands.yStar,
    { betaT: 1, betaSigma: 0, betaR: 0 }, { sSd: cands.sSd });
}

// ── retention scheduler (SM-2 default) ──
export class RetentionScheduler {
  private next = new Map<string, number>();
  private ivl = new Map<string, number>();
  constructor(private ease = 2.0, private first = 5.0) {}

  register(binKey: string, now: number): void {
    if (!this.next.has(binKey)) {
      this.ivl.set(binKey, this.first);
      this.next.set(binKey, now + this.first);
    }
  }
  due(binKey: string, now: number): boolean {
    return this.next.has(binKey) && now >= (this.next.get(binKey) as number);
  }
  dueBins(now: number): string[] {
    return [...this.next.keys()].filter((b) => now >= (this.next.get(b) as number));
  }
  updateOnRetrieval(binKey: string, correct: boolean, now: number): void {
    if (!this.ivl.has(binKey)) this.register(binKey, now);
    this.ivl.set(binKey, correct ? (this.ivl.get(binKey) as number) * this.ease : this.first);
    this.next.set(binKey, now + (this.ivl.get(binKey) as number));
  }
}

export interface ModeThresholds {
  tStar: number | null;
  sdFloor: number;
  cSd: number;
  alpha: number;
  Z: number;
  rho: number;
  meanskillGate: boolean;
}

export function defaultThresholds(): ModeThresholds {
  return { tStar: null, sdFloor: 0.23, cSd: 1.0, alpha: 0.05, Z: 2.0, rho: 0.5, meanskillGate: true };
}

const MEANSKILL_W = 20;
const MEANSKILL_Z = 1.645;

// Lag-1 autocorrelation = np.corrcoef(w[:-1], w[1:])[0,1] (the mean-skill gate's
// effective-sample-size adjustment). Falls back to 0.9 on a degenerate window.
function corr1(w: number[]): number {
  const a = w.slice(0, -1);
  const b = w.slice(1);
  const ma = a.reduce((x, y) => x + y, 0) / a.length;
  const mb = b.reduce((x, y) => x + y, 0) / b.length;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < a.length; i++) {
    cov += (a[i] - ma) * (b[i] - mb);
    va += (a[i] - ma) ** 2;
    vb += (b[i] - mb) ** 2;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0.9;
}

export class TaskModePolicy {
  private biasLabelBalance = 0;
  private skillSide = 1;
  private lastSkillS: number | null = null;
  private biasProbe = false;
  private mlHist: number[] = [];
  // Label-schedule randomization (docs/LABEL_SCHEDULE_PECR.md), attached by
  // TrainerPolicy when opts.labelSchedule === 'randomized'; null (default) ⇒
  // every legacy deterministic device runs identically. lsched = the
  // per-task label walk (skill sides + bias wants — one label stream per
  // task); flip = the cap-2 probe/corrective type walk; the mirror device is
  // the STATIC pairedBinChoice (no per-trial state). All walk/flip state
  // advances at record()/noteServed via info.sched — a step() whose choice
  // is never record()ed must be abandoned, not resumed.
  lsched: SideScheduler | null = null;
  flip: SideScheduler | null = null;
  binFallbacks = 0;

  constructor(public task: number, public ellStar: number,
    public sigmaStar: number, public th: ModeThresholds) {}

  effectiveTStar(filt: FilterLike, sigHat: number): number {
    if (this.th.tStar !== null) return this.th.tStar;
    return derivedTStar(filt.params.alphaT, filt.params.qT, sigHat);
  }

  chooseMode(filt: FilterLike): [string, [number, number]] {
    const [mt, ml] = filt.mean();
    const sigHat = Math.exp(-ml);
    const tHat = -mt;
    const sdT = filt.sd()[0];
    if (this.isMastered(filt)) return [RETENTION, [sigHat, tHat]];
    const tStar = this.effectiveTStar(filt, sigHat);
    if (Math.abs(tHat) > Math.max(tStar, this.th.cSd * sdT)) return [BIAS, [sigHat, tHat]];
    return [SKILL, [sigHat, tHat]];
  }

  notePosterior(filt: FilterLike): void {
    this.mlHist.push(filt.mean()[1]);
    if (this.mlHist.length > 3 * MEANSKILL_W) this.mlHist = this.mlHist.slice(-2 * MEANSKILL_W);
  }

  private meanskillOk(filt: FilterLike): boolean {
    if (this.mlHist.length < MEANSKILL_W) return false;
    const w = this.mlHist.slice(-MEANSKILL_W);
    let r1 = corr1(w);
    r1 = Math.min(Math.max(r1, 0.0), 0.98);
    const nEff = Math.max((MEANSKILL_W * (1 - r1)) / (1 + r1), 1);
    const sdL = filt.sd()[1];
    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    return sdL <= this.th.sdFloor && mean - MEANSKILL_Z * sdL / Math.sqrt(nEff) >= this.ellStar;
  }

  isMastered(filt: FilterLike): boolean {
    const [mt, ml] = filt.mean();
    const sigHat = Math.exp(-ml);
    const tHat = -mt;
    const sdT = filt.sd()[0];
    const skillOk = filt.isMastered(this.ellStar, this.th.sdFloor, this.th.alpha, this.th.Z)
      || (this.th.meanskillGate && this.meanskillOk(filt));
    const tStar = this.effectiveTStar(filt, sigHat);
    return skillOk && Math.abs(tHat) + 0.5 * sdT <= tStar;
  }

  private argmaxScore(score: number[], cands: TaskCandidates, wantLabel: number | null): number {
    if (wantLabel !== null) {
      const sub: number[] = [];
      for (let i = 0; i < cands.length; i++) if (cands.yStar[i] === wantLabel) sub.push(i);
      if (sub.length) return sub.reduce((best, i) => (score[i] > score[best] ? i : best), sub[0]);
    }
    let best = 0;
    for (let i = 1; i < score.length; i++) if (score[i] > score[best]) best = i;
    return best;
  }

  private nearest(cands: TaskCandidates, target: number, wantLabel: number | null = null): number {
    const s = cands.sMean;
    const cand = wantLabel !== null
      ? [...s.keys()].filter((i) => cands.yStar[i] === wantLabel)
      : [...s.keys()];
    const pool = cand.length ? cand : [...s.keys()];
    return pool.reduce((best, i) => (Math.abs(s[i] - target) < Math.abs(s[best] - target) ? i : best), pool[0]);
  }

  static binOf(s: number, sigHat: number, tHat: number): number {
    return Math.round((s - tHat) / Math.max(sigHat, 1e-6));
  }

  // Returns [idx, info] or null if the pool is empty.
  select(mode: string, est: [number, number], cands: TaskCandidates,
    retention: RetentionScheduler, now: number, filt: FilterLike | null): [number, Record<string, unknown>] | null {
    if (cands.length === 0) return null;
    const [sigHat, tHat] = est;
    if (mode === BIAS) {
      let want: number;
      let probeNow: boolean;
      if (this.lsched !== null) {
        // randomized schedule: want from the per-task label walk (its soft
        // anti-imbalance tilt IS the F6b envelope, with a proven guessing
        // bound); probe/corrective from the cap-2 type walk
        want = this.lsched.propose() > 0 ? 1 : 0;
        probeNow = (this.flip as SideScheduler).propose() > 0;
      } else {
        want = this.biasLabelBalance > 0 ? 0 : 1;
        probeNow = this.biasProbe;
      }
      let idx: number;
      if (filt !== null && !probeNow) {
        idx = this.argmaxScore(biasCorrectionScore(filt, cands), cands, want);
      } else {
        idx = this.argmaxScore(biasProbeScore(cands.sMean, cands.sSd, sigHat, tHat), cands, want);
      }
      const wasProbe = probeNow || filt === null;
      if (this.flip === null) this.biasProbe = !probeNow;
      const info: Record<string, unknown> = { target: tHat, wantLabel: want, probe: wasProbe };
      if (this.lsched !== null) info.sched = true;
      return [idx, info];
    }
    if (mode === SKILL) {
      const side = this.lsched !== null ? this.lsched.propose() : this.skillSide;
      const sigPlace = sigHat;
      if (this.lsched !== null) {
        // STATIC paired-bin serving (PECR §5c): side-blind bin choice +
        // within-bin want-side E[w] argmax (the F20 precision preference —
        // dropping it was the round-3a FG defect) — served magnitudes
        // side-matched by construction, no history-dependent channel
        const want = side > 0 ? 1 : 0;
        const bmask = pairedBinChoice(
          cands.sMean, cands.yStar, want,
          SKILL_MODE_MULTIPLIER * Math.max(sigPlace, 1e-6), cands.sSd);
        let view = cands;
        let pick: number[] | null = null;
        if (bmask === null) {
          this.binFallbacks += 1;
        } else {
          pick = [];
          for (let i = 0; i < bmask.length; i++) if (bmask[i]) pick.push(i);
          view = cands.subset(bmask);
        }
        const ew = expectedSkillWeight(view.sMean, view.sSd, sigPlace, 0.0,
          side, SKILL_MODE_MULTIPLIER, this.th.rho);
        const j = this.argmaxScore(ew, view, want);
        const idx = pick !== null ? pick[j] : j;
        this.lastSkillS = cands.sMean[idx];
        return [idx, {
          target: side * SKILL_MODE_MULTIPLIER * sigPlace,
          wantLabel: want, sched: true,
        }];
      }
      let mEff = SKILL_MODE_MULTIPLIER;
      if (this.lastSkillS !== null && (this.lastSkillS > 0) !== (side > 0)) {
        mEff = Math.abs(this.lastSkillS) / Math.max(sigPlace, 1e-6);
      }
      const target = side * mEff * sigPlace;
      const want = side > 0 ? 1 : 0;
      const ew = expectedSkillWeight(cands.sMean, cands.sSd, sigPlace, 0.0, side, mEff, this.th.rho);
      const idx = this.argmaxScore(ew, cands, want);
      this.lastSkillS = cands.sMean[idx];
      this.skillSide *= -1;
      return [idx, { target, wantLabel: want }];
    }
    // RETENTION
    const due = retention.dueBins(now);
    const target = tHat + (SKILL_MODE_MULTIPLIER + 0.5) * sigHat;
    const idx = this.nearest(cands, target);
    const binKey = `${this.task},${TaskModePolicy.binOf(cands.sMean[idx], sigHat, tHat)}`;
    return [idx, { target, bin: binKey, eligible: retention.due(binKey, now) || due.length === 0 }];
  }

  // Randomized-schedule commits happen here (single site, at record time):
  // only trials the schedule PROPOSED (info.sched) advance the walk/flip,
  // so retention serving never feeds the state, while pool-exhaustion
  // deviations self-correct because the SERVED label is what commits.
  noteServed(mode: string, yStar: number,
    info: Record<string, unknown> | null = null): void {
    if (mode === BIAS) this.biasLabelBalance += yStar === 1 ? 1 : -1;
    if (this.lsched !== null && info && info.sched) {
      this.lsched.commit(yStar === 1 ? 1 : -1);
      if (this.flip !== null && mode === BIAS && 'probe' in info) {
        this.flip.commit(info.probe ? 1 : -1);
      }
    }
  }
}

// ── cross-task scheduler ──
export class DeficiencyScheduler {
  private last: number | null = null;
  private consec = 0;
  everDeclared = new Set<number>();

  constructor(public K: number, private tStar = DEFAULT_T_STAR, private maxConsec = 5) {}

  deficiency(filt: FilterLike, ellStar: number): number {
    const [pi] = filt.passMass(ellStar);
    const tHat = -filt.mean()[0];
    const skillShort = 1 - pi;
    const biasOver = Math.min(Math.max(Math.abs(tHat) / this.tStar - 1, 0), 1);
    return Math.max(skillShort, biasOver);
  }

  markDeclared(k: number): void { this.everDeclared.add(k); }

  pick(filters: FilterLike[], ellStars: number[], modePolicies: TaskModePolicy[],
    exclude: Set<number> = new Set()): number | null {
    const defs = new Map<number, number>();
    for (let k = 0; k < this.K; k++) {
      if (exclude.has(k)) continue;
      if (modePolicies[k].isMastered(filters[k])) { this.everDeclared.add(k); continue; }
      defs.set(k, this.deficiency(filters[k], ellStars[k]));
    }
    if (defs.size === 0) return null;
    const order = [...defs.keys()].sort((a, b) => (defs.get(b) as number) - (defs.get(a) as number));
    let choice = order[0];
    if (this.last === choice && this.consec >= this.maxConsec && order.length > 1) choice = order[1];
    this.consec = choice === this.last ? this.consec + 1 : 1;
    this.last = choice;
    return choice;
  }
}

export interface Choice {
  task: number; mode: string; segId: number; s: number; sSd: number;
  yStar: number; margin: number; info: Record<string, unknown>; now: number;
}

// ── orchestrator ──
export class TrainerPolicy {
  readonly K: number;
  modePolicies: TaskModePolicy[];
  scheduler: DeficiencyScheduler;
  retention: RetentionScheduler;
  private served = new Set<number>();
  private trial = 0;
  private wasMastered: boolean[];
  private rng: Rng;

  constructor(
    public filters: FilterLike[],
    public ellStars: number[],
    public sigmaStars: number[],
    public bank: Bank,
    opts: { thresholds?: ModeThresholds; seed?: number; maxConsec?: number;
      excludeSegIds?: Set<number>; minMargin?: number;
      retention?: RetentionScheduler;
      // Label-schedule randomization (docs/LABEL_SCHEDULE_PECR.md §3):
      // undefined (default) ⇒ legacy deterministic devices, identical
      // behavior. 'randomized' ⇒ per-task capped soft-tilted label walks +
      // cap-2 type flips + static paired-bin serving. scheduleSeed MUST be
      // session-derived in prod (PECR §6.4) — a fixed seed replays the
      // label sequence across sessions.
      labelSchedule?: 'randomized'; scheduleSeed?: number | bigint;
      scheduleParams?: ScheduleParams } = {},
  ) {
    this.K = filters.length;
    const th = opts.thresholds ?? defaultThresholds();
    this.modePolicies = filters.map((_, k) => new TaskModePolicy(k, ellStars[k], sigmaStars[k], th));
    this.scheduler = new DeficiencyScheduler(this.K, th.tStar ?? DEFAULT_T_STAR, opts.maxConsec ?? 5);
    this.retention = opts.retention ?? new RetentionScheduler();
    this.exclude = opts.excludeSegIds ?? new Set();
    this.minMargin = opts.minMargin ?? 0.30;
    this.wasMastered = new Array(this.K).fill(false);
    this.rng = new Rng(opts.seed ?? 0);
    void this.rng;   // RNG reserved for the >4096 subsample (not on the browser path)
    if (opts.labelSchedule !== undefined) {
      if (opts.labelSchedule !== 'randomized') {
        throw new Error(`unknown labelSchedule: ${opts.labelSchedule}`);
      }
      const sp = opts.scheduleParams ?? defaultScheduleParams();
      const s0 = opts.scheduleSeed ?? opts.seed ?? 0;
      this.modePolicies.forEach((mp, k) => {
        mp.lsched = new SideScheduler(sp, new ScheduleRng(s0, 2 * k));
        mp.flip = new SideScheduler(flipParams(), new ScheduleRng(s0, 2 * k + 1));
      });
    }
  }

  private exclude: Set<number>;
  private minMargin: number;

  private cands(task: number, feedbackSafe = true): TaskCandidates {
    const excl = new Set<number>([...this.exclude, ...this.served]);
    return this.bank.candidates(task, { excludeSegIds: excl, feedbackSafe, minMargin: this.minMargin });
  }

  allMastered(): boolean {
    return this.filters.every((f, k) => this.modePolicies[k].isMastered(f));
  }

  step(now?: number): Choice | null {
    const nowVal = now ?? this.trial;
    let task: number | null = null;
    for (let k = 0; k < this.K; k++) {
      if (this.modePolicies[k].isMastered(this.filters[k])
        && this.retention.dueBins(nowVal).some((b) => b.startsWith(`${k},`))) { task = k; break; }
    }
    if (task === null) task = this.scheduler.pick(this.filters, this.ellStars, this.modePolicies);
    if (task === null) return null;
    const [mode, est] = this.modePolicies[task].chooseMode(this.filters[task]);
    let cands = this.cands(task, mode !== RETENTION);
    if (cands.length === 0) cands = this.cands(task, false);
    const sel = this.modePolicies[task].select(mode, est, cands, this.retention, nowVal, this.filters[task]);
    if (sel === null) return null;
    const [idx, info] = sel;
    return {
      task, mode, segId: cands.segId[idx], s: cands.sMean[idx], sSd: cands.sSd[idx],
      yStar: cands.yStar[idx], margin: cands.margin[idx], info, now: nowVal,
    };
  }

  record(choice: Choice, y: number, feedback = true): void {
    const { task, mode, s, sSd, yStar } = choice;
    this.filters[task].step(s, y, yStar, sSd, feedback);
    this.modePolicies[task].noteServed(mode, yStar, choice.info);
    this.modePolicies[task].notePosterior(this.filters[task]);
    if (mode === RETENTION && 'bin' in choice.info) {
      this.retention.updateOnRetrieval(choice.info.bin as string, y === yStar, choice.now);
    }
    const nowMastered = this.modePolicies[task].isMastered(this.filters[task]);
    if (nowMastered) this.scheduler.markDeclared(task);
    if (nowMastered && !this.wasMastered[task]) {
      const [mt, ml] = this.filters[task].mean();
      const binKey = `${task},${TaskModePolicy.binOf(s, Math.exp(-ml), -mt)}`;
      this.retention.register(binKey, choice.now);
    }
    this.wasMastered[task] = nowMastered;
    this.served.add(choice.segId);
    this.trial += 1;
  }
}
