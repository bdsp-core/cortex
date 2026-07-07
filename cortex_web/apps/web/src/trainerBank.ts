// Bridge from a web bundle manifest to the trainer's inputs (bank + cut-scores +
// initial clouds). The SDT-frame labelling (y* = 1[s>0]) matches the Python
// SimBankAdapter used in the G2 closed-loop, so exam↔train exposure is coherent.
import { Rng } from "../engine/rng";
import type { CandidateArrays, Cloud } from "../trainer/session";

interface ManifestSeg {
  segId: number;
  sMean: number[];
  sSd: number[];
  applicableTaskIdx?: number[];
}

const K = 7;

// Per-task candidate arrays. A segment is a candidate for task k iff k is in its
// applicableTaskIdx (i.e. it carries a valid signal for that task).
export function buildTrainerBank(manifest: { segments: ManifestSeg[] }): CandidateArrays[] {
  const per: CandidateArrays[] = Array.from({ length: K }, () => ({
    seg: [], sMean: [], sSd: [], yStar: [], margin: [], coherent: [],
  }));
  for (const s of manifest.segments) {
    // Pre-K=7 bundles omit applicableTaskIdx → fall back to "applies to all K"
    // (same backward-compat rule the exam engine uses).
    const applicable = s.applicableTaskIdx ?? Array.from({ length: K }, (_, i) => i);
    for (const k of applicable) {
      const sm = s.sMean[k];
      const y = sm > 0 ? 1 : 0;
      per[k].seg.push(s.segId);
      per[k].sMean.push(sm);
      per[k].sSd.push(s.sSd[k]);
      per[k].yStar.push(y);
      per[k].margin.push(1);
      per[k].coherent.push((y === 1) === (sm > 0));
    }
  }
  return per;
}

// ℓ* (mastery bar), σ* = exp(−ℓ*), and a per-task ceiling σ_∞ set above the cut
// (exp(−(ℓ*+0.3))) so trainable learners can clear the bar.
export function cutScores(ellStar: number[]) {
  return {
    ellStars: ellStar.slice(),
    sigmaStars: ellStar.map((e) => Math.exp(-e)),
    sigmaInf: ellStar.map((e) => Math.exp(-(e + 0.3))),
  };
}

// Initial per-task belief clouds. A pragmatic seeded prior (θ~N(0,0.4·),
// ℓ~N(ell0,0.3)); used as the fallback when no cert posterior is available.
export function seedClouds(ell0: number[], N = 200, seed = 0): Cloud[] {
  return ell0.map((e0, k) => {
    const r = new Rng(seed + 1000 * k + 1);
    return {
      theta: Array.from({ length: N }, () => 0.4 * r.gaussian()),
      ell: Array.from({ length: N }, () => e0 + 0.3 * r.gaussian()),
      w: new Array(N).fill(1 / N),
    };
  });
}

// The learner's measured per-task posterior handed off from the finished cert
// (engine coords: ℓ = skill, θ = criterion; sd = posterior SD of ℓ, or null).
export interface TaskPrior { taskK: number; ell: number; theta: number; sd: number | null; }

// Seed each task's belief cloud from the learner's MEASURED cert posterior
// (per-task ℓ/θ means) instead of a generic prior, variance-INFLATED so the
// trainer can still adapt (skill drifts between test and training, and the
// trainer is a different model — D1). The ℓ spread is the cert posterior SD when
// present, floored then inflated; θ uses a fixed moderate spread. Tasks with no
// prior fall back to the generic (ℓ~N(0,·), θ~N(0,·)) seed. Mastered tasks land
// above their cut so the scheduler deprioritises them; weak tasks land low and
// get trained.
export function seedCloudsFromPrior(
  prior: TaskPrior[], K: number, N = 200, seed = 0,
  opts: { inflate?: number; ellSdFloor?: number; thetaSd?: number } = {},
): Cloud[] {
  const inflate = opts.inflate ?? 1.6;
  const ellSdFloor = opts.ellSdFloor ?? 0.25;
  const thetaSd = opts.thetaSd ?? 0.4;
  const byK = new Map(prior.map((p) => [p.taskK, p]));
  return Array.from({ length: K }, (_v, k) => {
    const p = byK.get(k);
    const ell0 = p && Number.isFinite(p.ell) ? p.ell : 0.0;
    const theta0 = p && Number.isFinite(p.theta) ? p.theta : 0.0;
    const ellSd = Math.max(ellSdFloor, (p && p.sd != null && Number.isFinite(p.sd)) ? p.sd : ellSdFloor) * inflate;
    const r = new Rng(seed + 1000 * k + 1);
    return {
      theta: Array.from({ length: N }, () => theta0 + thetaSd * r.gaussian()),
      ell: Array.from({ length: N }, () => ell0 + ellSd * r.gaussian()),
      w: new Array(N).fill(1 / N),
    };
  });
}
