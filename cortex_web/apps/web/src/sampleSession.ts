// Per-session question sampling. The bundle ships a large POOL of segments;
// each test launch draws a fresh, difficulty-stratified, class-balanced
// sample of `target` (≈500) questions — so no two sittings see the same set
// (when the pool ≫ target), while every sitting still spans easy→hard for
// every IIIC class.
//
// Difficulty of a segment = its signal on its OWN (true) task: a high signal
// is a clear/easy example of that class, near-zero is ambiguous/hard. We
// stratify within each class across difficulty bins so the sample isn't all
// textbook cases.

import { EngineInputs, SegmentMeta } from "../engine/types";
import { Rng } from "../engine/rng";

const N_BINS = 8;

function difficultyOf(seg: SegmentMeta, taskPatternWords: string[]): number {
  const k = taskPatternWords.indexOf(seg.patternClass);
  return k >= 0 ? seg.sMean[k] : 0; // true-task signal; higher = easier
}

// Fisher–Yates shuffle in place using the seeded RNG.
function shuffle<T>(arr: T[], rng: Rng): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    const t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
}

export interface SampleInfo {
  seed: number;
  nPool: number;
  nSampled: number;
  perClass: Record<string, number>;
}

// Draw a fresh session sample. Returns the sampled EngineInputs plus a small
// info record (for telemetry / the results file). `seed` defaults to a random
// per-session value so each sitting differs; pass an explicit seed to
// reproduce a sitting.
export function sampleSession(
  pool: EngineInputs,
  target = 500,
  seed?: number,
): { inputs: EngineInputs; info: SampleInfo } {
  const s = seed ?? ((Math.floor(Math.random() * 2 ** 31) ^ Date.now()) >>> 0);
  const rng = new Rng(s);
  const words = pool.taskPatternWords;

  // group by class
  const byClass = new Map<string, SegmentMeta[]>();
  for (const seg of pool.segments) {
    const arr = byClass.get(seg.patternClass) ?? [];
    arr.push(seg);
    byClass.set(seg.patternClass, arr);
  }
  const classes = [...byClass.keys()];
  const perClassTarget = Math.ceil(target / classes.length);

  const chosen: SegmentMeta[] = [];
  for (const cls of classes) {
    const segs = byClass.get(cls)!.slice();
    // sort by difficulty, split into N_BINS contiguous bins, draw round-robin
    // across bins (shuffled within bin) until we have perClassTarget.
    segs.sort((a, b) => difficultyOf(a, words) - difficultyOf(b, words));
    const bins: SegmentMeta[][] = Array.from({ length: N_BINS }, () => []);
    segs.forEach((seg, i) => {
      const b = Math.min(N_BINS - 1, Math.floor((i / segs.length) * N_BINS));
      bins[b].push(seg);
    });
    bins.forEach((b) => shuffle(b, rng));
    const picks: SegmentMeta[] = [];
    let bi = 0;
    let exhausted = 0;
    while (picks.length < perClassTarget && exhausted < N_BINS) {
      const bin = bins[bi % N_BINS];
      if (bin.length) {
        picks.push(bin.pop()!);
        exhausted = 0;
      } else {
        exhausted++;
      }
      bi++;
    }
    chosen.push(...picks);
  }

  // trim to target (classes may overfill by rounding) with a shuffle so the
  // trim doesn't bias toward the last class
  shuffle(chosen, rng);
  const sampled = chosen.slice(0, Math.min(target, chosen.length));

  const perClass: Record<string, number> = {};
  for (const seg of sampled) perClass[seg.patternClass] = (perClass[seg.patternClass] ?? 0) + 1;

  return {
    inputs: { ...pool, segments: sampled },
    info: { seed: s, nPool: pool.segments.length, nSampled: sampled.length, perClass },
  };
}
