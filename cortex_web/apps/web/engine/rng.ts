// Seedable PRNG for the SMC engine.
//
// Reproducibility requirement: a session must replay identically given its
// seed (derived from the session id, as the desktop does). It does NOT need
// to bit-match NumPy's PCG64 — the test is statistically defined, not
// byte-defined (see PLAN.md §9). xoshiro256** + a splitmix64 seeder gives a
// fast, high-quality, fully-seedable stream.

function splitmix64(seed: bigint): () => bigint {
  let s = seed & 0xffffffffffffffffn;
  return () => {
    s = (s + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = s;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    return z ^ (z >> 31n);
  };
}

const MASK64 = 0xffffffffffffffffn;
function rotl(x: bigint, k: bigint): bigint {
  return ((x << k) | (x >> (64n - k))) & MASK64;
}

export class Rng {
  private s0: bigint;
  private s1: bigint;
  private s2: bigint;
  private s3: bigint;
  private gaussSpare: number | null = null;

  constructor(seed: number | bigint) {
    const sm = splitmix64(BigInt(seed));
    this.s0 = sm();
    this.s1 = sm();
    this.s2 = sm();
    this.s3 = sm();
  }

  // raw uint64
  private next64(): bigint {
    const result = (rotl((this.s1 * 5n) & MASK64, 7n) * 9n) & MASK64;
    const t = (this.s1 << 17n) & MASK64;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 45n);
    return result;
  }

  // uniform double in [0, 1)
  random(): number {
    // top 53 bits → double
    return Number(this.next64() >> 11n) / 9007199254740992; // 2^53
  }

  // standard normal via Marsaglia polar (caches the spare)
  gaussian(): number {
    if (this.gaussSpare !== null) {
      const v = this.gaussSpare;
      this.gaussSpare = null;
      return v;
    }
    let u: number, v: number, s: number;
    do {
      u = 2 * this.random() - 1;
      v = 2 * this.random() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    this.gaussSpare = v * mul;
    return u * mul;
  }

  // fill an array with standard normals
  fillGaussian(out: Float64Array): void {
    for (let i = 0; i < out.length; i++) out[i] = this.gaussian();
  }

  // integer in [0, n)
  int(n: number): number {
    return Math.floor(this.random() * n);
  }

  // multinomial resample indices: draw `n` indices in [0,n) ∝ weights w
  // (systematic/inverse-CDF; w must sum to 1). Returns Int32Array(n).
  resampleIndices(w: Float64Array, n: number): Int32Array {
    const idx = new Int32Array(n);
    // build cumulative
    const cum = new Float64Array(w.length);
    let acc = 0;
    for (let i = 0; i < w.length; i++) {
      acc += w[i];
      cum[i] = acc;
    }
    for (let j = 0; j < n; j++) {
      const u = this.random() * acc;
      // binary search
      let lo = 0;
      let hi = w.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] < u) lo = mid + 1;
        else hi = mid;
      }
      idx[j] = lo;
    }
    return idx;
  }

  // Exact copy of the generator state — for speculative branch isolation
  // (engine/advance.ts). A cloned Rng reproduces the identical stream from this
  // point, so a branch's draws don't disturb the live stream until adopted.
  clone(): Rng {
    const r = new Rng(0);
    r.s0 = this.s0;
    r.s1 = this.s1;
    r.s2 = this.s2;
    r.s3 = this.s3;
    r.gaussSpare = this.gaussSpare;
    return r;
  }
}
