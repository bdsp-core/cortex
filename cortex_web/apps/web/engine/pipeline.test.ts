import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { logPResponse, pResponseYes, signalZ } from "./likelihood";
import { update } from "./particles";
import { expectedLoss } from "./choose_item";
import { AD6Policy } from "./policy";
import { ParticleState, PriorPieces, PriorPair } from "./types";

const ref = JSON.parse(
  readFileSync(fileURLToPath(new URL("./__testdata__/reference.json", import.meta.url)), "utf8"),
);

// Flatten an (N×K) nested array to row-major Float64Array(N*K).
function flat(rows: number[][]): Float64Array {
  const N = rows.length;
  const K = rows[0].length;
  const a = new Float64Array(N * K);
  for (let n = 0; n < N; n++) for (let k = 0; k < K; k++) a[n * K + k] = rows[n][k];
  return a;
}

// A minimal state for update/expectedLoss/policy (none of which touch the
// prior; rejuvenation does, but it's not exercised here).
function makeFixtureState(t: number[][], l: number[][], w: number[]): ParticleState {
  const N = t.length;
  const K = t[0].length;
  const dummyPieces: PriorPieces = { K, sigmaInv: [], logDet: 0, L: [] };
  const dummyPrior: PriorPair = { tPieces: dummyPieces, lPieces: dummyPieces };
  const wSum = w.reduce((a, b) => a + b, 0);
  return {
    N,
    K,
    t: flat(t),
    l: flat(l),
    w: Float64Array.from(w.map((x) => x / wSum)),
    logPrior: new Float64Array(N),
    logLik: new Float64Array(N),
    history: [],
    prior: dummyPrior,
  };
}

describe("likelihood vs engine/core_mcmc.py", () => {
  for (const c of ref.likelihood as any[]) {
    it(`(l=${c.l}, t=${c.t}, s=${c.s})`, () => {
      const z = signalZ(c.l, c.t, c.s);
      expect(Math.abs(z - c.z)).toBeLessThan(1e-12);
      expect(Math.abs(logPResponse(z, 1) - c.logP1)).toBeLessThan(1e-7);
      expect(Math.abs(logPResponse(z, 0) - c.logP0)).toBeLessThan(1e-7);
      expect(Math.abs(pResponseYes(z) - c.pYes)).toBeLessThan(1e-7);
    });
  }
});

// Quantities that flow through Φ inherit erfc()'s ~1e-7 precision (see
// mathfns.ts). 1e-6 is comfortably above that floor yet far below any real
// porting bug (wrong formula/sign/index → ≫1e-2), so it's a meaningful
// agreement check, not a fake-loosened one. Upgrade erfc to the Cody form
// (PLAN §11) to recover 1e-12 here if exactness is ever required.
const LIK_TOL = 1e-6;

describe("update reweighting vs engine update()", () => {
  it("post-update weights match", () => {
    const f = ref.updateFixture;
    const st = makeFixtureState(f.t, f.l, f.w);
    update(st, f.k, f.s, f.y as 0 | 1, f.sSd);
    expect(st.w.length).toBe(f.wPost.length);
    for (let i = 0; i < f.wPost.length; i++) {
      expect(Math.abs(st.w[i] - f.wPost[i])).toBeLessThan(LIK_TOL);
    }
  });
});

describe("expectedLoss vs engine _expected_loss_vec", () => {
  it("EV total-posterior-variance matches", () => {
    const f = ref.updateFixture; // same fixed cloud, pre-update
    const ef = ref.expectedLossFixture;
    const st = makeFixtureState(f.t, f.l, f.w);
    const loss = expectedLoss(st, ef.k, ef.s, ef.sSd);
    expect(Math.abs(loss - ef.loss)).toBeLessThan(LIK_TOL);
  });

  it("total-variance identity matches the direct conditional-variance calculation", () => {
    const f = ref.updateFixture;
    const st = makeFixtureState(f.t, f.l, f.w);

    const direct = (k: number, s: number, sSd: number): number => {
      const probabilities = new Float64Array(st.N);
      let pYes = 0;
      for (let n = 0; n < st.N; n++) {
        const off = n * st.K + k;
        const pn = Math.min(1 - 1e-9, Math.max(
          1e-9,
          pResponseYes(signalZ(st.l[off], st.t[off], s, sSd)),
        ));
        probabilities[n] = pn;
        pYes += pn * st.w[n];
      }
      const conditionalTotal = (yes: boolean): number => {
        const norm = st.w.reduce(
          (sum, weight, n) => sum + weight * (yes ? probabilities[n] : 1 - probabilities[n]),
          0,
        );
        let total = 0;
        for (const values of [st.t, st.l]) {
          for (let kk = 0; kk < st.K; kk++) {
            let mean = 0;
            for (let n = 0; n < st.N; n++) {
              const weight = st.w[n] * (yes ? probabilities[n] : 1 - probabilities[n]);
              mean += weight * values[n * st.K + kk];
            }
            mean /= norm;
            let variance = 0;
            for (let n = 0; n < st.N; n++) {
              const weight = st.w[n] * (yes ? probabilities[n] : 1 - probabilities[n]);
              const delta = values[n * st.K + kk] - mean;
              variance += weight * delta * delta;
            }
            total += variance / norm;
          }
        }
        return total;
      };
      return pYes * conditionalTotal(true) + (1 - pYes) * conditionalTotal(false);
    };

    for (let k = 0; k < st.K; k++) {
      for (const [s, sSd] of [[-2, 0], [-0.4, 0.1], [0.8, 0.3], [2.2, 0.5]]) {
        expect(Math.abs(expectedLoss(st, k, s, sSd) - direct(k, s, sSd)))
          .toBeLessThan(1e-11);
      }
    }
  });
});

describe("AD6 policy π/mcse/R vs cortex_policy.py", () => {
  it("per-task pass-mass, mcse, info-gate match", () => {
    const f = ref.policyFixture;
    const st = makeFixtureState(f.l.map(() => f.varPrior), f.l, f.w); // t unused
    const policy = new AD6Policy(f.ellStar, f.varPrior);
    const res = policy.evaluate(st, [0, 0]);
    expect(Math.abs(res.ess - f.ess)).toBeLessThan(1e-9);
    for (let k = 0; k < f.pi.length; k++) {
      expect(Math.abs(res.pi[k] - f.pi[k])).toBeLessThan(1e-9);
      expect(Math.abs(res.mcse[k] - f.mcse[k])).toBeLessThan(1e-9);
      expect(Math.abs(res.R[k] - f.R[k])).toBeLessThan(1e-9);
    }
  });
});
