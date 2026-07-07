// Canonical parameter-convention bridge (G3 TS port of trainer/conventions.py).
// Deterministic; tolerance-parity gated by conventions.test.ts against the Python
// reference. Reuses the exam engine's normCdf (the shared λ-lapse likelihood);
// adds normPpf (Acklam + one Halley step) for the difficulty-placement constant.
//
//   plan:   z = (s − t)/σ,        (σ, t),  skill = 1/σ
//   engine: z = exp(ℓ)·(s + θ),   (θ, ℓ),  skill = exp(ℓ)
//   bridge: σ = exp(−ℓ),  t = −θ     P(y=1) = λ + (1−2λ)·Φ(z),  λ = 0.025
import { normCdf } from '../engine/mathfns';

export const LAPSE_RATE = 0.025;

export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// Inverse standard normal CDF (Acklam rational approximation + one Halley step).
export function normPpf(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let x: number;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= phigh) {
    const q = p - 0.5;
    const r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const e = normCdf(x) - p;                       // Halley refinement
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

export const WILSON_OPT_ACC = normCdf(1.0);

export function difficultyMultiplier(targetAcc: number, lapse = LAPSE_RATE): number {
  return normPpf((targetAcc - lapse) / (1 - 2 * lapse));
}

export function accuracyAtMultiplier(m: number, lapse = LAPSE_RATE): number {
  return lapse + (1 - 2 * lapse) * normCdf(m);
}

export const SKILL_MODE_MULTIPLIER = difficultyMultiplier(WILSON_OPT_ACC);

// ── parameter conversions ──
export function engineToPlan(theta: number[], ell: number[]): [number[], number[]] {
  return [ell.map((l) => Math.exp(-l)), theta.map((th) => -th)];
}
export function planToEngine(sigma: number[], t: number[]): [number[], number[]] {
  return [t.map((v) => -v), sigma.map((s) => -Math.log(s))];
}

// ── observation model ──
export function pYesPlan(s: number[], sigma: number[], t: number[], lapse = LAPSE_RATE): number[] {
  return s.map((sv, i) => lapse + (1 - 2 * lapse) * normCdf((sv - t[i]) / sigma[i]));
}
export function pYesEngine(s: number[], theta: number[], ell: number[], lapse = LAPSE_RATE): number[] {
  return s.map((sv, i) => lapse + (1 - 2 * lapse) * normCdf(Math.exp(ell[i]) * (sv + theta[i])));
}

// ── skill ↔ AUROC ──
export function aurocFromEll(ell: number[]): number[] {
  return ell.map((l) => normCdf(Math.SQRT2 / Math.sqrt(Math.exp(-2 * l) + 1)));
}
export function aurocFromSigma(sigma: number[]): number[] {
  return sigma.map((s) => normCdf(Math.SQRT2 / Math.sqrt(s * s + 1)));
}
export function sigmaStarFromEllStar(ellStar: number[]): number[] {
  return ellStar.map((e) => Math.exp(-e));
}
