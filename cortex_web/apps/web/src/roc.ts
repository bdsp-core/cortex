// ROC geometry for the results screen — faithful port of the desktop
// _binormal_steps / _oncurve_point / _empirical_point (scripts/eeg_bank_viewer.py
// 2847-2916). Equal-variance binormal model: d′ = √2·Φ⁻¹(AUROC). CORTEX records
// binary Yes/No, so there is ONE real operating point; the staircase is a
// cosmetic model curve at the given AUROC.

import { normCdf, type TrialDiag } from "../engine";

// Inverse normal CDF (Acklam's rational approximation, ~1e-9). normCdf lives in
// engine/mathfns; the engine never needed the inverse, so it lives here.
export function normPpf(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
             3.754408661907416];
  const plow = 0.02425, phigh = 1 - plow;
  let q: number, r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= phigh) {
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
}

// Deterministic binormal ROC staircase (steps-post path) for area = auroc.
export function binormalSteps(auroc: number, nEach = 48): { xs: number[]; ys: number[] } {
  const a = Math.min(Math.max(auroc, 0.5001), 0.9999);
  const dprime = Math.SQRT2 * normPpf(a);
  const present: number[] = [], absent: number[] = [];
  for (let i = 1; i <= nEach; i++) {
    const z = normPpf((i - 0.5) / nEach);
    present.push(z + dprime);
    absent.push(z);
  }
  const thr = [...present, ...absent].sort((x, y) => y - x); // high → low
  const far = [0], hr = [0];
  for (const t of thr) {
    far.push(absent.reduce((c, v) => c + (v >= t ? 1 : 0), 0) / nEach);
    hr.push(present.reduce((c, v) => c + (v >= t ? 1 : 0), 0) / nEach);
  }
  far.push(1); hr.push(1);
  const xs = [far[0]], ys = [hr[0]];
  for (let i = 1; i < far.length; i++) { xs.push(far[i], far[i]); ys.push(hr[i - 1], hr[i]); }
  return { xs, ys };
}

// Project a false-alarm rate onto the binormal ROC: HR = Φ(Φ⁻¹(far) + d′).
export function onCurvePoint(auroc: number, far: number): [number, number] {
  const a = Math.min(Math.max(auroc, 0.5001), 0.9999);
  const dprime = Math.SQRT2 * normPpf(a);
  const f = Math.min(Math.max(far, 1e-3), 1 - 1e-3);
  return [f, normCdf(normPpf(f) + dprime)];
}

// The taker's empirical (FAR, HR) for task k from their answers, or null if a
// SDT arm is empty. trueWord = the task's pattern word (taskPatternWords[k]);
// truthOf(segId) returns the segment's pattern class. response y is TrialDiag.y.
export function empiricalPoint(
  trials: TrialDiag[],
  k: number,
  trueWord: string,
  truthOf: (segId: number) => string | undefined,
  spikeTask = false,
): [number, number] | null {
  const pres: number[] = [], absn: number[] = [];
  for (const t of trials) {
    if (t.taskK !== k) continue;
    // spike truth = sign(s_mean) (eeg_bank_viewer.py:2906-2910); IIIC truth =
    // segment pattern class == this task's word.
    const present = spikeTask ? t.s > 0 : truthOf(t.segId) === trueWord;
    (present ? pres : absn).push(t.y);
  }
  if (!pres.length || !absn.length) return null;
  return [absn.reduce((a, b) => a + b, 0) / absn.length,
          pres.reduce((a, b) => a + b, 0) / pres.length];
}
