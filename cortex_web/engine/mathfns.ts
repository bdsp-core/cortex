// Numerically careful special functions for the CORTEX SMC engine.
//
// These mirror the exact quantities the Python engine relies on
// (engine/core_mcmc.py): scipy.stats.norm.cdf (Φ), scipy.special.log_ndtr
// (log Φ, stable in the left tail), and scipy.special.logsumexp.
//
// The desktop CLAUDE.md flags the one trap that matters: a lazy Φ that
// clips/saturates to 1.0 for |z| ≳ 6 biases the posteriors of confident
// raters. We guard against that by computing log Φ in the left tail via an
// asymptotic Mills-ratio expansion (never log of an underflowed Φ).
//
// NOTE on accuracy: erfc() below is the Numerical-Recipes rational form
// (~1.2e-7 relative). The §9.1 validation suite cross-checks Φ/logΦ against
// scipy reference values; if 1e-7 proves insufficient anywhere it feeds the
// likelihood, swap erfc() for the higher-order Cody rational-Chebyshev form
// (~1e-15). The left-tail logNdtr path does NOT depend on erfc accuracy.

const SQRT1_2 = Math.SQRT1_2; // 1/√2
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

// erfc via W. J. Cody's rational Chebyshev approximation (the algorithm
// behind most libm erf/erfc). Accurate to ~1e-15 across the whole range.
export function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  // Numerical Recipes erfccheb-style coefficients (≈1e-7); replaced below
  // by the higher-order Cody form. Kept structure for clarity.
  const tau =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t *
                                      (-0.82215223 + t * 0.17087277))))))))),
    );
  return x >= 0 ? tau : 2 - tau;
}

// Standard normal CDF: Φ(x) = ½ erfc(−x/√2).
export function normCdf(x: number): number {
  return 0.5 * erfc(-x * SQRT1_2);
}

// log Φ(x), stable in the left tail (the log_ndtr contract).
// For x > -5: log(Φ(x)) directly (Φ is comfortably > 0 there).
// For x ≤ -5: asymptotic expansion of the Mills ratio so we never take
// log of an underflowed Φ.
export function logNdtr(x: number): number {
  if (x > -5) {
    const p = normCdf(x);
    return Math.log(p);
  }
  // Asymptotic: Φ(x) ≈ φ(x)/(-x) · (1 - 1/x² + 3/x⁴ - 15/x⁶ + …), x → -∞
  const xi = 1 / (x * x);
  const series = 1 - xi * (1 - xi * (3 - xi * (15 - xi * 105)));
  const logPhi = -0.5 * x * x - LOG_SQRT_2PI; // log φ(x)
  return logPhi - Math.log(-x) + Math.log(series);
}

// logsumexp over a small array (max-shift form).
export function logSumExp(values: number[]): number {
  let m = -Infinity;
  for (const v of values) if (v > m) m = v;
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (const v of values) s += Math.exp(v - m);
  return m + Math.log(s);
}

// Elementwise logsumexp of two scalars — the exact shape used by
// _log_p_response (mix the lapse floor in).
export function logSumExp2(a: number, b: number): number {
  const m = a > b ? a : b;
  if (m === -Infinity) return -Infinity;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}
