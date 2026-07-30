const SQRT1_2 = Math.SQRT1_2;
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

/** Browser-portable complementary error-function approximation. */
export function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const coefficients = [
    -1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806,
    0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277,
  ];
  let polynomial = coefficients[coefficients.length - 1];
  for (let i = coefficients.length - 2; i >= 0; i -= 1) {
    polynomial = coefficients[i] + t * polynomial;
  }
  const tau = t * Math.exp(-z * z + polynomial);
  return x >= 0 ? tau : 2 - tau;
}

export function normCdf(x: number): number {
  return 0.5 * erfc(-x * SQRT1_2);
}

export function logNdtr(x: number): number {
  if (x > -5) return Math.log(normCdf(x));
  const xi = 1 / (x * x);
  const series = 1 - xi * (1 - xi * (3 - xi * (15 - xi * 105)));
  return -0.5 * x * x - LOG_SQRT_2PI - Math.log(-x) + Math.log(series);
}

export function logSumExp(values: readonly number[]): number {
  let maximum = -Infinity;
  for (const value of values) maximum = Math.max(maximum, value);
  if (maximum === -Infinity) return -Infinity;
  let total = 0;
  for (const value of values) total += Math.exp(value - maximum);
  return maximum + Math.log(total);
}

export function logSumExp2(a: number, b: number): number {
  const maximum = Math.max(a, b);
  if (maximum === -Infinity) return -Infinity;
  return maximum + Math.log(Math.exp(a - maximum) + Math.exp(b - maximum));
}

