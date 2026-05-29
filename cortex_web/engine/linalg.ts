// Small dense linear algebra for the SMC prior + MH proposal.
// Sizes are tiny and fixed: K=6 (prior blocks) and 2K=12 (rejuvenation
// covariance). Plain row-major number[][]; correctness over cleverness.
//
// Ports the pieces engine/core_mcmc.py gets from numpy:
//   _precompute_prior_pieces → cholesky + inverse + slogdet
//   mh_rejuvenate            → cov + cholesky (+ eigh fallback for non-PD)

export type Mat = number[][];

export function cholesky(A: Mat): Mat {
  const n = A.length;
  const L: Mat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 0) throw new Error("cholesky: matrix not positive-definite");
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

// inverse via Cholesky (A must be SPD).
export function invSPD(A: Mat): Mat {
  const n = A.length;
  const L = cholesky(A);
  // solve L y = e_i, then Lᵀ x = y for each column
  const inv: Mat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let col = 0; col < n; col++) {
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = i === col ? 1 : 0;
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    for (let i = 0; i < n; i++) inv[i][col] = x[i];
  }
  return inv;
}

// log|det A| for SPD A (2·Σ log L_ii).
export function logDetSPD(A: Mat): number {
  const L = cholesky(A);
  let s = 0;
  for (let i = 0; i < A.length; i++) s += Math.log(L[i][i]);
  return 2 * s;
}

// add jitter·I (matches the engine's 1e-9 I stabiliser).
export function addJitter(A: Mat, jitter = 1e-9): Mat {
  return A.map((row, i) => row.map((v, j) => (i === j ? v + jitter : v)));
}

// sample covariance of rows (each row is an observation, columns are dims) —
// the np.cov(theta, rowvar=False) the MH step uses on the (N × 2K) cloud.
export function covRows(data: Float64Array, nRows: number, nCols: number): Mat {
  const mean = new Array(nCols).fill(0);
  for (let r = 0; r < nRows; r++)
    for (let c = 0; c < nCols; c++) mean[c] += data[r * nCols + c];
  for (let c = 0; c < nCols; c++) mean[c] /= nRows;
  const cov: Mat = Array.from({ length: nCols }, () => new Array(nCols).fill(0));
  for (let r = 0; r < nRows; r++) {
    for (let i = 0; i < nCols; i++) {
      const di = data[r * nCols + i] - mean[i];
      for (let j = i; j < nCols; j++) {
        cov[i][j] += di * (data[r * nCols + j] - mean[j]);
      }
    }
  }
  const denom = nRows - 1;
  for (let i = 0; i < nCols; i++)
    for (let j = i; j < nCols; j++) {
      cov[i][j] /= denom;
      cov[j][i] = cov[i][j];
    }
  return cov;
}

// Jacobi eigendecomposition for symmetric matrices — the eigh fallback used
// when the MH covariance is not numerically PD (engine: L = eigvecs·√eigvals,
// eigvals floored at 1e-6). Returns a "square-root" factor F with F·Fᵀ ≈ A
// (clipped to PD), suitable as the proposal scale matrix.
export function symSqrtClipped(A: Mat, floor = 1e-6): Mat {
  const n = A.length;
  // copy
  const a: Mat = A.map((r) => r.slice());
  const v: Mat = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < 100; sweep++) {
    // largest off-diagonal
    let p = 0,
      q = 1,
      off = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(a[i][j]) > off) {
          off = Math.abs(a[i][j]);
          p = i;
          q = j;
        }
    if (off < 1e-14) break;
    const app = a[p][p],
      aqq = a[q][q],
      apq = a[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi),
      s = Math.sin(phi);
    for (let k = 0; k < n; k++) {
      const akp = a[k][p],
        akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[k][q] = s * akp + c * akq;
    }
    for (let k = 0; k < n; k++) {
      const apk = a[p][k],
        aqk = a[q][k];
      a[p][k] = c * apk - s * aqk;
      a[q][k] = s * apk + c * aqk;
    }
    for (let k = 0; k < n; k++) {
      const vkp = v[k][p],
        vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }
  const F: Mat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      F[i][j] = v[i][j] * Math.sqrt(Math.max(a[j][j], floor));
  return F;
}
