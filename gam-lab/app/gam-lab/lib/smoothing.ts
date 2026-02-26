// Stable 1D smoothing via second-derivative regularization:
// minimize ||z - y||^2 + lambda * ||D2 z||^2
export const smoothSeriesTikhonov = (values: number[], lambda: number): number[] => {
  const n = values.length;
  if (n <= 2 || lambda <= 0) return [...values];

  const A: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  const b = [...values];

  for (let r = 0; r <= n - 3; r += 1) {
    const i = r;
    const j = r + 1;
    const k = r + 2;
    A[i][i] += lambda;
    A[i][j] += -2 * lambda;
    A[i][k] += lambda;
    A[j][i] += -2 * lambda;
    A[j][j] += 4 * lambda;
    A[j][k] += -2 * lambda;
    A[k][i] += lambda;
    A[k][j] += -2 * lambda;
    A[k][k] += lambda;
  }

  // Gaussian elimination with partial pivoting.
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    let maxAbs = Math.abs(A[col][col]);
    for (let row = col + 1; row < n; row += 1) {
      const v = Math.abs(A[row][col]);
      if (v > maxAbs) {
        maxAbs = v;
        pivot = row;
      }
    }
    if (pivot !== col) {
      const rowTmp = A[col];
      A[col] = A[pivot];
      A[pivot] = rowTmp;
      const bTmp = b[col];
      b[col] = b[pivot];
      b[pivot] = bTmp;
    }

    const diag = A[col][col];
    if (!Number.isFinite(diag) || Math.abs(diag) < 1e-12) continue;

    for (let row = col + 1; row < n; row += 1) {
      const factor = A[row][col] / diag;
      if (!Number.isFinite(factor) || factor === 0) continue;
      A[row][col] = 0;
      for (let k = col + 1; k < n; k += 1) {
        A[row][k] -= factor * A[col][k];
      }
      b[row] -= factor * b[col];
    }
  }

  const x = Array.from({ length: n }, () => 0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let sum = b[row];
    for (let k = row + 1; k < n; k += 1) {
      sum -= A[row][k] * x[k];
    }
    const diag = A[row][row];
    x[row] = Number.isFinite(diag) && Math.abs(diag) >= 1e-12 ? sum / diag : values[row];
  }
  return x;
};

const reflectIndex = (idx: number, n: number): number => {
  if (n <= 1) return 0;
  let k = idx;
  while (k < 0 || k >= n) {
    if (k < 0) {
      k = -k;
    } else {
      k = 2 * n - 2 - k;
    }
  }
  return k;
};

// Gaussian smoothing with reflected boundaries (edge-stable).
export const smoothSeriesGaussianReflect = (values: number[], radius: number, sigma: number): number[] => {
  const n = values.length;
  if (n <= 2 || radius <= 0) return [...values];
  const r = Math.max(1, Math.round(radius));
  const s = Math.max(1e-3, sigma);

  const kernel = new Float64Array(r + 1);
  for (let d = 0; d <= r; d += 1) {
    kernel[d] = Math.exp(-(d * d) / (2 * s * s));
  }

  const out = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    let wSum = 0;
    let vSum = 0;
    for (let d = -r; d <= r; d += 1) {
      const j = reflectIndex(i + d, n);
      const v = values[j];
      if (!Number.isFinite(v)) continue;
      const w = kernel[Math.abs(d)] ?? 0;
      wSum += w;
      vSum += v * w;
    }
    out[i] = wSum > 0 ? vSum / wSum : values[i];
  }
  return out;
};
