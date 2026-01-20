export const interpolateFeature = (xTargets: number[], knots: { x: number[]; y: number[] }) => {
  if (knots.x.length !== knots.y.length || knots.x.length === 0) {
    return xTargets.map(() => 0);
  }
  const pairs = knots.x.map((x, idx) => ({ x, y: knots.y[idx] ?? 0 }));
  const sorted = pairs.sort((a, b) => a.x - b.x);
  if (sorted.length === 1) return xTargets.map(() => sorted[0].y);

  return xTargets.map((val) => {
    if (val <= sorted[0].x) return sorted[0].y;
    if (val >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;
    const idx = sorted.findIndex((pair) => pair.x > val);
    const left = sorted[idx - 1];
    const right = sorted[idx];
    const t = (val - left.x) / Math.max(1e-9, right.x - left.x);
    return left.y * (1 - t) + right.y * t;
  });
};

export type KnotSet = { x: number[]; y: number[] };
