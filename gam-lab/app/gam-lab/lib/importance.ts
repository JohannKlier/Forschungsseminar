import { interpolateFeature } from "./interpolate";
import { FeatureCurve, KnotSet } from "../types";

const finiteValues = (values: number[]) => values.filter((v) => Number.isFinite(v));

const computeStd = (values: number[]) => {
  if (!values.length) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const variance = values.reduce((sum, v) => {
    const delta = v - mean;
    return sum + delta * delta;
  }, 0) / values.length;
  return Math.sqrt(variance);
};

export const computeFeatureImportance = (partial: FeatureCurve, knots: KnotSet) => {
  const scatter = Array.isArray(partial.scatterX) ? partial.scatterX : [];
  if (partial.categories && partial.categories.length) {
    if (scatter.length) {
      const values: number[] = [];
      scatter.forEach((raw) => {
        if (!Number.isFinite(raw)) return;
        const idx = Math.round(raw);
        if (idx < 0 || idx >= knots.y.length) return;
        values.push(knots.y[idx] ?? 0);
      });
      return computeStd(values);
    }
    return computeStd(finiteValues(knots.y));
  }
  if (scatter.length) {
    const interpolated = interpolateFeature(scatter, knots);
    return computeStd(finiteValues(interpolated));
  }
  return computeStd(finiteValues(knots.y));
};
