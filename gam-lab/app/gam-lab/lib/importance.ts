import { interpolateFeature } from "./interpolate";
import { KnotSet, ShapeFunction } from "../types";

const finiteValues = (values: number[]) => values.filter((v) => Number.isFinite(v));

const computeVariance = (values: number[]) => {
  if (!values.length) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return values.reduce((sum, v) => {
    const delta = v - mean;
    return sum + delta * delta;
  }, 0) / values.length;
};

// scatterX is passed separately (from trainData.trainX[shape.key]) rather than
// being embedded in the shape function itself.
export const computeFeatureImportance = (shape: ShapeFunction, scatterX: number[], knots: KnotSet) => {
  if (shape.categories && shape.categories.length) {
    const valueByCategory = new Map<string, number>();
    shape.categories.forEach((cat, idx) => {
      valueByCategory.set(String(cat), knots.y[idx] ?? 0);
    });
    if (scatterX.length) {
      const values: number[] = [];
      scatterX.forEach((raw) => {
        const key = String(raw);
        values.push(valueByCategory.get(key) ?? 0);
      });
      return computeVariance(finiteValues(values));
    }
    return computeVariance(finiteValues(knots.y));
  }
  if (scatterX.length) {
    const numericScatter = scatterX.filter((v): v is number => Number.isFinite(v));
    const interpolated = interpolateFeature(numericScatter, knots);
    return computeVariance(finiteValues(interpolated));
  }
  return computeVariance(finiteValues(knots.y));
};
