import { interpolateFeature } from "./interpolate";
import { KnotSet, ShapeFunction } from "../types";

const finiteValues = (values: number[]) => values.filter((v) => Number.isFinite(v));

const computeStandardDeviation = (values: number[]) => {
  if (!values.length) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const variance = values.reduce((sum, v) => {
    const delta = v - mean;
    return sum + delta * delta;
  }, 0) / values.length;
  return Math.sqrt(variance);
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
      return computeStandardDeviation(finiteValues(values));
    }
    return computeStandardDeviation(finiteValues(knots.y));
  }
  if (scatterX.length) {
    const numericScatter = scatterX.filter((v): v is number => Number.isFinite(v));
    const interpolated = interpolateFeature(numericScatter, knots);
    return computeStandardDeviation(finiteValues(interpolated));
  }
  return computeStandardDeviation(finiteValues(knots.y));
};

export const computeShapeImportance = (
  shape: ShapeFunction,
  scatterX: Array<number | string>,
  knots: KnotSet,
) => {
  if (shape.editableZ) {
    const values = scatterX.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    return computeStandardDeviation(values);
  }
  return computeFeatureImportance(shape, scatterX as number[], knots);
};
