import { FeatureOperation, ShapeFunction, TrainData, TrainResponse } from "../types";
import { interpolateFeature } from "./interpolate";

const EPS = 1e-9;

const pearsonCorrelation = (xs: number[], ys: number[]) => {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  const xMean = xs.slice(0, n).reduce((sum, value) => sum + value, 0) / n;
  const yMean = ys.slice(0, n).reduce((sum, value) => sum + value, 0) / n;
  let num = 0;
  let xDen = 0;
  let yDen = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - xMean;
    const dy = ys[i] - yMean;
    num += dx * dy;
    xDen += dx * dx;
    yDen += dy * dy;
  }
  const den = Math.sqrt(xDen * yDen);
  return den > EPS ? num / den : 0;
};

const isCategorical = (key: string, trainData: TrainData) => Boolean(trainData.categories[key]?.length);

const asNumeric = (values: Array<number | string>) =>
  values.map((value) => (typeof value === "number" ? value : Number(value))).filter(Number.isFinite);

const buildOperationValues = (
  operator: FeatureOperation["operator"],
  left: Array<number | string>,
  right: Array<number | string>,
) => {
  const leftNum = asNumeric(left);
  const rightNum = asNumeric(right);
  const n = Math.min(leftNum.length, rightNum.length);
  if (n === 0) return [];
  const values: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const a = leftNum[i];
    const b = rightNum[i];
    if (operator === "product") values.push(a * b);
    else if (operator === "sum") values.push(a + b);
    else if (operator === "difference") values.push(a - b);
    else if (operator === "ratio") values.push(a / (Math.abs(b) < EPS ? EPS : b));
    else values.push(Math.abs(a - b));
  }
  return values;
};

const operationLabel = (leftLabel: string, rightLabel: string, operator: FeatureOperation["operator"]) => {
  if (operator === "product") return `${leftLabel} × ${rightLabel}`;
  if (operator === "sum") return `${leftLabel} + ${rightLabel}`;
  if (operator === "difference") return `${leftLabel} − ${rightLabel}`;
  if (operator === "ratio") return `${leftLabel} / ${rightLabel}`;
  return `|${leftLabel} − ${rightLabel}|`;
};

const operationKey = (left: string, right: string, operator: FeatureOperation["operator"]) =>
  operator === "product" ? `${left}__${right}` : `${left}__${operator}__${right}`;

const shapeContribution = (shape: ShapeFunction, trainData: TrainData) => {
  const scatterX = trainData.trainX[shape.key] ?? [];
  if (shape.categories?.length) {
    const valueByCategory = new Map<string, number>();
    shape.categories.forEach((cat, idx) => {
      valueByCategory.set(String(cat), shape.editableY?.[idx] ?? 0);
    });
    return scatterX.map((raw) => valueByCategory.get(String(raw)) ?? 0);
  }
  const numericScatter = scatterX.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return interpolateFeature(numericScatter, { x: shape.editableX ?? [], y: shape.editableY ?? [] });
};

export const residualsFromTrainResponse = (response: TrainResponse) => {
  const trainY = response.data.trainY ?? [];
  const contribsByShape = response.version.shapes.map((shape) => {
    if (shape.editableZ) {
      const interactionContribs = response.data.trainX[shape.key] ?? [];
      return interactionContribs.map((contrib) =>
        typeof contrib === "number" && Number.isFinite(contrib) ? contrib : 0,
      );
    }
    return shapeContribution(shape, response.data);
  });
  const totals = trainY.map((_, index) =>
    contribsByShape.reduce((sum, contribs) => sum + (contribs[index] ?? 0), 0),
  );
  const preds = response.model.task === "classification"
    ? totals.map((total) => 1 / (1 + Math.exp(-(total + response.version.intercept))))
    : totals.map((total) => total + response.version.intercept);
  return trainY.map((target, index) => target - (preds[index] ?? 0));
};

export const suggestInteractionOperations = ({
  residuals,
  trainData,
  selectedFeatures,
  limit = 8,
}: {
  residuals: number[];
  trainData: TrainData;
  selectedFeatures: string[];
  limit?: number;
}) => {
  const suggestionsByPair = new Map<string, FeatureOperation & { score: number }>();
  for (let i = 0; i < selectedFeatures.length; i += 1) {
    for (let j = i + 1; j < selectedFeatures.length; j += 1) {
      const leftKey = selectedFeatures[i];
      const rightKey = selectedFeatures[j];
      const pairKey = `${leftKey}__${rightKey}`;
      const leftVals = trainData.trainX[leftKey] ?? [];
      const rightVals = trainData.trainX[rightKey] ?? [];
      const leftLabel = trainData.featureLabels[leftKey] ?? leftKey;
      const rightLabel = trainData.featureLabels[rightKey] ?? rightKey;
      const categoricalPair = isCategorical(leftKey, trainData) || isCategorical(rightKey, trainData);
      const operators: FeatureOperation["operator"][] = categoricalPair
        ? ["product"]
        : ["product", "sum", "difference", "ratio", "absolute_difference"];

      operators.forEach((operator) => {
        const values = buildOperationValues(operator, leftVals, rightVals);
        const n = Math.min(values.length, residuals.length);
        if (n < 2) return;
        const score = Math.abs(pearsonCorrelation(values.slice(0, n), residuals.slice(0, n)));
        if (!Number.isFinite(score) || score <= 0) return;
        const candidate = {
          kind: "interaction",
          operator,
          sources: [leftKey, rightKey],
          key: operationKey(leftKey, rightKey, operator),
          label: operationLabel(leftLabel, rightLabel, operator),
          score,
        } as FeatureOperation & { score: number };
        const existing = suggestionsByPair.get(pairKey);
        if (!existing || candidate.score > existing.score) {
          suggestionsByPair.set(pairKey, candidate);
        }
      });
    }
  }
  return [...suggestionsByPair.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
};
