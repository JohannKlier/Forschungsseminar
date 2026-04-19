import { interpolateFeature } from "../lib/interpolate";
import type { KnotSet, ModelDetails, ModelInfo, ShapeFunction, ShapeFunctionVersion, TrainData } from "../types";

export {};

type WorkerPayload = {
  version: ShapeFunctionVersion | null;
  trainData: TrainData | null;
  modelInfo: ModelInfo | null;
  baselineKnots: Record<string, KnotSet>;
  knotEdits: Record<string, KnotSet>;
  comparisonKnotEdits?: Record<string, KnotSet> | null;
};

type WorkerCache = {
  contextKey: string;
  baseModel: ModelDetails;
  editedModel: ModelDetails;
  editedKnotsMap: Record<string, KnotSet>;
  comparisonModel: ModelDetails | null;
  comparisonKnotsMap: Record<string, KnotSet> | null;
};

let cache: WorkerCache | null = null;

const areKnotsEqual = (left?: KnotSet, right?: KnotSet) => {
  if (!left || !right) return left === right;
  return (
    left.x.length === right.x.length &&
    left.y.length === right.y.length &&
    left.x.every((value, index) => value === right.x[index]) &&
    left.y.every((value, index) => value === right.y[index])
  );
};

const getDefaultKnots = (shape: ShapeFunction): KnotSet => ({
  x: shape.editableX ?? [],
  y: shape.editableY ?? [],
});

const getShapeKnots = (shape: ShapeFunction, knotsMap: Record<string, KnotSet>) =>
  knotsMap[shape.key] ?? getDefaultKnots(shape);

const getContextKey = (
  version: ShapeFunctionVersion,
  trainData: TrainData,
  modelInfo: ModelInfo | null,
  baselineKnots: Record<string, KnotSet>,
) => {
  const shapeSignature = version.shapes
    .map((shape) => {
      const knots = getShapeKnots(shape, baselineKnots);
      return [
        shape.key,
        shape.editableZ ? "2d" : "1d",
        shape.categories?.join("|") ?? "",
        knots.x.join(","),
        knots.y.join(","),
      ].join(":");
    })
    .join(";");
  return [
    version.versionId,
    version.intercept,
    modelInfo?.task ?? "regression",
    trainData.trainY?.length ?? 0,
    shapeSignature,
  ].join("::");
};

const findChangedEditableKeys = (
  previousMap: Record<string, KnotSet>,
  nextMap: Record<string, KnotSet>,
  shapes: ShapeFunction[],
) =>
  shapes.flatMap((shape) => {
    if (shape.editableZ) return [];
    const previous = getShapeKnots(shape, previousMap);
    const next = getShapeKnots(shape, nextMap);
    return areKnotsEqual(previous, next) ? [] : [shape.key];
  });

const buildContrib = (
  shape: ShapeFunction,
  knotsMap: Record<string, KnotSet>,
  trainX: Record<string, Array<number | string>>,
) => {
  // Interaction shapes are not editable; trainX holds precomputed per-row contributions.
  if (shape.editableZ) {
    return (trainX[shape.key] ?? []).map((v) =>
      typeof v === "number" && Number.isFinite(v) ? v : 0
    );
  }

  const source = getShapeKnots(shape, knotsMap);
  const scatterX = trainX[shape.key] ?? [];
  if (shape.categories && shape.categories.length) {
    const mapping = new Map<number, number>();
    shape.categories.forEach((_cat: string, idx: number) => {
      const yVal = source.y[idx] ?? 0;
      mapping.set(idx, yVal);
    });
    return scatterX.map((raw) => {
      const idx = shape.categories?.indexOf(String(raw)) ?? -1;
      return mapping.get(idx) ?? 0;
    });
  }

  return interpolateFeature(scatterX, source);
};

const buildContribs = (
  shapes: ShapeFunction[],
  knotsMap: Record<string, KnotSet>,
  trainX: Record<string, Array<number | string>>,
) => shapes.map((shape) => buildContrib(shape, knotsMap, trainX));

const sumTotals = (contribs: number[][], rowCount: number) =>
  Array.from({ length: rowCount }, (_: unknown, idx: number) =>
    contribs.reduce((sum, arr) => sum + (arr[idx] ?? 0), 0)
  );

const applyLink = (values: number[], task: ModelInfo["task"] | undefined) => {
  if (task === "classification") {
    return values.map((val) => 1 / (1 + Math.exp(-val)));
  }
  return values;
};

const buildModel = (
  contribs: number[][],
  totals: number[],
  intercept: number,
  task: ModelInfo["task"] | undefined,
): ModelDetails => {
  const preds = applyLink(totals.map((total) => total + intercept), task);
  return { contribs, totals, intercept, preds };
};

const updateModelForFeature = (
  model: ModelDetails,
  shapes: ShapeFunction[],
  featureKey: string,
  knotsMap: Record<string, KnotSet>,
  trainX: Record<string, Array<number | string>>,
  task: ModelInfo["task"] | undefined,
) => {
  const shapeIndex = shapes.findIndex((shape) => shape.key === featureKey);
  if (shapeIndex < 0) return model;

  const shape = shapes[shapeIndex];
  const previousContrib = model.contribs[shapeIndex] ?? [];
  const nextContrib = buildContrib(shape, knotsMap, trainX);
  const nextTotals = model.totals.map((total, index) =>
    total - (previousContrib[index] ?? 0) + (nextContrib[index] ?? 0)
  );
  const nextContribs = model.contribs.slice();
  nextContribs[shapeIndex] = nextContrib;

  return buildModel(nextContribs, nextTotals, model.intercept, task);
};

self.onmessage = (event: MessageEvent) => {
  const { version, trainData, modelInfo, baselineKnots, knotEdits, comparisonKnotEdits } = event.data as WorkerPayload;
  if (!version || !trainData) {
    cache = null;
    self.postMessage(null);
    return;
  }

  const trainY: number[] = trainData.trainY ?? [];
  const trainX = trainData.trainX as Record<string, Array<number | string>>;
  const rowCount = trainY.length;
  const divisor = rowCount || 1;
  const contextKey = getContextKey(version, trainData, modelInfo, baselineKnots);
  const editedKnotsMap = { ...baselineKnots, ...knotEdits };
  const comparisonKnotsMap = comparisonKnotEdits
    ? { ...baselineKnots, ...comparisonKnotEdits }
    : null;

  const buildFullModel = (knotsMap: Record<string, KnotSet>, intercept: number) => {
    const contribs = buildContribs(version.shapes, knotsMap, trainX);
    const totals = sumTotals(contribs, rowCount);
    return buildModel(contribs, totals, intercept, modelInfo?.task);
  };

  const buildFullCache = (): WorkerCache => {
    const baseContribs = buildContribs(version.shapes, baselineKnots, trainX);
    const baseTotals = sumTotals(baseContribs, rowCount);
    const baseIntercept =
      version.intercept != null
        ? version.intercept
        : baseTotals.reduce((sum: number, totalVal: number, idx: number) => sum + (trainY[idx] - totalVal), 0) / divisor;

    const baseModel = buildModel(baseContribs, baseTotals, baseIntercept, modelInfo?.task);
    const editedModel = buildFullModel(editedKnotsMap, baseIntercept);
    const comparisonModel = comparisonKnotsMap ? buildFullModel(comparisonKnotsMap, baseIntercept) : null;
    return {
      contextKey,
      baseModel,
      editedModel,
      editedKnotsMap,
      comparisonModel,
      comparisonKnotsMap,
    };
  };

  if (!cache || cache.contextKey !== contextKey) {
    cache = buildFullCache();
  } else {
    const editedChangedKeys = findChangedEditableKeys(cache.editedKnotsMap, editedKnotsMap, version.shapes);
    const editedModel = editedChangedKeys.length === 0
      ? cache.editedModel
      : editedChangedKeys.length === 1
        ? updateModelForFeature(cache.editedModel, version.shapes, editedChangedKeys[0], editedKnotsMap, trainX, modelInfo?.task)
        : buildFullModel(editedKnotsMap, cache.baseModel.intercept);

    const comparisonChangedKeys = comparisonKnotsMap && cache.comparisonKnotsMap
      ? findChangedEditableKeys(cache.comparisonKnotsMap, comparisonKnotsMap, version.shapes)
      : [];
    const comparisonModel = (() => {
      if (!comparisonKnotsMap) return null;
      if (!cache?.comparisonKnotsMap || !cache.comparisonModel) {
        return buildFullModel(comparisonKnotsMap, cache?.baseModel.intercept ?? 0);
      }
      if (comparisonChangedKeys.length === 0) return cache.comparisonModel;
      if (comparisonChangedKeys.length === 1) {
        return updateModelForFeature(
          cache.comparisonModel,
          version.shapes,
          comparisonChangedKeys[0],
          comparisonKnotsMap,
          trainX,
          modelInfo?.task,
        );
      }
      return buildFullModel(comparisonKnotsMap, cache.baseModel.intercept);
    })();

    cache = {
      ...cache,
      editedModel,
      editedKnotsMap,
      comparisonModel,
      comparisonKnotsMap,
    };
  }

  const residuals = trainY.map((yVal: number, idx: number) => yVal - (cache?.editedModel.preds[idx] ?? 0));
  self.postMessage({
    baseModel: cache.baseModel,
    editedModel: cache.editedModel,
    comparisonModel: cache.comparisonModel,
    residuals,
  });
};
