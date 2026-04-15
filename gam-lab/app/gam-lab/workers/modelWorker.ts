import { interpolateFeature } from "../lib/interpolate";
import type { KnotSet, ModelInfo, ShapeFunction, ShapeFunctionVersion, TrainData } from "../types";

export {};

type WorkerPayload = {
  version: ShapeFunctionVersion | null;
  trainData: TrainData | null;
  modelInfo: ModelInfo | null;
  baselineKnots: Record<string, KnotSet>;
  knotEdits: Record<string, KnotSet>;
  comparisonKnotEdits?: Record<string, KnotSet> | null;
};

self.onmessage = (event: MessageEvent) => {
  const { version, trainData, modelInfo, baselineKnots, knotEdits, comparisonKnotEdits } = event.data as WorkerPayload;
  if (!version || !trainData) {
    self.postMessage(null);
    return;
  }

  const trainY: number[] = trainData.trainY ?? [];
  const trainX = trainData.trainX as Record<string, Array<number | string>>;

  const n = trainY.length || 1;

  const buildContribs = (knotsMap: Record<string, KnotSet>) =>
    version.shapes.map((shape: ShapeFunction) => {
      // Interaction shapes are not editable; trainX holds precomputed per-row contributions.
      if (shape.editableZ) {
        return (trainX[shape.key] ?? []).map((v) =>
          typeof v === "number" && Number.isFinite(v) ? v : 0
        );
      }
      const source = knotsMap[shape.key] ?? { x: shape.editableX ?? [], y: shape.editableY ?? [] };
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
    });

  const sumTotals = (contribs: number[][]) =>
    trainY.map((_: number, idx: number) => contribs.reduce((sum, arr) => sum + (arr[idx] ?? 0), 0));

  const baseContribs = buildContribs(baselineKnots);
  const baseTotals = sumTotals(baseContribs);
  const baseIntercept =
    version.intercept != null
      ? version.intercept
      : baseTotals.reduce((sum: number, totalVal: number, idx: number) => sum + (trainY[idx] - totalVal), 0) / n;

  const applyLink = (values: number[]) => {
    if (modelInfo?.task === "classification") {
      return values.map((val) => 1 / (1 + Math.exp(-val)));
    }
    return values;
  };

  const buildModel = (contribs: number[][], totals: number[]) => {
    const intercept = baseIntercept;
    const preds = applyLink(totals.map((t) => t + intercept));
    return { contribs, totals, intercept, preds };
  };

  const baseModel = buildModel(baseContribs, baseTotals);
  const editedKnotsMap = { ...baselineKnots, ...knotEdits };
  const editedContribs = buildContribs(editedKnotsMap);
  const editedTotals = sumTotals(editedContribs);
  const editedModel = buildModel(editedContribs, editedTotals);
  const comparisonModel = comparisonKnotEdits
    ? (() => {
        const comparisonKnotsMap = { ...baselineKnots, ...comparisonKnotEdits };
        const comparisonContribs = buildContribs(comparisonKnotsMap);
        const comparisonTotals = sumTotals(comparisonContribs);
        return buildModel(comparisonContribs, comparisonTotals);
      })()
    : null;
  const residuals = trainY.map((yVal: number, idx: number) => yVal - editedModel.preds[idx]);

  self.postMessage({ baseModel, editedModel, comparisonModel, residuals });
};
