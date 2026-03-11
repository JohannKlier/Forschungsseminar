import { interpolateFeature } from "../lib/interpolate";

export {};

self.onmessage = (event: MessageEvent) => {
  const { version, trainData, modelInfo, baselineKnots, knotEdits } = event.data as {
    version: any;
    trainData: any;
    modelInfo: any;
    baselineKnots: Record<string, { x: number[]; y: number[] }>;
    knotEdits: Record<string, { x: number[]; y: number[] }>;
  };
  if (!version || !trainData) {
    self.postMessage(null);
    return;
  }

  const trainY: number[] = trainData.trainY ?? [];
  const trainX: Record<string, number[]> = trainData.trainX ?? {};

  const n = trainY.length || 1;

  const buildContribs = (knotsMap: Record<string, { x: number[]; y: number[] }>) =>
    version.shapes.map((shape: any) => {
      const source = knotsMap[shape.key] ?? { x: shape.editableX ?? [], y: shape.editableY ?? [] };
      const scatterX: any[] = trainX[shape.key] ?? [];
      if (shape.categories && shape.categories.length) {
        const mapping = new Map<number, number>();
        shape.categories.forEach((_cat: string, idx: number) => {
          const yVal = source.y[idx] ?? 0;
          mapping.set(idx, yVal);
        });
        return scatterX.map((raw: any) => {
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
  const residuals = trainY.map((yVal: number, idx: number) => yVal - editedModel.preds[idx]);

  self.postMessage({ baseModel, editedModel, residuals });
};
