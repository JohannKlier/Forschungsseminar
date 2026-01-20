import { interpolateFeature } from "../lib/interpolate";

export {};

self.onmessage = (event: MessageEvent) => {
  const { result, baselineKnots, knotEdits } = event.data as {
    result: any;
    baselineKnots: Record<string, { x: number[]; y: number[] }>;
    knotEdits: Record<string, { x: number[]; y: number[] }>;
  };
  if (!result) {
    self.postMessage(null);
    return;
  }

  const n = result.y.length || 1;

  const buildContribs = (knotsMap: Record<string, { x: number[]; y: number[] }>) =>
    result.partials.map((partial: any) => {
      const source = knotsMap[partial.key] ?? { x: partial.editableX ?? [], y: partial.editableY ?? [] };
      if (partial.categories && partial.categories.length) {
        const mapping = new Map<number, number>();
        partial.categories.forEach((cat: string, idx: number) => {
          const yVal = source.y[idx] ?? 0;
          mapping.set(idx, yVal);
        });
        return partial.scatterX.map((raw: any) => {
          const idx = partial.categories?.indexOf(String(raw)) ?? -1;
          return mapping.get(idx) ?? 0;
        });
      }
      return interpolateFeature(partial.scatterX, source);
    });

  const sumTotals = (contribs: number[][]) => result.y.map((_: number, idx: number) => contribs.reduce((sum, arr) => sum + (arr[idx] ?? 0), 0));

  const baseContribs = buildContribs(baselineKnots);
  const baseTotals = sumTotals(baseContribs);
  const baseIntercept =
    result.intercept != null ? result.intercept : baseTotals.reduce((sum: number, totalVal: number, idx: number) => sum + (result.y[idx] - totalVal), 0) / n;
  const applyLink = (values: number[]) => {
    if (result.task === "classification") {
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
  const residuals = result.y.map((yVal: number, idx: number) => yVal - editedModel.preds[idx]);

  self.postMessage({ baseModel, editedModel, residuals });
};
