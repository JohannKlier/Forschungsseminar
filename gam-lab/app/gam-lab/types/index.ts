export type KnotSet = { x: number[]; y: number[] };

export type FeatureCurve = {
  key: string;
  label: string;
  scatterX: number[];
  editableX?: number[];
  editableY?: number[];
  gridX?: number[];
  curve?: number[];
  categories?: string[];
};

export type TrainResponse = {
  dataset: string;
  seed?: number;
  n_estimators?: number;
  boost_rate?: number;
  init_reg?: number;
  elm_alpha?: number;
  early_stopping?: number;
  scale_y?: boolean;
  intercept?: number;
  task?: "regression" | "classification";
  partials: FeatureCurve[];
  y: number[];
  testY?: number[];
  source?: "service" | "local" | "model" | "saved";
  points?: number;
  trainMetrics?: MetricSummary;
  testMetrics?: MetricSummary;
  testPreds?: number[];
};

export type DatasetOption = {
  id: string;
  label: string;
  summary: string;
};

export type ModelDetails = {
  contribs: number[][];
  totals: number[];
  intercept: number;
  preds: number[];
};

export type Models = {
  baseModel: ModelDetails;
  editedModel: ModelDetails;
  residuals: number[];
};

export type StatItem =
  | { label: string; kind: "value"; value: string }
  | { label: string; kind: "bar"; base: number | null; current: number | null; lowerIsBetter?: boolean; format?: string };

export type MetricSummary = { rmse: number | null; r2: number | null; count: number };
