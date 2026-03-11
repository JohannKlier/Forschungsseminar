export type KnotSet = { x: number[]; y: number[] };

// Shape function knots for a single feature — no raw data attached.
export type ShapeFunction = {
  key: string;
  label: string;
  editableX?: number[];
  editableY?: number[];
  categories?: string[];
};

// Raw training/test data, separated from model and shape functions.
// Kept stable across refits of the same dataset+seed so the frontend can
// recalculate metrics for edited shape functions without re-fetching.
export type TrainData = {
  trainX: Record<string, number[]>;    // feature key → raw training values (scatter + metric calc)
  trainY: number[];
  testY: number[];
  categories: Record<string, string[]>; // ordered category lists per categorical feature
  featureLabels: Record<string, string>; // display labels per feature key
};

// Model hyperparameters and dataset metadata.
export type ModelInfo = {
  dataset: string;
  model_type: "igann" | "igann_interactive";
  task: "regression" | "classification";
  seed: number;
  n_estimators: number;
  boost_rate: number;
  init_reg: number;
  elm_alpha: number;
  early_stopping: number;
  scale_y: boolean;
  points: number;
};

// One versioned snapshot of shape functions produced by a single train or refit call.
// Each retraining appends a new version; the frontend accumulates them.
export type ShapeFunctionVersion = {
  versionId: string;        // timestamp-based unique id
  timestamp: number;        // ms since epoch
  source: "train" | "refit";
  center_shapes: boolean;
  locked_features: string[];
  refit_from_edits: boolean;
  intercept: number;
  trainMetrics: MetricSummary;
  testMetrics: MetricSummary;
  shapes: ShapeFunction[];
};

// Full API response: model info + separated data + one version snapshot.
export type TrainResponse = {
  model: ModelInfo;
  data: TrainData;
  version: ShapeFunctionVersion;
  source?: "service" | "local" | "model" | "saved";
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
  // initial  = server metrics from the very first train (never changes)
  // latest   = server metrics from the most recent train/refit
  // current  = live frontend metrics recalculated from edited shape functions
  | { label: string; kind: "bar"; initial: number | null; latest: number | null; current: number | null; lowerIsBetter?: boolean; format?: string };

export type MetricSummary = {
  count: number;
  rmse?: number | null;
  mae?: number | null;
  r2?: number | null;
  acc?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
};
