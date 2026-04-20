export type KnotSet = { x: number[]; y: number[] };

export type FeatureOperation = {
  kind: "interaction";
  operator: "product" | "sum" | "difference" | "ratio" | "absolute_difference";
  sources: [string, string];
  key?: string;
  label?: string;
};

// Shape function knots for a single feature — no raw data attached.
// When editableZ is present this is a 2-D interaction surface (read-only).
export type ShapeFunction = {
  key: string;
  label: string;
  editableX?: number[];
  editableY?: number[];
  categories?: string[];
  // 2-D interaction fields — only set for pairwise interaction shapes
  label2?: string;
  gridX?: number[];
  gridX2?: number[];
  xCategories?: string[];
  yCategories?: string[];
  editableZ?: number[][];
};

// Raw training/test data, separated from model and shape functions.
export type TrainData = {
  trainX: Record<string, number[]>;    // feature key → raw training values (scatter + metric calc)
  trainY: number[];
  testY: number[];
  categories: Record<string, string[]>; // ordered category lists per categorical feature
  featureLabels: Record<string, string>; // display labels per feature key
  featureDescriptions?: Record<string, string>; // optional human-readable descriptions; sent by backend or injected from frontend catalog
};

// Model hyperparameters and dataset metadata.
export type ModelInfo = {
  dataset: string;
  model_type: "igann" | "igann_interactive";
  task: "regression" | "classification";
  selected_features?: string[];
  selected_interactions?: string[];
  selected_operations?: FeatureOperation[];
  seed: number;
  n_estimators: number;
  boost_rate: number;
  init_reg: number;
  elm_alpha: number;
  early_stopping: number;
  scale_y: boolean;
  points: number;
};

// One versioned snapshot of shape functions produced by a training call.
export type ShapeFunctionVersion = {
  versionId: string;        // timestamp-based unique id
  timestamp: number;        // ms since epoch
  source: "train";
  center_shapes: boolean;
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

export type ModelDetails = {
  contribs: number[][];
  totals: number[];
  intercept: number;
  preds: number[];
};

export type Models = {
  baseModel: ModelDetails;
  editedModel: ModelDetails;
  comparisonModel?: ModelDetails | null;
  residuals: number[];
};

export type DatasetOption = {
  id: string;
  label: string;
  summary: string;
};

// Tab identifiers for the sidebar panel.
export type SidebarTab = "edit" | "history" | "features";

export type StatItem =
  | { label: string; kind: "value"; value: string }
  // initial  = server metrics from the very first train (never changes)
  // last     = server metrics from the version just before the current one
  // latest   = server metrics from the most recent train
  // current  = live frontend metrics recalculated from edited shape functions
  | { label: string; kind: "bar"; initial: number | null; last: number | null; latest: number | null; current: number | null; lowerIsBetter?: boolean; format?: string };

export type HistoryChange = { x: number; before?: number; after?: number; delta?: number };
export type HistoryEntry = {
  featureKey: string;
  action: string;
  ts: number;
  changes: HistoryChange[];
};


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

export type MetricWarning = {
  action: string;
  details: {
    label: string;
    current: number;
    previous: number;
    delta: number;
    deltaPct: number | null;
    lowerIsBetter: boolean;
  }[];
};
