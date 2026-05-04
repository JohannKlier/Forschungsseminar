import { FeatureOperation, TrainResponse } from "../types";

const TRAINER_URL = process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

const fetchWithFallback = async (path: string, init?: RequestInit) => {
  if (path.includes("undefined")) {
    throw new Error("Model path invalid.");
  }
  const apiResponse = await fetch(`/api${path}`, init);
  if (apiResponse.status !== 404) return apiResponse;
  return fetch(`${TRAINER_URL}${path}`, init);
};

const normalizeLegacyModelPayload = (payload: Record<string, unknown>): TrainResponse => {
  if ("model" in payload && "data" in payload && "version" in payload) {
    return payload as unknown as TrainResponse;
  }

  const partials = Array.isArray(payload.partials) ? payload.partials as Array<Record<string, unknown>> : [];
  const trainY = Array.isArray(payload.y) ? payload.y as number[] : [];
  const testY = Array.isArray(payload.testY) ? payload.testY as number[] : [];
  const timestamp = Date.now();

  const shapes = partials
    .map((partial) => {
      const key = typeof partial.key === "string" ? partial.key : "";
      if (!key) return null;
      const categories = Array.isArray(partial.categories) ? partial.categories.map(String) : undefined;
      return {
        key,
        label: typeof partial.label === "string" ? partial.label : key,
        editableX: Array.isArray(partial.editableX) ? partial.editableX as number[] : undefined,
        editableY: Array.isArray(partial.editableY) ? partial.editableY as number[] : undefined,
        editableZ: Array.isArray(partial.editableZ) ? partial.editableZ as number[][] : undefined,
        categories,
      };
    })
    .filter((shape): shape is NonNullable<typeof shape> => shape !== null);

  const trainX = Object.fromEntries(
    partials
      .map((partial) => {
        const key = typeof partial.key === "string" ? partial.key : "";
        const scatterX = Array.isArray(partial.scatterX) ? partial.scatterX : [];
        return key ? [key, scatterX] : null;
      })
      .filter((entry): entry is [string, unknown[]] => entry !== null),
  );

  const categories = Object.fromEntries(
    partials
      .map((partial) => {
        const key = typeof partial.key === "string" ? partial.key : "";
        const values = Array.isArray(partial.categories) ? partial.categories.map(String) : null;
        return key && values?.length ? [key, values] : null;
      })
      .filter((entry): entry is [string, string[]] => entry !== null),
  );

  const featureLabels = Object.fromEntries(
    partials
      .map((partial) => {
        const key = typeof partial.key === "string" ? partial.key : "";
        return key ? [key, typeof partial.label === "string" ? partial.label : key] : null;
      })
      .filter((entry): entry is [string, string] => entry !== null),
  );

  const source = payload.source === "igann_interactive" ? "igann_interactive" : "igann";
  const task = payload.task === "classification" ? "classification" : "regression";

  return {
    model: {
      dataset: typeof payload.dataset === "string" ? payload.dataset : "unknown",
      model_type: source,
      task,
      selected_features: Object.keys(featureLabels),
      selected_interactions: shapes.filter((shape) => shape.editableZ).map((shape) => shape.key),
      selected_operations: shapes
        .filter((shape) => shape.editableZ)
        .map((shape) => ({
          kind: "interaction" as const,
          operator: "product" as const,
          sources: (shape.key.split("__").slice(0, 2) as [string, string]),
          key: shape.key,
          label: shape.label,
        })),
      seed: typeof payload.seed === "number" ? payload.seed : 3,
      n_estimators: typeof payload.n_estimators === "number" ? payload.n_estimators : 100,
      boost_rate: typeof payload.boost_rate === "number" ? payload.boost_rate : 0.1,
      init_reg: typeof payload.init_reg === "number" ? payload.init_reg : 1,
      elm_alpha: typeof payload.elm_alpha === "number" ? payload.elm_alpha : 1,
      early_stopping: typeof payload.early_stopping === "number" ? payload.early_stopping : 50,
      n_hid: typeof payload.n_hid === "number" ? payload.n_hid : 10,
      scale_y: typeof payload.scale_y === "boolean" ? payload.scale_y : true,
      points: typeof payload.points === "number" ? payload.points : 250,
    },
    data: {
      trainX: trainX as Record<string, number[]>,
      trainY,
      testY,
      categories,
      featureLabels,
      featureDescriptions:
        payload.featureDescriptions != null && typeof payload.featureDescriptions === "object"
          ? Object.fromEntries(
              Object.entries(payload.featureDescriptions as Record<string, unknown>).filter(
                (entry): entry is [string, string] => typeof entry[1] === "string",
              ),
            )
          : undefined,
    },
    version: {
      versionId: String(timestamp),
      timestamp,
      source: "train",
      center_shapes: typeof payload.center_shapes === "boolean" ? payload.center_shapes : false,
      intercept: typeof payload.intercept === "number" ? payload.intercept : 0,
      trainMetrics: (payload.trainMetrics as TrainResponse["version"]["trainMetrics"]) ?? { count: trainY.length },
      testMetrics: (payload.testMetrics as TrainResponse["version"]["testMetrics"]) ?? { count: testY.length },
      shapes,
    },
  };
};

export type TrainRequest = {
  dataset: string;
  model_type: "igann" | "igann_interactive";
  center_shapes: boolean;
  selected_features?: string[];
  seed: number;
  points: number;
  n_estimators: number;
  boost_rate: number;
  init_reg: number;
  elm_alpha: number;
  early_stopping: number;
  n_hid: number;
  scale_y: boolean;
  sample_size?: number;
};

export const trainModel = async (request: TrainRequest): Promise<TrainResponse> => {
  const response = await fetchWithFallback("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`Trainer responded with ${response.status}`);
  return (await response.json()) as TrainResponse;
};

export const listModels = async (): Promise<string[]> => {
  const response = await fetchWithFallback("/models");
  if (!response.ok) throw new Error(`Model list responded with ${response.status}`);
  const payload = (await response.json()) as { models?: string[] };
  return payload.models ?? [];
};

export const loadModel = async (name: string): Promise<TrainResponse> => {
  if (!name || name === "undefined") {
    throw new Error("Model name missing.");
  }
  const response = await fetchWithFallback(`/models/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`Model load responded with ${response.status}`);
  return normalizeLegacyModelPayload((await response.json()) as Record<string, unknown>);
};
