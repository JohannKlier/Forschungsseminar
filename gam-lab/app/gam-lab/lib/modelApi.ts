import { TrainResponse } from "../types";

const TRAINER_URL = process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

const fetchWithFallback = async (path: string, init?: RequestInit) => {
  if (path.includes("undefined")) {
    throw new Error("Model path invalid.");
  }
  const apiResponse = await fetch(`/api${path}`, init);
  if (apiResponse.status !== 404) return apiResponse;
  return fetch(`${TRAINER_URL}${path}`, init);
};

export type TrainRequest = {
  dataset: string;
  seed: number;
  points: number;
  n_estimators: number;
  boost_rate: number;
  init_reg: number;
  elm_alpha: number;
  early_stopping: number;
  scale_y: boolean;
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
  return (await response.json()) as TrainResponse;
};
