import { TrainResponse } from "../types";

const TRAINER_URL = process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

const fetchWithFallback = async (path: string, init?: RequestInit) => {
  if (path.includes("undefined")) {
    throw new Error("Saved model path invalid.");
  }
  const apiResponse = await fetch(`/api${path}`, init);
  if (apiResponse.status !== 404) return apiResponse;
  return fetch(`${TRAINER_URL}${path}`, init);
};

export const listSavedModels = async (): Promise<string[]> => {
  const response = await fetchWithFallback("/saved-models");
  if (!response.ok) throw new Error(`Saved model list responded with ${response.status}`);
  const payload = (await response.json()) as { models?: string[] };
  return payload.models ?? [];
};

export const loadSavedModel = async (name: string): Promise<TrainResponse> => {
  if (!name || name === "undefined") {
    throw new Error("Saved model name missing.");
  }
  const response = await fetchWithFallback(`/saved-models/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`Saved model load responded with ${response.status}`);
  return (await response.json()) as TrainResponse;
};

export const saveModel = async (name: string, payload: TrainResponse): Promise<void> => {
  const response = await fetchWithFallback("/saved-models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, payload }),
  });
  if (!response.ok) throw new Error(`Save responded with ${response.status}`);
};
