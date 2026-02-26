import { useEffect, useMemo, useRef, useState } from "react";
import { DatasetOption, FeatureCurve, KnotSet, Models, StatItem, TrainResponse } from "../types";
import { loadModel, refitModel, trainModel } from "../lib/modelApi";
import { loadSavedModel, saveModel } from "../lib/savedModelApi";

// Dataset registry used by the UI selectors and training requests.
const DATASETS: DatasetOption[] = [
  { id: "bike_hourly", label: "Bike sharing (hourly)", summary: "Hourly rentals with weather/seasonality." },
  { id: "adult_income", label: "Adult income", summary: "Census income classification (<=50K vs >50K)." },
  {
    id: "breast_cancer",
    label: "Breast cancer (Wisconsin)",
    summary: "Diagnostic features for malignant vs benign tumors.",
  },
];

// Fixed seed keeps demos deterministic across refreshes.
const DEFAULT_SEED = 3;
// Temporary switch to disable local cached edit restore/persist.
const CACHE_EDITS_ENABLED = false;

// Normalize the editable knot representation for categorical vs continuous features.
const buildEditableKnots = (partial: FeatureCurve): KnotSet => {
  if (partial.categories && partial.categories.length) {
    const yVals = partial.editableY?.length ? [...partial.editableY] : [];
    const xVals = partial.categories.map((_, idx) => idx);
    return { x: xVals, y: yVals };
  }
  const base =
    partial.editableX?.length && partial.editableY?.length
      ? { x: [...partial.editableX], y: [...partial.editableY] }
      : partial.gridX?.length && partial.curve?.length
        ? { x: [...partial.gridX], y: [...partial.curve] }
        : { x: [], y: [] };

  if (base.x.length === 0 || base.y.length === 0) return { x: [], y: [] };
  return base;
};

// Main state/logic hook that powers the GAM Lab page.
type InitOptions = {
  initialModel?: string | null;
  initialTrain?: {
    dataset: string;
    model_type: "igann" | "igann_interactive";
    center_shapes: boolean;
    points: number;
    seed: number;
    n_estimators: number;
    boost_rate: number;
    init_reg: number;
    elm_alpha: number;
    early_stopping: number;
    scale_y: boolean;
  } | null;
};

type HistoryChange = { x: number; before?: number; after?: number; delta?: number };
type HistoryEntry = {
  featureKey: string;
  action: string;
  ts: number;
  changes: HistoryChange[];
};

export const useGamLab = (options: InitOptions = {}) => {
  // User-configurable training inputs.
  const [dataset, setDataset] = useState(DATASETS[0].id);
  const [modelType, setModelType] = useState<"igann" | "igann_interactive">("igann");
  const [centerShapes, setCenterShapes] = useState(false);
  const [shapePoints, setShapePoints] = useState(250);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [nEstimators, setNEstimators] = useState(100);
  const [boostRate, setBoostRate] = useState(0.1);
  const [initReg, setInitReg] = useState(1);
  const [elmAlpha, setElmAlpha] = useState(1);
  const [earlyStopping, setEarlyStopping] = useState(50);
  const [scaleY, setScaleY] = useState(true);
  // Global training/loading status and payloads.
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [debugPayload, setDebugPayload] = useState<TrainResponse | null>(null);
  const [debugError, setDebugError] = useState<string | null>(null);
  // Editing state for the current feature and its knot history.
  const [activePartialIdx, setActivePartialIdx] = useState(0);
  const [baselineKnots, setBaselineKnots] = useState<Record<string, KnotSet>>({});
  const [knots, setKnots] = useState<KnotSet>({ x: [], y: [] });
  const [knotEdits, setKnotEdits] = useState<Record<string, KnotSet>>({});
  const [committedEdits, setCommittedEdits] = useState<Record<string, KnotSet>>({});
  const [selectedKnots, setSelectedKnots] = useState<number[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyCursor, setHistoryCursor] = useState(0);
  const [lockedFeatures, setLockedFeatures] = useState<string[]>([]);
  // Derived model predictions and worker wiring.
  const [models, setModels] = useState<Models | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerDebounceRef = useRef<number | null>(null);
  // Model/source selector and sidebar tab state.
  const [modelSource, setModelSource] = useState<string>("");
  const [sidebarTab, setSidebarTab] = useState<"edit" | "history">("edit");
  const cacheLoadedRef = useRef<string | null>(null);

  // Convenience selectors for the UI.
  const selectedDataset = useMemo(() => DATASETS.find((item) => item.id === dataset) ?? DATASETS[0], [dataset]);
  const partial = useMemo(() => (result ? result.partials[activePartialIdx] ?? result.partials[0] ?? null : null), [
    result,
    activePartialIdx,
  ]);

  const displayLabel = useMemo(() => {
    if (!partial || !result) return "";
    return partial.label || partial.key || `x${(result.partials.indexOf(partial) ?? 0) + 1}`;
  }, [partial, result]);

  // Trigger a fresh training run for the selected dataset and parameters.
  const train = async (
    overrides?: {
      dataset?: string;
      model_type?: "igann" | "igann_interactive";
      center_shapes?: boolean;
      points?: number;
      seed?: number;
      n_estimators?: number;
      boost_rate?: number;
      init_reg?: number;
      elm_alpha?: number;
      early_stopping?: number;
      scale_y?: boolean;
    },
  ) => {
    setStatus("loading");
    setDebugError(null);
    setModelSource("train");
    try {
      const payload = await trainModel({
        dataset: overrides?.dataset ?? dataset,
        model_type: overrides?.model_type ?? modelType,
        center_shapes: overrides?.center_shapes ?? centerShapes,
        seed: overrides?.seed ?? seed,
        points: overrides?.points ?? shapePoints,
        n_estimators: overrides?.n_estimators ?? nEstimators,
        boost_rate: overrides?.boost_rate ?? boostRate,
        init_reg: overrides?.init_reg ?? initReg,
        elm_alpha: overrides?.elm_alpha ?? elmAlpha,
        early_stopping: overrides?.early_stopping ?? earlyStopping,
        scale_y: overrides?.scale_y ?? scaleY,
      });
      setResult({ ...payload, source: "service" });
      setDebugPayload(payload);
      setStatus("idle");
    } catch (error) {
      console.warn("Trainer service unavailable.", error);
      setResult(null);
      setDebugPayload(null);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

  // Public wrapper for UI buttons.
  const manualTrain = async (
    overrides?: {
      dataset?: string;
      model_type?: "igann" | "igann_interactive";
      center_shapes?: boolean;
      points?: number;
      seed?: number;
      n_estimators?: number;
      boost_rate?: number;
      init_reg?: number;
      elm_alpha?: number;
      early_stopping?: number;
      scale_y?: boolean;
    },
  ) => {
    await train(overrides);
  };

  // Initialize from landing-page query parameters.
  useEffect(() => {
    if (options.initialTrain) {
      setDataset(options.initialTrain.dataset);
      setModelType(options.initialTrain.model_type);
      setCenterShapes(options.initialTrain.center_shapes);
      setShapePoints(options.initialTrain.points);
      setSeed(options.initialTrain.seed);
      setNEstimators(options.initialTrain.n_estimators);
      setBoostRate(options.initialTrain.boost_rate);
      setInitReg(options.initialTrain.init_reg);
      setElmAlpha(options.initialTrain.elm_alpha);
      setEarlyStopping(options.initialTrain.early_stopping);
      setScaleY(options.initialTrain.scale_y);
      manualTrain({
        dataset: options.initialTrain.dataset,
        model_type: options.initialTrain.model_type,
        center_shapes: options.initialTrain.center_shapes,
        points: options.initialTrain.points,
        seed: options.initialTrain.seed,
        n_estimators: options.initialTrain.n_estimators,
        boost_rate: options.initialTrain.boost_rate,
        init_reg: options.initialTrain.init_reg,
        elm_alpha: options.initialTrain.elm_alpha,
        early_stopping: options.initialTrain.early_stopping,
        scale_y: options.initialTrain.scale_y,
      });
      return;
    }
    if (options.initialModel && options.initialModel !== "undefined") {
      handleModelSelect(options.initialModel);
      return;
    }
    setModelSource("train");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load a model when the selector changes.
  const handleModelSelect = async (value: string) => {
    
    setModelSource(value);
    if (value === "train") return;
    setStatus("loading");
    setDebugError(null);
    try {
      const isSaved = value.startsWith("saved:");
      const isBase = value.startsWith("model:");
      const name = isSaved
        ? value.replace("saved:", "")
        : isBase
          ? value.replace("model:", "")
          : value;
      
      const payload = isSaved ? await loadSavedModel(name) : await loadModel(name);
      setResult({ ...payload, source: isSaved ? "saved" : "model" });
      setDebugPayload(payload);
      setDataset(payload.dataset);
      if (payload.model_type === "igann" || payload.model_type === "igann_interactive") {
        setModelType(payload.model_type);
      }
      if (typeof payload.center_shapes === "boolean") {
        setCenterShapes(payload.center_shapes);
      }
      setLockedFeatures(Array.isArray(payload.locked_features) ? payload.locked_features.map(String) : []);
      setShapePoints(payload.points ?? shapePoints);
      if (typeof payload.seed === "number") setSeed(payload.seed);
      if (typeof payload.n_estimators === "number") setNEstimators(payload.n_estimators);
      if (typeof payload.boost_rate === "number") setBoostRate(payload.boost_rate);
      if (typeof payload.init_reg === "number") setInitReg(payload.init_reg);
      if (typeof payload.elm_alpha === "number") setElmAlpha(payload.elm_alpha);
      if (typeof payload.early_stopping === "number") setEarlyStopping(payload.early_stopping);
      if (typeof payload.scale_y === "boolean") setScaleY(payload.scale_y);
      setStatus("idle");
    } catch (error) {
      console.warn("Failed to load saved model.", error);
      setResult(null);
      setDebugPayload(null);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

  // Convert the current edits into a payload compatible with the backend saver.
  const buildSavePayload = () => {
    if (!result) return null;
    const edits = committedEdits;
    const updatedPartials = result.partials.map((partialItem) => {
      const edit = edits[partialItem.key];
      if (!edit) return partialItem;
      return {
        ...partialItem,
        editableX: [...edit.x],
        editableY: [...edit.y],
      };
    });
    return { ...result, partials: updatedPartials };
  };

  // Persist edits and refresh model options.
  const handleSave = async () => {
    if (!result) return;
    const baseName = `${dataset}-edited-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}`;
    const name = window.prompt("Save edited model as:", baseName);
    if (!name) return;
    const payload = buildSavePayload();
    if (!payload) return;
    try {
      await saveModel(name, payload);
      const savedValue = `saved:${name.replace(/\\.json$/, "")}`;
      setModelSource(savedValue);
      handleModelSelect(savedValue);
    } catch (error) {
      console.warn("Failed to save model.", error);
    }
  };

  const toggleFeatureLock = (featureKey: string) => {
    setLockedFeatures((prev) =>
      prev.includes(featureKey) ? prev.filter((key) => key !== featureKey) : [...prev, featureKey],
    );
  };

  const manualRefitFromEdits = async () => {
    if (!result) return;
    if (modelType !== "igann_interactive") {
      setDebugError("Refit from edited shape functions requires model type IGANN interactive.");
      return;
    }
    const payload = buildSavePayload();
    if (!payload) return;
    setStatus("loading");
    setDebugError(null);
    setModelSource("train");
    try {
      const refitPoints = typeof result.points === "number" ? result.points : shapePoints;
      const refitPayload = await refitModel({
        dataset,
        model_type: modelType,
        center_shapes: centerShapes,
        seed,
        points: refitPoints,
        n_estimators: nEstimators,
        boost_rate: boostRate,
        init_reg: initReg,
        elm_alpha: elmAlpha,
        early_stopping: earlyStopping,
        scale_y: scaleY,
        partials: payload.partials.map((partialItem) => ({
          key: partialItem.key,
          categories: partialItem.categories,
          editableX: partialItem.editableX,
          editableY: partialItem.editableY,
        })),
        refit_estimators: nEstimators,
        refit_early_stopping: earlyStopping,
        locked_features: lockedFeatures,
      });
      setResult({ ...refitPayload, source: "service" });
      setDebugPayload(refitPayload);
      setLockedFeatures(Array.isArray(refitPayload.locked_features) ? refitPayload.locked_features.map(String) : lockedFeatures);
      if (typeof refitPayload.center_shapes === "boolean") {
        setCenterShapes(refitPayload.center_shapes);
      }
      if (typeof refitPayload.points === "number") {
        setShapePoints(refitPayload.points);
      }
      setStatus("idle");
    } catch (error) {
      console.warn("Refit failed.", error);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

  // Rebuild editable knot state when a new model is loaded or trained.
  useEffect(() => {
    setActivePartialIdx(0);
    if (!result) return;
    // Use the trained model's point count; avoid resampling after training.
    const next: Record<string, KnotSet> = {};
    result.partials.forEach((partialItem) => {
      next[partialItem.key] = buildEditableKnots(partialItem);
    });
    setBaselineKnots((prev) => {
      // Keep the original baseline across refit iterations so "Before" always
      // refers to the initial model, not only the previous refit step.
      if (!result.refit_from_edits) return next;
      const merged: Record<string, KnotSet> = {};
      Object.keys(next).forEach((key) => {
        const existing = prev[key];
        merged[key] =
          existing && existing.x?.length && existing.y?.length ? existing : next[key];
      });
      return merged;
    });
    // Current view should reflect the latest model result (including refit output),
    // while baselineKnots may stay anchored to the initial model.
    setKnotEdits(next);
    setCommittedEdits(next);
    setHistory([]);
    setHistoryCursor(0);
    setLockedFeatures((prev) =>
      prev.filter((featureKey) => result.partials.some((partialItem) => partialItem.key === featureKey)),
    );
  }, [result]);

  // Update the visible knot set when switching active features.
  useEffect(() => {
    if (!result) return;
    const active = result.partials[activePartialIdx];
    const baseline = baselineKnots[active.key];
    const cached = active.key ? knotEdits[active.key] : undefined;
    if (cached) {
      setKnots(cached);
    } else if (baseline) {
      setKnots(baseline);
    } else {
      setKnots(buildEditableKnots(active));
    }
    setSelectedKnots([]);
  }, [result, activePartialIdx, baselineKnots]);

  const buildChanges = (before: KnotSet, after: KnotSet) => {
    const beforeMap = new Map<number, number>();
    const afterMap = new Map<number, number>();
    before.x.forEach((x, i) => beforeMap.set(x, before.y[i] ?? 0));
    after.x.forEach((x, i) => afterMap.set(x, after.y[i] ?? 0));
    const keys = new Set<number>([...beforeMap.keys(), ...afterMap.keys()]);
    const changes: HistoryChange[] = [];
    keys.forEach((x) => {
      const b = beforeMap.get(x);
      const a = afterMap.get(x);
      if (b !== a) {
        const delta = b != null && a != null ? a - b : undefined;
        changes.push({ x, before: b, after: a, delta });
      }
    });
    return changes;
  };

  const applyChanges = (
    base: KnotSet,
    changes: HistoryChange[] | undefined,
    direction: "undo" | "redo",
  ) => {
    const next = { x: [...base.x], y: [...base.y] };
    const indexByX = new Map<number, number>();
    next.x.forEach((x, i) => indexByX.set(x, i));
    (changes ?? []).forEach((change) => {
      const idx = indexByX.get(change.x);
      const target = direction === "undo" ? change.before : change.after;
      const delta = change.delta;
      if (target === undefined && delta == null) {
        if (idx != null) {
          next.x.splice(idx, 1);
          next.y.splice(idx, 1);
          indexByX.delete(change.x);
        }
        return;
      }
      if (delta != null) {
        if (idx == null) {
          const seed = direction === "undo" ? change.after : change.before;
          const baseVal = seed ?? 0;
          const nextVal = direction === "undo" ? baseVal - delta : baseVal + delta;
          next.x.push(change.x);
          next.y.push(nextVal);
          indexByX.set(change.x, next.x.length - 1);
        } else {
          next.y[idx] = (next.y[idx] ?? 0) + (direction === "undo" ? -delta : delta);
        }
      } else if (target !== undefined) {
        if (idx == null) {
          next.x.push(change.x);
          next.y.push(target);
          indexByX.set(change.x, next.x.length - 1);
        } else {
          next.y[idx] = target;
        }
      }
    });
    const pairs = next.x.map((x, i) => ({ x, y: next.y[i] ?? 0 })).sort((a, b) => a.x - b.x);
    return { x: pairs.map((p) => p.x), y: pairs.map((p) => p.y) };
  };

  // Record every edit action so the history sidebar can display it.
  const recordAction = (featureKey: string, before: KnotSet, after: KnotSet, action = "edit") => {
    const changes = buildChanges(before, after);
    if (!changes.length) return;
    setHistory((prev) => {
      const entry = { featureKey, action, ts: Date.now(), changes };
      const truncated = prev.slice(0, historyCursor);
      const next = [...truncated, entry].slice(-200);
      setHistoryCursor(next.length);
      return next;
    });
  };

  // Save a committed version used by the worker to recompute predictions.
  const commitEdits = (featureKey: string, next: KnotSet) => {
    setCommittedEdits((prev) => ({ ...prev, [featureKey]: { x: [...next.x], y: [...next.y] } }));
  };

  const applyHistoryEntry = (
    current: KnotSet,
    entry: { changes?: HistoryChange[] },
  ) => {
    return applyChanges(current, entry.changes, "redo");
  };

  const rebuildEditsFromHistory = (entries: typeof history) => {
    const nextEdits: Record<string, KnotSet> = {};
    entries.forEach((entry) => {
      const base = nextEdits[entry.featureKey] ?? baselineKnots[entry.featureKey] ?? { x: [], y: [] };
      nextEdits[entry.featureKey] = applyHistoryEntry(base, entry);
    });
    return nextEdits;
  };

  const applyHistoryCursor = (nextCursor: number, entries: typeof history) => {
    const clamped = Math.max(0, Math.min(entries.length, nextCursor));
    const rebuilt = rebuildEditsFromHistory(entries.slice(0, clamped));
    setHistoryCursor(clamped);
    setKnotEdits(rebuilt);
    setCommittedEdits(rebuilt);
    if (result) {
      const active = result.partials[activePartialIdx];
      const fallback = baselineKnots[active.key] ?? { x: [], y: [] };
      setKnots(rebuilt[active.key] ?? fallback);
      setSelectedKnots([]);
    }
  };

  const updateFeatureEdits = (featureKey: string, next: KnotSet) => {
    setKnotEdits((prev) => ({ ...prev, [featureKey]: next }));
    setCommittedEdits((prev) => ({ ...prev, [featureKey]: next }));
    if (result) {
      const active = result.partials[activePartialIdx];
      if (active.key === featureKey) {
        setKnots(next);
        setSelectedKnots([]);
      }
    }
  };

  const applyHistoryStep = (entryIndex: number, direction: "undo" | "redo") => {
    const entry = history[entryIndex];
    if (!entry) return;
    const base = knotEdits[entry.featureKey] ?? baselineKnots[entry.featureKey] ?? { x: [], y: [] };
    const next = applyChanges(base, entry.changes, direction);
    updateFeatureEdits(entry.featureKey, next);
    setHistoryCursor(direction === "undo" ? entryIndex : entryIndex + 1);
  };

  // Undo/redo apply per-feature patches without replaying full history.
  const undoLast = () => {
    if (historyCursor <= 0) return;
    applyHistoryStep(historyCursor - 1, "undo");
  };

  const redoLast = () => {
    if (historyCursor >= history.length) return;
    applyHistoryStep(historyCursor, "redo");
  };

  // Remove a single history entry by its array index.
  const deleteHistoryEntry = (index: number) => {
    const nextHistory = history.filter((_, i) => i !== index);
    const nextCursor = historyCursor > index ? historyCursor - 1 : historyCursor;
    setHistory(nextHistory);
    applyHistoryCursor(nextCursor, nextHistory);
  };

  // Background worker keeps prediction plots responsive during edits.
  useEffect(() => {
    if (!result) {
      setModels(null);
      return;
    }
    if (typeof window === "undefined") return;
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("../workers/modelWorker.ts", import.meta.url));
      workerRef.current.onmessage = (e) => {
        setModels(e.data);
      };
    }
    if (workerDebounceRef.current) {
      window.clearTimeout(workerDebounceRef.current);
    }
    workerDebounceRef.current = window.setTimeout(() => {
      workerRef.current?.postMessage({ result, baselineKnots, knotEdits: committedEdits });
    }, 120);
    return () => {};
  }, [result, baselineKnots, committedEdits]);

  // Compute dashboard metrics from the latest model predictions.
  const stats = useMemo<StatItem[] | null>(() => {
    if (!result || !models) return null;
    const items: StatItem[] = [];
    const buildPairs = (yTrue: number[], yPred: number[]) =>
      yTrue
        .map((y, i) => ({ y, p: yPred[i] }))
        .filter((pair) => Number.isFinite(pair.y) && Number.isFinite(pair.p)) as { y: number; p: number }[];

    const calcRegression = (yTrue: number[], yPred: number[]) => {
      const pairs = buildPairs(yTrue, yPred);
      if (!pairs.length) return null;
      const diffs = pairs.map((pair) => pair.y - pair.p);
      const mse = diffs.reduce((s, d) => s + d * d, 0) / pairs.length;
      const mae = diffs.reduce((s, d) => s + Math.abs(d), 0) / pairs.length;
      const rmse = Math.sqrt(mse);
      const mean = pairs.reduce((s, pair) => s + pair.y, 0) / pairs.length;
      const denom = pairs.reduce((s, pair) => s + (pair.y - mean) * (pair.y - mean), 0);
      const r2 = denom > 0 ? 1 - diffs.reduce((s, d) => s + d * d, 0) / denom : 0;
      return { rmse, r2, mae, count: pairs.length };
    };
    const calcClassification = (yTrue: number[], yPred: number[]) => {
      const pairs = buildPairs(yTrue, yPred);
      if (!pairs.length) return null;
      const yBin = pairs.map((pair) => (pair.y >= 0.5 ? 1 : 0));
      const pBin = pairs.map((pair) => (pair.p >= 0.5 ? 1 : 0));
      const correct = yBin.reduce((s, v, i) => s + (v === pBin[i] ? 1 : 0), 0);
      const acc = correct / yBin.length;
      let tp = 0;
      let fp = 0;
      let fn = 0;
      yBin.forEach((v, i) => {
        const p = pBin[i];
        if (v === 1 && p === 1) tp += 1;
        if (v === 0 && p === 1) fp += 1;
        if (v === 1 && p === 0) fn += 1;
      });
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { acc, precision, recall, f1, count: yBin.length };
    };

    const isClassification = result.task === "classification";
    const makeBarItem = (
      label: string,
      base: number | null,
      current: number | null,
      lowerIsBetter = false,
      format = "0.000",
    ): StatItem => ({
      label,
      kind: "bar",
      base,
      current,
      lowerIsBetter,
      format,
    });

    const makeValueItem = (label: string, value: string): StatItem => ({ label, kind: "value", value });

    const basePreds = Array.isArray(result.predictions) ? result.predictions : models.baseModel.preds;
    const baseMetric = isClassification ? calcClassification(result.y, basePreds) : calcRegression(result.y, basePreds);
    const editedMetric = isClassification
      ? calcClassification(result.y, models.editedModel.preds)
      : calcRegression(result.y, models.editedModel.preds);
    if (baseMetric && editedMetric) {
      if (isClassification && "acc" in baseMetric && "acc" in editedMetric) {
        items.push(makeBarItem("Accuracy", baseMetric.acc, editedMetric.acc));
        items.push(makeBarItem("F1", baseMetric.f1, editedMetric.f1));
        items.push(makeBarItem("Precision", baseMetric.precision, editedMetric.precision));
        items.push(makeBarItem("Recall", baseMetric.recall, editedMetric.recall));
        items.push(makeValueItem("n", baseMetric.count.toString()));
      } else if (!isClassification && "rmse" in baseMetric && "rmse" in editedMetric) {
        items.push(makeBarItem("RMSE", baseMetric.rmse, editedMetric.rmse, true));
        items.push(makeBarItem("MAE", baseMetric.mae, editedMetric.mae, true));
        items.push(makeBarItem("R²", baseMetric.r2, editedMetric.r2));
        items.push(makeValueItem("n", baseMetric.count.toString()));
      }
    }

    return items;
  }, [result, models]);

  // Restore cached history/edits for the current model once per key.
  useEffect(() => {
    if (!CACHE_EDITS_ENABLED) return;
    if (typeof window === "undefined") return;
    if (!result || !modelSource) return;
    const key = getCacheKey(
      modelSource,
      dataset,
      modelType,
      shapePoints,
      seed,
      nEstimators,
      boostRate,
      initReg,
      elmAlpha,
      earlyStopping,
      scaleY,
    );
    if (cacheLoadedRef.current === key) return;
    cacheLoadedRef.current = key;
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) return;
      const parsed = JSON.parse(raw) as {
        history?: {
          featureKey: string;
          action: string;
          ts: number;
          changes?: HistoryChange[];
          before?: KnotSet;
          after?: KnotSet;
        }[];
        historyCursor?: number;
        activePartialIdx?: number;
      };
      if (!parsed.history || !Array.isArray(parsed.history)) return;
      const normalized: HistoryEntry[] = parsed.history.map((entry) => {
        if (entry.changes) {
          const changes = entry.changes.map((c) =>
            c.delta == null && c.before != null && c.after != null ? { ...c, delta: c.after - c.before } : c,
          );
          return { featureKey: entry.featureKey, action: entry.action, ts: entry.ts, changes };
        }
        if (entry.before && entry.after) {
          return { featureKey: entry.featureKey, action: entry.action, ts: entry.ts, changes: buildChanges(entry.before, entry.after) };
        }
        return { featureKey: entry.featureKey, action: entry.action, ts: entry.ts, changes: [] };
      });
      setHistory(normalized);
      const cursor = typeof parsed.historyCursor === "number" ? parsed.historyCursor : parsed.history.length;
      setHistoryCursor(cursor);
      if (typeof parsed.activePartialIdx === "number") {
        setActivePartialIdx(parsed.activePartialIdx);
      }
      const rebuilt = rebuildEditsFromHistory(normalized.slice(0, cursor));
      setKnotEdits(rebuilt);
      setCommittedEdits(rebuilt);
      const active = result.partials[parsed.activePartialIdx ?? activePartialIdx];
      const fallback = baselineKnots[active.key] ?? { x: [], y: [] };
      setKnots(rebuilt[active.key] ?? fallback);
      setSelectedKnots([]);
    } catch (error) {
      console.warn("Failed to restore cached edits.", error);
    }
  }, [
    result,
    modelSource,
    dataset,
    shapePoints,
    seed,
    nEstimators,
    boostRate,
    initReg,
    elmAlpha,
    earlyStopping,
    scaleY,
    baselineKnots,
    activePartialIdx,
  ]);

  // Persist history/edits per model so a refresh can resume work.
  useEffect(() => {
    if (!CACHE_EDITS_ENABLED) return;
    if (typeof window === "undefined") return;
    if (!result || !modelSource) return;
    const key = getCacheKey(
      modelSource,
      dataset,
      modelType,
      shapePoints,
      seed,
      nEstimators,
      boostRate,
      initReg,
      elmAlpha,
      earlyStopping,
      scaleY,
    );
    try {
      window.localStorage.setItem(
        key,
        JSON.stringify({
          history,
          historyCursor,
          activePartialIdx,
        }),
      );
    } catch (error) {
      console.warn("Failed to persist cached edits.", error);
    }
  }, [
    history,
    historyCursor,
    activePartialIdx,
    result,
    modelSource,
    dataset,
    shapePoints,
    seed,
    nEstimators,
    boostRate,
    initReg,
    elmAlpha,
    earlyStopping,
    scaleY,
  ]);

  return {
    datasets: DATASETS,
    dataset,
    modelType,
    centerShapes,
    setDataset,
    setModelType,
    setCenterShapes,
    shapePoints,
    setShapePoints,
    seed,
    setSeed,
    nEstimators,
    setNEstimators,
    boostRate,
    setBoostRate,
    initReg,
    setInitReg,
    elmAlpha,
    setElmAlpha,
    earlyStopping,
    setEarlyStopping,
    scaleY,
    setScaleY,
    status,
    result,
    debugPayload,
    debugError,
    activePartialIdx,
    setActivePartialIdx,
    baselineKnots,
    knots,
    setKnots,
    knotEdits,
    setKnotEdits,
    committedEdits,
    selectedKnots,
    setSelectedKnots,
    lockedFeatures,
    history,
    historyCursor,
    recordAction,
    commitEdits,
    undoLast,
    redoLast,
    deleteHistoryEntry,
    stats,
    models,
    modelSource,
    handleModelSelect,
    manualTrain,
    manualRefitFromEdits,
    toggleFeatureLock,
    handleSave,
    sidebarTab,
    setSidebarTab,
    selectedDataset,
    partial,
    displayLabel,
  };
};

const getCacheKey = (
  modelSource: string,
  dataset: string,
  modelType: "igann" | "igann_interactive",
  points: number,
  seed: number,
  nEstimators: number,
  boostRate: number,
  initReg: number,
  elmAlpha: number,
  earlyStopping: number,
  scaleY: boolean,
) => {
  if (modelSource.startsWith("model:") || modelSource.startsWith("saved:")) {
    return `gam-lab:${modelSource}`;
  }
  return `gam-lab:train:${dataset}:${modelType}:${points}:${seed}:${nEstimators}:${boostRate}:${initReg}:${elmAlpha}:${earlyStopping}:${scaleY}`;
};

export type GamLabState = ReturnType<typeof useGamLab>;
