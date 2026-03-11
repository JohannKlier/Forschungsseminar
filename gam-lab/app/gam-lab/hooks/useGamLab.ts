import { useEffect, useMemo, useRef, useState } from "react";
import { DatasetOption, KnotSet, ModelInfo, Models, ShapeFunction, ShapeFunctionVersion, StatItem, TrainData, TrainResponse } from "../types";
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

// Normalize the editable knot representation for categorical vs continuous features.
const buildEditableKnots = (shape: ShapeFunction): KnotSet => {
  if (shape.categories && shape.categories.length) {
    const yVals = shape.editableY?.length ? [...shape.editableY] : [];
    const xVals = shape.categories.map((_, idx) => idx);
    return { x: xVals, y: yVals };
  }
  if (shape.editableX?.length && shape.editableY?.length) {
    return { x: [...shape.editableX], y: [...shape.editableY] };
  }
  return { x: [], y: [] };
};

const cloneKnots = (knots: KnotSet): KnotSet => ({ x: [...knots.x], y: [...knots.y] });

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
type FixedLineSnapshot = { id: string; knots: KnotSet };

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

  // Global training/loading status.
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [debugError, setDebugError] = useState<string | null>(null);

  // Core separated state: model metadata, raw data, and accumulated versions.
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [trainData, setTrainData] = useState<TrainData | null>(null);
  // Versions accumulate across retrains; the latest is always versions[versions.length - 1].
  const [versions, setVersions] = useState<ShapeFunctionVersion[]>([]);

  // Editing state for the current feature and its knot history.
  const [activePartialIdx, setActivePartialIdx] = useState(0);
  const [baselineKnots, setBaselineKnots] = useState<Record<string, KnotSet>>({});
  const [knots, setKnots] = useState<KnotSet>({ x: [], y: [] });
  const [knotEdits, setKnotEdits] = useState<Record<string, KnotSet>>({});
  const [committedEdits, setCommittedEdits] = useState<Record<string, KnotSet>>({});
  const [previousCommittedEdits, setPreviousCommittedEdits] = useState<Record<string, KnotSet>>({});
  const [selectedKnots, setSelectedKnots] = useState<number[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyCursor, setHistoryCursor] = useState(0);
  const [lockedFeatures, setLockedFeatures] = useState<string[]>([]);

  // Derived model predictions and worker wiring.
  const [models, setModels] = useState<Models | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerDebounceRef = useRef<number | null>(null);
  // Suppress worker result application while the user is dragging/interacting.
  const isInteractingRef = useRef(false);
  const pendingModelsRef = useRef<Models | null>(null);

  // Model/source selector and sidebar tab state.
  const [modelSource, setModelSource] = useState<string>("");
  const [sidebarTab, setSidebarTab] = useState<"edit" | "history">("edit");

  // Convenience: latest version and active shape.
  const currentVersion = useMemo<ShapeFunctionVersion | null>(
    () => versions[versions.length - 1] ?? null,
    [versions],
  );

  const selectedDataset = useMemo(() => DATASETS.find((item) => item.id === dataset) ?? DATASETS[0], [dataset]);

  const partial = useMemo<ShapeFunction | null>(
    () => (currentVersion ? currentVersion.shapes[activePartialIdx] ?? currentVersion.shapes[0] ?? null : null),
    [currentVersion, activePartialIdx],
  );

  const displayLabel = useMemo(() => {
    if (!partial || !currentVersion) return "";
    return partial.label || partial.key || `x${(currentVersion.shapes.indexOf(partial) ?? 0) + 1}`;
  }, [partial, currentVersion]);

  // Apply a new API response: update model info, data, and append the version.
  const applyResponse = (payload: TrainResponse, isRefit: boolean) => {
    setModelInfo(payload.model);
    // Data stays stable across refits of the same dataset+seed; always update on fresh train.
    if (!isRefit) {
      setTrainData(payload.data);
    }
    setVersions((prev) => (isRefit ? [...prev, payload.version] : [payload.version]));
  };

  // Trigger a fresh training run.
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
      applyResponse({ ...payload, source: "service" }, false);
      setStatus("idle");
    } catch (error) {
      console.warn("Trainer service unavailable.", error);
      setModelInfo(null);
      setTrainData(null);
      setVersions([]);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

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
      applyResponse({ ...payload, source: isSaved ? "saved" : "model" }, false);

      // Restore UI parameters from stored model info.
      setDataset(payload.model.dataset);
      if (payload.model.model_type === "igann" || payload.model.model_type === "igann_interactive") {
        setModelType(payload.model.model_type);
      }
      if (typeof payload.model.scale_y === "boolean") setScaleY(payload.model.scale_y);
      setShapePoints(payload.model.points ?? shapePoints);
      if (typeof payload.model.seed === "number") setSeed(payload.model.seed);
      if (typeof payload.model.n_estimators === "number") setNEstimators(payload.model.n_estimators);
      if (typeof payload.model.boost_rate === "number") setBoostRate(payload.model.boost_rate);
      if (typeof payload.model.init_reg === "number") setInitReg(payload.model.init_reg);
      if (typeof payload.model.elm_alpha === "number") setElmAlpha(payload.model.elm_alpha);
      if (typeof payload.model.early_stopping === "number") setEarlyStopping(payload.model.early_stopping);
      if (typeof payload.version.center_shapes === "boolean") setCenterShapes(payload.version.center_shapes);
      setLockedFeatures(Array.isArray(payload.version.locked_features) ? payload.version.locked_features.map(String) : []);
      setStatus("idle");
    } catch (error) {
      console.warn("Failed to load saved model.", error);
      setModelInfo(null);
      setTrainData(null);
      setVersions([]);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

  // Build the save payload from the current state.
  const buildSavePayload = (): TrainResponse | null => {
    if (!modelInfo || !trainData || !currentVersion) return null;
    const edits = committedEdits;
    const updatedShapes = currentVersion.shapes.map((shape) => {
      const edit = edits[shape.key];
      if (!edit) return shape;
      return { ...shape, editableX: [...edit.x], editableY: [...edit.y] };
    });
    return {
      model: modelInfo,
      data: trainData,
      version: { ...currentVersion, shapes: updatedShapes },
    };
  };

  // Persist edits and reload.
  const handleSave = async () => {
    if (!modelInfo) return;
    const baseName = `${dataset}-edited-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}`;
    const name = window.prompt("Save edited model as:", baseName);
    if (!name) return;
    const payload = buildSavePayload();
    if (!payload) return;
    try {
      await saveModel(name, payload);
      const savedValue = `saved:${name.replace(/\.json$/, "")}`;
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
    if (!modelInfo || !currentVersion) return;
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
      const refitPoints = typeof modelInfo.points === "number" ? modelInfo.points : shapePoints;
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
        partials: payload.version.shapes.map((shape) => ({
          key: shape.key,
          categories: shape.categories,
          editableX: shape.editableX,
          editableY: shape.editableY,
        })),
        locked_features: lockedFeatures,
      });
      // Refit: append new version, keep existing data.
      applyResponse({ ...refitPayload, source: "service" }, true);
      setLockedFeatures(
        Array.isArray(refitPayload.version.locked_features)
          ? refitPayload.version.locked_features.map(String)
          : lockedFeatures,
      );
      if (typeof refitPayload.version.center_shapes === "boolean") {
        setCenterShapes(refitPayload.version.center_shapes);
      }
      if (typeof refitPayload.model.points === "number") {
        setShapePoints(refitPayload.model.points);
      }
      setStatus("idle");
    } catch (error) {
      console.warn("Refit failed.", error);
      setDebugError(error instanceof Error ? error.message : "Unknown error");
      setStatus("error");
    }
  };

  // Rebuild editable knot state when a new version arrives.
  useEffect(() => {
    setActivePartialIdx(0);
    if (!currentVersion) return;
    const next: Record<string, KnotSet> = {};
    currentVersion.shapes.forEach((shape) => {
      next[shape.key] = buildEditableKnots(shape);
    });
    setBaselineKnots((prev) => {
      // Keep original baseline across refit iterations so "Before" always
      // refers to the initial model, not only the previous refit step.
      if (!currentVersion.refit_from_edits) return next;
      const merged: Record<string, KnotSet> = {};
      Object.keys(next).forEach((key) => {
        const existing = prev[key];
        merged[key] =
          existing && existing.x?.length && existing.y?.length ? existing : next[key];
      });
      return merged;
    });
    setKnotEdits(next);
    setCommittedEdits(next);
    setPreviousCommittedEdits(next);
    setHistory([]);
    setHistoryCursor(0);
    setLockedFeatures((prev) =>
      prev.filter((featureKey) => currentVersion.shapes.some((s) => s.key === featureKey)),
    );
  }, [currentVersion]);

  // Update the visible knot set when switching active features.
  useEffect(() => {
    if (!currentVersion) return;
    const active = currentVersion.shapes[activePartialIdx];
    if (!active) return;
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
  }, [currentVersion, activePartialIdx, baselineKnots]);

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
    setCommittedEdits((prev) => {
      const previous = prev[featureKey] ?? baselineKnots[featureKey] ?? next;
      setPreviousCommittedEdits((prevPrevious) => ({
        ...prevPrevious,
        [featureKey]: cloneKnots(previous),
      }));
      return { ...prev, [featureKey]: cloneKnots(next) };
    });
  };

  // Interaction gate: while dragging/smoothing, buffer worker results and apply them on release.
  const notifyInteractionStart = () => {
    isInteractingRef.current = true;
  };
  const notifyInteractionEnd = () => {
    isInteractingRef.current = false;
    if (pendingModelsRef.current) {
      setModels(pendingModelsRef.current);
      pendingModelsRef.current = null;
    }
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
    setPreviousCommittedEdits(rebuilt);
    if (currentVersion) {
      const active = currentVersion.shapes[activePartialIdx];
      const fallback = baselineKnots[active?.key ?? ""] ?? { x: [], y: [] };
      setKnots(rebuilt[active?.key ?? ""] ?? fallback);
      setSelectedKnots([]);
    }
  };

  const updateFeatureEdits = (featureKey: string, next: KnotSet) => {
    setKnotEdits((prev) => ({ ...prev, [featureKey]: next }));
    setCommittedEdits((prev) => {
      const previous = prev[featureKey] ?? baselineKnots[featureKey] ?? next;
      setPreviousCommittedEdits((prevPrevious) => ({
        ...prevPrevious,
        [featureKey]: cloneKnots(previous),
      }));
      return { ...prev, [featureKey]: cloneKnots(next) };
    });
    if (currentVersion) {
      const active = currentVersion.shapes[activePartialIdx];
      if (active?.key === featureKey) {
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

  const undoLast = () => {
    if (historyCursor <= 0) return;
    applyHistoryStep(historyCursor - 1, "undo");
  };

  const redoLast = () => {
    if (historyCursor >= history.length) return;
    applyHistoryStep(historyCursor, "redo");
  };

  const deleteHistoryEntry = (index: number) => {
    const featureKey = history[index]?.featureKey;
    // Remove the entry and all subsequent entries for the same feature.
    const nextHistory = history.filter((e, i) => i !== index && !(i > index && e.featureKey === featureKey));
    const removed = history.length - nextHistory.length;
    const nextCursor = Math.max(0, historyCursor - (historyCursor > index ? removed : 0));
    setHistory(nextHistory);
    applyHistoryCursor(nextCursor, nextHistory);
  };

  // Background worker keeps prediction plots responsive during edits.
  // Passes trainData and currentVersion separately so the worker can use
  // trainX for scatter/contributions and intercept from the version.
  useEffect(() => {
    if (!currentVersion || !trainData) {
      setModels(null);
      return;
    }
    if (typeof window === "undefined") return;
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("../workers/modelWorker.ts", import.meta.url));
      workerRef.current.onmessage = (e) => {
        if (isInteractingRef.current) {
          pendingModelsRef.current = e.data;
        } else {
          setModels(e.data);
        }
      };
    }
    if (workerDebounceRef.current) {
      window.clearTimeout(workerDebounceRef.current);
    }
    workerDebounceRef.current = window.setTimeout(() => {
      workerRef.current?.postMessage({
        version: currentVersion,
        trainData,
        modelInfo,
        baselineKnots,
        knotEdits: committedEdits,
      });
    }, 120);
    return () => {};
  }, [currentVersion, trainData, modelInfo, baselineKnots, committedEdits]);

  // Compute dashboard metrics showing 3 bars: initial / latest / current.
  // initial  = server-computed metrics from the very first training run (versions[0])
  // latest   = server-computed metrics from the most recent train/refit (currentVersion)
  // current  = live frontend metrics recalculated from edited knots via worker
  const stats = useMemo<StatItem[] | null>(() => {
    if (!modelInfo || !trainData || !models || !currentVersion) return null;
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
      const correct = yBin.reduce((s: number, v, i) => s + (v === pBin[i] ? 1 : 0), 0);
      const acc = correct / yBin.length;
      let tp = 0; let fp = 0; let fn = 0;
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

    const isClassification = modelInfo.task === "classification";
    const makeBarItem = (
      label: string,
      initial: number | null,
      latest: number | null,
      current: number | null,
      lowerIsBetter = false,
      format = "0.000",
    ): StatItem => ({ label, kind: "bar", initial, latest, current, lowerIsBetter, format });
    const makeValueItem = (label: string, value: string): StatItem => ({ label, kind: "value", value });

    // Server metrics from first and latest versions.
    const initialVersion = versions[0];
    const initTrain = initialVersion?.trainMetrics;
    const latestTrain = currentVersion.trainMetrics;

    // Live frontend metrics from worker (edited knots applied).
    const liveMetric = isClassification
      ? calcClassification(trainData.trainY, models.editedModel.preds)
      : calcRegression(trainData.trainY, models.editedModel.preds);

    const n = liveMetric?.count ?? initTrain?.count ?? latestTrain?.count ?? 0;

    if (isClassification) {
      const initAcc = initTrain?.acc ?? null;
      const latestAcc = latestTrain?.acc ?? null;
      const curAcc = liveMetric && "acc" in liveMetric ? liveMetric.acc : null;
      items.push(makeBarItem("Accuracy", initAcc, latestAcc, curAcc));
      if (liveMetric && "f1" in liveMetric) {
        items.push(makeBarItem("F1", null, null, liveMetric.f1));
        items.push(makeBarItem("Precision", null, null, liveMetric.precision));
        items.push(makeBarItem("Recall", null, null, liveMetric.recall));
      }
    } else {
      const initRmse = initTrain?.rmse ?? null;
      const latestRmse = latestTrain?.rmse ?? null;
      const curRmse = liveMetric && "rmse" in liveMetric ? liveMetric.rmse : null;
      const initMae = initTrain?.mae ?? null;
      const latestMae = latestTrain?.mae ?? null;
      const initR2 = initTrain?.r2 ?? null;
      const latestR2 = latestTrain?.r2 ?? null;
      const curR2 = liveMetric && "r2" in liveMetric ? liveMetric.r2 : null;
      const curMae = liveMetric && "mae" in liveMetric ? liveMetric.mae : null;
      items.push(makeBarItem("RMSE", initRmse, latestRmse, curRmse, true));
      items.push(makeBarItem("MAE", initMae, latestMae, curMae, true));
      items.push(makeBarItem("R²", initR2, latestR2, curR2));
    }
    items.push(makeValueItem("n", n.toString()));

    return items;
  }, [modelInfo, trainData, versions, currentVersion, models]);

  // Expose a synthetic `result` object so pages/components that pass `result` as a prop
  // get a consistent view without needing individual model/data/version props.
  const result = useMemo<TrainResponse | null>(() => {
    if (!modelInfo || !trainData || !currentVersion) return null;
    return { model: modelInfo, data: trainData, version: currentVersion };
  }, [modelInfo, trainData, currentVersion]);

  const fixedLinesByFeature = useMemo<Record<string, FixedLineSnapshot[]>>(() => {
    const snapshots: Record<string, FixedLineSnapshot[]> = {};
    const currentByFeature: Record<string, KnotSet> = {};
    history.slice(0, historyCursor).forEach((entry, index) => {
      const base = currentByFeature[entry.featureKey] ?? baselineKnots[entry.featureKey] ?? { x: [], y: [] };
      const next = applyHistoryEntry(base, entry);
      currentByFeature[entry.featureKey] = next;
      snapshots[entry.featureKey] = [
        ...(snapshots[entry.featureKey] ?? []),
        { id: `${entry.ts}-${index}`, knots: cloneKnots(next) },
      ].slice(-2);
    });
    return snapshots;
  }, [history, historyCursor, baselineKnots]);

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
    modelInfo,
    trainData,
    versions,
    currentVersion,
    debugError,
    activePartialIdx,
    setActivePartialIdx,
    baselineKnots,
    knots,
    setKnots,
    knotEdits,
    setKnotEdits,
    committedEdits,
    previousCommittedEdits,
    fixedLinesByFeature,
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
    notifyInteractionStart,
    notifyInteractionEnd,
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

export type GamLabState = ReturnType<typeof useGamLab>;
