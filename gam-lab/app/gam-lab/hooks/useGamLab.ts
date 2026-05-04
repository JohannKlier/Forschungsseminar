import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DatasetOption, HistoryChange, HistoryEntry, KnotSet, MetricSummary, MetricWarning, ModelInfo, Models, ShapeFunction, ShapeFunctionVersion, SidebarTab, TrainData, TrainResponse } from "../types";
import { loadModel, trainModel } from "../lib/modelApi";
import { loadSavedModel, saveModel } from "../lib/savedModelApi";
import { type AuditLogFn } from "../lib/audit";

// Dataset registry used by the UI selectors and training requests.
const DATASETS: DatasetOption[] = [
  { id: "bike_hourly", label: "Bike sharing (hourly)", summary: "Hourly rentals with weather/seasonality." },
  {
    id: "mimic4_mean_100_full",
    label: "MIMIC-IV mortality",
    summary: "ICU cohort with demographics, length of stay, and mean vital/lab features.",
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

const calcRegressionMetrics = (yTrue: number[], yPred: number[]): MetricSummary | null => {
  const pairs = yTrue.map((y, i) => ({ y, p: yPred[i] })).filter((pair) => Number.isFinite(pair.y) && Number.isFinite(pair.p));
  if (!pairs.length) return null;
  const count = pairs.length;
  const mse = pairs.reduce((sum, { y, p }) => sum + (y - p) ** 2, 0) / count;
  const mae = pairs.reduce((sum, { y, p }) => sum + Math.abs(y - p), 0) / count;
  const mean = pairs.reduce((sum, { y }) => sum + y, 0) / count;
  const totalVariance = pairs.reduce((sum, { y }) => sum + (y - mean) ** 2, 0);
  const r2 = totalVariance > 0 ? 1 - (mse * count) / totalVariance : 0;
  return { count, rmse: Math.sqrt(mse), mae, r2 };
};

const calcClassificationMetrics = (yTrue: number[], yPred: number[]): MetricSummary | null => {
  const pairs = yTrue.map((y, i) => ({ y, p: yPred[i] })).filter((pair) => Number.isFinite(pair.y) && Number.isFinite(pair.p));
  if (!pairs.length) return null;
  const yBin = pairs.map(({ y }) => (y > 0.5 ? 1 : 0));
  const pBin = pairs.map(({ p }) => (p > 0.5 ? 1 : 0));
  const tp = yBin.reduce((sum, y, i) => sum + (y === 1 && pBin[i] === 1 ? 1 : 0), 0 as number);
  const fp = yBin.reduce((sum, y, i) => sum + (y === 0 && pBin[i] === 1 ? 1 : 0), 0 as number);
  const fn = yBin.reduce((sum, y, i) => sum + (y === 1 && pBin[i] === 0 ? 1 : 0), 0 as number);
  const acc = yBin.filter((y, i) => y === pBin[i]).length / yBin.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  return { count: yBin.length, acc, precision, recall };
};

// Main state/logic hook that powers the GAM Lab page.
type InitOptions = {
  initialModel?: string | null;
  initialTrain?: {
    dataset: string;
    model_type: "igann" | "igann_interactive";
    center_shapes: boolean;
    selected_features?: string[];
    points: number;
    seed: number;
    n_estimators: number;
    boost_rate: number;
    init_reg: number;
    elm_alpha: number;
    early_stopping: number;
    n_hid: number;
    scale_y: boolean;
  } | null;
  auditLogger?: AuditLogFn;
};

type FixedLineSnapshot = { id: string; knots: KnotSet };

const noopAuditLog: AuditLogFn = () => {};

export const useGamLab = (options: InitOptions = {}) => {
  const logEvent = options.auditLogger ?? noopAuditLog;

  // ─── Training settings ────────────────────────────────────────────────────
  const [dataset, setDataset] = useState(DATASETS[1].id);
  const [modelType, setModelType] = useState<"igann" | "igann_interactive">("igann_interactive");
  const [centerShapes, setCenterShapes] = useState(true);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [shapePoints, setShapePoints] = useState(250);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [nEstimators, setNEstimators] = useState(100);
  const [boostRate, setBoostRate] = useState(0.1);
  const [initReg, setInitReg] = useState(1);
  const [elmAlpha, setElmAlpha] = useState(1);
  const [earlyStopping, setEarlyStopping] = useState(50);
  const [nHid, setNHid] = useState(10);
  const [scaleY, setScaleY] = useState(true);
  const [sampleSize, setSampleSize] = useState(1000);

  // ─── Model state ─────────────────────────────────────────────────────────
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [trainData, setTrainData] = useState<TrainData | null>(null);
  const [versions, setVersions] = useState<ShapeFunctionVersion[]>([]);
  const [modelSource, setModelSource] = useState<string>("");

  const [savedModels, setSavedModels] = useState<ShapeFunctionVersion[]>([]);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("gam-lab-saved-models");
      if (raw) setSavedModels(JSON.parse(raw));
    } catch {}
  }, []);

  // ─── Knot editing state ───────────────────────────────────────────────────
  const [activePartialIdx, setActivePartialIdx] = useState(0);
  const [baselineKnots, setBaselineKnots] = useState<Record<string, KnotSet>>({});
  const [knots, setKnots] = useState<KnotSet>({ x: [], y: [] });
  const [knotEdits, setKnotEdits] = useState<Record<string, KnotSet>>({});
  const [committedEdits, setCommittedEdits] = useState<Record<string, KnotSet>>({});
  const [selectedKnots, setSelectedKnots] = useState<number[]>([]);

  // ─── Edit history ─────────────────────────────────────────────────────────
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyCursor, setHistoryCursor] = useState(0);
  const persistedEditsRef = useRef<Record<string, KnotSet>>({});
  const lastSelectionContextRef = useRef<{ versionId: string | null; featureKey: string | null }>({
    versionId: null,
    featureKey: null,
  });

  // ─── Background worker / predictions ─────────────────────────────────────
  const [models, setModels] = useState<Models | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerDebounceRef = useRef<number | null>(null);
  useEffect(() => () => { workerRef.current?.terminate(); workerRef.current = null; }, []);

  // ─── UI state ─────────────────────────────────────────────────────────────
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("edit");

  // ─── Derived selectors ────────────────────────────────────────────────────
  const currentVersion = useMemo<ShapeFunctionVersion | null>(
    () => versions[versions.length - 1] ?? null,
    [versions],
  );

  const selectedDataset = useMemo(() => DATASETS.find((item) => item.id === dataset) ?? DATASETS[0], [dataset]);

  const partial = useMemo<ShapeFunction | null>(
    () => (currentVersion ? currentVersion.shapes[activePartialIdx] ?? currentVersion.shapes[0] ?? null : null),
    [currentVersion, activePartialIdx],
  );

  // ─── Saved-model management ───────────────────────────────────────────────
  const addSavedModel = (version: ShapeFunctionVersion, info: ModelInfo) => {
    const entry: ShapeFunctionVersion = { ...version, modelInfo: info };
    setSavedModels((prev) => {
      const next = [...prev, entry];
      try { localStorage.setItem("gam-lab-saved-models", JSON.stringify(next)); } catch {}
      return next;
    });
  };

  const saveSnapshot = (label?: string) => {
    if (!currentVersion || !modelInfo) return;
    const editedShapes = currentVersion.shapes.map((shape) => {
      const edit = committedEdits[shape.key];
      return edit ? { ...shape, editableX: edit.x, editableY: edit.y } : shape;
    });
    const snapshot: ShapeFunctionVersion = {
      ...currentVersion,
      versionId: `edit-${Date.now()}`,
      timestamp: Date.now(),
      source: "edit",
      isEdited: true,
      label,
      shapes: editedShapes,
      modelInfo,
    };
    addSavedModel(snapshot, modelInfo);
  };

  const clearSavedModels = () => {
    setSavedModels([]);
    try { localStorage.removeItem("gam-lab-saved-models"); } catch {}
  };

  // ─── Model lifecycle: train / load / save ─────────────────────────────────
  const applyResponse = (payload: TrainResponse) => {
    setModelInfo(payload.model);
    setTrainData(payload.data);
    setVersions([payload.version]);
    addSavedModel(payload.version, payload.model);
  };

  const summarizeChanges = (changes: HistoryChange[]) => {
    if (!changes.length) {
      return { count: 0 };
    }
    const xs = changes.map((change) => change.x);
    const deltas = changes
      .map((change) => change.delta)
      .filter((delta): delta is number => typeof delta === "number" && Number.isFinite(delta));
    return {
      count: changes.length,
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      deltaSum: deltas.reduce((sum, delta) => sum + delta, 0),
      deltaAbsSum: deltas.reduce((sum, delta) => sum + Math.abs(delta), 0),
    };
  };

  const train = async (
    overrides?: {
      dataset?: string;
      model_type?: "igann" | "igann_interactive";
      center_shapes?: boolean;
      selected_features?: string[];
      points?: number;
      seed?: number;
      n_estimators?: number;
      boost_rate?: number;
      init_reg?: number;
      elm_alpha?: number;
      early_stopping?: number;
      n_hid?: number;
      scale_y?: boolean;
      sample_size?: number;
    },
  ) => {
    const requestedParams = {
      dataset: overrides?.dataset ?? dataset,
      model_type: overrides?.model_type ?? modelType,
      center_shapes: overrides?.center_shapes ?? centerShapes,
      selected_features: overrides?.selected_features ?? selectedFeatures,
      seed: overrides?.seed ?? seed,
      points: overrides?.points ?? shapePoints,
      n_estimators: overrides?.n_estimators ?? nEstimators,
      boost_rate: overrides?.boost_rate ?? boostRate,
      init_reg: overrides?.init_reg ?? initReg,
      elm_alpha: overrides?.elm_alpha ?? elmAlpha,
      early_stopping: overrides?.early_stopping ?? earlyStopping,
      n_hid: overrides?.n_hid ?? nHid,
      scale_y: overrides?.scale_y ?? scaleY,
      sample_size: overrides?.sample_size ?? sampleSize,
    };
    logEvent({
      category: "model",
      action: "model.train_requested",
      detail: requestedParams,
    });
    setStatus("loading");
    setModelSource("train");
    try {
      const payload = await trainModel(requestedParams);
      applyResponse({ ...payload, source: "service" });
      logEvent({
        category: "model",
        action: "model.train_succeeded",
        detail: {
          dataset: payload.model.dataset,
          modelType: payload.model.model_type,
          versionId: payload.version.versionId,
        },
      });
      setStatus("idle");
    } catch (error) {
      console.warn("Trainer service unavailable.", error);
      setModelInfo(null);
      setTrainData(null);
      setVersions([]);
      logEvent({
        category: "model",
        action: "model.train_failed",
        detail: {
          ...requestedParams,
          error: error instanceof Error ? error.message : "Unknown error",
        },
      });
      setStatus("error");
    }
  };

  const handleModelSelect = async (value: string) => {
    logEvent({
      category: "model",
      action: "model.load_requested",
      detail: { source: value },
    });
    setModelSource(value);
    if (value === "train") return;
    setStatus("loading");
    try {
      const isSaved = value.startsWith("saved:");
      const isBase = value.startsWith("model:");
      const name = isSaved
        ? value.replace("saved:", "")
        : isBase
          ? value.replace("model:", "")
          : value;

      const payload = isSaved ? await loadSavedModel(name) : await loadModel(name);
      applyResponse({ ...payload, source: isSaved ? "saved" : "model" });

      // Restore UI parameters from stored model info.
      setDataset(payload.model.dataset);
      setSelectedFeatures(payload.model.selected_features ?? Object.keys(payload.data.featureLabels));
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
      if (typeof payload.model.n_hid === "number") setNHid(payload.model.n_hid);
      if (typeof payload.version.center_shapes === "boolean") setCenterShapes(payload.version.center_shapes);
      logEvent({
        category: "model",
        action: "model.load_succeeded",
        detail: {
          source: value,
          dataset: payload.model.dataset,
          versionId: payload.version.versionId,
        },
      });
      setStatus("idle");
    } catch (error) {
      console.warn("Failed to load saved model.", error);
      setModelInfo(null);
      setTrainData(null);
      setVersions([]);
      logEvent({
        category: "model",
        action: "model.load_failed",
        detail: {
          source: value,
          error: error instanceof Error ? error.message : "Unknown error",
        },
      });
      setStatus("error");
    }
  };

  // Stable refs used only in the mount-time init effect — avoids stale-closure lint suppression.
  const initTrainRef = useRef(train);
  const initModelSelectRef = useRef(handleModelSelect);

  // Initialize from landing-page query parameters (runs once on mount).
  useEffect(() => {
    if (options.initialTrain) {
      setDataset(options.initialTrain.dataset);
      setModelType(options.initialTrain.model_type);
      setCenterShapes(options.initialTrain.center_shapes);
      setSelectedFeatures(options.initialTrain.selected_features ?? []);
      setShapePoints(options.initialTrain.points);
      setSeed(options.initialTrain.seed);
      setNEstimators(options.initialTrain.n_estimators);
      setBoostRate(options.initialTrain.boost_rate);
      setInitReg(options.initialTrain.init_reg);
      setElmAlpha(options.initialTrain.elm_alpha);
      setEarlyStopping(options.initialTrain.early_stopping);
      setNHid(options.initialTrain.n_hid);
      setScaleY(options.initialTrain.scale_y);
      initTrainRef.current({
        dataset: options.initialTrain.dataset,
        model_type: options.initialTrain.model_type,
        center_shapes: options.initialTrain.center_shapes,
        selected_features: options.initialTrain.selected_features ?? [],
        points: options.initialTrain.points,
        seed: options.initialTrain.seed,
        n_estimators: options.initialTrain.n_estimators,
        boost_rate: options.initialTrain.boost_rate,
        init_reg: options.initialTrain.init_reg,
        elm_alpha: options.initialTrain.elm_alpha,
        early_stopping: options.initialTrain.early_stopping,
        n_hid: options.initialTrain.n_hid,
        scale_y: options.initialTrain.scale_y,
      });
      return;
    }
    if (options.initialModel && options.initialModel !== "undefined") {
      initModelSelectRef.current(options.initialModel);
      return;
    }
    setModelSource("train");
  }, []);

  // Build the save payload from the current state.
  const buildSavePayload = (): TrainResponse | null => {
    if (!modelInfo || !trainData || !currentVersion) return null;
    const allKeys = Object.keys(trainData.featureLabels);
    const updatedShapes: ShapeFunction[] = allKeys.flatMap((key) => {
      const versionShape = currentVersion.shapes.find((s) => s.key === key);
      const edit = committedEdits[key] ?? persistedEditsRef.current[key];
      if (!edit && !versionShape) return [];
      const categories = trainData.categories[key];
      const label = trainData.featureLabels[key] ?? key;
      return [{
        key,
        label,
        ...(categories ? { categories } : {}),
        editableX: edit ? [...edit.x] : (versionShape?.editableX ?? []),
        editableY: edit ? [...edit.y] : (versionShape?.editableY ?? []),
      }];
    });
    return {
      model: modelInfo,
      data: trainData,
      version: { ...currentVersion, shapes: updatedShapes },
    };
  };

  const defaultSaveName = () =>
    `${dataset}-edited-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}`;

  const handleSave = async (name: string) => {
    if (!modelInfo || !name) return;
    const payload = buildSavePayload();
    if (!payload) return;
    logEvent({
      category: "model",
      action: "model.save_requested",
      detail: {
        name,
        dataset: payload.model.dataset,
        versionId: payload.version.versionId,
      },
    });
    try {
      await saveModel(name, payload);
      const savedValue = `saved:${name.replace(/\.json$/, "")}`;
      setModelSource(savedValue);
      logEvent({
        category: "model",
        action: "model.save_succeeded",
        detail: {
          name: savedValue,
          dataset: payload.model.dataset,
          versionId: payload.version.versionId,
        },
      });
      void handleModelSelect(savedValue);
    } catch (error) {
      console.warn("Failed to save model.", error);
      logEvent({
        category: "model",
        action: "model.save_failed",
        detail: {
          name,
          error: error instanceof Error ? error.message : "Unknown error",
        },
      });
    }
  };

  // ─── Knot sync effects ────────────────────────────────────────────────────
  useEffect(() => {
    setActivePartialIdx(0);
    if (!currentVersion) return;
    const next: Record<string, KnotSet> = {};
    currentVersion.shapes.forEach((shape) => {
      if (shape.editableZ) return; // 2-D interaction shapes have no editable 1-D knots
      next[shape.key] = buildEditableKnots(shape);
    });
    setBaselineKnots(next);
    currentVersion.shapes.forEach((shape) => {
      if (shape.editableZ) return;
      persistedEditsRef.current[shape.key] = buildEditableKnots(shape);
    });
    setKnotEdits(next);
    setCommittedEdits(next);
    setHistory([]);
    setHistoryCursor(0);
    logEvent({
      category: "model",
      action: "model.version_loaded",
      detail: {
        versionId: currentVersion.versionId,
        source: currentVersion.source,
        shapeCount: currentVersion.shapes.length,
      },
    });
  }, [currentVersion, logEvent]);

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
    const last = lastSelectionContextRef.current;
    const nextContext = {
      versionId: currentVersion.versionId,
      featureKey: active.key,
    };
    if (last.versionId !== nextContext.versionId || last.featureKey !== nextContext.featureKey) {
      setSelectedKnots([]);
    }
    lastSelectionContextRef.current = nextContext;
  }, [currentVersion, activePartialIdx, baselineKnots, knotEdits]);

  // ─── Edit history logic ───────────────────────────────────────────────────
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

  const recordAction = (featureKey: string, before: KnotSet, after: KnotSet, action = "edit") => {
    const changes = buildChanges(before, after);
    if (!changes.length) return;
    logEvent({
      category: "edit",
      action: "edit.recorded",
      featureKey,
      detail: {
        editAction: action,
        summary: summarizeChanges(changes),
        changes,
      },
    });
    setHistory((prev) => {
      const entry = { featureKey, action, ts: Date.now(), changes };
      const truncated = prev.slice(0, historyCursor);
      const next = [...truncated, entry].slice(-200);
      setHistoryCursor(next.length);
      return next;
    });
  };

  const commitEdits = (featureKey: string, next: KnotSet) => {
    setCommittedEdits((prev) => ({ ...prev, [featureKey]: cloneKnots(next) }));
  };

  const rebuildEditsFromHistory = useCallback((entries: HistoryEntry[]) => {
    const nextEdits: Record<string, KnotSet> = {};
    entries.forEach((entry) => {
      const base = nextEdits[entry.featureKey] ?? baselineKnots[entry.featureKey] ?? { x: [], y: [] };
      nextEdits[entry.featureKey] = applyChanges(base, entry.changes, "redo");
    });
    return nextEdits;
  }, [baselineKnots]);

  const applyHistoryCursor = (nextCursor: number, entries: typeof history) => {
    const clamped = Math.max(0, Math.min(entries.length, nextCursor));
    const rebuilt = rebuildEditsFromHistory(entries.slice(0, clamped));
    setHistoryCursor(clamped);
    setKnotEdits(rebuilt);
    setCommittedEdits(rebuilt);
    if (currentVersion) {
      const active = currentVersion.shapes[activePartialIdx];
      const fallback = baselineKnots[active?.key ?? ""] ?? { x: [], y: [] };
      setKnots(rebuilt[active?.key ?? ""] ?? fallback);
      setSelectedKnots([]);
    }
  };

  const updateFeatureEdits = (featureKey: string, next: KnotSet) => {
    setKnotEdits((prev) => ({ ...prev, [featureKey]: next }));
    setCommittedEdits((prev) => ({ ...prev, [featureKey]: cloneKnots(next) }));
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
    const entry = history[historyCursor - 1];
    if (entry) {
      logEvent({
        category: "history",
        action: "history.undo",
        featureKey: entry.featureKey,
        detail: {
          entryIndex: historyCursor - 1,
          editAction: entry.action,
          summary: summarizeChanges(entry.changes),
        },
      });
    }
    applyHistoryStep(historyCursor - 1, "undo");
  };

  const redoLast = () => {
    if (historyCursor >= history.length) return;
    const entry = history[historyCursor];
    if (entry) {
      logEvent({
        category: "history",
        action: "history.redo",
        featureKey: entry.featureKey,
        detail: {
          entryIndex: historyCursor,
          editAction: entry.action,
          summary: summarizeChanges(entry.changes),
        },
      });
    }
    applyHistoryStep(historyCursor, "redo");
  };

  const countCascadingDeletes = (index: number) => {
    const featureKey = history[index]?.featureKey;
    if (!featureKey) return 0;
    return history.filter((e, i) => i > index && e.featureKey === featureKey).length;
  };

  const deleteHistoryEntry = (index: number) => {
    const featureKey = history[index]?.featureKey;
    const entry = history[index];
    if (!featureKey || !entry) return;
    const laterEntriesForFeature = countCascadingDeletes(index);
    // Remove the entry and all subsequent entries for the same feature.
    const nextHistory = history.filter((e, i) => i !== index && !(i > index && e.featureKey === featureKey));
    const removed = history.length - nextHistory.length;
    const nextCursor = Math.max(0, historyCursor - (historyCursor > index ? removed : 0));
    logEvent({
      category: "history",
      action: "history.deleted",
      featureKey: entry.featureKey,
      detail: {
        entryIndex: index,
        removedEntries: removed,
        editAction: entry.action,
        summary: summarizeChanges(entry.changes),
      },
    });
    setHistory(nextHistory);
    applyHistoryCursor(nextCursor, nextHistory);
  };

  // ─── Worker predictions & metric warning ──────────────────────────────────
  const comparisonEdits = useMemo(
    () => (historyCursor > 0 ? rebuildEditsFromHistory(history.slice(0, historyCursor - 1)) : null),
    [history, historyCursor, rebuildEditsFromHistory],
  );

  const metricWarning = useMemo<MetricWarning | null>(() => {
    if (!modelInfo || !trainData || !models?.comparisonModel || historyCursor <= 0) return null;

    const currentMetric = modelInfo.task === "classification"
      ? calcClassificationMetrics(trainData.trainY, models.editedModel.preds)
      : calcRegressionMetrics(trainData.trainY, models.editedModel.preds);
    const previousMetric = modelInfo.task === "classification"
      ? calcClassificationMetrics(trainData.trainY, models.comparisonModel.preds)
      : calcRegressionMetrics(trainData.trainY, models.comparisonModel.preds);
    if (!currentMetric || !previousMetric) return null;

    const latestEntry = history[historyCursor - 1];
    if (!latestEntry) return null;

    const details = modelInfo.task === "classification"
      ? (() => {
          const current = currentMetric.acc ?? NaN;
          const previous = previousMetric.acc ?? NaN;
          if (!Number.isFinite(current) || !Number.isFinite(previous)) return [];
          const delta = current - previous;
          return delta <= -0.015
            ? [{
                label: "Accuracy",
                current,
                previous,
                delta,
                deltaPct: previous !== 0 ? (delta / Math.abs(previous)) * 100 : null,
                lowerIsBetter: false,
              }]
            : [];
        })()
      : [
          { label: "RMSE", current: currentMetric.rmse ?? NaN, previous: previousMetric.rmse ?? NaN, lowerIsBetter: true },
          { label: "MAE", current: currentMetric.mae ?? NaN, previous: previousMetric.mae ?? NaN, lowerIsBetter: true },
          { label: "R²", current: currentMetric.r2 ?? NaN, previous: previousMetric.r2 ?? NaN, lowerIsBetter: false },
        ].flatMap((item) => {
          if (!Number.isFinite(item.current) || !Number.isFinite(item.previous)) return [];
          const delta = item.current - item.previous;
          if (item.lowerIsBetter) {
            const relativeIncrease = item.previous !== 0 ? (item.current - item.previous) / Math.abs(item.previous) : Number.POSITIVE_INFINITY;
            return delta > 0 && relativeIncrease >= 0.03
              ? [{
                  ...item,
                  delta,
                  deltaPct: Number.isFinite(relativeIncrease) ? relativeIncrease * 100 : null,
                }]
              : [];
          }
          return delta <= -0.02
            ? [{
                ...item,
                delta,
                deltaPct: item.previous !== 0 ? (delta / Math.abs(item.previous)) * 100 : null,
              }]
            : [];
        });

    return details.length ? { action: latestEntry.action, details } : null;
  }, [history, historyCursor, modelInfo, models, trainData]);

  // Worker fires on every committed edit (debounced 40ms); incremental update
  // recomputes only the one changed feature's contribution array.
  useEffect(() => {
    if (!currentVersion || !trainData) {
      if (workerDebounceRef.current != null) {
        window.clearTimeout(workerDebounceRef.current);
        workerDebounceRef.current = null;
      }
      setModels(null);
      return;
    }
    if (typeof window === "undefined") return;
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("../workers/modelWorker.ts", import.meta.url));
      workerRef.current.onmessage = (e) => { setModels(e.data); };
    }
    if (workerDebounceRef.current != null) window.clearTimeout(workerDebounceRef.current);
    workerDebounceRef.current = window.setTimeout(() => {
      workerRef.current?.postMessage({
        version: currentVersion,
        trainData,
        modelInfo,
        baselineKnots,
        knotEdits: committedEdits,
        comparisonKnotEdits: comparisonEdits,
      });
      workerDebounceRef.current = null;
    }, 40);
    return () => {
      if (workerDebounceRef.current != null) {
        window.clearTimeout(workerDebounceRef.current);
        workerDebounceRef.current = null;
      }
    };
  }, [baselineKnots, committedEdits, comparisonEdits, currentVersion, modelInfo, trainData]);

  // ─── Derived exports ──────────────────────────────────────────────────────
  const result = useMemo<TrainResponse | null>(() => {
    if (!modelInfo || !trainData || !currentVersion) return null;
    return { model: modelInfo, data: trainData, version: currentVersion };
  }, [modelInfo, trainData, currentVersion]);

  const fixedLinesByFeature = useMemo<Record<string, FixedLineSnapshot[]>>(() => {
    const snapshots: Record<string, FixedLineSnapshot[]> = {};
    const currentByFeature: Record<string, KnotSet> = {};
    history.slice(0, historyCursor).forEach((entry, index) => {
      const base = currentByFeature[entry.featureKey] ?? baselineKnots[entry.featureKey] ?? { x: [], y: [] };
      const next = applyChanges(base, entry.changes, "redo");
      currentByFeature[entry.featureKey] = next;
      snapshots[entry.featureKey] = [
        ...(snapshots[entry.featureKey] ?? []),
        { id: `${entry.ts}-${index}`, knots: cloneKnots(next) },
      ].slice(-2);
    });
    return snapshots;
  }, [history, historyCursor, baselineKnots]);

  const handleKnotSelectionChange = useCallback((next: number[]) => {
    setSelectedKnots(next);
    logEvent({
      category: "edit",
      action: "edit.selection_changed",
      detail: { count: next.length },
    });
  }, [logEvent]);

  // ─── Public API ───────────────────────────────────────────────────────────
  return {
    datasets: DATASETS,
    dataset,
    modelType,
    centerShapes,
    setDataset,
    setModelType,
    setCenterShapes,
    selectedFeatures,
    setSelectedFeatures,
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
    nHid,
    setNHid,
    scaleY,
    setScaleY,
    sampleSize,
    setSampleSize,
    status,
    result,
    models,
    modelInfo,
    trainData,
    currentVersion,
    activePartialIdx,
    setActivePartialIdx,
    baselineKnots,
    knots,
    setKnots,
    knotEdits,
    setKnotEdits,
    fixedLinesByFeature,
    selectedKnots,
    setSelectedKnots: handleKnotSelectionChange,
    history,
    historyCursor,
    recordAction,
    commitEdits,
    undoLast,
    redoLast,
    deleteHistoryEntry,
    countCascadingDeletes,
    metricWarning,
    modelSource,
    train,
    handleSave,
    defaultSaveName,
    sidebarTab,
    setSidebarTab,
    selectedDataset,
    partial,
    savedModels,
    saveSnapshot,
    clearSavedModels,
    handleModelSelect,
  };
};
