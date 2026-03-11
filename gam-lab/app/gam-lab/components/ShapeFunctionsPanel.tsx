import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import VisxShapeEditor, { type DragCurve, type SmoothingAlgorithm } from "./VisxShapeEditor";
import styles from "../page.module.css";
import { KnotSet, ShapeFunction, TrainData } from "../types";
import CategoricalShapePlot from "./CategoricalShapePlot";
import { useShapeFunctionActions } from "../hooks/useShapeFunctionActions";
import ShapeFunctionsGridView from "./ShapeFunctionsGridView";
import { computeFeatureImportance } from "../lib/importance";

type ContinuousActionTool = "drag" | "smooth";

const ACTION_ICON_URLS = {
  align: "/action-icons/align.drawio.png",
  drag: "/action-icons/drag.drawio.png",
  monInc: "/action-icons/mon_inc.drawio.png",
  monDec: "/action-icons/mon_dec.drawio.png",
};

type Props = {
  shapes: ShapeFunction[];
  trainData: TrainData;
  baselineKnots: Record<string, KnotSet>;
  fixedLinesByFeature: Record<string, Array<{ id: string; knots: KnotSet }>>;
  knots: KnotSet;
  setKnots: Dispatch<SetStateAction<KnotSet>>;
  knotEdits: Record<string, KnotSet>;
  setKnotEdits: Dispatch<SetStateAction<Record<string, KnotSet>>>;
  selectedKnots: number[];
  setSelectedKnots: Dispatch<SetStateAction<number[]>>;
  activePartialIdx: number;
  setActivePartialIdx: Dispatch<SetStateAction<number>>;
  lockedFeatures: string[];
  onToggleFeatureLock: (featureKey: string) => void;
  onRecordAction: (featureKey: string, before: KnotSet, after: KnotSet, action?: string) => void;
  onCommitEdits: (featureKey: string, next: KnotSet) => void;
  onUndo: () => void;
  canUndo: boolean;
  onRedo: () => void;
  canRedo: boolean;
  onSave: () => void;
  onInteractionStart?: () => void;
  onInteractionEnd?: () => void;
  applyMonotonic: (direction: "increasing" | "decreasing") => void;
  activeContinuousTool: ContinuousActionTool;
  setActiveContinuousTool: Dispatch<SetStateAction<ContinuousActionTool>>;
  dragFalloffRadius: number;
  setDragFalloffRadius: Dispatch<SetStateAction<number>>;
  dragRangeBoost: number;
  setDragRangeBoost: Dispatch<SetStateAction<number>>;
  smoothAmount: number;
  setSmoothAmount: Dispatch<SetStateAction<number>>;
  smoothingRangeMax: number;
  setSmoothingRangeMax: Dispatch<SetStateAction<number>>;
  smoothingSpeed: number;
  setSmoothingSpeed: Dispatch<SetStateAction<number>>;
  dragCurve: DragCurve;
  setDragCurve: Dispatch<SetStateAction<DragCurve>>;
  smoothingAlgorithm: SmoothingAlgorithm;
  setSmoothingAlgorithm: Dispatch<SetStateAction<SmoothingAlgorithm>>;
};

export default function ShapeFunctionsPanel({
  shapes,
  trainData,
  baselineKnots,
  fixedLinesByFeature,
  knots,
  setKnots,
  knotEdits,
  setKnotEdits,
  selectedKnots,
  setSelectedKnots,
  activePartialIdx,
  setActivePartialIdx,
  lockedFeatures,
  onToggleFeatureLock,
  onRecordAction,
  onCommitEdits,
  onUndo,
  canUndo,
  onRedo,
  canRedo,
  onSave,
  onInteractionStart,
  onInteractionEnd,
  applyMonotonic,
  activeContinuousTool,
  setActiveContinuousTool,
  dragFalloffRadius,
  setDragFalloffRadius,
  dragRangeBoost,
  setDragRangeBoost,
  smoothAmount,
  setSmoothAmount,
  smoothingRangeMax,
  setSmoothingRangeMax,
  smoothingSpeed,
  setSmoothingSpeed,
  dragCurve,
  setDragCurve,
  smoothingAlgorithm,
  setSmoothingAlgorithm,
}: Props) {
  const [panLocked, setPanLocked] = useState(false);
  const [spacePanActive, setSpacePanActive] = useState(false);
  const [viewMode, setViewMode] = useState<"single" | "grid">("single");
  const partial = useMemo(() => shapes[activePartialIdx] ?? shapes[0] ?? null, [shapes, activePartialIdx]);
  const interactionMode: "select" | "zoom" = panLocked || spacePanActive ? "zoom" : "select";
  const isPanning = interactionMode === "zoom";
  const smoothingMode = activeContinuousTool === "smooth";

  useEffect(() => {
    const isEditableTarget = (target: EventTarget | null) => {
      const el = target instanceof HTMLElement ? target : null;
      if (!el) return false;
      if (el.isContentEditable) return true;
      return Boolean(el.closest("input, textarea, select, [contenteditable=\"true\"]"));
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      if (isEditableTarget(event.target)) return;
      event.preventDefault();
      setSpacePanActive(true);
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      setSpacePanActive(false);
    };

    const clearSpacePan = () => setSpacePanActive(false);

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearSpacePan);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearSpacePan);
    };
  }, []);

  const featureImportance = useMemo(() => {
    const rawByKey: Record<string, number> = {};
    shapes.forEach((s) => {
      const scatterX = trainData.trainX[s.key] ?? [];
      const shape = baselineKnots[s.key] ?? { x: s.editableX ?? [], y: s.editableY ?? [] };
      rawByKey[s.key] = Math.max(0, computeFeatureImportance(s, scatterX, shape));
    });
    const total = Object.values(rawByKey).reduce((sum, value) => sum + value, 0);
    const normalizedByKey: Record<string, number> = {};
    Object.entries(rawByKey).forEach(([key, value]) => {
      normalizedByKey[key] = total > 0 ? value / total : 0;
    });
    return { rawByKey, normalizedByKey };
  }, [shapes, trainData.trainX, baselineKnots]);
  const formatImportance = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "0.000");

  const catRanges = useMemo(() => {
    const ranges: Record<string, { min: number; max: number }> = {};
    const allVals: number[] = [];
    shapes.forEach((s) => {
      if (!s.categories || !s.categories.length) return;
      const baseKnots = baselineKnots[s.key] ?? { x: s.editableX ?? [], y: s.editableY ?? [] };
      (baseKnots.y ?? []).forEach((v) => {
        if (Number.isFinite(v)) allVals.push(v as number);
      });
    });
    const fallback = { min: -1, max: 1 };
    const minValRaw = allVals.length ? Math.min(...allVals) : fallback.min;
    const maxValRaw = allVals.length ? Math.max(...allVals) : fallback.max;
    const minVal = Math.min(minValRaw, 0);
    const maxVal = Math.max(maxValRaw, 0);
    const span = maxVal - minVal || Math.max(Math.abs(minVal), Math.abs(maxVal), 1);
    const pad = Math.max(0.1 * span, 0.05 * Math.max(Math.abs(maxVal), Math.abs(minVal), 1));
    const shared = { min: minVal - pad, max: maxVal + pad };
    shapes.forEach((s) => {
      if (s.categories && s.categories.length) {
        ranges[s.key] = shared;
      }
    });
    return ranges;
  }, [shapes, baselineKnots]);

  const {
    handleKnotChange,
    handleDragStart,
    handleDragEnd,
    alignSelection,
    interpolateSelection,
    setSelectionToZero,
    handleCatValueChange,
    handleCatMultiValueChange,
    handleCatDragStart,
    handleCatDragEnd,
    handleSmoothStart,
    handleSmoothEnd,
  } = useShapeFunctionActions({
    partial,
    knots,
    knotEdits,
    selectedKnots,
    setKnots,
    setKnotEdits,
    setSelectedKnots,
    onRecordAction,
    onCommitEdits,
    onInteractionStart,
    onInteractionEnd,
  });

  if (!partial) return null;
  const activeFeatureLocked = lockedFeatures.includes(partial.key);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <p className={styles.panelEyebrow}>Shape functions</p>
        <div className={styles.panelControlRow}>
          <div className={styles.panelToggle}>
            <button
              type="button"
              className={`${styles.panelToggleButton} ${viewMode === "single" ? styles.panelToggleButtonActive : ""}`}
              onClick={() => setViewMode("single")}
            >
              Single
            </button>
            <button
              type="button"
              className={`${styles.panelToggleButton} ${viewMode === "grid" ? styles.panelToggleButtonActive : ""}`}
              onClick={() => setViewMode("grid")}
            >
              Grid
            </button>
          </div>
          {viewMode === "single" ? (
            <div className={styles.featureHeaderRow}>
              <button
                type="button"
                className={styles.navButtonInline}
                onClick={() => setActivePartialIdx((prev) => (prev - 1 + shapes.length) % shapes.length)}
                aria-label="Previous feature"
              >
                ‹
              </button>
              <select
                className={styles.featureSelect}
                value={activePartialIdx}
                onChange={(event) => setActivePartialIdx(Number(event.target.value))}
                aria-label="Feature"
              >
                {shapes.map((s, idx) => (
                  <option key={s.key} value={idx}>
                    {s.categories && s.categories.length ? "Cat • " : "Cont • "}
                    {s.label || s.key || `x${idx + 1}`} • I={formatImportance(featureImportance.normalizedByKey[s.key] ?? 0)}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className={styles.navButtonInline}
                onClick={() => setActivePartialIdx((prev) => (prev + 1) % shapes.length)}
                aria-label="Next feature"
              >
                ›
              </button>
            </div>
          ) : (
            <div />
          )}
          {viewMode === "single" ? (
            <div className={styles.panelActions}>
              <button
                className={`${styles.panelToggleButton} ${styles.lockButton} ${activeFeatureLocked ? styles.panelToggleButtonActive : ""}`}
                type="button"
                onClick={() => onToggleFeatureLock(partial.key)}
                aria-label={activeFeatureLocked ? "Unlock feature" : "Lock feature"}
              >
                {activeFeatureLocked ? "🔒" : "🔓"}
              </button>
            </div>
          ) : null}
        </div>
      </div>
      <div className={styles.shapeLegend} aria-label="Shape function legend">
        <span className={styles.shapeLegendItem}>
          <span className={`${styles.shapeLegendSwatch} ${styles.shapeLegendSwatchInitial}`} />
          Initial
        </span>
        <span className={styles.shapeLegendItem}>
          <span className={`${styles.shapeLegendSwatch} ${styles.shapeLegendSwatchBefore}`} />
          Previous
        </span>
        <span className={styles.shapeLegendItem}>
          <span className={`${styles.shapeLegendSwatch} ${styles.shapeLegendSwatchCurrent}`} />
          Current
        </span>
      </div>
      {viewMode === "grid" ? (
        <ShapeFunctionsGridView
          shapes={shapes}
          baselineKnots={baselineKnots}
          knotEdits={knotEdits}
          onSelectFeature={(idx) => {
            setActivePartialIdx(idx);
            setViewMode("single");
          }}
        />
      ) : (() => {
        const s = shapes[activePartialIdx];
        if (!s) return null;
        const label = s.label || s.key || `x${activePartialIdx + 1}`;
        const baseKnots = knotEdits[s.key] ?? baselineKnots[s.key] ?? { x: s.editableX ?? [], y: s.editableY ?? [] };
        const scatterX = trainData.trainX[s.key] ?? [];
        const plot = s.categories && s.categories.length ? (
          <CategoricalShapePlot
            categories={s.categories}
            scatterX={scatterX}
            knots={baseKnots}
            baseline={baselineKnots[s.key]}
            title={label}
            fixedRange={catRanges[s.key]}
            interactionMode={interactionMode}
            selectedIdxs={selectedKnots}
            onSelect={(nextSel) => setSelectedKnots(nextSel)}
            onValueChange={(idxSel, value) => handleCatValueChange(s.key, idxSel, value)}
            onMultiValueChange={(indices, values) => handleCatMultiValueChange(s.key, indices, values)}
            onDragStart={() => handleCatDragStart(s.key)}
            onDragEnd={() => handleCatDragEnd(s.key)}
          />
        ) : (
          <VisxShapeEditor
            key={s.key}
            knots={baseKnots}
            baseline={baselineKnots[s.key]}
            fixedLines={fixedLinesByFeature[s.key] ?? []}
            scatterX={scatterX}
            featureKey={s.key}
            interactionMode={interactionMode}
            selected={selectedKnots}
            onSelectionChange={(nextSel) => setSelectedKnots(nextSel)}
            onKnotChange={handleKnotChange}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            onSmoothStart={handleSmoothStart}
            onSmoothEnd={(start, end) => handleSmoothEnd(s.key, start, end)}
            title={label}
            dragFalloffRadius={dragFalloffRadius}
            dragRangeBoost={dragRangeBoost}
            dragCurve={dragCurve}
            smoothingMode={smoothingMode}
            smoothAmount={smoothAmount}
            smoothingRangeMax={smoothingRangeMax}
            smoothingSpeed={smoothingSpeed}
            smoothingAlgorithm={smoothingAlgorithm}
          />
        );
        const showContinuousTools = !s.categories?.length;
        return (
          <>
            <div className={styles.plotWithActionsRow}>
              <div className={styles.plotArea}>{plot}</div>
              <div className={styles.actionsScroll}>
                <div className={styles.actionsStack}>
                  <div className={styles.actionsGroup}>
                    <span className={styles.actionsGroupLabel}>navigate</span>
                    <button
                      className={`${styles.panButton} ${panLocked ? styles.actionButtonActive : ""}`}
                      type="button"
                      aria-label="Pan mode"
                      aria-pressed={panLocked}
                      onClick={() => setPanLocked((prev) => !prev)}
                    >
                      Pan
                    </button>
                    <span className={styles.actionsStatus}>
                      {spacePanActive && !panLocked ? "Space-pan active" : panLocked ? "Pan locked" : "Hold Space to pan"}
                    </span>
                  </div>
                  <hr className={styles.actionsDivider} />
                  {showContinuousTools ? (
                    <div className={styles.actionsGroup}>
                      <div className={styles.toolButtonGrid}>
                        <button
                          className={`${styles.actionButton} ${styles.toolButton} ${activeContinuousTool === "drag" ? styles.actionButtonActive : ""}`}
                          type="button"
                          disabled={isPanning}
                          aria-pressed={activeContinuousTool === "drag"}
                          onClick={() => setActiveContinuousTool("drag")}
                        >
                          Drag
                        </button>
                        <button
                          className={`${styles.actionButton} ${styles.toolButton} ${activeContinuousTool === "smooth" ? styles.actionButtonActive : ""}`}
                          type="button"
                          disabled={isPanning}
                          aria-pressed={activeContinuousTool === "smooth"}
                          onClick={() => setActiveContinuousTool("smooth")}
                        >
                          Smooth
                        </button>
                      </div>
                      <div className={styles.actionSettingsCard}>
                        {activeContinuousTool === "drag" ? (
                          <>
                            <span className={styles.actionSettingLabel}>Falloff curve</span>
                            <div className={styles.actionChips}>
                              {(
                                [
                                  { value: "gaussian", label: "Gaussian" },
                                  { value: "linear", label: "Linear" },
                                  { value: "cosine", label: "Cosine" },
                                  { value: "sharp", label: "Sharp" },
                                ] as const
                              ).map(({ value, label }) => (
                                <button
                                  key={value}
                                  type="button"
                                  className={`${styles.actionChip} ${dragCurve === value ? styles.actionChipActive : ""}`}
                                  onClick={() => setDragCurve(value)}
                                >
                                  {label}
                                </button>
                              ))}
                            </div>
                            <span className={styles.actionSettingDesc}>
                              {dragCurve === "gaussian" && "Bell-shaped falloff — smooth, natural feel"}
                              {dragCurve === "linear" && "Even ramp from center to edge — predictable"}
                              {dragCurve === "cosine" && "S-shaped rolloff — softer than linear"}
                              {dragCurve === "sharp" && "Tight bell — localized, precise pull"}
                            </span>
                            <hr className={styles.actionSettingsDivider} />
                            <label className={styles.actionSettingLabel}>
                              <span>Influence radius</span>
                              <span>{dragFalloffRadius}</span>
                            </label>
                            <input
                              className={styles.actionRange}
                              type="range"
                              min="0"
                              max="24"
                              step="1"
                              value={dragFalloffRadius}
                              onChange={(event) => setDragFalloffRadius(Number(event.target.value))}
                            />
                            <label className={styles.actionSettingLabel}>
                              <span>Spread on drag right</span>
                              <span>{dragRangeBoost.toFixed(1)}x</span>
                            </label>
                            <input
                              className={styles.actionRange}
                              type="range"
                              min="0"
                              max="3"
                              step="0.1"
                              value={dragRangeBoost}
                              onChange={(event) => setDragRangeBoost(Number(event.target.value))}
                            />
                          </>
                        ) : (
                          <>
                            <span className={styles.actionSettingLabel}>Algorithm</span>
                            <div className={styles.actionChips}>
                              {(
                                [
                                  { value: "gaussian", label: "Gaussian" },
                                  { value: "box", label: "Box avg" },
                                  { value: "median", label: "Median" },
                                  { value: "exponential", label: "Exp" },
                                ] as const
                              ).map(({ value, label }) => (
                                <button
                                  key={value}
                                  type="button"
                                  className={`${styles.actionChip} ${smoothingAlgorithm === value ? styles.actionChipActive : ""}`}
                                  onClick={() => setSmoothingAlgorithm(value)}
                                >
                                  {label}
                                </button>
                              ))}
                            </div>
                            <span className={styles.actionSettingDesc}>
                              {smoothingAlgorithm === "gaussian" && "Weighted average — smooth, general purpose"}
                              {smoothingAlgorithm === "box" && "Uniform average — stronger, flatter result"}
                              {smoothingAlgorithm === "median" && "Removes spikes without blurring the shape"}
                              {smoothingAlgorithm === "exponential" && "Decays from center — preserves overall trend"}
                            </span>
                            <hr className={styles.actionSettingsDivider} />
                            <label className={styles.actionSettingLabel}>
                              <span>Strength</span>
                              <span>{smoothAmount.toFixed(2)}</span>
                            </label>
                            <input
                              className={styles.actionRange}
                              type="range"
                              min="0.1"
                              max="1"
                              step="0.05"
                              value={smoothAmount}
                              onChange={(event) => setSmoothAmount(Number(event.target.value))}
                            />
                            <label className={styles.actionSettingLabel}>
                              <span>Range</span>
                              <span>{smoothingRangeMax}</span>
                            </label>
                            <input
                              className={styles.actionRange}
                              type="range"
                              min="4"
                              max="64"
                              step="1"
                              value={smoothingRangeMax}
                              onChange={(event) => setSmoothingRangeMax(Number(event.target.value))}
                            />
                            <label className={styles.actionSettingLabel}>
                              <span>Speed</span>
                              <span>{smoothingSpeed.toFixed(1)}x</span>
                            </label>
                            <input
                              className={styles.actionRange}
                              type="range"
                              min="0.3"
                              max="3"
                              step="0.1"
                              value={smoothingSpeed}
                              onChange={(event) => setSmoothingSpeed(Number(event.target.value))}
                            />
                          </>
                        )}
                      </div>
                    </div>
                  ) : null}
                  <hr className={styles.actionsDivider} />
                  <div className={styles.actionsGroup}>
                    <span className={styles.actionsGroupLabel}>{`selection (${selectedKnots.length})`}</span>
                    <div className={styles.selectionActionsGrid}>
                      {!s.categories?.length ? (
                        <button
                          className={styles.selectionActionButton}
                          type="button"
                          disabled={isPanning}
                          onClick={interpolateSelection}
                        >
                          <img src={ACTION_ICON_URLS.drag} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                          <span className={styles.selectionActionLabel}>Interp.</span>
                        </button>
                      ) : null}
                      <button
                        className={styles.selectionActionButton}
                        type="button"
                        disabled={isPanning || selectedKnots.length === 0}
                        onClick={alignSelection}
                      >
                        <img src={ACTION_ICON_URLS.align} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                        <span className={styles.selectionActionLabel}>Align</span>
                      </button>
                      {s.categories?.length ? (
                        <button
                          className={styles.selectionActionButton}
                          type="button"
                          disabled={isPanning || selectedKnots.length === 0}
                          onClick={setSelectionToZero}
                        >
                          <span style={{ fontSize: "1rem", fontWeight: 700, lineHeight: 1 }}>0</span>
                          <span className={styles.selectionActionLabel}>Zero</span>
                        </button>
                      ) : null}
                      {!s.categories?.length ? (
                        <>
                          <button
                            className={styles.selectionActionButton}
                            type="button"
                            disabled={isPanning || selectedKnots.length === 0}
                            onClick={() => applyMonotonic("increasing")}
                          >
                            <img src={ACTION_ICON_URLS.monInc} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                            <span className={styles.selectionActionLabel}>Mon ↑</span>
                          </button>
                          <button
                            className={styles.selectionActionButton}
                            type="button"
                            disabled={isPanning || selectedKnots.length === 0}
                            onClick={() => applyMonotonic("decreasing")}
                          >
                            <img src={ACTION_ICON_URLS.monDec} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                            <span className={styles.selectionActionLabel}>Mon ↓</span>
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className={styles.shapePanelFooter}>
              <div className={styles.shapeEditActions}>
                <button type="button" className={styles.undoButton} onClick={onUndo} disabled={!canUndo}>
                  Undo
                </button>
                <button type="button" className={styles.undoButton} onClick={onRedo} disabled={!canRedo}>
                  Redo
                </button>
                <button type="button" className={styles.undoButton} onClick={onSave}>
                  Save edits
                </button>
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
}
