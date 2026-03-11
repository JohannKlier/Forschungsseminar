import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import VisxShapeEditor from "./VisxShapeEditor";
import styles from "../page.module.css";
import { KnotSet, ShapeFunction, TrainData } from "../types";
import CategoricalShapePlot from "./CategoricalShapePlot";
import { useShapeFunctionActions } from "../hooks/useShapeFunctionActions";
import ShapeFunctionsGridView from "./ShapeFunctionsGridView";
import { computeFeatureImportance } from "../lib/importance";

const ACTION_ICON_URLS = {
  align: "/action-icons/align.drawio.png",
  drag: "/action-icons/drag.drawio.png",
  monInc: "/action-icons/mon_inc.drawio.png",
  monDec: "/action-icons/mon_dec.drawio.png",
};

type Props = {
  activeTourFocus?: "shape-overview" | "shape-plot" | "shape-actions" | "shape-views" | null;
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
  addPointsInSelection: () => void;
  smoothAmount: number;
  smoothingMode: boolean;
  setSmoothingMode: Dispatch<SetStateAction<boolean>>;
  smoothingRangeMax: number;
};

export default function ShapeFunctionsPanel({
  activeTourFocus = null,
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
  addPointsInSelection,
  smoothAmount,
  smoothingMode,
  setSmoothingMode,
  smoothingRangeMax,
}: Props) {
  const [panLocked, setPanLocked] = useState(false);
  const [spacePanActive, setSpacePanActive] = useState(false);
  const [viewMode, setViewMode] = useState<"single" | "grid">("single");
  const partial = useMemo(() => shapes[activePartialIdx] ?? shapes[0] ?? null, [shapes, activePartialIdx]);
  const interactionMode: "select" | "zoom" = panLocked || spacePanActive ? "zoom" : "select";
  const isPanning = interactionMode === "zoom";

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
    <div className={`${styles.panel} ${activeTourFocus === "shape-overview" ? styles.tourFocus : ""}`}>
      <div
        className={`${styles.panelHeader} ${
          activeTourFocus === "shape-actions" || activeTourFocus === "shape-views" ? styles.tourFocus : ""
        }`}
      >
        <p className={styles.panelEyebrow}>Shape functions</p>
        <div className={styles.panelControlRow}>
          <div className={`${styles.panelToggle} ${activeTourFocus === "shape-views" ? styles.tourFocus : ""}`}>
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
            <div className={`${styles.featureHeaderRow} ${activeTourFocus === "shape-actions" ? styles.tourFocus : ""}`}>
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
            <div className={`${styles.panelActions} ${activeTourFocus === "shape-actions" ? styles.tourFocus : ""}`}>
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
            smoothingMode={smoothingMode}
            smoothAmount={smoothAmount}
            smoothingRangeMax={smoothingRangeMax}
          />
        );
        return (
          <>
            <div className={styles.plotWithActionsRow}>
              <div className={`${styles.plotArea} ${activeTourFocus === "shape-plot" ? styles.tourFocus : ""}`}>{plot}</div>
              <div className={`${styles.actionsScroll} ${activeTourFocus === "shape-actions" ? styles.tourFocus : ""}`}>
                <div className={styles.actionsStack}>
                  <button
                    className={`${styles.actionButton} ${styles.actionButtonWide} ${styles.actionIconButton} ${panLocked ? styles.actionButtonActive : ""}`}
                    type="button"
                    aria-label="Pan mode"
                    aria-pressed={panLocked}
                    title="Pan the plot. You can also hold Space temporarily."
                    onClick={() => setPanLocked((prev) => !prev)}
                  >
                    Pan
                  </button>
                  <hr className={styles.actionsDivider} />
                  <div className={styles.actionsGroup}>
                    <span className={styles.actionsGroupLabel}>navigate</span>
                    <span className={styles.actionsStatus}>
                      {spacePanActive && !panLocked ? "Space-pan active" : panLocked ? "Pan locked" : "Hold Space to pan"}
                    </span>
                  </div>
                  {!s.categories?.length ? (
                    <div className={styles.actionsGroup}>
                      <span className={styles.actionsGroupLabel}>tool</span>
                      <button
                        className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton} ${smoothingMode ? styles.actionButtonActive : ""}`}
                        type="button"
                        disabled={isPanning}
                        aria-label={smoothingMode ? "Disable smoothing mode" : "Enable smoothing mode"}
                        aria-pressed={smoothingMode}
                        onClick={() => setSmoothingMode((prev) => !prev)}
                      >
                        {smoothingMode ? "◎" : "○"}
                      </button>
                    </div>
                  ) : null}
                  <div className={styles.actionsGroup}>
                    <span className={styles.actionsGroupLabel}>{`selection (${selectedKnots.length})`}</span>
                    {!s.categories?.length ? (
                      <button
                        className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                        type="button"
                        disabled={isPanning}
                        aria-label="Interpolate line"
                        onClick={interpolateSelection}
                      >
                        <img src={ACTION_ICON_URLS.drag} alt="" aria-hidden="true" className={styles.actionIconImage} />
                      </button>
                    ) : null}
                    <button
                      className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                      type="button"
                      disabled={isPanning || selectedKnots.length === 0}
                      aria-label="Align selection"
                      onClick={alignSelection}
                    >
                      <img src={ACTION_ICON_URLS.align} alt="" aria-hidden="true" className={styles.actionIconImage} />
                    </button>
                    {s.categories?.length ? (
                      <button
                        className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                        type="button"
                        disabled={isPanning || selectedKnots.length === 0}
                        aria-label="Set to zero"
                        onClick={setSelectionToZero}
                      >
                        0
                      </button>
                    ) : null}
                    {!s.categories?.length ? (
                      <>
                        <button
                          className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                          type="button"
                          disabled={isPanning || selectedKnots.length === 0}
                          aria-label="Monotonic increase"
                          onClick={() => applyMonotonic("increasing")}
                        >
                          <img src={ACTION_ICON_URLS.monInc} alt="" aria-hidden="true" className={styles.actionIconImage} />
                        </button>
                        <button
                          className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                          type="button"
                          disabled={isPanning || selectedKnots.length === 0}
                          aria-label="Monotonic decrease"
                          onClick={() => applyMonotonic("decreasing")}
                        >
                          <img src={ACTION_ICON_URLS.monDec} alt="" aria-hidden="true" className={styles.actionIconImage} />
                        </button>
                        <button
                          className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                          type="button"
                          disabled={isPanning || selectedKnots.length < 2}
                          aria-label="Add points between"
                          onClick={() => addPointsInSelection()}
                        >
                          ＋
                        </button>
                      </>
                    ) : null}
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
