import { Dispatch, SetStateAction, useMemo, useState } from "react";
import VisxShapeEditor from "./VisxShapeEditor";
import styles from "../page.module.css";
import { KnotSet, TrainResponse } from "../types";
import CategoricalShapePlot from "./CategoricalShapePlot";
import { useShapeFunctionActions } from "../hooks/useShapeFunctionActions";
import ShapeFunctionsGridView from "./ShapeFunctionsGridView";

const ACTION_ICON_URLS = {
  align: "/action-icons/align.drawio.png",
  drag: "/action-icons/drag.drawio.png",
  monInc: "/action-icons/mon_inc.drawio.png",
  monDec: "/action-icons/mon_dec.drawio.png",
};

type Props = {
  result: TrainResponse;
  baselineKnots: Record<string, KnotSet>;
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
  applyMonotonic: (direction: "increasing" | "decreasing") => void;
  addPointsInSelection: () => void;
  smoothAmount: number;
  smoothingMode: boolean;
  setSmoothingMode: Dispatch<SetStateAction<boolean>>;
  smoothingRangeMax: number;
};

export default function ShapeFunctionsPanel({
  result,
  baselineKnots,
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
  applyMonotonic,
  addPointsInSelection,
  smoothAmount,
  smoothingMode,
  setSmoothingMode,
  smoothingRangeMax,
}: Props) {
  const [interactionMode, setInteractionMode] = useState<"select" | "zoom">("select");
  const [viewMode, setViewMode] = useState<"single" | "grid">("single");
  const partial = useMemo(() => result.partials[activePartialIdx] ?? result.partials[0] ?? null, [result, activePartialIdx]);

  const catRanges = useMemo(() => {
    const ranges: Record<string, { min: number; max: number }> = {};
    const allVals: number[] = [];
    result.partials.forEach((p) => {
      if (!p.categories || !p.categories.length) return;
      const baseKnots = baselineKnots[p.key] ?? { x: p.editableX ?? [], y: p.editableY ?? [] };
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
    result.partials.forEach((p) => {
      if (p.categories && p.categories.length) {
        ranges[p.key] = shared;
      }
    });
    return ranges;
  }, [result.partials, baselineKnots]);
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
  });

  if (!partial) return null;
  const activeFeatureLocked = lockedFeatures.includes(partial.key);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <p className={styles.panelEyebrow}>Shape functions</p>
        <div className={styles.panelControlRow}>
          {viewMode === "single" ? (
            <div className={styles.featureHeaderRow}>
              <button
                type="button"
                className={styles.navButtonInline}
                onClick={() => setActivePartialIdx((prev) => (prev - 1 + result.partials.length) % result.partials.length)}
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
                {result.partials.map((p, idx) => (
                  <option key={p.key} value={idx}>
                    {p.categories && p.categories.length ? "Cat • " : "Cont • "}
                    {p.label || p.key || `x${idx + 1}`}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className={styles.navButtonInline}
                onClick={() => setActivePartialIdx((prev) => (prev + 1) % result.partials.length)}
                aria-label="Next feature"
              >
                ›
              </button>
            </div>
          ) : (
            <div />
          )}
          <div className={styles.panelActions}>
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
              <>
                <button
                  className={styles.panelButton}
                  type="button"
                  onClick={() => onToggleFeatureLock(partial.key)}
                >
                  {activeFeatureLocked ? "Unlock feature" : "Lock feature"}
                </button>
                <div className={styles.panelToggle}>
                  <button
                    type="button"
                    className={`${styles.panelToggleButton} ${interactionMode === "select" ? styles.panelToggleButtonActive : ""}`}
                    onClick={() => setInteractionMode("select")}
                  >
                    ◎
                  </button>
                  <button
                    type="button"
                    className={`${styles.panelToggleButton} ${interactionMode === "zoom" ? styles.panelToggleButtonActive : ""}`}
                    onClick={() => setInteractionMode("zoom")}
                  >
                    ⛶
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
      {viewMode === "single" ? (
      <div className={styles.actionsScroll}>
        <div className={styles.actionsGroup}>
          <div className={styles.actionsStack}>
            <button
              className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
              type="button"
              disabled={selectedKnots.length === 0}
              aria-label="Align selection"
              onClick={alignSelection}
            >
              <img src={ACTION_ICON_URLS.align} alt="" aria-hidden="true" className={styles.actionIconImage} />
            </button>
              {!partial.categories?.length ? (
                <button
                  className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                  type="button"
                  disabled={selectedKnots.length < 2}
                  aria-label="Interpolate line"
                  onClick={interpolateSelection}
                >
                  <img src={ACTION_ICON_URLS.drag} alt="" aria-hidden="true" className={styles.actionIconImage} />
                </button>
              ) : null}
              {partial.categories?.length ? (
                <button
                  className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                  type="button"
                  disabled={selectedKnots.length === 0}
                  aria-label="Set to zero"
                  onClick={setSelectionToZero}
                >
                  0
                </button>
              ) : null}
            <div className={styles.actionsRow}>
              {!partial.categories?.length ? (
                <>
                  <button
                    className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                    type="button"
                    disabled={selectedKnots.length === 0}
                    aria-label="Monotonic increase"
                    onClick={() => {
                      applyMonotonic("increasing");
                    }}
                  >
                    <img src={ACTION_ICON_URLS.monInc} alt="" aria-hidden="true" className={styles.actionIconImage} />
                  </button>
                  <button
                    className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                    type="button"
                    disabled={selectedKnots.length === 0}
                    aria-label="Monotonic decrease"
                    onClick={() => {
                      applyMonotonic("decreasing");
                    }}
                  >
                    <img src={ACTION_ICON_URLS.monDec} alt="" aria-hidden="true" className={styles.actionIconImage} />
                  </button>
                </>
              ) : null}
            </div>
            {!partial.categories?.length ? (
              <button
                className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                type="button"
                aria-label={smoothingMode ? "Disable smoothing mode" : "Enable smoothing mode"}
                onClick={() => setSmoothingMode((prev) => !prev)}
              >
                {smoothingMode ? "◎" : "○"}
              </button>
            ) : null}
            {!partial.categories?.length ? (
              <button
                className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                type="button"
                disabled={selectedKnots.length < 2}
                aria-label="Add points between"
                onClick={() => {
                  addPointsInSelection();
                }}
              >
                ＋
              </button>
            ) : null}
          </div>
        </div>
      </div>
      ) : null}
      <div className={styles.shapeLegend} aria-label="Shape function legend">
        <span className={styles.shapeLegendItem}>
          <span className={`${styles.shapeLegendSwatch} ${styles.shapeLegendSwatchBefore}`} />
          Initial
        </span>
        <span className={styles.shapeLegendItem}>
          <span className={`${styles.shapeLegendSwatch} ${styles.shapeLegendSwatchCurrent}`} />
          Current
        </span>
      </div>
      {viewMode === "grid" ? (
        <ShapeFunctionsGridView
          result={result}
          baselineKnots={baselineKnots}
          knotEdits={knotEdits}
          onSelectFeature={(idx) => {
            setActivePartialIdx(idx);
            setViewMode("single");
          }}
        />
      ) : (() => {
        const p = result.partials[activePartialIdx];
        if (!p) return null;
        const label = p.label || p.key || `x${activePartialIdx + 1}`;
        const baseKnots = knotEdits[p.key] ?? baselineKnots[p.key] ?? { x: p.editableX ?? [], y: p.editableY ?? [] };
        if (p.categories && p.categories.length) {
          const catRange = catRanges[p.key];
                return (
          <CategoricalShapePlot
            categories={p.categories}
            knots={baseKnots}
            baseline={baselineKnots[p.key]}
            title={label}
            fixedRange={catRange}
            interactionMode={interactionMode}
            selectedIdxs={selectedKnots}
            onSelect={(nextSel) => setSelectedKnots(nextSel)}
            onValueChange={(idxSel, value) => handleCatValueChange(p.key, idxSel, value)}
            onMultiValueChange={(indices, values) => handleCatMultiValueChange(p.key, indices, values)}
            onDragStart={() => handleCatDragStart(p.key)}
            onDragEnd={() => handleCatDragEnd(p.key)}
            />
          );
        }

        return (
          <VisxShapeEditor
            knots={baseKnots}
            baseline={baselineKnots[p.key]}
            scatterX={p.scatterX}
            featureKey={p.key}
            interactionMode={interactionMode}
            selected={selectedKnots}
            onSelectionChange={(nextSel) => setSelectedKnots(nextSel)}
            onKnotChange={handleKnotChange}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            onSmoothEnd={(start, end) => handleSmoothEnd(p.key, start, end)}
            title={label}
            smoothingMode={smoothingMode}
            smoothAmount={smoothAmount}
            smoothingRangeMax={smoothingRangeMax}
          />
        );
      })()}
    </div>
  );
}
