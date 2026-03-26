import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import styles from "../page.module.css";
import { KnotSet, ShapeFunction, TrainData } from "../types";
import CategoricalShapePlot from "./CategoricalShapePlot";
import VisxShapeEditor from "./VisxShapeEditor";
import { InteractionHeatmap } from "./InteractionHeatmap";
import { useShapeFunctionActions, type AlignSelectionMode } from "../hooks/useShapeFunctionActions";
import ShapeFunctionsGridView from "./ShapeFunctionsGridView";
import { computeShapeImportance } from "../lib/importance";
import TourLabel from "./TourLabel";
import { type ToolSettings } from "../hooks/useToolSettings";

const ACTION_ICON_URLS = {
  align: "/action-icons/align.drawio.png",
  drag: "/action-icons/drag.drawio.png",
};

const ALIGN_ACTIONS: { value: AlignSelectionMode; label: string; shortLabel: string }[] = [
  { value: "left", label: "Align left", shortLabel: "Left" },
  { value: "center", label: "Align center", shortLabel: "Center" },
  { value: "right", label: "Align right", shortLabel: "Right" },
];

type Props = {
  showTourLabels?: boolean;
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
  toolSettings: ToolSettings;
  hideLock?: boolean;
};

export default function ShapeFunctionsPanel({
  showTourLabels = false,
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
  toolSettings,
  hideLock = false,
}: Props) {
  const {
    activeContinuousTool, setActiveContinuousTool,
    dragFalloffRadius, dragRangeBoost, dragCurve, setDragCurve,
    smoothAmount, smoothingRangeMax, setSmoothingRangeMax,
    smoothingSpeed, smoothingAlgorithm, setSmoothingAlgorithm,
  } = toolSettings;
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
      rawByKey[s.key] = Math.max(0, computeShapeImportance(s, scatterX, shape));
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
      if (s.editableZ) return; // skip 2-D interaction shapes
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
    <div className={`${styles.panel} ${styles.panelFillHeight} ${showTourLabels ? styles.tourFocus : ""}`}>
      <div className={styles.panelHeader}>
        <div className={styles.panelEyebrowRow}>
          <p className={styles.panelEyebrow}>Shape functions</p>
          <div className={styles.panelEyebrowActions}>
            {viewMode === "single" && !hideLock ? (
              <button
                className={`${styles.panelToggleButton} ${styles.lockButton} ${activeFeatureLocked ? styles.panelToggleButtonActive : ""}`}
                type="button"
                onClick={() => onToggleFeatureLock(partial.key)}
                aria-label={activeFeatureLocked ? "Unlock feature" : "Lock feature"}
              >
                {activeFeatureLocked ? "🔒" : "🔓"}
              </button>
            ) : null}
            <div className={`${styles.panelToggle} ${showTourLabels ? styles.tourLabelAnchor : ""}`}>
              {showTourLabels ? (
                <TourLabel
                  label="View mode"
                  title="Choose the editing lens"
                  description="Switch between focused editing of one shape and a scan across all shapes."
                  details={[
                    "Single view shows the active feature with the full editor and tool panel.",
                    "Grid view helps compare many feature shapes quickly and jump into one with a click.",
                  ]}
                  placement="top-right"
                />
              ) : null}
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
          </div>
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
        const plot = s.editableZ ? (
          <InteractionHeatmap shape={s} />
        ) : s.categories && s.categories.length ? (
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
        const showContinuousTools = !s.editableZ && !s.categories?.length;
        return (
          <>
            <div className={`${styles.featureNavCentered} ${showTourLabels ? styles.tourLabelAnchor : ""}`}>
              {showTourLabels ? (
                <TourLabel
                  label="Feature selector"
                  title="Move between features"
                  description="The selector and arrows change the active feature without leaving the editor."
                  details={[
                    "Each option shows whether the feature is continuous or categorical.",
                    "The I value is the normalized importance score used for quick prioritization.",
                  ]}
                  placement="top-left"
                />
              ) : null}
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
                {shapes.map((sf, idx) => (
                  <option key={sf.key} value={idx}>
                    {sf.editableZ ? "2D • " : sf.categories && sf.categories.length ? "Cat • " : "Cont • "}
                    {sf.label || sf.key || `x${idx + 1}`}
                    {` • I=${formatImportance(featureImportance.normalizedByKey[sf.key] ?? 0)}`}
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
            <div className={styles.plotWithActionsRow}>
              <div className={`${styles.plotArea} ${showTourLabels ? styles.tourLabelAnchor : ""}`}>
                {showTourLabels ? (
                  <TourLabel
                    label="Plot"
                    title="Edit directly on the shape"
                    description="Most edits happen in the plot itself: drag, select, smooth, or inspect the current curve against earlier states."
                    details={[
                      "Continuous features let you drag knots and apply selection-based actions.",
                      "Categorical features update bar heights directly.",
                      "Initial, previous, and current lines help you see what changed.",
                    ]}
                    placement="top-left"
                  />
                ) : null}
                {plot}
              </div>
              <div className={`${styles.actionsScroll} ${showTourLabels ? styles.tourLabelAnchor : ""}`}>
                {showTourLabels ? (
                <TourLabel
                  label="Tools"
                  title="Adjust how edits behave"
                  description="The tool rail controls navigation, drag or smoothing behavior, and selection-based operations."
                    details={[
                      "Pan mode changes the interaction from editing to navigation.",
                      "Continuous features expose drag and smooth settings.",
                      "Selection actions apply constraints or bulk edits to the current selection.",
                    ]}
                  placement="top-right"
                />
              ) : null}
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
                    <div className={styles.toolTabContainer}>
                      <div className={styles.toolTabBar} role="tablist">
                        <button
                          className={`${styles.toolTab} ${activeContinuousTool === "drag" ? styles.toolTabActive : ""}`}
                          type="button"
                          role="tab"
                          aria-selected={activeContinuousTool === "drag"}
                          disabled={isPanning}
                          onClick={() => setActiveContinuousTool("drag")}
                        >
                          Drag
                        </button>
                        <button
                          className={`${styles.toolTab} ${activeContinuousTool === "smooth" ? styles.toolTabActive : ""}`}
                          type="button"
                          role="tab"
                          aria-selected={activeContinuousTool === "smooth"}
                          disabled={isPanning}
                          onClick={() => setActiveContinuousTool("smooth")}
                        >
                          Smooth
                        </button>
                      </div>
                      <div className={styles.toolTabPanel} role="tabpanel">
                        {activeContinuousTool === "drag" ? (
                          <>
                            <span className={styles.actionSettingLabel}>Falloff curve</span>
                            <div className={styles.actionChips}>
                              {(
                                [
                                  { value: "gaussian", label: "Gaussian", icon: "G" },
                                  { value: "linear", label: "Linear", icon: "L" },
                                  { value: "sharp", label: "Sharp", icon: "S" },
                                ] as const
                              ).map(({ value, label }) => (
                                <button
                                  key={value}
                                  type="button"
                                  className={`${styles.actionChip} ${dragCurve === value ? styles.actionChipActive : ""}`}
                                  onClick={() => setDragCurve(value)}
                                  aria-label={label}
                                  title={label}
                                >
                                  <span className={styles.actionChipIcon} aria-hidden="true">
                                    {label.charAt(0)}
                                  </span>
                                </button>
                              ))}
                            </div>
                          </>
                        ) : (
                          <>
                            <span className={styles.actionSettingLabel}>Mode</span>
                            <div className={styles.actionChips}>
                              {(
                                [
                                  { value: "gaussian", label: "Gaussian", icon: "G" },
                                  { value: "median", label: "Median", icon: "M" },
                                  { value: "exponential", label: "Exponential", icon: "E" },
                                ] as const
                              ).map(({ value, label }) => (
                                <button
                                  key={value}
                                  type="button"
                                  className={`${styles.actionChip} ${smoothingAlgorithm === value ? styles.actionChipActive : ""}`}
                                  onClick={() => setSmoothingAlgorithm(value)}
                                  aria-label={label}
                                  title={label}
                                >
                                  <span className={styles.actionChipIcon} aria-hidden="true">
                                    {label.charAt(0)}
                                  </span>
                                </button>
                              ))}
                            </div>
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
                          </>
                        )}
                      </div>
                    </div>
                  ) : null}
                  <hr className={styles.actionsDivider} />
                  <div className={`${styles.actionsGroup} ${selectedKnots.length === 0 ? styles.actionsGroupMuted : ""}`}>
                    <span className={styles.actionsGroupLabel}>Selection</span>
                    <div className={styles.selectionActionsGrid}>
                      {!s.categories?.length ? (
                        <button
                          className={styles.selectionActionButton}
                          type="button"
                          disabled={isPanning || selectedKnots.length === 0}
                          onClick={interpolateSelection}
                        >
                          <img src={ACTION_ICON_URLS.drag} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                          <span className={styles.selectionActionLabel}>Straight</span>
                        </button>
                      ) : null}
                      {!s.categories?.length
                        ? ALIGN_ACTIONS.map(({ value, label, shortLabel }) => (
                            <button
                              key={value}
                              className={styles.selectionActionButton}
                              type="button"
                              disabled={isPanning || selectedKnots.length === 0}
                              onClick={() => alignSelection(value)}
                              aria-label={label}
                              title={label}
                            >
                              <img src={ACTION_ICON_URLS.align} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                              <span className={styles.selectionActionLabel}>{shortLabel}</span>
                            </button>
                          ))
                        : (
                          <div className={`${styles.selectionActionButton} ${styles.selectionActionButtonWide}`}>
                            <button
                              className={styles.selectionActionMenuTrigger}
                              type="button"
                              disabled={isPanning || selectedKnots.length === 0}
                            >
                              <img src={ACTION_ICON_URLS.align} alt="" aria-hidden="true" className={styles.selectionActionIcon} />
                              <span className={styles.selectionActionLabel}>Align</span>
                              <span className={styles.selectionActionHint}>Select values</span>
                            </button>
                          </div>
                        )}
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
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className={`${styles.shapePanelFooter} ${styles.tourLabelAnchor}`}>
              {showTourLabels ? (
                <TourLabel
                  label="Undo / Save"
                  title="Commit or revisit edits"
                  description="Use these actions to step backward, restore a change, or save the edited model snapshot."
                  details={[
                    "Undo and redo operate on the recorded edit history.",
                    "Save writes the current edited state as a reusable model snapshot.",
                  ]}
                  placement="top-right"
                />
              ) : null}
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
