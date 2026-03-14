"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import PredictionFitPlot from "../components/PredictionFitPlot";
import ShapeFunctionsPanel from "../components/ShapeFunctionsPanel";
import { type DragCurve, type SmoothingAlgorithm } from "../components/VisxShapeEditor";
import SidebarPanel from "../components/SidebarPanel";
import styles from "../page.module.css";
import trainStyles from "./train.module.css";
import { useGamLab } from "../hooks/useGamLab";
import { useSidebarActions } from "../hooks/useSidebarActions";
import { useAuditLogger } from "../hooks/useAuditLogger";
import { useUiAuditLogger } from "../hooks/useUiAuditLogger";

export default function TrainPage() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const search = searchParams.toString();
  const { logEvent } = useAuditLogger();
  useUiAuditLogger(logEvent);
  const {
    status,
    result,
    datasets,
    dataset,
    setDataset,
    modelType,
    setModelType,
    centerShapes,
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
    selectedDataset,
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
    stats,
    models,
    lockedFeatures,
    handleSave,
    manualTrain,
    manualRefitFromEdits,
    toggleFeatureLock,
    sidebarTab,
    setSidebarTab,
    partial,
    history,
    historyCursor,
    recordAction,
    commitEdits,
    undoLast,
    redoLast,
    deleteHistoryEntry,
  } = useGamLab({ auditLogger: logEvent });

  useEffect(() => {
    logEvent({
      category: "system",
      action: "page.view",
      detail: {
        pathname,
        search,
      },
    });
  }, [logEvent, pathname, search]);

  const { formatHistoryAction, formatHistoryDetail, applyMonotonic, addPointsInSelection } = useSidebarActions({
    partial,
    knotEdits,
    knots,
    selectedKnots,
    setKnots,
    setKnotEdits,
    setSelectedKnots,
    recordAction,
    commitEdits,
    history,
    baselineKnots,
  });

  const [activeContinuousTool, setActiveContinuousTool] = useState<"drag" | "smooth">("drag");
  const [dragFalloffRadius, setDragFalloffRadius] = useState(4);
  const [dragRangeBoost, setDragRangeBoost] = useState(1);
  const [dragCurve, setDragCurve] = useState<DragCurve>("gaussian");
  const [smoothAmount, setSmoothAmount] = useState(0.5);
  const [smoothingRangeMax, setSmoothingRangeMax] = useState(32);
  const [smoothingSpeed, setSmoothingSpeed] = useState(1);
  const [smoothingAlgorithm, setSmoothingAlgorithm] = useState<SmoothingAlgorithm>("gaussian");
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className={styles.pageFrame}>
      <div className={styles.page}>
        <section className={styles.dashboard}>
          {result ? (
            <div className={styles.datasetBanner}>
              <div>
                <p className={styles.datasetLabel}>Dataset</p>
                <p className={styles.datasetTitle}>{selectedDataset.label}</p>
                <p className={styles.datasetSummary}>{selectedDataset.summary}</p>
              </div>
              <Link className={styles.selectModelButton} href="/">
                Choose another model
              </Link>
            </div>
          ) : null}
          <section className={styles.panel}>
            <div className={trainStyles.trainHeader}>
              <div className={trainStyles.trainTitleBlock}>
                <p className={styles.panelEyebrow}>Training</p>
                <h2 className={styles.panelTitle}>Hyperparameters</h2>
              </div>
              <div className={styles.panelActions}>
                <button
                  type="button"
                  className={styles.panelButton}
                  onClick={() => manualRefitFromEdits()}
                  disabled={status === "loading" || !result || modelType !== "igann_interactive"}
                >
                  {status === "loading" ? "Refitting…" : "Refit from edits"}
                </button>
                <button
                  type="button"
                  className={trainStyles.trainButton}
                  onClick={() => manualTrain()}
                  disabled={status === "loading"}
                >
                  {status === "loading" ? "Training…" : "Train model"}
                </button>
              </div>
            </div>

            {/* Primary fields */}
            <div className={trainStyles.primaryRow}>
              <div className={trainStyles.field}>
                <label className={trainStyles.fieldLabel} htmlFor="train-dataset">Dataset</label>
                <select
                  id="train-dataset"
                  className={trainStyles.select}
                  value={dataset}
                  onChange={(event) => setDataset(event.target.value)}
                >
                  {datasets.map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
                </select>
              </div>
              <div className={trainStyles.field}>
                <label className={trainStyles.fieldLabel} htmlFor="train-model">Model</label>
                <select
                  id="train-model"
                  className={trainStyles.select}
                  value={modelType}
                  onChange={(event) => setModelType(event.target.value as "igann" | "igann_interactive")}
                >
                  <option value="igann">IGANN</option>
                  <option value="igann_interactive">IGANN interactive</option>
                </select>
              </div>
            </div>

            {/* Toggle options */}
            <div className={trainStyles.toggleRow}>
              <label className={trainStyles.toggleField}>
                <input
                  className={styles.toggleInput}
                  type="checkbox"
                  checked={centerShapes}
                  disabled={modelType !== "igann_interactive"}
                  onChange={(event) => setCenterShapes(event.target.checked)}
                />
                <span className={styles.toggleTrack}><span className={styles.toggleThumb} /></span>
                <span className={trainStyles.toggleFieldText}>
                  <span className={trainStyles.toggleFieldLabel}>Center shapes</span>
                  <span className={trainStyles.toggleFieldHint}>Enforce E[f<sub>j</sub>(X<sub>j</sub>)] = 0</span>
                </span>
              </label>
              <label className={trainStyles.toggleField}>
                <input
                  className={styles.toggleInput}
                  type="checkbox"
                  checked={scaleY}
                  onChange={(event) => setScaleY(event.target.checked)}
                />
                <span className={styles.toggleTrack}><span className={styles.toggleThumb} /></span>
                <span className={trainStyles.toggleFieldText}>
                  <span className={trainStyles.toggleFieldLabel}>Scale target</span>
                  <span className={trainStyles.toggleFieldHint}>Normalize y for training</span>
                </span>
              </label>
            </div>

            {/* Advanced settings toggle */}
            <button
              type="button"
              className={trainStyles.advancedToggle}
              onClick={() => setShowAdvanced((v) => !v)}
            >
              <span>{showAdvanced ? "Hide" : "Show"} advanced settings</span>
              <span className={trainStyles.advancedToggleChevron} style={{ transform: showAdvanced ? "rotate(180deg)" : undefined }}>▾</span>
            </button>

            {/* Advanced settings grid */}
            {showAdvanced && (
              <div className={trainStyles.advancedGrid}>
                {[
                  { id: "train-points", label: "Shape points", value: shapePoints, step: 1, min: 2, max: 250, set: setShapePoints },
                  { id: "train-seed", label: "Seed", value: seed, step: 1, min: 0, max: 9999, set: setSeed },
                  { id: "train-estimators", label: "Estimators", value: nEstimators, step: 1, min: 10, max: 500, set: setNEstimators },
                  { id: "train-boostrate", label: "Boost rate", value: boostRate, step: 0.01, min: 0.01, max: 1, set: setBoostRate },
                  { id: "train-initreg", label: "Init reg", value: initReg, step: 0.01, min: 0.01, max: 10, set: setInitReg },
                  { id: "train-elmalpha", label: "ELM alpha", value: elmAlpha, step: 0.01, min: 0.01, max: 10, set: setElmAlpha },
                  { id: "train-earlystop", label: "Early stopping", value: earlyStopping, step: 1, min: 5, max: 200, set: setEarlyStopping },
                ].map(({ id, label, value, step, min, max, set }) => (
                  <div key={id} className={trainStyles.field}>
                    <label className={trainStyles.fieldLabel} htmlFor={id}>{label}</label>
                    <input
                      id={id}
                      className={trainStyles.numberInput}
                      type="number"
                      step={step}
                      min={min}
                      max={max}
                      value={value}
                      onChange={(e) => set(Number(e.target.value))}
                    />
                  </div>
                ))}
              </div>
            )}
          </section>
          {result ? (
            <>
              <ShapeFunctionsPanel
                shapes={result.version.shapes}
                trainData={result.data}
                baselineKnots={baselineKnots}
                fixedLinesByFeature={fixedLinesByFeature}
                knots={knots}
                setKnots={setKnots}
                knotEdits={knotEdits}
                setKnotEdits={setKnotEdits}
                selectedKnots={selectedKnots}
                setSelectedKnots={setSelectedKnots}
                activePartialIdx={activePartialIdx}
                setActivePartialIdx={setActivePartialIdx}
                lockedFeatures={lockedFeatures}
                onToggleFeatureLock={toggleFeatureLock}
                onRecordAction={recordAction}
                onCommitEdits={commitEdits}
                onUndo={undoLast}
                canUndo={historyCursor > 0}
                onRedo={redoLast}
                canRedo={historyCursor < history.length}
                onSave={handleSave}
                applyMonotonic={applyMonotonic}
                addPointsInSelection={addPointsInSelection}
                activeContinuousTool={activeContinuousTool}
                setActiveContinuousTool={setActiveContinuousTool}
                dragFalloffRadius={dragFalloffRadius}
                setDragFalloffRadius={setDragFalloffRadius}
                dragRangeBoost={dragRangeBoost}
                setDragRangeBoost={setDragRangeBoost}
                dragCurve={dragCurve}
                setDragCurve={setDragCurve}
                smoothAmount={smoothAmount}
                setSmoothAmount={setSmoothAmount}
                smoothingRangeMax={smoothingRangeMax}
                setSmoothingRangeMax={setSmoothingRangeMax}
                smoothingSpeed={smoothingSpeed}
                setSmoothingSpeed={setSmoothingSpeed}
                smoothingAlgorithm={smoothingAlgorithm}
                setSmoothingAlgorithm={setSmoothingAlgorithm}
              />
              {models ? <PredictionFitPlot result={result} models={models} /> : null}
              {partial ? (
                <SidebarPanel
                  sidebarTab={sidebarTab}
                  setSidebarTab={setSidebarTab}
                  stats={stats}
                  history={history}
                  formatHistoryAction={formatHistoryAction}
                  formatHistoryDetail={formatHistoryDetail}
                  onDeleteHistoryEntry={deleteHistoryEntry}
                />
              ) : null}
            </>
          ) : (
            <div className={styles.placeholder}>Press &quot;Train model&quot; to load shapes.</div>
          )}
        </section>
      </div>
    </div>
  );
}
