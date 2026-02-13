"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import PredictionFitPlot from "./components/PredictionFitPlot";
import ShapeFunctionsPanel from "./components/ShapeFunctionsPanel";
import SidebarPanel from "./components/SidebarPanel";
import styles from "./page.module.css";
import { useGamLab } from "./hooks/useGamLab";
import { useSidebarActions } from "./hooks/useSidebarActions";

export default function GamLabPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawModel = searchParams.get("model");
  const initialModel = rawModel && rawModel !== "undefined" ? rawModel : null;
  const trainMode = searchParams.get("train") === "1";
  const trainDataset = searchParams.get("dataset") ?? "bike_hourly";
  const trainPoints = Number(searchParams.get("points") ?? "10");
  const trainSeed = Number(searchParams.get("seed") ?? "3");
  const trainEstimators = Number(searchParams.get("n_estimators") ?? "100");
  const trainBoostRate = Number(searchParams.get("boost_rate") ?? "0.1");
  const trainInitReg = Number(searchParams.get("init_reg") ?? "1");
  const trainElmAlpha = Number(searchParams.get("elm_alpha") ?? "1");
  const trainEarlyStopping = Number(searchParams.get("early_stopping") ?? "50");
  const trainScaleY = searchParams.get("scale_y") !== "false";

  const {
    status,
    result,
    datasets,
    dataset,
    setDataset,
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
    handleSave,
    manualTrain,
    sidebarTab,
    setSidebarTab,
    partial,
    displayLabel,
    history,
    historyCursor,
    recordAction,
    commitEdits,
    undoLast,
    redoLast,
    deleteHistoryEntry,
    modelSource,
  } = useGamLab({
    initialModel,
    initialTrain: trainMode
      ? {
          dataset: trainDataset,
          points: Number.isFinite(trainPoints) ? trainPoints : 10,
          seed: Number.isFinite(trainSeed) ? trainSeed : 3,
          n_estimators: Number.isFinite(trainEstimators) ? trainEstimators : 100,
          boost_rate: Number.isFinite(trainBoostRate) ? trainBoostRate : 0.1,
          init_reg: Number.isFinite(trainInitReg) ? trainInitReg : 1,
          elm_alpha: Number.isFinite(trainElmAlpha) ? trainElmAlpha : 1,
          early_stopping: Number.isFinite(trainEarlyStopping) ? trainEarlyStopping : 50,
          scale_y: trainScaleY,
        }
      : null,
  });

  useEffect(() => {
    if (!initialModel && !trainMode) {
      router.replace("/");
    }
  }, [initialModel, trainMode, router]);

  if (!initialModel && !trainMode) {
    return null;
  }

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

  const [smoothAmount, setSmoothAmount] = useState(0.7);
  const [smoothingMode, setSmoothingMode] = useState(false);
  const [smoothingRangeMax, setSmoothingRangeMax] = useState(6);
  const [smoothingNeighbors, setSmoothingNeighbors] = useState(4);
  const [smoothingRate, setSmoothingRate] = useState(0.4);
  const [smoothingStepPerSec, setSmoothingStepPerSec] = useState(0.3);

  return (
    <div className={styles.pageFrame}>
      <div className={styles.page}>
        {result ? (
          <section className={styles.dashboard}>
            <div className={styles.datasetBanner}>
              <div>
                <p className={styles.datasetLabel}>Dataset</p>
                <p className={styles.datasetTitle}>{selectedDataset.label}</p>
                <p className={styles.datasetSummary}>{selectedDataset.summary}</p>
              </div>
              <a className={styles.selectModelButton} href="/">
                Choose another model
              </a>
            </div>
            {modelSource === "train" ? (
              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div className={styles.panelTitleRow}>
                    <div>
                      <p className={styles.panelEyebrow}>Training</p>
                      <h2 className={styles.panelTitle}>Hyperparameters</h2>
                    </div>
                    <div className={styles.panelActions}>
                      <button
                        type="button"
                        className={styles.panelButton}
                        onClick={() => {
                          manualTrain();
                        }}
                        disabled={status === "loading"}
                      >
                        {status === "loading" ? "Training..." : "Train model"}
                      </button>
                    </div>
                  </div>
                </div>
                <div className={styles.controlGrid}>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Dataset</span>
                    <select
                      className={styles.datasetSelect}
                      value={dataset}
                      onChange={(event) => setDataset(event.target.value)}
                    >
                      {datasets.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Points</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="1"
                      min="2"
                      max="200"
                      value={shapePoints}
                      onChange={(event) => setShapePoints(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Seed</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="1"
                      min="0"
                      max="9999"
                      value={seed}
                      onChange={(event) => setSeed(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Estimators</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="1"
                      min="10"
                      max="500"
                      value={nEstimators}
                      onChange={(event) => setNEstimators(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Boost rate</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="0.01"
                      min="0.01"
                      max="1"
                      value={boostRate}
                      onChange={(event) => setBoostRate(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Init reg</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="0.01"
                      min="0.01"
                      max="10"
                      value={initReg}
                      onChange={(event) => setInitReg(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>ELM alpha</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="0.01"
                      min="0.01"
                      max="10"
                      value={elmAlpha}
                      onChange={(event) => setElmAlpha(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Early stopping</span>
                    <input
                      className={styles.datasetSelect}
                      type="number"
                      step="1"
                      min="5"
                      max="200"
                      value={earlyStopping}
                      onChange={(event) => setEarlyStopping(Number(event.target.value))}
                    />
                  </label>
                  <label className={styles.sliderLabel}>
                    <span className={styles.controlLabel}>Scale target</span>
                    <label className={styles.toggleLabel}>
                      <input
                        className={styles.toggleInput}
                        type="checkbox"
                        checked={scaleY}
                        onChange={(event) => setScaleY(event.target.checked)}
                      />
                      <span className={styles.toggleTrack}>
                        <span className={styles.toggleThumb} />
                      </span>
                      <span className={styles.toggleText}>Normalize y for training</span>
                    </label>
                  </label>
                </div>
              </section>
            ) : null}
            <ShapeFunctionsPanel
              result={result}
              baselineKnots={baselineKnots}
              knots={knots}
              setKnots={setKnots}
              knotEdits={knotEdits}
              setKnotEdits={setKnotEdits}
              selectedKnots={selectedKnots}
              setSelectedKnots={setSelectedKnots}
              activePartialIdx={activePartialIdx}
              setActivePartialIdx={setActivePartialIdx}
              onRecordAction={recordAction}
              onCommitEdits={commitEdits}
              applyMonotonic={applyMonotonic}
              addPointsInSelection={addPointsInSelection}
              smoothAmount={smoothAmount}
              setSmoothAmount={setSmoothAmount}
              smoothingMode={smoothingMode}
              setSmoothingMode={setSmoothingMode}
              smoothingRangeMax={smoothingRangeMax}
              setSmoothingRangeMax={setSmoothingRangeMax}
              smoothingNeighbors={smoothingNeighbors}
              setSmoothingNeighbors={setSmoothingNeighbors}
              smoothingRate={smoothingRate}
              setSmoothingRate={setSmoothingRate}
              smoothingStepPerSec={smoothingStepPerSec}
              setSmoothingStepPerSec={setSmoothingStepPerSec}
            />
            {models ? <PredictionFitPlot result={result} models={models} /> : null}
            {partial ? (
              <SidebarPanel
                sidebarTab={sidebarTab}
                setSidebarTab={setSidebarTab}
                displayLabel={displayLabel}
                partial={partial}
                onUndo={undoLast}
                canUndo={historyCursor > 0}
                onRedo={redoLast}
                canRedo={historyCursor < history.length}
                stats={stats}
                history={history}
                formatHistoryAction={formatHistoryAction}
                formatHistoryDetail={formatHistoryDetail}
                onDeleteHistoryEntry={deleteHistoryEntry}
                onSave={handleSave}
              />
            ) : null}
          </section>
        ) : (
          <div className={styles.placeholder}>Press "Train model" to load shapes.</div>
        )}
      </div>
    </div>
  );
}
