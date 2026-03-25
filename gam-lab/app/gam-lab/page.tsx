"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ShapeFunctionsPanel from "./components/ShapeFunctionsPanel";
import SidebarPanel from "./components/SidebarPanel";
import FeatureModePanel from "./components/FeatureModePanel";
import HelpCallout from "./components/HelpCallout";
import styles from "./page.module.css";
import { useGamLab } from "./hooks/useGamLab";
import { useSidebarActions } from "./hooks/useSidebarActions";
import { useToolSettings } from "./hooks/useToolSettings";
import { useHelpTour } from "./hooks/useHelpTour";
import { useAuditLogger } from "./hooks/useAuditLogger";
import { useUiAuditLogger } from "./hooks/useUiAuditLogger";

function GamLabPageContent() {
  const [showAdvancedTraining, setShowAdvancedTraining] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const search = searchParams.toString();
  const { logEvent } = useAuditLogger();
  useUiAuditLogger(logEvent);
  const rawModel = searchParams.get("model");
  const initialModel = rawModel && rawModel !== "undefined" ? rawModel : null;
  const trainMode = searchParams.get("train") === "1";
  const trainDataset = searchParams.get("dataset") ?? "bike_hourly";
  const trainModelType = (searchParams.get("model_type") ?? "igann") as "igann" | "igann_interactive";
  const trainCenterShapes = searchParams.get("center_shapes") === "true";
  const trainPoints = Number(searchParams.get("points") ?? "250");
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
    lockedFeatures,
    featureModes,
    setFeatureMode,
    handleSave,
    train,
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
    modelSource,
    currentVersion,
    trainData,
  } = useGamLab({
    initialModel,
    initialTrain: trainMode
      ? {
          dataset: trainDataset,
          model_type: trainModelType === "igann_interactive" ? "igann_interactive" : "igann",
          center_shapes: trainCenterShapes,
          points: Number.isFinite(trainPoints) ? trainPoints : 250,
          seed: Number.isFinite(trainSeed) ? trainSeed : 3,
          n_estimators: Number.isFinite(trainEstimators) ? trainEstimators : 100,
          boost_rate: Number.isFinite(trainBoostRate) ? trainBoostRate : 0.1,
          init_reg: Number.isFinite(trainInitReg) ? trainInitReg : 1,
          elm_alpha: Number.isFinite(trainElmAlpha) ? trainElmAlpha : 1,
          early_stopping: Number.isFinite(trainEarlyStopping) ? trainEarlyStopping : 50,
          scale_y: trainScaleY,
        }
      : null,
    auditLogger: logEvent,
  });

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

  useEffect(() => {
    if (!initialModel && !trainMode) {
      router.replace("/");
    }
  }, [initialModel, trainMode, router]);

  const helpTour = useHelpTour(setSidebarTab);

  const { formatHistoryAction, formatHistoryDetail } = useSidebarActions({ history });

  const toolSettings = useToolSettings();

  if (!initialModel && !trainMode) {
    return null;
  }

  return (
    <div className={styles.pageFrame}>
      <div className={styles.page}>
        {result ? (
          <section className={styles.dashboard}>
            {helpTour.showHelp ? <div className={styles.helpOverlay} aria-hidden="true" /> : null}
            <div className={styles.datasetBanner}>
              <div className={styles.bannerTitle}>
                <span className={styles.datasetLabel}>Dataset</span>
                <span className={styles.datasetTitle}>{selectedDataset.label}</span>
              </div>
              <div className={styles.bannerActions}>
                <button
                  type="button"
                  className={styles.helpButton}
                  onClick={helpTour.openHelp}
                >
                  Help
                </button>
                <Link className={styles.selectModelButton} href="/">
                  Choose another model
                </Link>
              </div>
            </div>
            <div className={styles.topPanels}>
            {modelSource === "train" ? (
              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div className={styles.trainingHeader}>
                    <div className={styles.trainingTitleBlock}>
                      <p className={styles.panelEyebrow}>Training</p>
                      <h2 className={styles.panelTitle}>Hyperparameters</h2>
                      <p className={styles.trainingIntro}>
                        Keep the defaults for a quick run. Open advanced settings only when you need to tune the
                        model.
                      </p>
                    </div>
                    <div className={styles.trainingActions}>
                      <button
                        type="button"
                        className={styles.panelButton}
                        onClick={() => {
                          train();
                        }}
                        disabled={status === "loading"}
                      >
                        {status === "loading" ? "Training..." : "Train model"}
                      </button>
                    </div>
                  </div>
                </div>

                <div className={styles.trainingPrimaryRow}>
                  <label className={styles.trainingField} htmlFor="gam-lab-dataset">
                    <span className={styles.trainingFieldLabel}>Dataset</span>
                    <select
                      id="gam-lab-dataset"
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
                  <label className={styles.trainingField} htmlFor="gam-lab-model">
                    <span className={styles.trainingFieldLabel}>Model</span>
                    <select
                      id="gam-lab-model"
                      className={styles.datasetSelect}
                      value={modelType}
                      onChange={(event) => setModelType(event.target.value as "igann" | "igann_interactive")}
                    >
                      <option value="igann">IGANN</option>
                      <option value="igann_interactive">IGANN interactive</option>
                    </select>
                  </label>
                </div>

                <div className={styles.trainingToggleRow}>
                  <label className={styles.trainingToggleCard}>
                    <input
                      className={styles.toggleInput}
                      type="checkbox"
                      checked={centerShapes}
                      disabled={modelType !== "igann_interactive"}
                      onChange={(event) => setCenterShapes(event.target.checked)}
                    />
                    <span className={styles.toggleTrack}>
                      <span className={styles.toggleThumb} />
                    </span>
                    <span className={styles.trainingToggleText}>
                      <span className={styles.trainingToggleLabel}>Center shapes</span>
                      <span className={styles.trainingToggleHint}>Enforce E[fj(Xj)] = 0</span>
                    </span>
                  </label>
                </div>

                <div className={styles.trainingFooter}>
                  <button
                    type="button"
                    className={styles.trainingAdvancedToggle}
                    aria-expanded={showAdvancedTraining}
                    onClick={() => setShowAdvancedTraining((current) => !current)}
                  >
                    {showAdvancedTraining ? "Hide advanced settings" : "Show advanced settings"}
                    <span
                      className={`${styles.trainingAdvancedChevron} ${
                        showAdvancedTraining ? styles.trainingAdvancedChevronOpen : ""
                      }`}
                    >
                      ▾
                    </span>
                  </button>
                  {!showAdvancedTraining ? (
                    <p className={styles.trainingCompactSummary}>
                      Defaults: {shapePoints} points, seed {seed}, {nEstimators} estimators.
                    </p>
                  ) : null}
                </div>

                {showAdvancedTraining ? (
                  <div className={styles.trainingAdvancedGrid}>
                    <label className={styles.trainingField} htmlFor="gam-lab-points">
                      <span className={styles.trainingFieldLabel}>Shape points</span>
                      <input
                        id="gam-lab-points"
                        className={styles.datasetSelect}
                        type="number"
                        step="1"
                        min="2"
                        max="250"
                        value={shapePoints}
                        onChange={(event) => setShapePoints(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-seed">
                      <span className={styles.trainingFieldLabel}>Seed</span>
                      <input
                        id="gam-lab-seed"
                        className={styles.datasetSelect}
                        type="number"
                        step="1"
                        min="0"
                        max="9999"
                        value={seed}
                        onChange={(event) => setSeed(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-estimators">
                      <span className={styles.trainingFieldLabel}>Estimators</span>
                      <input
                        id="gam-lab-estimators"
                        className={styles.datasetSelect}
                        type="number"
                        step="1"
                        min="10"
                        max="500"
                        value={nEstimators}
                        onChange={(event) => setNEstimators(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-boost-rate">
                      <span className={styles.trainingFieldLabel}>Boost rate</span>
                      <input
                        id="gam-lab-boost-rate"
                        className={styles.datasetSelect}
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="1"
                        value={boostRate}
                        onChange={(event) => setBoostRate(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-init-reg">
                      <span className={styles.trainingFieldLabel}>Init reg</span>
                      <input
                        id="gam-lab-init-reg"
                        className={styles.datasetSelect}
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="10"
                        value={initReg}
                        onChange={(event) => setInitReg(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-elm-alpha">
                      <span className={styles.trainingFieldLabel}>ELM alpha</span>
                      <input
                        id="gam-lab-elm-alpha"
                        className={styles.datasetSelect}
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="10"
                        value={elmAlpha}
                        onChange={(event) => setElmAlpha(Number(event.target.value))}
                      />
                    </label>
                    <label className={styles.trainingField} htmlFor="gam-lab-early-stopping">
                      <span className={styles.trainingFieldLabel}>Early stopping</span>
                      <input
                        id="gam-lab-early-stopping"
                        className={styles.datasetSelect}
                        type="number"
                        step="1"
                        min="5"
                        max="200"
                        value={earlyStopping}
                        onChange={(event) => setEarlyStopping(Number(event.target.value))}
                      />
                    </label>
                  </div>
                ) : null}

              </section>
            ) : null}
            {result?.model?.model_type === "igann_interactive" && trainData ? (
              <section className={styles.panel} style={{ flexShrink: 0 }}>
                <div className={styles.panelHeader}>
                  <div className={styles.trainingHeader}>
                    <div className={styles.trainingTitleBlock}>
                      <p className={styles.panelEyebrow}>Interactive</p>
                      <h2 className={styles.panelTitle}>Feature Modes</h2>
                    </div>
                    {modelSource === "train" ? (
                      <div className={styles.trainingActions}>
                        <button
                          type="button"
                          className={styles.panelButton}
                          onClick={manualRefitFromEdits}
                          disabled={status === "loading"}
                        >
                          {status === "loading" ? "Refitting..." : "Refit from edits"}
                        </button>
                      </div>
                    ) : null}
                  </div>
                </div>
                <FeatureModePanel
                  trainData={trainData}
                  shapes={currentVersion?.shapes ?? []}
                  featureModes={featureModes}
                  onSetFeatureMode={setFeatureMode}
                />
              </section>
            ) : null}
            </div>
            <div className={`${styles.tourTarget} ${styles.shapePanelSlot} ${helpTour.isStepActive("shapePanel") ? styles.tourTargetActive : ""}`}>
              <ShapeFunctionsPanel
                showTourLabels={helpTour.isStepActive("shapePanel")}
                shapes={currentVersion?.shapes ?? []}
                trainData={trainData!}
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
                toolSettings={toolSettings}
                hideLock={modelSource.startsWith("model:")}
              />
              <HelpCallout
                target="shapePanel"
                currentStep={helpTour.currentStep}
                stepIndex={helpTour.stepIndex}
                totalSteps={helpTour.totalSteps}
                isFirstStep={helpTour.isFirstStep}
                isLastStep={helpTour.isLastStep}
                isStepActive={helpTour.isStepActive}
                onClose={helpTour.closeHelp}
                onNext={helpTour.goNext}
                onPrev={helpTour.goPrev}
              />
            </div>
            {partial ? (
              <div className={`${styles.sidebarTourTarget} ${helpTour.isStepActive("sidebar") ? styles.tourTargetActive : ""}`}>
                <SidebarPanel
                  showTourLabels={helpTour.isStepActive("sidebar")}
                  sidebarTab={sidebarTab}
                  setSidebarTab={setSidebarTab}
                  stats={stats}
                  history={history}
                  formatHistoryAction={formatHistoryAction}
                  formatHistoryDetail={formatHistoryDetail}
                  onDeleteHistoryEntry={deleteHistoryEntry}
                  shapes={currentVersion?.shapes ?? []}
                  trainData={trainData!}
                  activeFeatureKey={partial?.key ?? null}
                  activeKnots={partial ? knots : null}
                  selectedPointIndices={selectedKnots}
                  activeFeatureCategories={partial?.categories ?? null}
                />
                <HelpCallout
                  target="sidebar"
                  currentStep={helpTour.currentStep}
                  stepIndex={helpTour.stepIndex}
                  totalSteps={helpTour.totalSteps}
                  isFirstStep={helpTour.isFirstStep}
                  isLastStep={helpTour.isLastStep}
                  isStepActive={helpTour.isStepActive}
                  onClose={helpTour.closeHelp}
                  onNext={helpTour.goNext}
                  onPrev={helpTour.goPrev}
                />
              </div>
            ) : null}
          </section>
        ) : (
          <div className={styles.placeholder}>Press &quot;Train model&quot; to load shapes.</div>
        )}
      </div>
    </div>
  );
}

export default function GamLabPage() {
  return (
    <Suspense fallback={null}>
      <GamLabPageContent />
    </Suspense>
  );
}
