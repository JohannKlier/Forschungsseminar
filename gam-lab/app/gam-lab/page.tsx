"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import ShapeFunctionsPanel from "./components/ShapeFunctionsPanel";
import SidebarPanel from "./components/SidebarPanel";
import styles from "./page.module.css";
import { useGamLab } from "./hooks/useGamLab";
import { useSidebarActions } from "./hooks/useSidebarActions";

const helpSteps = [
  {
    eyebrow: "Step 1",
    target: "shapePanel",
    placementClass: "helpCalloutPanelRight",
    focus: "shape-overview",
    title: "Shape Functions Panel",
    text: "This is the main workspace for changing the model. Each shape shows how one feature contributes to the prediction.",
    bullets: [
      "Select a feature and inspect its current shape.",
      "Use this panel to decide which relationships look plausible or implausible.",
      "Most of your work in the study happens here.",
    ],
  },
  {
    eyebrow: "Step 2",
    target: "shapePanel",
    placementClass: "helpCalloutPanelRight",
    focus: "shape-plot",
    title: "The Plot",
    text: "The plot shows the current effect of the selected feature. This is the curve or bar pattern you will inspect and revise.",
    bullets: [
      "For continuous features, the line shows how the prediction changes across the feature range.",
      "For categorical features, bars show the contribution of each category.",
      "Look for shapes that conflict with your expectations about the domain.",
    ],
  },
  {
    eyebrow: "Step 3",
    target: "shapePanel",
    placementClass: "helpCalloutPanelRight",
    focus: "shape-actions",
    title: "Actions And Modes",
    text: "The controls around the plot let you move between features and edit their behavior.",
    bullets: [
      "Use the feature selector and arrows to move through the model one feature at a time.",
      "Use the action buttons to edit points, apply constraints, and save your work.",
      "Undo and redo help you compare alternatives without losing progress.",
    ],
  },
  {
    eyebrow: "Step 4",
    target: "shapePanel",
    placementClass: "helpCalloutPanelRight",
    focus: "shape-views",
    title: "Single And Grid View",
    text: "The panel supports both focused inspection and broad comparison.",
    bullets: [
      "Single view lets you work on one feature in detail.",
      "Grid view helps you scan many shape functions quickly.",
      "Use grid view to spot suspicious features, then switch back to single view to edit them.",
    ],
  },
  {
    eyebrow: "Step 5",
    target: "sidebar",
    placementClass: "helpCalloutSidebarLeft",
    sidebarTab: "edit",
    focus: "sidebar-edit",
    title: "Sidebar Edit View",
    text: "The edit view summarizes the currently selected feature and gives you supporting context for your decisions.",
    bullets: [
      "Review statistics for the selected feature.",
      "Use this view when you want to understand the feature before changing it.",
      "This area complements the plot rather than replacing it.",
    ],
  },
  {
    eyebrow: "Step 6",
    target: "sidebar",
    placementClass: "helpCalloutSidebarLeft",
    sidebarTab: "history",
    focus: "sidebar-history",
    title: "Sidebar History View",
    text: "The history view keeps track of edits you already made, so you can review and compare your changes.",
    bullets: [
      "Use it to see which actions were applied to a feature.",
      "History helps you reason about whether your revisions improved the model.",
      "Switch between edit and history while iterating on the same feature.",
    ],
  },
] as const;

function GamLabPageContent() {
  const [showAdvancedTraining, setShowAdvancedTraining] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [helpStepIndex, setHelpStepIndex] = useState(0);
  const router = useRouter();
  const searchParams = useSearchParams();
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
    notifyInteractionStart,
    notifyInteractionEnd,
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
  });

  useEffect(() => {
    if (!initialModel && !trainMode) {
      router.replace("/");
    }
  }, [initialModel, trainMode, router]);

  useEffect(() => {
    if (!showHelp) return;

    const previousBodyOverflow = document.body.style.overflow;
    const previousHtmlOverflow = document.documentElement.style.overflow;
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowHelp(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousBodyOverflow;
      document.documentElement.style.overflow = previousHtmlOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [showHelp]);

  const currentHelpStep = helpSteps[helpStepIndex];
  const isFirstHelpStep = helpStepIndex === 0;
  const isLastHelpStep = helpStepIndex === helpSteps.length - 1;

  useEffect(() => {
    if (!showHelp) return;
    if (currentHelpStep.target !== "sidebar") return;
    if (!("sidebarTab" in currentHelpStep) || !currentHelpStep.sidebarTab) return;
    setSidebarTab(currentHelpStep.sidebarTab);
  }, [showHelp, currentHelpStep, setSidebarTab]);

  const renderHelpCallout = (target: (typeof helpSteps)[number]["target"]) => {
    if (!showHelp || currentHelpStep.target !== target) return null;

    return (
      <div
        className={`${styles.helpCallout} ${styles[currentHelpStep.placementClass]}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dashboard-help-title"
      >
        <div className={styles.helpHeader}>
          <div>
            <p className={styles.helpEyebrow}>{currentHelpStep.eyebrow}</p>
            <h2 id="dashboard-help-title" className={styles.helpTitle}>
              {currentHelpStep.title}
            </h2>
          </div>
          <button
            type="button"
            className={styles.helpCloseButton}
            onClick={() => setShowHelp(false)}
            aria-label="Close help"
          >
            Close
          </button>
        </div>
        <div className={styles.helpBody}>
          <div className={styles.helpProgressRow}>
            <p className={styles.helpProgressLabel}>
              Page {helpStepIndex + 1} of {helpSteps.length}
            </p>
            <div className={styles.helpProgressDots} aria-hidden="true">
              {helpSteps.map((step, index) => (
                <span
                  key={step.title}
                  className={`${styles.helpProgressDot} ${index === helpStepIndex ? styles.helpProgressDotActive : ""}`}
                />
              ))}
            </div>
          </div>
          <section className={styles.helpSpotlight}>
            <p className={styles.helpLead}>{currentHelpStep.text}</p>
            <ul className={styles.helpList}>
              {currentHelpStep.bullets.map((bullet) => (
                <li key={bullet} className={styles.helpListItem}>
                  {bullet}
                </li>
              ))}
            </ul>
          </section>
        </div>
        <div className={styles.helpFooter}>
          <button
            type="button"
            className={styles.helpNavButton}
            onClick={() => setHelpStepIndex((current) => Math.max(0, current - 1))}
            disabled={isFirstHelpStep}
          >
            Previous
          </button>
          <button
            type="button"
            className={styles.helpNavButtonPrimary}
            onClick={() => {
              if (isLastHelpStep) {
                setShowHelp(false);
                return;
              }
              setHelpStepIndex((current) => Math.min(helpSteps.length - 1, current + 1));
            }}
          >
            {isLastHelpStep ? "Start editing" : "Next"}
          </button>
        </div>
      </div>
    );
  };

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

  const smoothAmount = 0.5;
  const [smoothingMode, setSmoothingMode] = useState(false);
  const smoothingRangeMax = 32;

  if (!initialModel && !trainMode) {
    return null;
  }

  return (
    <div className={styles.pageFrame}>
      <div className={styles.page}>
        {result ? (
          <section
            className={`${styles.dashboard} ${styles.tourTarget} ${
              showHelp && currentHelpStep.target === "dashboard" ? styles.tourTargetActive : ""
            }`}
          >
            {showHelp ? <div className={styles.helpOverlay} aria-hidden="true" /> : null}
            <div className={styles.datasetBanner}>
              <div>
                <p className={styles.datasetLabel}>Dataset</p>
                <p className={styles.datasetTitle}>{selectedDataset.label}</p>
                <p className={styles.datasetSummary}>{selectedDataset.summary}</p>
              </div>
              <div className={styles.bannerActions}>
                <button
                  type="button"
                  className={styles.helpButton}
                  onClick={() => {
                    setHelpStepIndex(0);
                    setShowHelp(true);
                  }}
                >
                  Help
                </button>
                <Link className={styles.selectModelButton} href="/">
                  Choose another model
                </Link>
              </div>
            </div>
            {renderHelpCallout("dashboard")}
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
                          manualRefitFromEdits();
                        }}
                        disabled={status === "loading" || !result || modelType !== "igann_interactive"}
                      >
                        {status === "loading" ? "Refitting..." : "Refit from edits"}
                      </button>
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
                  <label className={styles.trainingToggleCard}>
                    <input
                      className={styles.toggleInput}
                      type="checkbox"
                      checked={scaleY}
                      onChange={(event) => setScaleY(event.target.checked)}
                    />
                    <span className={styles.toggleTrack}>
                      <span className={styles.toggleThumb} />
                    </span>
                    <span className={styles.trainingToggleText}>
                      <span className={styles.trainingToggleLabel}>Scale target</span>
                      <span className={styles.trainingToggleHint}>Normalize y for training</span>
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
            <div className={`${styles.tourTarget} ${showHelp && currentHelpStep.target === "shapePanel" ? styles.tourTargetActive : ""}`}>
              <ShapeFunctionsPanel
                activeTourFocus={showHelp && currentHelpStep.target === "shapePanel" ? currentHelpStep.focus : null}
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
                onInteractionStart={notifyInteractionStart}
                onInteractionEnd={notifyInteractionEnd}
                applyMonotonic={applyMonotonic}
                addPointsInSelection={addPointsInSelection}
                smoothAmount={smoothAmount}
                smoothingMode={smoothingMode}
                setSmoothingMode={setSmoothingMode}
                smoothingRangeMax={smoothingRangeMax}
              />
              {renderHelpCallout("shapePanel")}
            </div>
            {partial ? (
              <div className={`${styles.sidebarTourTarget} ${showHelp && currentHelpStep.target === "sidebar" ? styles.tourTargetActive : ""}`}>
                <SidebarPanel
                  activeTourFocus={showHelp && currentHelpStep.target === "sidebar" ? currentHelpStep.focus : null}
                  sidebarTab={sidebarTab}
                  setSidebarTab={setSidebarTab}
                  stats={stats}
                  history={history}
                  formatHistoryAction={formatHistoryAction}
                  formatHistoryDetail={formatHistoryDetail}
                  onDeleteHistoryEntry={deleteHistoryEntry}
                />
                {renderHelpCallout("sidebar")}
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
