"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import ShapeFunctionsPanel from "../components/ShapeFunctionsPanel";
import SidebarPanel from "../components/SidebarPanel";
import FeatureModePanel from "../components/FeatureModePanel";
import styles from "../page.module.css";
import trainStyles from "./train.module.css";
import { useGamLab } from "../hooks/useGamLab";
import { useSidebarActions } from "../hooks/useSidebarActions";
import { useToolSettings } from "../hooks/useToolSettings";
import { useAuditLogger } from "../hooks/useAuditLogger";
import { useUiAuditLogger } from "../hooks/useUiAuditLogger";
import { suggestInteractionOperations } from "../lib/interactionSuggestions";

type FeatureCatalogEntry = {
  key: string;
  label: string;
  kind: "continuous" | "categorical";
};

const FEATURE_CATALOG: Record<string, FeatureCatalogEntry[]> = {
  bike_hourly: [
    { key: "Time of Day", label: "Time of Day", kind: "categorical" },
    { key: "Windspeed", label: "Windspeed", kind: "continuous" },
    { key: "Temperature", label: "Temperature", kind: "continuous" },
    { key: "Humidity", label: "Humidity", kind: "continuous" },
    { key: "Weathersituation", label: "Weathersituation", kind: "categorical" },
    { key: "Type of Day", label: "Type of Day", kind: "categorical" },
  ],
  adult_income: [
    { key: "age", label: "age", kind: "continuous" },
    { key: "fnlwgt", label: "fnlwgt", kind: "continuous" },
    { key: "education-num", label: "education-num", kind: "continuous" },
    { key: "capital-gain", label: "capital-gain", kind: "continuous" },
    { key: "capital-loss", label: "capital-loss", kind: "continuous" },
    { key: "hours-per-week", label: "hours-per-week", kind: "continuous" },
    { key: "workclass", label: "workclass", kind: "categorical" },
    { key: "education", label: "education", kind: "categorical" },
    { key: "marital-status", label: "marital-status", kind: "categorical" },
    { key: "occupation", label: "occupation", kind: "categorical" },
    { key: "relationship", label: "relationship", kind: "categorical" },
    { key: "race", label: "race", kind: "categorical" },
    { key: "sex", label: "sex", kind: "categorical" },
    { key: "native-country", label: "native-country", kind: "categorical" },
  ],
  breast_cancer: [
    { key: "radius_mean", label: "radius_mean", kind: "continuous" },
    { key: "texture_mean", label: "texture_mean", kind: "continuous" },
    { key: "perimeter_mean", label: "perimeter_mean", kind: "continuous" },
    { key: "area_mean", label: "area_mean", kind: "continuous" },
    { key: "smoothness_mean", label: "smoothness_mean", kind: "continuous" },
    { key: "compactness_mean", label: "compactness_mean", kind: "continuous" },
    { key: "concavity_mean", label: "concavity_mean", kind: "continuous" },
    { key: "concave points_mean", label: "concave points_mean", kind: "continuous" },
    { key: "symmetry_mean", label: "symmetry_mean", kind: "continuous" },
    { key: "fractal_dimension_mean", label: "fractal_dimension_mean", kind: "continuous" },
    { key: "radius_se", label: "radius_se", kind: "continuous" },
    { key: "texture_se", label: "texture_se", kind: "continuous" },
    { key: "perimeter_se", label: "perimeter_se", kind: "continuous" },
    { key: "area_se", label: "area_se", kind: "continuous" },
    { key: "smoothness_se", label: "smoothness_se", kind: "continuous" },
    { key: "compactness_se", label: "compactness_se", kind: "continuous" },
    { key: "concavity_se", label: "concavity_se", kind: "continuous" },
    { key: "concave points_se", label: "concave points_se", kind: "continuous" },
    { key: "symmetry_se", label: "symmetry_se", kind: "continuous" },
    { key: "fractal_dimension_se", label: "fractal_dimension_se", kind: "continuous" },
    { key: "radius_worst", label: "radius_worst", kind: "continuous" },
    { key: "texture_worst", label: "texture_worst", kind: "continuous" },
    { key: "perimeter_worst", label: "perimeter_worst", kind: "continuous" },
    { key: "area_worst", label: "area_worst", kind: "continuous" },
    { key: "smoothness_worst", label: "smoothness_worst", kind: "continuous" },
    { key: "compactness_worst", label: "compactness_worst", kind: "continuous" },
    { key: "concavity_worst", label: "concavity_worst", kind: "continuous" },
    { key: "concave points_worst", label: "concave points_worst", kind: "continuous" },
    { key: "symmetry_worst", label: "symmetry_worst", kind: "continuous" },
    { key: "fractal_dimension_worst", label: "fractal_dimension_worst", kind: "continuous" },
  ],
};

export default function TrainPage() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const search = searchParams.toString();
  const { logEvent } = useAuditLogger();
  useUiAuditLogger(logEvent);
  const {
    status,
    result,
    models,
    datasets,
    dataset,
    setDataset,
    modelType,
    centerShapes,
    setCenterShapes,
    selectedFeatures,
    setSelectedFeatures,
    setSelectedInteractions,
    setSelectedOperations,
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
    metricWarning,
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
    currentVersion,
    trainData,
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

  const { formatHistoryAction, formatHistoryDetail } = useSidebarActions({ history });
  const toolSettings = useToolSettings();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showRefitSettings, setShowRefitSettings] = useState(false);
  const [activeTab, setActiveTab] = useState<"shapes" | "features">("shapes");

  const availableFeatures = FEATURE_CATALOG[dataset] ?? [];
  const featureKeys = availableFeatures.map((feature) => feature.key);

  useEffect(() => {
    setSelectedFeatures((prev) => {
      const validPrev = prev.filter((feature) => featureKeys.includes(feature));
      if (validPrev.length > 0) {
        return validPrev;
      }
      return featureKeys;
    });
  }, [featureKeys, setSelectedFeatures]);

  const featureSelection = (() => {
    const selected = new Set(selectedFeatures);
    return availableFeatures.reduce(
      (acc, feature) => {
        if (feature.kind === "categorical") {
          acc[1].push({ ...feature, checked: selected.has(feature.key) });
        } else {
          acc[0].push({ ...feature, checked: selected.has(feature.key) });
        }
        return acc;
      },
      [[], []] as [
        Array<FeatureCatalogEntry & { checked: boolean }>,
        Array<FeatureCatalogEntry & { checked: boolean }>,
      ],
    );
  })();

  const togglePretrainFeature = (featureKey: string, checked: boolean) => {
    setSelectedFeatures((prev) => {
      if (checked) {
        return prev.includes(featureKey) ? prev : [...prev, featureKey];
      }
      return prev.filter((key) => key !== featureKey);
    });
  };

  const applySuggestedOperations = () => {
    if (!models?.residuals || !trainData) return;
    const suggestions = suggestInteractionOperations({
      residuals: models.residuals,
      trainData,
      selectedFeatures,
    });
    setSelectedInteractions(suggestions.map((operation) => operation.sources.join("__")));
    setSelectedOperations(suggestions);
  };

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
              <div className={trainStyles.bannerRight}>
                <div className={trainStyles.tabBar} role="tablist">
                  <button
                    role="tab"
                    aria-selected={activeTab === "shapes"}
                    className={`${trainStyles.tabButton} ${activeTab === "shapes" ? trainStyles.tabButtonActive : ""}`}
                    onClick={() => setActiveTab("shapes")}
                  >
                    Shapes
                  </button>
                  <button
                    role="tab"
                    aria-selected={activeTab === "features"}
                    className={`${trainStyles.tabButton} ${activeTab === "features" ? trainStyles.tabButtonActive : ""}`}
                    onClick={() => setActiveTab("features")}
                  >
                    Features
                  </button>
                </div>
                <Link className={styles.selectModelButton} href="/">
                  Choose another model
                </Link>
              </div>
            </div>
          ) : null}

          <div className={styles.topPanels} style={result && activeTab === "shapes" ? { display: "none" } : undefined}>
            {!result ? (
              <section className={styles.panel}>
                <div className={trainStyles.stageHeader}>
                  <div className={trainStyles.stagePill}>Stage 1</div>
                  <div>
                    <p className={styles.panelEyebrow}>Setup</p>
                    <h2 className={styles.panelTitle}>Select features, then train</h2>
                    <p className={trainStyles.stageSummary}>
                      Choose the input features you want in the initial model. After training, you can still
                      deactivate features, edit shape functions, and run a final refinement.
                    </p>
                  </div>
                </div>

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
                </div>

                <div className={trainStyles.featureSelectionCard}>
                  <div className={trainStyles.featureSelectionHeader}>
                    <div>
                      <p className={trainStyles.fieldLabel}>Training features</p>
                      <p className={trainStyles.featureSelectionSummary}>
                        {selectedFeatures.length} of {featureKeys.length} selected
                      </p>
                    </div>
                    <div className={trainStyles.featureSelectionActions}>
                      <button
                        type="button"
                        className={trainStyles.selectionButton}
                        onClick={() => setSelectedFeatures(featureKeys)}
                      >
                        Select all
                      </button>
                      <button
                        type="button"
                        className={trainStyles.selectionButton}
                        onClick={() => setSelectedFeatures([])}
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  <div className={trainStyles.featureColumns}>
                    {(["Continuous", "Categorical"] as const).map((heading, columnIdx) => (
                      <div key={heading} className={trainStyles.featureColumn}>
                        <p className={trainStyles.featureColumnTitle}>{heading}</p>
                        <div className={trainStyles.featureChecklist}>
                          {featureSelection[columnIdx].map((feature) => (
                            <label key={feature.key} className={trainStyles.featureOption}>
                              <input
                                type="checkbox"
                                checked={feature.checked}
                                onChange={(event) => togglePretrainFeature(feature.key, event.target.checked)}
                              />
                              <span>{feature.label}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

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
                </div>

                <button
                  type="button"
                  className={trainStyles.advancedToggle}
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  <span>{showAdvanced ? "Hide" : "Show"} advanced settings</span>
                  <span className={trainStyles.advancedToggleChevron} style={{ transform: showAdvanced ? "rotate(180deg)" : undefined }}>▾</span>
                </button>

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

                <div className={trainStyles.trainActionRow}>
                  <button
                    type="button"
                    className={trainStyles.trainButton}
                    onClick={() => {
                      setSelectedInteractions([]);
                      setSelectedOperations([]);
                      train({
                        selected_features: selectedFeatures,
                        selected_interactions: [],
                        selected_operations: [],
                      });
                    }}
                    disabled={status === "loading" || selectedFeatures.length === 0}
                  >
                    {status === "loading" ? "Training…" : "Train selected features"}
                  </button>
                  <p className={trainStyles.actionHint}>
                    The initial model is trained only on the selected features. Interaction suggestions remain
                    available later during refinement.
                  </p>
                </div>
              </section>
            ) : (
              <section className={`${styles.panel} ${trainStyles.featurePanel}`}>
                <div className={trainStyles.stageHeader}>
                  <div className={trainStyles.stagePill}>Stage 2</div>
                  <div>
                    <p className={styles.panelEyebrow}>Interactive</p>
                    <h2 className={styles.panelTitle}>Edit and deactivate</h2>
                    <p className={trainStyles.stageSummary}>
                      Change feature modes or edit shape functions. When the current version looks right, run the
                      final refinement to fit the model around those edits.
                    </p>
                  </div>
                </div>

                {trainData && (
                  <FeatureModePanel
                    trainData={trainData}
                    shapes={currentVersion?.shapes ?? []}
                    featureModes={featureModes}
                    onSetFeatureMode={setFeatureMode}
                  />
                )}

                <button
                  type="button"
                  className={trainStyles.advancedToggle}
                  onClick={() => setShowRefitSettings((v) => !v)}
                >
                  <span>{showRefitSettings ? "Hide" : "Show"} refinement settings</span>
                  <span className={trainStyles.advancedToggleChevron} style={{ transform: showRefitSettings ? "rotate(180deg)" : undefined }}>▾</span>
                </button>

                {showRefitSettings && (
                  <div className={trainStyles.advancedGrid}>
                    {[
                      { id: "refit-points", label: "Shape points", value: shapePoints, step: 1, min: 2, max: 250, set: setShapePoints },
                      { id: "refit-seed", label: "Seed", value: seed, step: 1, min: 0, max: 9999, set: setSeed },
                      { id: "refit-estimators", label: "Estimators", value: nEstimators, step: 1, min: 10, max: 500, set: setNEstimators },
                      { id: "refit-boostrate", label: "Boost rate", value: boostRate, step: 0.01, min: 0.01, max: 1, set: setBoostRate },
                      { id: "refit-initreg", label: "Init reg", value: initReg, step: 0.01, min: 0.01, max: 10, set: setInitReg },
                      { id: "refit-elmalpha", label: "ELM alpha", value: elmAlpha, step: 0.01, min: 0.01, max: 10, set: setElmAlpha },
                      { id: "refit-earlystop", label: "Early stopping", value: earlyStopping, step: 1, min: 5, max: 200, set: setEarlyStopping },
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

                <div className={trainStyles.finalStageCard}>
                  <div>
                    <div className={trainStyles.stagePill}>Stage 3</div>
                    <h3 className={trainStyles.finalStageTitle}>Final refinement</h3>
                    <p className={trainStyles.stageSummary}>
                      Refit the model using your edited shapes and current feature deactivations.
                    </p>
                  </div>
                  <div className={trainStyles.finalStageActions}>
                    <button
                      type="button"
                      className={trainStyles.selectionButton}
                      onClick={applySuggestedOperations}
                      disabled={!trainData || !models?.residuals?.length}
                    >
                      Suggest interactions
                    </button>
                    <button
                      type="button"
                      className={trainStyles.trainButton}
                      onClick={() => manualRefitFromEdits()}
                      disabled={status === "loading"}
                    >
                      {status === "loading" ? "Refining…" : "Run final refinement"}
                    </button>
                  </div>
                </div>
              </section>
            )}
          </div>

          {result && activeTab === "shapes" ? (
            <div className={styles.shapePanelSlot}>
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
                toolSettings={toolSettings}
              />
              {partial ? (
                <SidebarPanel
                  sidebarTab={sidebarTab}
                  setSidebarTab={setSidebarTab}
                  metricWarning={metricWarning}
                  history={history}
                  formatHistoryAction={formatHistoryAction}
                  formatHistoryDetail={formatHistoryDetail}
                  onDeleteHistoryEntry={deleteHistoryEntry}
                  shapes={currentVersion?.shapes ?? []}
                  trainData={trainData!}
                  activeFeatureKey={partial.key}
                  activeKnots={knots}
                  selectedPointIndices={selectedKnots}
                  activeFeatureCategories={partial.categories ?? null}
                />
              ) : null}
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
