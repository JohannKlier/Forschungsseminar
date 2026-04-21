"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import ShapeFunctionsPanel from "../components/ShapeFunctionsPanel";
import SidebarPanel from "../components/SidebarPanel";
import FeatureMiniHistogram from "../components/FeatureMiniHistogram";
import styles from "../page.module.css";
import trainStyles from "./train.module.css";
import { useGamLab } from "../hooks/useGamLab";
import { useSidebarActions } from "../hooks/useSidebarActions";
import { useToolSettings } from "../hooks/useToolSettings";
import { useAuditLogger } from "../hooks/useAuditLogger";
import { useUiAuditLogger } from "../hooks/useUiAuditLogger";

type FeatureCatalogEntry = {
  key: string;
  label: string;
  kind: "continuous" | "categorical";
};

type FeatureSummary =
  | {
      key: string;
      label: string;
      kind: "continuous";
      bins: number[];
      min: number | null;
      max: number | null;
    }
  | {
      key: string;
      label: string;
      kind: "categorical";
      categories: { label: string; count: number }[];
    };

const EMPTY_FEATURES: FeatureCatalogEntry[] = [];

const FEATURE_DESCRIPTIONS: Record<string, Record<string, string>> = {
  bike_hourly: {
    "Time of Day": "Hour of the day when rentals were counted.",
    "Windspeed": "Normalized wind speed converted to an estimated km/h scale.",
    "Temperature": "Air temperature converted to an estimated Celsius scale.",
    "Humidity": "Relative humidity on a 0 to 100 scale.",
    "Weathersituation": "Observed weather condition, from clear to rain.",
    "Type of Day": "Whether the observation falls on a working day, weekend, or holiday.",
  },
  mimic4_mean_100_full: {
    "LOS": "Length of ICU stay in days.",
    "Age": "Patient age at ICU admission.",
    "Weight+100%mean": "Mean body weight during ICU stay (kg).",
    "Height+100%mean": "Mean height during ICU stay (cm).",
    "Bmi+100%mean": "Mean body mass index during stay.",
    "Temp+100%mean": "Mean body temperature (°C).",
    "RR+100%mean": "Mean respiratory rate (breaths/min).",
    "HR+100%mean": "Mean heart rate (beats/min).",
    "GLU+100%mean": "Mean blood glucose level (mg/dL).",
    "SBP+100%mean": "Mean systolic blood pressure (mmHg).",
    "DBP+100%mean": "Mean diastolic blood pressure (mmHg).",
    "MBP+100%mean": "Mean mean arterial pressure (mmHg).",
    "Ph+100%mean": "Mean arterial blood pH.",
    "GCST+100%mean": "Mean Glasgow Coma Scale total score (3–15); lower = more impaired.",
    "PaO2+100%mean": "Mean partial pressure of arterial oxygen (mmHg).",
    "Kreatinin+100%mean": "Mean serum creatinine (mg/dL) — kidney function marker.",
    "FiO2+100%mean": "Mean fraction of inspired oxygen (0–1).",
    "Kalium+100%mean": "Mean serum potassium (mEq/L).",
    "Natrium+100%mean": "Mean serum sodium (mEq/L).",
    "Leukocyten+100%mean": "Mean white blood cell count (10³/μL).",
    "Thrombocyten+100%mean": "Mean platelet count (10³/μL).",
    "Bilirubin+100%mean": "Mean total bilirubin (mg/dL) — liver function marker.",
    "HCO3+100%mean": "Mean serum bicarbonate (mEq/L) — acid-base balance.",
    "Hb+100%mean": "Mean hemoglobin concentration (g/dL).",
    "Quick+100%mean": "Mean Quick / prothrombin time (%) — coagulation marker.",
    "ALAT+100%mean": "Mean alanine aminotransferase (U/L) — liver enzyme.",
    "ASAT+100%mean": "Mean aspartate aminotransferase (U/L) — liver enzyme.",
    "PaCO2+100%mean": "Mean partial pressure of arterial CO₂ (mmHg).",
    "Albumin+100%mean": "Mean serum albumin (g/dL) — nutritional and hepatic marker.",
    "AnionGAP+100%mean": "Mean anion gap (mEq/L) — metabolic acidosis indicator.",
    "Lactate+100%mean": "Mean blood lactate (mmol/L) — tissue perfusion marker.",
    "Urea+100%mean": "Mean blood urea nitrogen (mg/dL).",
    "Eth": "Patient ethnicity.",
    "Sex": "Patient sex.",
  },
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
  mimic4_mean_100_full: [
    { key: "LOS", label: "LOS", kind: "continuous" },
    { key: "Age", label: "Age", kind: "continuous" },
    { key: "Weight+100%mean", label: "Weight+100%mean", kind: "continuous" },
    { key: "Height+100%mean", label: "Height+100%mean", kind: "continuous" },
    { key: "Bmi+100%mean", label: "Bmi+100%mean", kind: "continuous" },
    { key: "Temp+100%mean", label: "Temp+100%mean", kind: "continuous" },
    { key: "RR+100%mean", label: "RR+100%mean", kind: "continuous" },
    { key: "HR+100%mean", label: "HR+100%mean", kind: "continuous" },
    { key: "GLU+100%mean", label: "GLU+100%mean", kind: "continuous" },
    { key: "SBP+100%mean", label: "SBP+100%mean", kind: "continuous" },
    { key: "DBP+100%mean", label: "DBP+100%mean", kind: "continuous" },
    { key: "MBP+100%mean", label: "MBP+100%mean", kind: "continuous" },
    { key: "Ph+100%mean", label: "Ph+100%mean", kind: "continuous" },
    { key: "GCST+100%mean", label: "GCST+100%mean", kind: "continuous" },
    { key: "PaO2+100%mean", label: "PaO2+100%mean", kind: "continuous" },
    { key: "Kreatinin+100%mean", label: "Kreatinin+100%mean", kind: "continuous" },
    { key: "FiO2+100%mean", label: "FiO2+100%mean", kind: "continuous" },
    { key: "Kalium+100%mean", label: "Kalium+100%mean", kind: "continuous" },
    { key: "Natrium+100%mean", label: "Natrium+100%mean", kind: "continuous" },
    { key: "Leukocyten+100%mean", label: "Leukocyten+100%mean", kind: "continuous" },
    { key: "Thrombocyten+100%mean", label: "Thrombocyten+100%mean", kind: "continuous" },
    { key: "Bilirubin+100%mean", label: "Bilirubin+100%mean", kind: "continuous" },
    { key: "HCO3+100%mean", label: "HCO3+100%mean", kind: "continuous" },
    { key: "Hb+100%mean", label: "Hb+100%mean", kind: "continuous" },
    { key: "Quick+100%mean", label: "Quick+100%mean", kind: "continuous" },
    { key: "ALAT+100%mean", label: "ALAT+100%mean", kind: "continuous" },
    { key: "ASAT+100%mean", label: "ASAT+100%mean", kind: "continuous" },
    { key: "PaCO2+100%mean", label: "PaCO2+100%mean", kind: "continuous" },
    { key: "Albumin+100%mean", label: "Albumin+100%mean", kind: "continuous" },
    { key: "AnionGAP+100%mean", label: "AnionGAP+100%mean", kind: "continuous" },
    { key: "Lactate+100%mean", label: "Lactate+100%mean", kind: "continuous" },
    { key: "Urea+100%mean", label: "Urea+100%mean", kind: "continuous" },
    { key: "Eth", label: "Eth", kind: "categorical" },
    { key: "Sex", label: "Sex", kind: "categorical" },
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
    handleSave,
    train,
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
  const [featureSummaryState, setFeatureSummaryState] = useState<{
    dataset: string;
    summaries: Record<string, FeatureSummary>;
  } | null>(null);

  const availableFeatures = FEATURE_CATALOG[dataset] ?? EMPTY_FEATURES;
  const featureKeys = useMemo(() => availableFeatures.map((feature) => feature.key), [availableFeatures]);

  useEffect(() => {
    let cancelled = false;

    const loadFeatureSummaries = async () => {
      try {
        const response = await fetch(`/api/datasets/${encodeURIComponent(dataset)}/features?seed=${encodeURIComponent(String(seed))}`);
        if (!response.ok) return;
        const payload = (await response.json()) as { features?: FeatureSummary[] };
        if (cancelled) return;
        const summaries = Object.fromEntries((payload.features ?? []).map((feature) => [feature.key, feature]));
        setFeatureSummaryState({ dataset, summaries });
      } catch (error) {
        console.warn("Failed to load dataset feature summaries.", error);
      }
    };

    void loadFeatureSummaries();

    return () => {
      cancelled = true;
    };
  }, [dataset, seed]);

  const featureSummaries = featureSummaryState?.dataset === dataset ? featureSummaryState.summaries : {};

  useEffect(() => {
    setSelectedFeatures((prev) => {
      const validPrev = prev.filter((feature) => featureKeys.includes(feature));
      if (validPrev.length > 0) {
        if (validPrev.length === prev.length && validPrev.every((feature, index) => feature === prev[index])) {
          return prev;
        }
        return validPrev;
      }
      if (featureKeys.length === prev.length && featureKeys.every((feature, index) => feature === prev[index])) {
        return prev;
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

  const renderFeatureDistribution = (summary: FeatureSummary | undefined) => {
    if (!summary) {
      return <span className={trainStyles.featureDistributionPlaceholder} aria-hidden="true" />;
    }

    if (summary.kind === "categorical") {
      const maxCount = Math.max(...summary.categories.map((c) => c.count), 1);
      return (
        <div className={trainStyles.featureCatBlock} aria-hidden="true">
          {summary.categories.map((category) => (
            <div key={category.label} className={trainStyles.featureCatCol}>
              <div className={trainStyles.featureCatBarArea}>
                <div
                  className={trainStyles.featureCatBar}
                  style={{ height: `${Math.max(5, (category.count / maxCount) * 100)}%` }}
                  title={`${category.label}: ${category.count}`}
                />
              </div>
              <span className={trainStyles.featureCatLabel}>{category.label}</span>
            </div>
          ))}
        </div>
      );
    }

    return (
      <div className={trainStyles.featureHistogramBlock} aria-hidden="true">
        <div className={trainStyles.featureOptionHistogram}>
          <FeatureMiniHistogram bins={summary.bins} />
        </div>
        {summary.min != null && summary.max != null && (
          <div className={trainStyles.featureHistogramRange}>
            <span>{summary.min.toFixed(1)}</span>
            <span>{summary.max.toFixed(1)}</span>
          </div>
        )}
      </div>
    );
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
              <Link className={styles.selectModelButton} href="/">
                Choose another model
              </Link>
            </div>
          ) : null}

          <div
            className={`${styles.topPanels} ${!result ? trainStyles.preTrainTopPanels : ""}`}
            style={result ? { display: "none" } : undefined}
          >
            {!result ? (
              <section className={`${styles.panel} ${trainStyles.preTrainFeaturePanel}`}>
                <h2 className={styles.panelTitle}>Select features, then train</h2>

                <div className={trainStyles.field} style={{ marginBottom: "0.75rem" }}>
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
                          {featureSelection[columnIdx].map((feature) => {
                            const desc = FEATURE_DESCRIPTIONS[dataset]?.[feature.key];
                            const summary = featureSummaries[feature.key];
                            return (
                              <label key={feature.key} className={trainStyles.featureOption}>
                                <span className={trainStyles.featureOptionTop}>
                                  <input
                                    type="checkbox"
                                    checked={feature.checked}
                                    onChange={(event) => togglePretrainFeature(feature.key, event.target.checked)}
                                  />
                                  <span className={trainStyles.featureOptionLabel}>{feature.label}</span>
                                </span>
                                {desc && <span className={trainStyles.featureOptionDesc}>{desc}</span>}
                                {renderFeatureDistribution(summary)}
                              </label>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

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
                    {status === "loading" ? "Training…" : "Train"}
                  </button>
                </div>
              </section>
            ) : null}
          </div>

          {!result && (
            <aside className={trainStyles.preTrainSidebar}>
              <div className={trainStyles.preTrainSidebarSection}>
                <p className={trainStyles.preTrainSectionLabel}>Model options</p>
                <label className={trainStyles.preTrainToggle}>
                  <span className={trainStyles.preTrainToggleText}>
                    <span className={trainStyles.toggleFieldLabel}>Center shapes</span>
                    <span className={trainStyles.toggleFieldHint}>Enforce E[f<sub>j</sub>(X<sub>j</sub>)] = 0</span>
                  </span>
                  <span className={trainStyles.preTrainToggleControl}>
                    <input
                      className={trainStyles.preTrainToggleInput}
                      type="checkbox"
                      checked={centerShapes}
                      disabled={modelType !== "igann_interactive"}
                      onChange={(event) => setCenterShapes(event.target.checked)}
                    />
                    <span className={trainStyles.preTrainToggleTrack}>
                      <span className={trainStyles.preTrainToggleThumb} />
                    </span>
                  </span>
                </label>
              </div>

              <div className={trainStyles.preTrainSidebarSection}>
                <button
                  type="button"
                  className={trainStyles.preTrainSectionToggle}
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  <p className={trainStyles.preTrainSectionLabel}>Advanced settings</p>
                  <span className={trainStyles.advancedToggleChevron} style={{ transform: showAdvanced ? "rotate(180deg)" : undefined }}>▾</span>
                </button>
                {showAdvanced && (
                  <div className={`${trainStyles.advancedGrid} ${trainStyles.advancedGridSidebar}`}>
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
              </div>
            </aside>
          )}

          {result ? (
            <div className={styles.shapePanelSlot}>
              <ShapeFunctionsPanel
                shapes={result.version.shapes}
                trainData={result.data}
                featureDescriptions={result.data.featureDescriptions ?? FEATURE_DESCRIPTIONS[result.model.dataset]}
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
                  intercept={currentVersion?.intercept ?? null}
                />
              ) : null}
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
