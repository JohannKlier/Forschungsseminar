"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
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

type FeatureSummary =
  | {
      key: string;
      label: string;
      description?: string;
      kind: "continuous";
      bins: number[];
      min: number | null;
      max: number | null;
    }
  | {
      key: string;
      label: string;
      description?: string;
      kind: "categorical";
      categories: { label: string; count: number }[];
    };


export default function TrainPage() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const search = searchParams.toString();
  const { logEvent, kuerzel, setKuerzel } = useAuditLogger();
  const [kuerzelInput, setKuerzelInput] = useState("");
  const [kuerzelError, setKuerzelError] = useState("");
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
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
    selectedFeatures,
    setSelectedFeatures,
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
    sampleSize,
    setSampleSize,
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
    modelInfo,
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
  const [showIntro, setShowIntro] = useState(false);
  const prevResultRef = useRef<typeof result>(null);

  useEffect(() => {
    if (result && !prevResultRef.current) {
      setShowIntro(true);
    }
    prevResultRef.current = result;
  }, [result]);

  const [featureSummaryState, setFeatureSummaryState] = useState<{
    dataset: string;
    summaries: Record<string, FeatureSummary>;
  } | null>(null);

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
  const availableFeatures = useMemo(() => Object.values(featureSummaries), [featureSummaries]);
  const featureKeys = useMemo(() => availableFeatures.map((feature) => feature.key), [availableFeatures]);

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
        Array<FeatureSummary & { checked: boolean }>,
        Array<FeatureSummary & { checked: boolean }>,
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

  const handleKuerzelSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!kuerzelInput.trim()) {
      setKuerzelError("Bitte einen Kürzel eingeben.");
      return;
    }
    setKuerzel(kuerzelInput.trim());
  };

  return (
    <>
    {mounted && !kuerzel && (
      <div className={trainStyles.kuerzelOverlay}>
        <div className={trainStyles.kuerzelCard}>
          <p className={trainStyles.introEyebrow}>User Study — GAM Lab</p>
          <h1 className={trainStyles.introHeading}>Kürzel eingeben</h1>
          <p className={trainStyles.introBody}>
            Ihnen wurde für diese Studie ein persönlicher Kürzel zugewiesen.
            Geben Sie ihn ein, um Ihre Sitzung zu starten.
          </p>
          <form onSubmit={handleKuerzelSubmit} className={trainStyles.kuerzelForm}>
            <input
              className={trainStyles.kuerzelInput}
              type="text"
              placeholder="z. B. TN01"
              value={kuerzelInput}
              onChange={(e) => { setKuerzelInput(e.target.value); setKuerzelError(""); }}
              autoFocus
              maxLength={30}
            />
            {kuerzelError && <p className={trainStyles.kuerzelError}>{kuerzelError}</p>}
            <button className={trainStyles.introConfirmButton} type="submit">
              Bestätigen
            </button>
          </form>
        </div>
      </div>
    )}
    {showIntro && (
      <div className={trainStyles.introOverlay}>
        <div className={trainStyles.introCard}>
          <div className={trainStyles.introHeader}>
            <p className={trainStyles.introEyebrow}>User Study — GAM Lab</p>
            <h1 className={trainStyles.introHeading}>Ihre Aufgabe</h1>
            <p className={trainStyles.introLead}>
              Das Modell wurde trainiert und zeigt Ihnen nun, wie es jeden Messwert
              bei der Vorhersage von <strong>Intensivstationsmortalität</strong> gewichtet.
              Beurteilen Sie, ob diese Gewichtung <strong>medizinisch plausibel</strong> ist —
              und korrigieren Sie sie, wenn nötig.
            </p>
          </div>

          <div className={trainStyles.introSteps}>
            <div className={trainStyles.introStep}>
              <div className={trainStyles.introStepIcon}>
                <span className={trainStyles.introStepNum}>1</span>
              </div>
              <div className={trainStyles.introStepBody}>
                <p className={trainStyles.introStepTitle}>Kurven lesen</p>
                <p className={trainStyles.introStepText}>
                  Jedes Merkmal hat eine Kurve. Sie zeigt: Wie verändert sich das
                  vorhergesagte Sterberisiko, wenn dieser Messwert steigt oder fällt?
                </p>
              </div>
            </div>
            <div className={trainStyles.introStep}>
              <div className={trainStyles.introStepIcon}>
                <span className={trainStyles.introStepNum}>2</span>
              </div>
              <div className={trainStyles.introStepBody}>
                <p className={trainStyles.introStepTitle}>y-Achse verstehen</p>
                <p className={trainStyles.introStepText}>
                  Werte <strong>über null</strong> erhöhen das vorhergesagte Risiko,
                  Werte <strong>unter null</strong> senken es.
                  Bei null hat der Messwert an dieser Stelle keinen Einfluss.
                </p>
              </div>
            </div>
            <div className={trainStyles.introStep}>
              <div className={trainStyles.introStepIcon}>
                <span className={trainStyles.introStepNum}>3</span>
              </div>
              <div className={trainStyles.introStepBody}>
                <p className={trainStyles.introStepTitle}>Kurven korrigieren</p>
                <p className={trainStyles.introStepText}>
                  Wirkt eine Kurve klinisch falsch? Ziehen Sie die Punkte nach oben
                  oder unten. Die Seitenleiste zeigt, wie sich Ihre Änderungen auf
                  die Modellgüte auswirken.
                </p>
              </div>
            </div>
          </div>

          <div className={trainStyles.introFooter}>
            <button
              className={trainStyles.introConfirmButton}
              onClick={() => setShowIntro(false)}
            >
              Verstanden — Modell anzeigen
            </button>
          </div>
        </div>
      </div>
    )}
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

                <div className={trainStyles.field} style={{ marginBottom: "0.75rem" }}>
                  <label className={trainStyles.fieldLabel} htmlFor="train-modeltype">Model type</label>
                  <select
                    id="train-modeltype"
                    className={trainStyles.select}
                    value={modelType}
                    onChange={(event) => setModelType(event.target.value as "igann" | "igann_interactive")}
                  >
                    <option value="igann_interactive">IGANN Interactive</option>
                    <option value="igann">IGANN</option>
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
                            const desc = featureSummaries[feature.key]?.description;
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
                      train({
                        selected_features: selectedFeatures,
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
              {selectedDataset.id === "mimic4_mean_100_full" && (
                <div className={trainStyles.preTrainSidebarSection}>
                  <p className={trainStyles.preTrainSectionLabel}>Dataset options</p>
                  <div className={trainStyles.field}>
                    <label className={trainStyles.fieldLabel} htmlFor="train-samplesize">
                      Sample size
                    </label>
                    <input
                      id="train-samplesize"
                      className={trainStyles.numberInput}
                      type="number"
                      step={100}
                      min={100}
                      max={65350}
                      value={sampleSize}
                      onChange={(e) => setSampleSize(Number(e.target.value))}
                    />
                  </div>
                </div>
              )}

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
                featureDescriptions={result.data.featureDescriptions}
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
                  modelInfo={modelInfo}
                  currentVersion={currentVersion}
                />
              ) : null}
            </div>
          ) : null}
        </section>
      </div>
    </div>
    </>
  );
}
