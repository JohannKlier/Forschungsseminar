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
            /* ── Initial state: full hyperparameter setup ── */
            <section className={styles.panel}>
              <div className={trainStyles.trainHeader}>
                <div className={trainStyles.trainTitleBlock}>
                  <p className={styles.panelEyebrow}>Training</p>
                  <h2 className={styles.panelTitle}>Hyperparameters</h2>
                </div>
                <div className={styles.panelActions}>
                  <button
                    type="button"
                    className={trainStyles.trainButton}
                    onClick={() => train()}
                    disabled={status === "loading"}
                  >
                    {status === "loading" ? "Training…" : "Train model"}
                  </button>
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
            </section>
          ) : (
            /* ── Trained state: feature modes + optional settings ── */
            <section className={`${styles.panel} ${trainStyles.featurePanel}`}>
              <div className={trainStyles.trainHeader}>
                <div className={trainStyles.trainTitleBlock}>
                  <p className={styles.panelEyebrow}>Interactive</p>
                  <h2 className={styles.panelTitle}>Feature Modes</h2>
                </div>
              </div>

              {result.model.model_type === "igann_interactive" && trainData && (
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
                <span>{showRefitSettings ? "Hide" : "Show"} settings</span>
                <span className={trainStyles.advancedToggleChevron} style={{ transform: showRefitSettings ? "rotate(180deg)" : undefined }}>▾</span>
              </button>

              {showRefitSettings && (
                <>
                  <div className={trainStyles.primaryRow}>
                    <div className={trainStyles.field}>
                      <label className={trainStyles.fieldLabel} htmlFor="refit-dataset">Dataset</label>
                      <select
                        id="refit-dataset"
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
                      <label className={trainStyles.fieldLabel} htmlFor="refit-model">Model</label>
                      <select
                        id="refit-model"
                        className={trainStyles.select}
                        value={modelType}
                        onChange={(event) => setModelType(event.target.value as "igann" | "igann_interactive")}
                      >
                        <option value="igann">IGANN</option>
                        <option value="igann_interactive">IGANN interactive</option>
                      </select>
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
                  <button
                    type="button"
                    className={trainStyles.trainButton}
                    onClick={() => manualRefitFromEdits()}
                    disabled={status === "loading"}
                    style={{ marginTop: "0.75rem" }}
                  >
                    {status === "loading" ? "Refitting…" : "Refit from edits"}
                  </button>
                </>
              )}
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
              ) : null}
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
