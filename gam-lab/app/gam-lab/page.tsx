"use client";

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
  const trainBandwidth = Number(searchParams.get("bandwidth") ?? "0.12");
  const trainPoints = Number(searchParams.get("points") ?? "10");

  const {
    status,
    result,
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
    sidebarTab,
    setSidebarTab,
    partial,
    displayLabel,
    featureImportances,
    history,
    historyCursor,
    recordAction,
    commitEdits,
    undoLast,
    redoLast,
    deleteHistoryEntry,
  } = useGamLab({
    initialModel,
    initialTrain: trainMode
      ? {
          dataset: trainDataset,
          bandwidth: Number.isFinite(trainBandwidth) ? trainBandwidth : 0.12,
          points: Number.isFinite(trainPoints) ? trainPoints : 10,
        }
      : null,
  });

  if (!initialModel && !trainMode) {
    router.replace("/");
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
              featureImportances={featureImportances}
            />
            {models ? <PredictionFitPlot result={result} models={models} /> : null}
            {partial ? (
              <SidebarPanel
                sidebarTab={sidebarTab}
                setSidebarTab={setSidebarTab}
                displayLabel={displayLabel}
                partial={partial}
                selectedKnots={selectedKnots}
                knotEdits={knotEdits}
                knots={knots}
                onRecordAction={recordAction}
                setKnots={setKnots}
                setKnotEdits={setKnotEdits}
                setSelectedKnots={setSelectedKnots}
                onCommitEdits={commitEdits}
                applyMonotonic={applyMonotonic}
                addPointsInSelection={addPointsInSelection}
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
