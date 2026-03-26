"use client";

import { useMemo } from "react";
import styles from "../page.module.css";
import { FeatureMode, ShapeFunction, TrainData } from "../types";
import { computeFeatureImportance } from "../lib/importance";
import FeatureMiniHistogram from "./FeatureMiniHistogram";
import { InteractionHeatmap } from "./InteractionHeatmap";

type FeatureHistData =
  | { type: "continuous"; bins: number[]; min: number | null; max: number | null }
  | { type: "categorical"; bars: { label: string; count: number }[] };

type Props = {
  trainData: TrainData;
  shapes: ShapeFunction[];
  featureModes: Record<string, FeatureMode>;
  onSetFeatureMode: (key: string, mode: FeatureMode | undefined) => void;
};

export default function FeatureModePanel({ trainData, shapes, featureModes, onSetFeatureMode }: Props) {
  const regularKeys = useMemo(() => Object.keys(trainData.featureLabels), [trainData.featureLabels]);
  const interactionShapes = useMemo(() => shapes.filter((s) => s.editableZ), [shapes]);

  const shapeByKey = useMemo(() => {
    const map: Record<string, ShapeFunction> = {};
    shapes.forEach((s) => { map[s.key] = s; });
    return map;
  }, [shapes]);

  const globalAbsMax = useMemo(() => {
    let max = 1e-9;
    interactionShapes.forEach((s) => {
      s.editableZ?.flat().forEach((v) => { if (Math.abs(v) > max) max = Math.abs(v); });
    });
    return max;
  }, [interactionShapes]);

  const importanceByKey = useMemo(() => {
    const raw: Record<string, number> = {};
    shapes.forEach((s) => {
      if (s.editableZ) {
        // Variance of per-row interaction contributions — reflects actual data distribution
        // by summing dummy shape outputs for each row's category level.
        const contribs = (trainData.trainX[s.key] ?? []).filter(Number.isFinite);
        if (contribs.length > 0) {
          const mean = contribs.reduce((a: number, b: number) => a + b, 0) / contribs.length;
          raw[s.key] = contribs.reduce((sum: number, v: number) => sum + (v - mean) ** 2, 0) / contribs.length;
        } else {
          raw[s.key] = 0;
        }
      } else {
        const scatterX = trainData.trainX[s.key] ?? [];
        const knots = { x: s.editableX ?? [], y: s.editableY ?? [] };
        raw[s.key] = Math.max(0, computeFeatureImportance(s, scatterX, knots));
      }
    });
    const total = Object.values(raw).reduce((sum, v) => sum + v, 0);
    const normalized: Record<string, number> = {};
    shapes.forEach((s) => {
      normalized[s.key] = total > 0 ? (raw[s.key] ?? 0) / total : 0;
    });
    return normalized;
  }, [shapes, trainData.trainX]);

  const histByKey = useMemo(() => {
    const map: Record<string, FeatureHistData> = {};
    shapes.forEach((shape) => {
      if (shape.editableZ) return;
      const values = trainData.trainX[shape.key] ?? [];
      if (shape.categories?.length) {
        const counts = new Map<string, number>();
        shape.categories.forEach((cat) => counts.set(String(cat), 0));
        values.forEach((v) => {
          const k = String(v);
          counts.set(k, (counts.get(k) ?? 0) + 1);
        });
        map[shape.key] = {
          type: "categorical",
          bars: shape.categories.map((cat) => ({ label: String(cat), count: counts.get(String(cat)) ?? 0 })),
        };
      } else {
        const numeric = values.map(Number).filter(Number.isFinite);
        if (!numeric.length) {
          map[shape.key] = { type: "continuous", bins: [], min: null, max: null };
          return;
        }
        const min = Math.min(...numeric);
        const max = Math.max(...numeric);
        const binCount = Math.max(8, Math.min(18, Math.round(Math.sqrt(numeric.length))));
        const safeWidth = max === min ? 1 : (max - min) / binCount;
        const bins = Array.from({ length: binCount }, () => 0);
        numeric.forEach((v) => {
          const idx = Math.max(0, Math.min(binCount - 1, max === min ? 0 : Math.floor((v - min) / safeWidth)));
          bins[idx] += 1;
        });
        map[shape.key] = { type: "continuous", bins, min, max };
      }
    });
    return map;
  }, [shapes, trainData.trainX]);

  const sortedRegularKeys = useMemo(() => {
    return [...regularKeys].sort((a, b) => {
      const aDeact = featureModes[a] === "deactivate";
      const bDeact = featureModes[b] === "deactivate";
      if (aDeact !== bDeact) return aDeact ? 1 : -1;
      return (importanceByKey[b] ?? 0) - (importanceByKey[a] ?? 0);
    });
  }, [regularKeys, featureModes, importanceByKey]);

  const continuousKeys = useMemo(
    () => sortedRegularKeys.filter((k) => !shapeByKey[k]?.categories?.length),
    [sortedRegularKeys, shapeByKey]
  );
  const categoricalKeys = useMemo(
    () => sortedRegularKeys.filter((k) => shapeByKey[k]?.categories?.length),
    [sortedRegularKeys, shapeByKey]
  );

  const renderRegularKey = (key: string) => {
    const label = trainData.featureLabels[key] ?? key;
    const mode = featureModes[key];
    const isLocked = mode === "lock";
    const isDeactivated = mode === "deactivate";
    const importance = importanceByKey[key] ?? 0;
    const hist = histByKey[key];
    const bgAlpha = 0.08 + importance * 0.55;
    const importanceBg = `rgba(242, 95, 76, ${bgAlpha.toFixed(3)})`;
    return (
      <div key={key} className={`${styles.featureModeRow} ${isDeactivated ? styles.featureModeRowDeactivated : ""}`}>
        <span className={styles.featureModeRowLabel} title={key}>{label}</span>
        <span className={styles.importanceCell} style={{ background: importanceBg }}>
          {importance.toFixed(3)}
        </span>
        <div className={styles.featureModeControls}>
          <label className={styles.featureModeLockLabel} title="Lock: freeze shape, skip boosting">
            <input
              type="checkbox"
              className={styles.featureModeLockInput}
              checked={isLocked}
              disabled={isDeactivated}
              onChange={(e) => onSetFeatureMode(key, e.target.checked ? "lock" : undefined)}
            />
            <span className={styles.featureModeLockTrack}><span className={styles.featureModeLockThumb} /></span>
            <span className={styles.featureModeLockText}>Lock</span>
          </label>
          <label className={styles.featureModeDeactivateLabel} title="Off: remove from model entirely">
            <input
              type="checkbox"
              className={styles.featureModeDeactivateInput}
              checked={isDeactivated}
              onChange={(e) => onSetFeatureMode(key, e.target.checked ? "deactivate" : undefined)}
            />
            <span className={styles.featureModeDeactivateTrack}><span className={styles.featureModeDeactivateThumb} /></span>
            <span className={styles.featureModeLockText}>Off</span>
          </label>
        </div>
        {hist && (
          <div className={styles.featureModeHist}>
            {hist.type === "continuous" ? (
              <>
                <div className={styles.featureDistributionHistogram}><FeatureMiniHistogram bins={hist.bins} /></div>
                <div className={styles.featureDistributionRange}>
                  <span>{hist.min?.toFixed(2) ?? "—"}</span>
                  <span>{hist.max?.toFixed(2) ?? "—"}</span>
                </div>
              </>
            ) : (
              <div className={styles.featureDistributionBars} aria-hidden="true">
                {(() => {
                  const maxCount = Math.max(...hist.bars.map((b) => b.count), 1);
                  return hist.bars.map((bar) => (
                    <div key={bar.label} className={styles.featureDistributionBarGroup}>
                      <div className={styles.featureDistributionBar} style={{ height: `${(bar.count / maxCount) * 100}%` }} />
                      <span className={styles.featureDistributionBarLabel}>{bar.label}</span>
                    </div>
                  ));
                })()}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={styles.featureModePanel}>
      <p className={styles.settingsLabel}>Feature modes</p>
      <div className={styles.featureModeRows}>
        {continuousKeys.length > 0 && (
          <>
            <p className={styles.featureModeSectionHeader}>Continuous</p>
            {continuousKeys.map(renderRegularKey)}
          </>
        )}
        {categoricalKeys.length > 0 && (
          <>
            <p className={styles.featureModeSectionHeader}>Categorical</p>
            {categoricalKeys.map(renderRegularKey)}
          </>
        )}
        {interactionShapes.length > 0 && (
          <>
            <p className={styles.featureModeSectionHeader}>Interactions</p>
            {interactionShapes.map((shape) => {
              const mode = featureModes[shape.key];
              const isDeactivated = mode === "deactivate";
              const importance = importanceByKey[shape.key] ?? 0;
              const bgAlpha = 0.08 + importance * 0.55;
              const importanceBg = `rgba(242, 95, 76, ${bgAlpha.toFixed(3)})`;
              return (
                <div key={shape.key} className={`${styles.featureModeRow} ${isDeactivated ? styles.featureModeRowDeactivated : ""}`}>
                  <span className={styles.featureModeRowLabel} title={shape.key}>{shape.label}</span>
                  <span className={styles.importanceCell} style={{ background: importanceBg }}>
                    {importance.toFixed(3)}
                  </span>
                  <div className={styles.featureModeControls}>
                    <label className={styles.featureModeDeactivateLabel} title="Off: exclude this interaction from the model">
                      <input
                        type="checkbox"
                        className={styles.featureModeDeactivateInput}
                        checked={isDeactivated}
                        onChange={(e) => onSetFeatureMode(shape.key, e.target.checked ? "deactivate" : undefined)}
                      />
                      <span className={styles.featureModeDeactivateTrack}><span className={styles.featureModeDeactivateThumb} /></span>
                      <span className={styles.featureModeLockText}>Off</span>
                    </label>
                  </div>
                  <div className={styles.featureModeHist}>
                    <div className={styles.featureDistributionHistogram}>
                      <InteractionHeatmap shape={shape} width={200} height={46} absMax={globalAbsMax} />
                    </div>
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
