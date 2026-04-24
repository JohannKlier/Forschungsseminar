import { Dispatch, SetStateAction, useEffect, useMemo, useRef } from "react";
import styles from "../page.module.css";
import { HistoryEntry, KnotSet, MetricSummary, MetricWarning, ModelInfo, ShapeFunction, ShapeFunctionVersion, SidebarTab, TrainData } from "../types";
import FeatureMiniHistogram from "./FeatureMiniHistogram";
import { InteractionHeatmap } from "./InteractionHeatmap";

const formatMetricChange = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const formatMetricPercent = (value: number | null) => (value == null ? null : `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`);

function HistoryPanel({
  history,
  formatHistoryAction,
  formatHistoryDetail,
  onDeleteHistoryEntry,
}: {
  history: HistoryEntry[];
  formatHistoryAction: (action: string) => string;
  formatHistoryDetail: (entryIndex: number) => string | null;
  onDeleteHistoryEntry: (index: number) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history.length]);

  // Group entries by featureKey, preserving insertion order of first occurrence.
  const groups = useMemo(() => {
    const result: { featureKey: string; entries: { entry: HistoryEntry; index: number }[] }[] = [];
    const groupIdx: Record<string, number> = {};
    history.forEach((entry, index) => {
      if (groupIdx[entry.featureKey] === undefined) {
        groupIdx[entry.featureKey] = result.length;
        result.push({ featureKey: entry.featureKey, entries: [] });
      }
      result[groupIdx[entry.featureKey]].entries.push({ entry, index });
    });
    return result;
  }, [history]);

  return (
    <div className={styles.historyScroll}>
      <p className={styles.settingsLabel}>History</p>
      <div className={styles.historyList}>
        {groups.length ? groups.map(({ featureKey, entries }) => (
          <div key={featureKey} className={styles.historyGroup}>
            <div className={styles.historyGroupHeader}>
              <span className={styles.historyGroupLabel}>{featureKey}</span>
              <span className={styles.historyGroupCount}>{entries.length}</span>
            </div>
            {entries.map(({ entry, index }) => {
              const detail = formatHistoryDetail(index);
              return (
                <div key={`${entry.ts}-${entry.action}`} className={styles.historyItem}>
                  <div className={styles.historyActionRow}>
                    <span className={styles.settingsValue}>{formatHistoryAction(entry.action)}</span>
                    <button
                      type="button"
                      className={styles.historyDeleteButton}
                      onClick={() => onDeleteHistoryEntry(index)}
                      aria-label="Delete history entry"
                    >
                      ×
                    </button>
                  </div>
                  {detail ? (
                    <span className={styles.historyDetail}>{detail}</span>
                  ) : null}
                </div>
              );
            })}
          </div>
        )) : (
          <p className={styles.settingsHint}>No actions yet.</p>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function FeatureDistributionsPanel({
  shapes,
  trainData,
  activeFeatureKey,
  activeKnots,
  selectedPointIndices,
  activeFeatureCategories,
}: {
  shapes: ShapeFunction[];
  trainData: TrainData;
  activeFeatureKey: string | null;
  activeKnots: KnotSet | null;
  selectedPointIndices: number[];
  activeFeatureCategories: string[] | null;
}) {
  const selectedRowMask = useMemo(() => {
    if (!activeFeatureKey || !selectedPointIndices.length) return null;
    const activeValues = trainData.trainX[activeFeatureKey] ?? [];
    if (!activeValues.length) return null;

    if (activeFeatureCategories?.length) {
      const selectedCategories = new Set(
        selectedPointIndices
          .map((idx) => activeFeatureCategories[idx])
          .filter((value): value is string => typeof value === "string"),
      );
      return activeValues.map((value) => selectedCategories.has(String(value)));
    }

    if (!activeKnots) return null;
    const allSorted = activeKnots.x
      .map((x, idx) => ({ x, idx }))
      .filter((point) => Number.isFinite(point.x))
      .sort((a, b) => a.x - b.x);
    if (!allSorted.length) return null;
    const intervals = allSorted
      .filter((point) => selectedPointIndices.includes(point.idx))
      .map((point) => {
        const sortedIndex = allSorted.findIndex((candidate) => candidate.idx === point.idx);
        const prev = allSorted[sortedIndex - 1];
        const next = allSorted[sortedIndex + 1];
        return {
          left: prev ? (prev.x + point.x) / 2 : Number.NEGATIVE_INFINITY,
          right: next ? (next.x + point.x) / 2 : Number.POSITIVE_INFINITY,
        };
      });
    if (!intervals.length) return null;
    return activeValues.map((raw) => {
      const value = Number(raw);
      if (!Number.isFinite(value)) return false;
      return intervals.some((interval) => value >= interval.left && value < interval.right);
    });
  }, [activeFeatureCategories, activeFeatureKey, activeKnots, selectedPointIndices, trainData.trainX]);

  const featureRows = useMemo(
    () =>
      shapes.map((shape) => {
        const label = trainData.featureLabels[shape.key] ?? shape.label ?? shape.key;

        // Interaction shapes: show a mini heatmap instead of a distribution
        if (shape.editableZ) {
          return { key: shape.key, label, type: "interaction" as const, shape };
        }

        const values = trainData.trainX[shape.key] ?? [];
        if (shape.categories?.length) {
          const counts = new Map<string, number>();
          const selectedCounts = new Map<string, number>();
          shape.categories.forEach((category) => counts.set(String(category), 0));
          shape.categories.forEach((category) => selectedCounts.set(String(category), 0));
          values.forEach((value, index) => {
            const key = String(value);
            counts.set(key, (counts.get(key) ?? 0) + 1);
            if (selectedRowMask?.[index]) {
              selectedCounts.set(key, (selectedCounts.get(key) ?? 0) + 1);
            }
          });
          const bars = shape.categories.map((category) => ({
            label: String(category),
            count: counts.get(String(category)) ?? 0,
          }));
          const selectedBars = shape.categories.map((category) => ({
            label: String(category),
            count: selectedCounts.get(String(category)) ?? 0,
          }));
          return { key: shape.key, label, type: "categorical" as const, bars, selectedBars, total: values.length };
        }

        const numericValues = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
        if (!numericValues.length) {
          return {
            key: shape.key,
            label,
            type: "continuous" as const,
            bins: [],
            selectedBins: [],
            min: null,
            max: null,
            total: 0,
          };
        }
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const binCount = Math.max(8, Math.min(18, Math.round(Math.sqrt(numericValues.length))));
        const safeWidth = max === min ? 1 : (max - min) / binCount;
        const bins = Array.from({ length: binCount }, () => 0);
        const selectedBins = Array.from({ length: binCount }, () => 0);
        values.forEach((raw, valueIndex) => {
          const value = Number(raw);
          if (!Number.isFinite(value)) return;
          const rawIndex = max === min ? 0 : Math.floor((value - min) / safeWidth);
          const index = Math.max(0, Math.min(binCount - 1, rawIndex));
          bins[index] += 1;
          if (selectedRowMask?.[valueIndex]) {
            selectedBins[index] += 1;
          }
        });
        return { key: shape.key, label, type: "continuous" as const, bins, selectedBins, min, max, total: numericValues.length };
      }),
    [selectedRowMask, shapes, trainData],
  );

  return (
    <div className={styles.featureDistributionScroll}>
      <p className={styles.settingsLabel}>Features</p>
      <div className={styles.featureDistributionLegend} aria-hidden="true">
        <span className={styles.featureDistributionLegendItem}>
          <span className={`${styles.featureDistributionLegendSwatch} ${styles.featureDistributionLegendSwatchAll}`} />
          All
        </span>
        <span className={styles.featureDistributionLegendItem}>
          <span className={`${styles.featureDistributionLegendSwatch} ${styles.featureDistributionLegendSwatchSelected}`} />
          Selected
        </span>
      </div>
      <div className={styles.featureDistributionList}>
        {featureRows.map((feature) => {
          if (feature.type === "interaction") {
            return (
              <div key={feature.key} className={styles.featureDistributionCard}>
                <div className={styles.featureDistributionHeader}>
                  <span className={styles.featureDistributionName}>{feature.label}</span>
                </div>
                <div className={styles.featureDistributionHistogram}>
                  <InteractionHeatmap shape={feature.shape} width={312} height={46} />
                </div>
              </div>
            );
          }
          return (
            <div
              key={feature.key}
              className={styles.featureDistributionCard}
            >
              <div className={styles.featureDistributionHeader}>
                <span className={styles.featureDistributionName}>{feature.label}</span>
              </div>
              {feature.type === "categorical" ? (
                <div className={styles.featureDistributionBars} aria-hidden="true">
                  {(() => {
                    const maxCount = Math.max(...feature.bars.map((entry) => entry.count), 1);
                    const maxSelectedCount = Math.max(...feature.selectedBars.map((entry) => entry.count), 1);
                    return feature.bars.map((bar) => {
                      const selectedBar = feature.selectedBars.find((entry) => entry.label === bar.label);
                      return (
                        <div key={bar.label} className={styles.featureDistributionBarGroup}>
                          <div
                            className={styles.featureDistributionBar}
                            style={{ height: `${(bar.count / maxCount) * 100}%` }}
                          >
                            {selectedBar && selectedBar.count > 0 ? (
                              <div
                                className={styles.featureDistributionBarSelected}
                                style={{ height: `${(selectedBar.count / maxSelectedCount) * 100}%` }}
                              />
                            ) : null}
                          </div>
                          <span className={styles.featureDistributionBarLabel}>{bar.label}</span>
                        </div>
                      );
                    });
                  })()}
                </div>
              ) : (
                <>
                  <div className={styles.featureDistributionHistogram} aria-hidden="true">
                    <FeatureMiniHistogram bins={feature.bins} selectedBins={feature.selectedBins} />
                  </div>
                  <div className={styles.featureDistributionRange}>
                    <span>{feature.min?.toFixed(2) ?? "—"}</span>
                    <span>{feature.max?.toFixed(2) ?? "—"}</span>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}


function MetricLine({ label, value }: { label: string; value: number | null | undefined }) {
  if (value == null) return null;
  return (
    <div className={styles.modelInfoRow}>
      <span className={styles.modelInfoKey}>{label}</span>
      <span className={styles.modelInfoVal}>{value.toFixed(4)}</span>
    </div>
  );
}

function ModelInfoPanel({ modelInfo, version }: { modelInfo: ModelInfo; version: ShapeFunctionVersion }) {
  const rows: [string, string | number | boolean][] = [
    ["Dataset",        modelInfo.dataset],
    ["Model type",     modelInfo.model_type],
    ["Task",           modelInfo.task],
    ["n_estimators",   modelInfo.n_estimators],
    ["boost_rate",     modelInfo.boost_rate],
    ["init_reg",       modelInfo.init_reg],
    ["elm_alpha",      modelInfo.elm_alpha],
    ["early_stopping", modelInfo.early_stopping],
    ["seed",           modelInfo.seed],
    ["scale_y",        String(modelInfo.scale_y)],
    ["center_shapes",  String(version.center_shapes)],
    ["shape_points",   modelInfo.points],
  ];

  const trainM: MetricSummary = version.trainMetrics;
  const testM: MetricSummary  = version.testMetrics;

  return (
    <div className={styles.modelInfoPanel}>
      <p className={styles.settingsLabel}>Hyperparameters</p>
      <div className={styles.modelInfoTable}>
        {rows.map(([k, v]) => (
          <div key={k} className={styles.modelInfoRow}>
            <span className={styles.modelInfoKey}>{k}</span>
            <span className={styles.modelInfoVal}>{String(v)}</span>
          </div>
        ))}
      </div>

      {modelInfo.selected_features && modelInfo.selected_features.length > 0 && (
        <>
          <p className={styles.settingsLabel} style={{ marginTop: "0.75rem" }}>
            Features ({modelInfo.selected_features.length})
          </p>
          <div className={styles.modelInfoFeatures}>
            {modelInfo.selected_features.map((f) => (
              <span key={f} className={styles.modelInfoFeatureTag}>{f}</span>
            ))}
          </div>
        </>
      )}

      <p className={styles.settingsLabel} style={{ marginTop: "0.75rem" }}>Train metrics</p>
      <div className={styles.modelInfoTable}>
        <MetricLine label="RMSE"      value={trainM.rmse} />
        <MetricLine label="MAE"       value={trainM.mae} />
        <MetricLine label="R²"        value={trainM.r2} />
        <MetricLine label="Accuracy"  value={trainM.acc} />
        <MetricLine label="F1"        value={trainM.f1} />
        <MetricLine label="Precision" value={trainM.precision} />
        <MetricLine label="Recall"    value={trainM.recall} />
        <div className={styles.modelInfoRow}>
          <span className={styles.modelInfoKey}>n</span>
          <span className={styles.modelInfoVal}>{trainM.count}</span>
        </div>
      </div>

      <p className={styles.settingsLabel} style={{ marginTop: "0.75rem" }}>Test metrics</p>
      <div className={styles.modelInfoTable}>
        <MetricLine label="RMSE"      value={testM.rmse} />
        <MetricLine label="MAE"       value={testM.mae} />
        <MetricLine label="R²"        value={testM.r2} />
        <MetricLine label="Accuracy"  value={testM.acc} />
        <MetricLine label="F1"        value={testM.f1} />
        <MetricLine label="Precision" value={testM.precision} />
        <MetricLine label="Recall"    value={testM.recall} />
        <div className={styles.modelInfoRow}>
          <span className={styles.modelInfoKey}>n</span>
          <span className={styles.modelInfoVal}>{testM.count}</span>
        </div>
      </div>
    </div>
  );
}

type Props = {
  sidebarTab: SidebarTab;
  setSidebarTab: Dispatch<SetStateAction<SidebarTab>>;
  metricWarning: MetricWarning | null;
  history: HistoryEntry[];
  formatHistoryAction: (action: string) => string;
  formatHistoryDetail: (entryIndex: number) => string | null;
  onDeleteHistoryEntry: (index: number) => void;
  shapes: ShapeFunction[];
  trainData: TrainData;
  activeFeatureKey: string | null;
  activeKnots: KnotSet | null;
  selectedPointIndices: number[];
  activeFeatureCategories: string[] | null;
  modelInfo?: ModelInfo | null;
  currentVersion?: ShapeFunctionVersion | null;
};

export default function SidebarPanel({
  sidebarTab,
  setSidebarTab,
  metricWarning,
  history,
  formatHistoryAction,
  formatHistoryDetail,
  onDeleteHistoryEntry,
  shapes,
  trainData,
  activeFeatureKey,
  activeKnots,
  selectedPointIndices,
  activeFeatureCategories,
  modelInfo,
  currentVersion,
}: Props) {
  return (
    <div className={styles.settingsRail}>
      <div className={styles.sidebarHeader}>
        <span className={styles.logo}>GAM Lab</span>
      </div>
      <div className={styles.sidebarTabs}>
        <button
          type="button"
          className={`${styles.sidebarTabButton} ${sidebarTab === "edit" ? styles.sidebarTabButtonActive : ""}`}
          onClick={() => setSidebarTab("edit")}
        >
          Edit
        </button>
        <button
          type="button"
          className={`${styles.sidebarTabButton} ${sidebarTab === "history" ? styles.sidebarTabButtonActive : ""}`}
          onClick={() => setSidebarTab("history")}
        >
          History
        </button>
        <button
          type="button"
          className={`${styles.sidebarTabButton} ${sidebarTab === "features" ? styles.sidebarTabButtonActive : ""}`}
          onClick={() => setSidebarTab("features")}
        >
          Features
        </button>
      </div>
      {sidebarTab === "edit" ? (
        <>
          <div className={styles.settingsSection}>
            <div>
              <p className={styles.settingsLabel}>Warnings</p>
              {metricWarning ? (
                <div className={styles.warningCard} role="alert" aria-live="polite">
                  <div className={styles.warningHeader}>
                    <span className={styles.warningBadge}>Warning</span>
                    <span className={styles.warningAction}>{formatHistoryAction(metricWarning.action)}</span>
                  </div>
                  <p className={styles.warningText}>
                    The latest applied action pushed the live fit past the warning margin.
                  </p>
                  <div className={styles.warningMetrics}>
                    {metricWarning.details.map((detail) => (
                      <div key={detail.label} className={styles.warningMetricRow}>
                        <span className={styles.warningMetricLabel}>{detail.label}</span>
                        <span className={styles.warningMetricDelta}>
                          {formatMetricChange(detail.delta)}
                          {detail.deltaPct != null ? ` (${formatMetricPercent(detail.deltaPct)})` : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className={styles.warningIdleCard}>
                  <p className={styles.warningIdleTitle}>No warning</p>
                  <p className={styles.settingsHint}>
                    Warnings appear here only when the latest applied action degrades the live metrics beyond the guardrail.
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      ) : sidebarTab === "history" ? (
        <div>
          <HistoryPanel
            history={history}
            formatHistoryAction={formatHistoryAction}
            formatHistoryDetail={formatHistoryDetail}
            onDeleteHistoryEntry={onDeleteHistoryEntry}
          />
        </div>
      ) : (
        <div>
          <FeatureDistributionsPanel
            shapes={shapes}
            trainData={trainData}
            activeFeatureKey={activeFeatureKey}
            activeKnots={activeKnots}
            selectedPointIndices={selectedPointIndices}
            activeFeatureCategories={activeFeatureCategories}
          />
        </div>
      )}
    </div>
  );
}
