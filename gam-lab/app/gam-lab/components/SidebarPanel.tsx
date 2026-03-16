import { Dispatch, SetStateAction, useEffect, useMemo, useRef } from "react";
import styles from "../page.module.css";
import { KnotSet, ShapeFunction, StatItem, TrainData } from "../types";
import TourLabel from "./TourLabel";

type HistoryEntry = { featureKey: string; action: string; ts: number; changes: { x: number; before?: number; after?: number; delta?: number }[] };

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
  const groups: { featureKey: string; entries: { entry: HistoryEntry; index: number }[] }[] = [];
  const groupIdx: Record<string, number> = {};
  history.forEach((entry, index) => {
    if (groupIdx[entry.featureKey] === undefined) {
      groupIdx[entry.featureKey] = groups.length;
      groups.push({ featureKey: entry.featureKey, entries: [] });
    }
    groups[groupIdx[entry.featureKey]].entries.push({ entry, index });
  });

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
            {entries.map(({ entry, index }) => (
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
                {formatHistoryDetail(index) ? (
                  <span className={styles.historyDetail}>{formatHistoryDetail(index)}</span>
                ) : null}
              </div>
            ))}
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
  onSelectFeature,
}: {
  shapes: ShapeFunction[];
  trainData: TrainData;
  activeFeatureKey: string | null;
  activeKnots: KnotSet | null;
  selectedPointIndices: number[];
  activeFeatureCategories: string[] | null;
  onSelectFeature: (featureKey: string) => void;
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
        const values = trainData.trainX[shape.key] ?? [];
        const label = trainData.featureLabels[shape.key] ?? shape.label ?? shape.key;
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
      <div className={styles.featureDistributionList}>
        {featureRows.map((feature) => {
          return (
            <button
              key={feature.key}
              type="button"
              className={styles.featureDistributionCard}
              onClick={() => onSelectFeature(feature.key)}
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
                    {(() => {
                      const maxBin = Math.max(...feature.bins, 1);
                      const maxSelectedBin = Math.max(...feature.selectedBins, 1);
                      return feature.bins.map((bin, index) => {
                        return (
                          <div
                            key={`${feature.key}-${index}`}
                            className={styles.featureDistributionBin}
                            style={{ height: `${(bin / maxBin) * 100}%` }}
                          >
                            {(feature.selectedBins[index] ?? 0) > 0 ? (
                              <div
                                className={styles.featureDistributionBinSelected}
                                style={{ height: `${((feature.selectedBins[index] ?? 0) / maxSelectedBin) * 100}%` }}
                              />
                            ) : null}
                          </div>
                        );
                      });
                    })()}
                  </div>
                  <div className={styles.featureDistributionRange}>
                    <span>{feature.min?.toFixed(2) ?? "—"}</span>
                    <span>{feature.max?.toFixed(2) ?? "—"}</span>
                  </div>
                </>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

type Props = {
  showTourLabels?: boolean;
  sidebarTab: "edit" | "history" | "features";
  setSidebarTab: Dispatch<SetStateAction<"edit" | "history" | "features">>;
  stats: StatItem[] | null;
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
  onSelectFeature: (featureKey: string) => void;
};

export default function SidebarPanel({
  showTourLabels = false,
  sidebarTab,
  setSidebarTab,
  stats,
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
  onSelectFeature,
}: Props) {
  return (
    <div className={`${styles.settingsRail} ${showTourLabels ? styles.tourFocus : ""}`}>
      <div className={styles.sidebarHeader}>
        <span className={styles.logo}>GAM Lab</span>
      </div>
      <div className={`${styles.sidebarTabs} ${styles.tourLabelAnchor}`}>
        {showTourLabels ? (
          <TourLabel
            label="Tabs"
            title="Switch the sidebar mode"
            description="The sidebar has one mode for live feedback and another for reviewing edits."
            details={[
              "Edit shows the stats summary for the current model state.",
              "History lists recorded edit actions and lets you prune them.",
              "Features shows the raw feature distributions from the dataset.",
            ]}
            placement="top-left"
          />
        ) : null}
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
            <div className={styles.tourLabelAnchor}>
              {showTourLabels ? (
                <TourLabel
                  label="Statistics"
                  title="Read model quality at a glance"
                  description="These bars compare the original model, the latest trained or refit version, and the current live edited state."
                  details={[
                    "Initial comes from the very first training run.",
                    "Latest comes from the most recent train or refit result.",
                    "Current is recomputed in the frontend while you edit.",
                  ]}
                  placement="top-left"
                />
              ) : null}
            <p className={styles.settingsLabel}>Stats</p>
            <div className={styles.statsLegend}>
              <span className={styles.statsLegendItem}>
                <span className={`${styles.statsLegendSwatch} ${styles.statsLegendSwatchInitial}`} />
                Initial
              </span>
              <span className={styles.statsLegendItem}>
                <span className={`${styles.statsLegendSwatch} ${styles.statsLegendSwatchLatest}`} />
                Latest
              </span>
              <span className={styles.statsLegendItem}>
                <span className={`${styles.statsLegendSwatch} ${styles.statsLegendSwatchCurrent}`} />
                Current
              </span>
            </div>
            {stats?.map((stat) => {
            if (stat.kind === "value") {
              return (
                <div key={stat.label} className={styles.settingsItem}>
                  <span className={styles.settingsHint}>{stat.label}</span>
                  <span className={styles.settingsValue}>{stat.value}</span>
                </div>
              );
            }
            const { initial, latest, current } = stat;
            const format = stat.format ?? "0.000";
            const digits = format.includes("0.00") ? 2 : 3;
            const fmt = (v: number | null) => (Number.isFinite(v) ? (v as number).toFixed(digits) : "—");
            const maxVal = Math.max(Math.abs(initial ?? 0), Math.abs(latest ?? 0), Math.abs(current ?? 0), 1e-6);
            const pct = (v: number | null) =>
              Number.isFinite(v) ? `${(Math.abs(v as number) / maxVal) * 100}%` : "0%";
            return (
              <div key={stat.label} className={styles.statsBarRow}>
                <div className={styles.statsBarHeader}>
                  <span className={styles.settingsHint}>{stat.label}</span>
                </div>
                <div className={styles.statsMetricRows}>
                  {initial !== null ? (
                    <div className={styles.statsMetricRow}>
                      <div className={styles.statsBar}>
                        <div className={`${styles.statsBarValue} ${styles.statsBarValueInitial}`} style={{ width: pct(initial) }}>
                          <span className={styles.statsBarNumber}>{fmt(initial)}</span>
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {latest !== null ? (
                    <div className={styles.statsMetricRow}>
                      <div className={styles.statsBar}>
                        <div className={`${styles.statsBarValue} ${styles.statsBarValueLatest}`} style={{ width: pct(latest) }}>
                          <span className={styles.statsBarNumber}>{fmt(latest)}</span>
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {current !== null ? (
                    <div className={styles.statsMetricRow}>
                      <div className={styles.statsBar}>
                        <div className={`${styles.statsBarValue} ${styles.statsBarValueCurrent}`} style={{ width: pct(current) }}>
                          <span className={styles.statsBarNumber}>{fmt(current)}</span>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            );
            })}
            </div>
          </div>
        </>
      ) : sidebarTab === "history" ? (
        <div className={styles.tourLabelAnchor}>
          {showTourLabels ? (
            <TourLabel
              label="Edit history"
              title="Inspect and remove recorded edits"
              description="Every recorded action is grouped by feature so you can understand how the current state was built."
              details={[
                "Deleting an older history item also removes later entries for the same feature.",
                "That cascade happens because later edits depend on the earlier state.",
              ]}
              placement="top-left"
            />
          ) : null}
          <HistoryPanel
            history={history}
            formatHistoryAction={formatHistoryAction}
            formatHistoryDetail={formatHistoryDetail}
            onDeleteHistoryEntry={onDeleteHistoryEntry}
          />
        </div>
      ) : (
        <div className={styles.tourLabelAnchor}>
          {showTourLabels ? (
            <TourLabel
              label="Feature index"
              title="See the dataset behind each feature"
              description="This tab lists all features and shows a compact view of their distribution in the training data."
              details={[
                "Continuous features use mini histograms.",
                "Categorical features use per-category bars.",
                "Click a card to jump the editor to that feature.",
              ]}
              placement="top-left"
            />
          ) : null}
          <FeatureDistributionsPanel
            shapes={shapes}
            trainData={trainData}
            activeFeatureKey={activeFeatureKey}
            activeKnots={activeKnots}
            selectedPointIndices={selectedPointIndices}
            activeFeatureCategories={activeFeatureCategories}
            onSelectFeature={onSelectFeature}
          />
        </div>
      )}
    </div>
  );
}
