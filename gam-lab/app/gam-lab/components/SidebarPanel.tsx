import { Dispatch, SetStateAction, useEffect, useRef } from "react";
import styles from "../page.module.css";
import { StatItem } from "../types";

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

type Props = {
  sidebarTab: "edit" | "history";
  setSidebarTab: Dispatch<SetStateAction<"edit" | "history">>;
  stats: StatItem[] | null;
  history: HistoryEntry[];
  formatHistoryAction: (action: string) => string;
  formatHistoryDetail: (entryIndex: number) => string | null;
  onDeleteHistoryEntry: (index: number) => void;
};

export default function SidebarPanel({
  sidebarTab,
  setSidebarTab,
  stats,
  history,
  formatHistoryAction,
  formatHistoryDetail,
  onDeleteHistoryEntry,
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
      </div>
      {sidebarTab === "edit" ? (
        <>
          <div className={styles.settingsSection}>
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
        </>
      ) : (
        <HistoryPanel
          history={history}
          formatHistoryAction={formatHistoryAction}
          formatHistoryDetail={formatHistoryDetail}
          onDeleteHistoryEntry={onDeleteHistoryEntry}
        />
      )}
    </div>
  );
}
