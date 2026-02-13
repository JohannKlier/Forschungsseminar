import { Dispatch, SetStateAction } from "react";
import styles from "../page.module.css";
import { FeatureCurve, StatItem } from "../types";

type HistoryEntry = { featureKey: string; action: string; ts: number; changes: { x: number; before?: number; after?: number; delta?: number }[] };

type Props = {
  sidebarTab: "edit" | "history";
  setSidebarTab: Dispatch<SetStateAction<"edit" | "history">>;
  displayLabel: string;
  partial: FeatureCurve;
  onUndo: () => void;
  canUndo: boolean;
  onRedo: () => void;
  canRedo: boolean;
  stats: StatItem[] | null;
  history: HistoryEntry[];
  formatHistoryAction: (action: string) => string;
  formatHistoryDetail: (entryIndex: number) => string | null;
  onDeleteHistoryEntry: (index: number) => void;
  onSave: () => void;
};

export default function SidebarPanel({
  sidebarTab,
  setSidebarTab,
  displayLabel,
  partial,
  onUndo,
  canUndo,
  onRedo,
  canRedo,
  stats,
  history,
  formatHistoryAction,
  formatHistoryDetail,
  onDeleteHistoryEntry,
  onSave,
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
            <p className={styles.settingsLabel}>Selected feature</p>
            <div className={styles.settingsItem}>
              <span className={styles.settingsHint}>Feature</span>
              <span className={styles.settingsValue}>{displayLabel}</span>
            </div>
          </div>
          <div className={styles.settingsSection}>
            <p className={styles.settingsLabel}>Edits</p>
            <div className={styles.undoRow}>
              <button type="button" className={styles.undoButton} onClick={onUndo} disabled={!canUndo}>
                ← Undo
              </button>
              <button type="button" className={styles.undoButton} onClick={onRedo} disabled={!canRedo}>
                Redo →
              </button>
              <button className={styles.undoButton} onClick={onSave} disabled={!partial}>
                Save edits
              </button>
            </div>
          </div>
          <div className={styles.settingsSection}>
            <p className={styles.settingsLabel}>Stats</p>
            {stats?.map((stat) => {
            if (stat.kind === "value") {
              return (
                <div key={stat.label} className={styles.settingsItem}>
                  <span className={styles.settingsHint}>{stat.label}</span>
                  <span className={styles.settingsValue}>{stat.value}</span>
                </div>
              );
            }
            const base = stat.base ?? null;
            const current = stat.current ?? null;
            const format = stat.format ?? "0.000";
            const digits = format.includes("0.00") ? 2 : 3;
            const baseLabel = Number.isFinite(base) ? (base as number).toFixed(digits) : "—";
            const currentLabel = Number.isFinite(current) ? (current as number).toFixed(digits) : "—";
            const maxVal = Math.max(Math.abs(base ?? 0), Math.abs(current ?? 0), 1e-6);
            const baseWidth = Number.isFinite(base) ? `${(Math.abs(base as number) / maxVal) * 100}%` : "0%";
            const currentWidth = Number.isFinite(current) ? `${(Math.abs(current as number) / maxVal) * 100}%` : "0%";
            return (
              <div key={stat.label} className={styles.statsBarRow}>
                <div className={styles.statsBarHeader}>
                  <span className={styles.settingsHint}>{stat.label}</span>
                </div>
                <div className={styles.statsBarTrack}>
                  <div className={styles.statsBar}>
                    <div className={styles.statsBarValue} style={{ width: baseWidth }}>
                      <span className={styles.statsBarNumber}>{baseLabel}</span>
                    </div>
                  </div>
                  <div className={styles.statsBar}>
                    <div className={`${styles.statsBarValue} ${styles.statsBarValueCurrent}`} style={{ width: currentWidth }}>
                      <span className={styles.statsBarNumber}>{currentLabel}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
            })}
          </div>
        </>
      ) : (
        <div className={styles.historyScroll}>
          <p className={styles.settingsLabel}>History</p>
          <div className={styles.historyList}>
            {history.length ? (
              [...history].reverse().map((entry, reversedIndex) => {
                const entryIndex = history.length - 1 - reversedIndex;
                return (
                  <div key={`${entry.ts}-${entry.featureKey}-${entry.action}`} className={styles.historyItem}>
                    <div className={styles.historyText}>
                      <span className={styles.settingsHint}>{entry.featureKey}</span>
                      <div className={styles.historyActionRow}>
                        <span className={styles.settingsValue}>{formatHistoryAction(entry.action)}</span>
                        <button
                          type="button"
                          className={styles.historyDeleteButton}
                          onClick={() => onDeleteHistoryEntry(entryIndex)}
                          aria-label="Delete history entry"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                    {formatHistoryDetail(entryIndex) ? (
                      <span className={styles.historyDetail}>{formatHistoryDetail(entryIndex)}</span>
                    ) : null}
                  </div>
                );
              })
            ) : (
              <p className={styles.settingsHint}>No actions yet.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
