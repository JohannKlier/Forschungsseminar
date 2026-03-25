type HistoryEntry = { featureKey: string; action: string; ts: number; changes: { x: number; before?: number; after?: number; delta?: number }[] };

type Params = {
  history: HistoryEntry[];
};

// Encapsulates sidebar-specific formatting of history entries.
export const useSidebarActions = ({ history }: Params) => {
  // Translate internal action keys into short labels for the history list.
  const formatHistoryAction = (action: string) => {
    switch (action) {
      case "drag-end":
        return "Drag";
      case "align-left":
        return "Align left";
      case "align-center":
        return "Align center";
      case "align-right":
        return "Align right";
      case "cat-zero":
        return "Set to zero";
      case "interpolate":
        return "Interpolate line";
      case "monotonic-increasing":
        return "Mono ↑";
      case "monotonic-decreasing":
        return "Mono ↓";
      case "add-points":
        return "Add points between";
      case "cat-edit":
        return "Edit category";
      default:
        return action.replace(/-/g, " ");
    }
  };

  // Summarize a history item by showing the changed point range and key action values.
  const formatHistoryDetail = (entryIndex: number) => {
    const entry = history[entryIndex];
    if (!entry) return null;
    const changes = entry.changes ?? [];
    if (!changes.length) return null;
    const xs = changes.map((c) => c.x);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const rangeLabel = `Range ${minX.toFixed(2)}–${maxX.toFixed(2)} (${changes.length} pts)`;
    if (entry.action === "drag-end") {
      const deltas = changes.map((c) => (c.delta != null ? c.delta : (c.after ?? 0) - (c.before ?? 0)));
      const avgDelta = deltas.reduce((s, v) => s + v, 0) / Math.max(1, deltas.length);
      const sign = avgDelta >= 0 ? "+" : "";
      return `${rangeLabel} ${sign}${avgDelta.toFixed(3)}`;
    }
    if (entry.action === "cat-zero") {
      return rangeLabel;
    }
    if (entry.action.startsWith("align-")) {
      const targetVals = changes.map((c) => c.after ?? 0);
      const avg = targetVals.reduce((s, v) => s + v, 0) / Math.max(1, targetVals.length);
      return `${rangeLabel} → ${avg.toFixed(3)}`;
    }
    if (entry.action === "interpolate") {
      const start = changes.find((c) => c.x === minX);
      const end = changes.find((c) => c.x === maxX);
      const startY = start?.after ?? 0;
      const endY = end?.after ?? 0;
      return `${rangeLabel} from ${startY.toFixed(3)} to ${endY.toFixed(3)}`;
    }
    if (entry.action.startsWith("monotonic")) {
      return rangeLabel;
    }
    return rangeLabel;
  };

  return {
    formatHistoryAction,
    formatHistoryDetail,
  };
};
