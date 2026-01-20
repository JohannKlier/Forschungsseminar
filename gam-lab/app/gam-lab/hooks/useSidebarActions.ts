import { Dispatch, SetStateAction } from "react";
import { FeatureCurve, KnotSet } from "../types";

type HistoryEntry = { featureKey: string; action: string; ts: number; changes: { x: number; before?: number; after?: number; delta?: number }[] };

type Params = {
  partial: FeatureCurve | null;
  knotEdits: Record<string, KnotSet>;
  knots: KnotSet;
  selectedKnots: number[];
  setKnots: Dispatch<SetStateAction<KnotSet>>;
  setKnotEdits: Dispatch<SetStateAction<Record<string, KnotSet>>>;
  setSelectedKnots: Dispatch<SetStateAction<number[]>>;
  recordAction: (featureKey: string, before: KnotSet, after: KnotSet, action?: string) => void;
  commitEdits: (featureKey: string, next: KnotSet) => void;
  history: HistoryEntry[];
  baselineKnots: Record<string, KnotSet>;
};

// Encapsulates sidebar-specific formatting and knot editing actions.
export const useSidebarActions = ({
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
}: Params) => {
  // Translate internal action keys into short labels for the history list.
  const formatHistoryAction = (action: string) => {
    switch (action) {
      case "drag-end":
        return "Drag";
      case "align":
        return "Align selection";
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
    if (entry.action === "align") {
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

  // Enforce monotonicity across the selected knots.
  const applyMonotonic = (direction: "increasing" | "decreasing") => {
    if (!partial) return;
    const current = knotEdits[partial.key] ?? knots;
    const targets = selectedKnots.length
      ? current.x.map((x, i) => ({ x, y: current.y[i] ?? 0, i })).filter((p) => selectedKnots.includes(p.i))
      : [];
    if (!targets.length) return;
    const sorted = targets.slice().sort((a, b) => a.x - b.x);
    const nextY = [...current.y];
    if (direction === "increasing") {
      let last = -Infinity;
      sorted.forEach((p) => {
        last = Math.max(last, p.y);
        nextY[p.i] = last;
      });
    } else {
      let last = Infinity;
      sorted.forEach((p) => {
        last = Math.min(last, p.y);
        nextY[p.i] = last;
      });
    }
    const changed = nextY.some((v, i) => v !== current.y[i]);
    if (!changed) return;
    const next = { x: [...current.x], y: nextY };
    recordAction(partial.key, current, next, `monotonic-${direction}`);
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
    commitEdits(partial.key, next);
  };

  // Insert midpoints between selected knots without resampling the full curve.
  const addPointsInSelection = () => {
    if (!partial) return;
    const current = knotEdits[partial.key] ?? knots;
    const selected = selectedKnots.slice();
    if (selected.length < 2) return;
    const pairs = selected
      .map((i) => ({ x: current.x[i], y: current.y[i] ?? 0, i }))
      .sort((a, b) => a.x - b.x);
    const additions = [];
    for (let i = 0; i < pairs.length - 1; i += 1) {
      const a = pairs[i];
      const b = pairs[i + 1];
      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2;
      additions.push({ x: midX, y: midY });
    }
    if (!additions.length) return;
    const merged = current.x
      .map((x, i) => ({ x, y: current.y[i] ?? 0 }))
      .concat(additions)
      .sort((a, b) => a.x - b.x);
    const next = { x: merged.map((p) => p.x), y: merged.map((p) => p.y) };
    recordAction(partial.key, current, next, "add-points");
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots([]);
    commitEdits(partial.key, next);
  };

  return {
    formatHistoryAction,
    formatHistoryDetail,
    applyMonotonic,
    addPointsInSelection,
  };
};
