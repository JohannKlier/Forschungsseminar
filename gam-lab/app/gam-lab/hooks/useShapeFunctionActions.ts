import { Dispatch, SetStateAction, useRef } from "react";
import { FeatureCurve, KnotSet } from "../types";

type Params = {
  partial: FeatureCurve | null;
  knots: KnotSet;
  knotEdits: Record<string, KnotSet>;
  selectedKnots: number[];
  setKnots: Dispatch<SetStateAction<KnotSet>>;
  setKnotEdits: Dispatch<SetStateAction<Record<string, KnotSet>>>;
  setSelectedKnots: Dispatch<SetStateAction<number[]>>;
  onRecordAction: (featureKey: string, before: KnotSet, after: KnotSet, action?: string) => void;
  onCommitEdits: (featureKey: string, next: KnotSet) => void;
};

export const useShapeFunctionActions = ({
  partial,
  knots,
  knotEdits,
  selectedKnots,
  setKnots,
  setKnotEdits,
  setSelectedKnots,
  onRecordAction,
  onCommitEdits,
}: Params) => {
  const dragStartRef = useRef<KnotSet | null>(null);
  const catDragStartRef = useRef<KnotSet | null>(null);
  const catPendingRef = useRef<KnotSet | null>(null);

  const handleKnotChange = (next: KnotSet) => {
    if (!partial) return;
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
  };

  const handleDragStart = () => {
    dragStartRef.current = { x: [...knots.x], y: [...knots.y] };
  };

  const handleDragEnd = (next: KnotSet) => {
    if (!partial) return;
    const start = dragStartRef.current;
    const compare = start ?? knots;
    const changed = next.x.some((v, i) => v !== compare.x[i]) || next.y.some((v, i) => v !== compare.y[i]);
    if (changed) {
      onRecordAction(partial.key, start ?? knots, next, "drag-end");
    }
    onCommitEdits(partial.key, next);
    dragStartRef.current = null;
  };

  const alignSelection = () => {
    if (!partial) return;
    const current = knotEdits[partial.key] ?? knots;
    const sel = selectedKnots.length ? selectedKnots : [];
    if (sel.length === 0) return;
    const avgY = sel.reduce((sum, idx) => sum + (current.y[idx] ?? 0), 0) / sel.length;
    const next = { x: [...current.x], y: current.y.map((val, idx) => (sel.includes(idx) ? avgY : val)) };
    const changed = next.y.some((v, i) => v !== current.y[i]);
    if (changed) onRecordAction(partial.key, current, next, "align");
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
    onCommitEdits(partial.key, next);
  };

  const interpolateSelection = () => {
    if (!partial) return;
    const current = knotEdits[partial.key] ?? knots;
    const sel = [...selectedKnots].sort((a, b) => (current.x[a] ?? 0) - (current.x[b] ?? 0));
    if (sel.length < 2) return;
    const y0 = current.y[sel[0]] ?? 0;
    const y1 = current.y[sel[sel.length - 1]] ?? 0;
    const nextY = [...current.y];
    sel.forEach((idx, pos) => {
      const t = sel.length === 1 ? 0 : pos / (sel.length - 1);
      nextY[idx] = y0 * (1 - t) + y1 * t;
    });
    const changed = nextY.some((v, i) => v !== current.y[i]);
    if (!changed) return;
    const next = { x: [...current.x], y: nextY };
    onRecordAction(partial.key, current, next, "interpolate");
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
    onCommitEdits(partial.key, next);
  };

  const setSelectionToZero = () => {
    if (!partial || !selectedKnots.length) return;
    const current = knotEdits[partial.key] ?? knots;
    const next = {
      x: [...current.x],
      y: current.y.map((val, idx) => (selectedKnots.includes(idx) ? 0 : val)),
    };
    const changed = next.y.some((v, i) => v !== current.y[i]);
    if (changed) onRecordAction(partial.key, current, next, "cat-zero");
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
    onCommitEdits(partial.key, next);
  };

  const handleCatValueChange = (featureKey: string, idxSel: number, value: number) => {
    const current = knotEdits[featureKey] ?? knots;
    const next = { x: [...current.x], y: [...current.y] };
    if (idxSel < 0 || idxSel >= next.y.length) return;
    if (next.y[idxSel] === value) return;
    next.y[idxSel] = value;
    catPendingRef.current = next;
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [featureKey]: next }));
    setSelectedKnots([idxSel]);
  };

  const handleCatMultiValueChange = (featureKey: string, indices: number[], values: Record<number, number>) => {
    const current = knotEdits[featureKey] ?? knots;
    const next = { x: [...current.x], y: [...current.y] };
    const filtered = indices.filter((idx) => idx >= 0 && idx < next.y.length);
    if (!filtered.length) return;
    let changed = false;
    filtered.forEach((idx) => {
      const nextVal = values[idx];
      if (nextVal === undefined) return;
      if (next.y[idx] !== nextVal) {
        next.y[idx] = nextVal;
        changed = true;
      }
    });
    if (!changed) return;
    catPendingRef.current = next;
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [featureKey]: next }));
    setSelectedKnots(filtered);
  };

  const handleCatDragStart = (featureKey: string) => {
    const current = knotEdits[featureKey] ?? knots;
    catDragStartRef.current = { x: [...current.x], y: [...current.y] };
  };

  const handleCatDragEnd = (featureKey: string) => {
    const start = catDragStartRef.current;
    const current = catPendingRef.current ?? knotEdits[featureKey] ?? knots;
    if (start) {
      onRecordAction(featureKey, start, current, "cat-edit");
    }
    onCommitEdits(featureKey, current);
    catDragStartRef.current = null;
    catPendingRef.current = null;
  };

  const handleSmoothEnd = (featureKey: string, start: KnotSet, end: KnotSet) => {
    const changed = end.y.some((v, i) => v !== start.y[i]);
    if (!changed) return;
    const before = { x: [...start.x], y: [...start.y] };
    const after = { x: [...end.x], y: [...end.y] };
    onRecordAction(featureKey, before, after, "smooth-selection");
    onCommitEdits(featureKey, after);
  };

  return {
    handleKnotChange,
    handleDragStart,
    handleDragEnd,
    alignSelection,
    interpolateSelection,
    setSelectionToZero,
    handleCatValueChange,
    handleCatMultiValueChange,
    handleCatDragStart,
    handleCatDragEnd,
    handleSmoothEnd,
  };
};
