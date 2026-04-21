import { Dispatch, SetStateAction, useEffect, useRef } from "react";
import { ShapeFunction, KnotSet } from "../types";
import { preserveSelectionForNextKnots } from "../lib/selection";

export type AlignSelectionMode = "left" | "center" | "right";

const areKnotsEqual = (left: KnotSet, right: KnotSet) => (
  left.x.length === right.x.length &&
  left.y.length === right.y.length &&
  left.x.every((value, index) => value === right.x[index]) &&
  left.y.every((value, index) => value === right.y[index])
);

type Params = {
  partial: ShapeFunction | null;
  knots: KnotSet;
  knotEdits: Record<string, KnotSet>;
  selectedKnots: number[];
  setKnots: Dispatch<SetStateAction<KnotSet>>;
  setKnotEdits: Dispatch<SetStateAction<Record<string, KnotSet>>>;
  setSelectedKnots: (next: number[]) => void;
  onRecordAction: (featureKey: string, before: KnotSet, after: KnotSet, action?: string) => void;
  onCommitEdits: (featureKey: string, next: KnotSet) => void;
  onInteractionStart?: () => void;
  onInteractionEnd?: () => void;
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
  onInteractionStart,
  onInteractionEnd,
}: Params) => {
  const dragStartRef = useRef<KnotSet | null>(null);
  const catDragStartRef = useRef<KnotSet | null>(null);
  const catPendingRef = useRef<KnotSet | null>(null);
  const selectionReferenceRef = useRef<{ leftY: number; rightY: number } | null>(null);
  const selectionIdentityRef = useRef<string | null>(null);
  const latestKnotsRef = useRef(knots);
  const latestKnotEditsRef = useRef(knotEdits);

  useEffect(() => {
    latestKnotsRef.current = knots;
    latestKnotEditsRef.current = knotEdits;
  }, [knots, knotEdits]);

  const getContinuousSelectionContext = (current: KnotSet, selection: number[]) => {
    const sortedSelection = [...selection]
      .filter((idx) => idx >= 0 && idx < current.x.length)
      .sort((a, b) => (current.x[a] ?? 0) - (current.x[b] ?? 0));
    if (!sortedSelection.length) return null;
    const startIdx = sortedSelection[0];
    const endIdx = sortedSelection[sortedSelection.length - 1];
    if (startIdx == null || endIdx == null) return null;
    const leftAnchorIdx = startIdx > 0 ? startIdx - 1 : null;
    const rightAnchorIdx = endIdx < current.x.length - 1 ? endIdx + 1 : null;
    return {
      sortedSelection,
      startIdx,
      endIdx,
      leftAnchorIdx,
      rightAnchorIdx,
    };
  };
  const currentContinuousKnots = partial ? (knotEdits[partial.key] ?? knots) : null;
  const currentSelectionIdentity = (() => {
    if (!partial || !currentContinuousKnots || !selectedKnots.length) return null;
    const sortedSelection = [...selectedKnots]
      .filter((idx) => idx >= 0 && idx < currentContinuousKnots.x.length)
      .sort((a, b) => (currentContinuousKnots.x[a] ?? 0) - (currentContinuousKnots.x[b] ?? 0));
    if (!sortedSelection.length) return null;
    return `${partial.key}:${sortedSelection.join(",")}`;
  })();
  useEffect(() => {
    if (currentSelectionIdentity === selectionIdentityRef.current) return;
    selectionIdentityRef.current = currentSelectionIdentity;
    if (!partial || !currentContinuousKnots || !selectedKnots.length) {
      selectionReferenceRef.current = null;
      return;
    }
    const sortedSelection = [...selectedKnots]
      .filter((idx) => idx >= 0 && idx < currentContinuousKnots.x.length)
      .sort((a, b) => (currentContinuousKnots.x[a] ?? 0) - (currentContinuousKnots.x[b] ?? 0));
    const selectedYs = sortedSelection.map((idx) => currentContinuousKnots.y[idx] ?? 0);
    if (!selectedYs.length) {
      selectionReferenceRef.current = null;
      return;
    }
    selectionReferenceRef.current = {
      leftY: selectedYs[0] ?? 0,
      rightY: selectedYs[selectedYs.length - 1] ?? 0,
    };
  }, [currentContinuousKnots, currentSelectionIdentity, partial, selectedKnots]);

  const applyKnotUpdate = (featureKey: string, next: KnotSet) => {
    const current = latestKnotEditsRef.current[featureKey] ?? latestKnotsRef.current;
    if (areKnotsEqual(current, next)) return;

    latestKnotsRef.current = next;
    latestKnotEditsRef.current = { ...latestKnotEditsRef.current, [featureKey]: next };

    setKnots((prev) => (areKnotsEqual(prev, next) ? prev : next));
    setKnotEdits((prev) => {
      const previous = prev[featureKey];
      if (previous && areKnotsEqual(previous, next)) return prev;
      return { ...prev, [featureKey]: next };
    });
    onCommitEdits(featureKey, next);
  };

  const handleKnotChange = (next: KnotSet) => {
    if (!partial) return;
    applyKnotUpdate(partial.key, next);
    const filtered = selectedKnots.filter((idx) => idx < next.x.length);
    if (filtered.length !== selectedKnots.length) setSelectedKnots(filtered);
  };

  const handleDragStart = () => {
    onInteractionStart?.();
    dragStartRef.current = { x: [...knots.x], y: [...knots.y] };
  };

  const handleDragEnd = (next: KnotSet) => {
    onInteractionEnd?.();
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

  const alignSelection = (mode: AlignSelectionMode) => {
    if (!partial) return;
    const current = knotEdits[partial.key] ?? knots;
    if (!selectedKnots.length) return;
    const context = getContinuousSelectionContext(current, selectedKnots);
    if (!context) return;
    const { sortedSelection, leftAnchorIdx, rightAnchorIdx } = context;
    const reference = selectionReferenceRef.current;
    const targetY = (() => {
      if (mode === "left") {
        return reference?.leftY ?? current.y[leftAnchorIdx ?? sortedSelection[0] ?? -1] ?? 0;
      }
      if (mode === "right") {
        return reference?.rightY ?? current.y[rightAnchorIdx ?? sortedSelection[sortedSelection.length - 1] ?? -1] ?? 0;
      }
      const leftReference = reference?.leftY ?? current.y[leftAnchorIdx ?? sortedSelection[0] ?? -1] ?? 0;
      const rightReference = reference?.rightY ?? current.y[rightAnchorIdx ?? sortedSelection[sortedSelection.length - 1] ?? -1] ?? 0;
      return (leftReference + rightReference) / 2;
    })();
    const selectedSet = new Set(sortedSelection);
    const next = { x: [...current.x], y: current.y.map((val, idx) => (selectedSet.has(idx) ? targetY : val)) };
    const changed = next.y.some((v, i) => v !== current.y[i]);
    if (changed) onRecordAction(partial.key, current, next, `align-${mode}`);
    applyKnotUpdate(partial.key, next);
    setSelectedKnots(preserveSelectionForNextKnots(current, next, selectedKnots));
  };

  const interpolateSelection = () => {
    if (!partial) return;
    if (selectedKnots.length < 2) return;
    const current = knotEdits[partial.key] ?? knots;
    const context = getContinuousSelectionContext(current, selectedKnots);
    if (!context) return;
    const { sortedSelection, startIdx, endIdx, leftAnchorIdx, rightAnchorIdx } = context;
    if (startIdx == null || endIdx == null || startIdx === endIdx) return;
    const sel = current.x
      .map((_, idx) => idx)
      .filter((idx) => idx >= startIdx && idx <= endIdx);
    if (sel.length < 2) return;
    const reference = selectionReferenceRef.current;
    const y0 = reference?.leftY ?? current.y[leftAnchorIdx ?? sortedSelection[0] ?? sel[0]] ?? 0;
    const y1 = reference?.rightY ?? current.y[rightAnchorIdx ?? sortedSelection[sortedSelection.length - 1] ?? sel[sel.length - 1]] ?? 0;
    const nextY = [...current.y];
    sel.forEach((idx, pos) => {
      const t = sel.length === 1 ? 0 : pos / (sel.length - 1);
      nextY[idx] = y0 * (1 - t) + y1 * t;
    });
    const changed = nextY.some((v, i) => v !== current.y[i]);
    if (!changed) return;
    const next = { x: [...current.x], y: nextY };
    onRecordAction(partial.key, current, next, "interpolate");
    applyKnotUpdate(partial.key, next);
    setSelectedKnots(preserveSelectionForNextKnots(current, next, selectedKnots));
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
    applyKnotUpdate(partial.key, next);
    setSelectedKnots(preserveSelectionForNextKnots(current, next, selectedKnots));
  };

  const handleCatValueChange = (featureKey: string, idxSel: number, value: number) => {
    const current = knotEdits[featureKey] ?? knots;
    const next = { x: [...current.x], y: [...current.y] };
    if (idxSel < 0 || idxSel >= next.y.length) return;
    if (next.y[idxSel] === value) return;
    next.y[idxSel] = value;
    catPendingRef.current = next;
    applyKnotUpdate(featureKey, next);
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
    applyKnotUpdate(featureKey, next);
    setSelectedKnots(filtered);
  };

  const handleCatDragStart = (featureKey: string) => {
    onInteractionStart?.();
    const current = knotEdits[featureKey] ?? knots;
    catDragStartRef.current = { x: [...current.x], y: [...current.y] };
  };

  const handleCatDragEnd = (featureKey: string) => {
    onInteractionEnd?.();
    const start = catDragStartRef.current;
    const current = catPendingRef.current ?? knotEdits[featureKey] ?? knots;
    if (start) {
      onRecordAction(featureKey, start, current, "cat-edit");
    }
    onCommitEdits(featureKey, current);
    catDragStartRef.current = null;
    catPendingRef.current = null;
  };

  const handleSmoothStart = () => {
    onInteractionStart?.();
  };

  const handleSmoothEnd = (featureKey: string, start: KnotSet, end: KnotSet) => {
    onInteractionEnd?.();
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
    handleSmoothStart,
    handleSmoothEnd,
  };
};
