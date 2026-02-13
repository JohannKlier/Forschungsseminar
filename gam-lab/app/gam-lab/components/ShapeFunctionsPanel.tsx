import { Dispatch, SetStateAction, useMemo, useRef, useState } from "react";
import VisxShapeEditor from "./VisxShapeEditor";
import styles from "../page.module.css";
import { KnotSet, TrainResponse } from "../types";
import CategoricalShapePlot from "./CategoricalShapePlot";

type Props = {
  result: TrainResponse;
  baselineKnots: Record<string, KnotSet>;
  knots: KnotSet;
  setKnots: Dispatch<SetStateAction<KnotSet>>;
  knotEdits: Record<string, KnotSet>;
  setKnotEdits: Dispatch<SetStateAction<Record<string, KnotSet>>>;
  selectedKnots: number[];
  setSelectedKnots: Dispatch<SetStateAction<number[]>>;
  activePartialIdx: number;
  setActivePartialIdx: Dispatch<SetStateAction<number>>;
  onRecordAction: (featureKey: string, before: KnotSet, after: KnotSet, action?: string) => void;
  onCommitEdits: (featureKey: string, next: KnotSet) => void;
  applyMonotonic: (direction: "increasing" | "decreasing") => void;
  addPointsInSelection: () => void;
  smoothAmount: number;
  setSmoothAmount: Dispatch<SetStateAction<number>>;
  smoothingMode: boolean;
  setSmoothingMode: Dispatch<SetStateAction<boolean>>;
  smoothingRangeMax: number;
  setSmoothingRangeMax: Dispatch<SetStateAction<number>>;
  smoothingNeighbors: number;
  setSmoothingNeighbors: Dispatch<SetStateAction<number>>;
  smoothingRate: number;
  setSmoothingRate: Dispatch<SetStateAction<number>>;
  smoothingStepPerSec: number;
  setSmoothingStepPerSec: Dispatch<SetStateAction<number>>;
};

export default function ShapeFunctionsPanel({
  result,
  baselineKnots,
  knots,
  setKnots,
  knotEdits,
  setKnotEdits,
  selectedKnots,
  setSelectedKnots,
  activePartialIdx,
  setActivePartialIdx,
  onRecordAction,
  onCommitEdits,
  applyMonotonic,
  addPointsInSelection,
  smoothAmount,
  setSmoothAmount,
  smoothingMode,
  setSmoothingMode,
  smoothingRangeMax,
  setSmoothingRangeMax,
  smoothingNeighbors,
  setSmoothingNeighbors,
  smoothingRate,
  setSmoothingRate,
  smoothingStepPerSec,
  setSmoothingStepPerSec,
}: Props) {
  const catDragStartRef = useRef<KnotSet | null>(null);
  const catPendingRef = useRef<KnotSet | null>(null);
  const dragStartRef = useRef<KnotSet | null>(null);
  const smoothBaseRef = useRef<KnotSet | null>(null);
  const smoothingActiveRef = useRef(false);
  const [interactionMode, setInteractionMode] = useState<"select" | "zoom">("select");
  const partial = useMemo(() => result.partials[activePartialIdx] ?? result.partials[0] ?? null, [result, activePartialIdx]);

  const catRanges = useMemo(() => {
    const ranges: Record<string, { min: number; max: number }> = {};
    const allVals: number[] = [];
    result.partials.forEach((p) => {
      if (!p.categories || !p.categories.length) return;
      const baseKnots = baselineKnots[p.key] ?? { x: p.editableX ?? [], y: p.editableY ?? [] };
      (baseKnots.y ?? []).forEach((v) => {
        if (Number.isFinite(v)) allVals.push(v as number);
      });
    });
    const fallback = { min: -1, max: 1 };
    const minValRaw = allVals.length ? Math.min(...allVals) : fallback.min;
    const maxValRaw = allVals.length ? Math.max(...allVals) : fallback.max;
    const minVal = Math.min(minValRaw, 0);
    const maxVal = Math.max(maxValRaw, 0);
    const span = maxVal - minVal || Math.max(Math.abs(minVal), Math.abs(maxVal), 1);
    const pad = Math.max(0.1 * span, 0.05 * Math.max(Math.abs(maxVal), Math.abs(minVal), 1));
    const shared = { min: minVal - pad, max: maxVal + pad };
    result.partials.forEach((p) => {
      if (p.categories && p.categories.length) {
        ranges[p.key] = shared;
      }
    });
    return ranges;
  }, [result.partials, baselineKnots]);
  const handleKnotChange = (next: KnotSet) => {
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    setSelectedKnots((prev) => prev.filter((idx) => idx < next.x.length));
  };
  const handleDragStart = () => {
    dragStartRef.current = { x: [...knots.x], y: [...knots.y] };
  };
  const handleDragEnd = (next: KnotSet) => {
    const start = dragStartRef.current;
    const compare = start ?? knots;
    const changed = next.x.some((v, i) => v !== compare.x[i]) || next.y.some((v, i) => v !== compare.y[i]);
    if (changed) {
      onRecordAction(partial.key, start ?? knots, next, "drag-end");
    }
    onCommitEdits(partial.key, next);
    dragStartRef.current = null;
  };
  const smoothSelection = (amount: number) => {
    if (!partial || partial.categories?.length) return;
    const indices = selectedKnots.filter((idx) => idx >= 0 && idx < knots.x.length);
    if (indices.length < 2) return;
    const indexSet = new Set(indices);
    const minIdx = Math.min(...indices);
    const maxIdx = Math.max(...indices);
    if (minIdx === maxIdx) return;
    const base = smoothBaseRef.current ?? knotEdits[partial.key] ?? knots;
    const radius = Math.max(1, Math.round(smoothingRangeMax * amount));
    const next: KnotSet = { x: [...base.x], y: [...base.y] };
    for (let i = minIdx; i <= maxIdx; i += 1) {
      if (!indexSet.has(i)) continue;
      const current = base.y[i];
      if (!Number.isFinite(current)) continue;
      let wSum = 0;
      let vSum = 0;
      const start = Math.max(minIdx, i - radius);
      const end = Math.min(maxIdx, i + radius);
      const sigma = Math.max(1e-3, radius / 2);
      for (let j = start; j <= end; j += 1) {
        if (!indexSet.has(j)) continue;
        const v = base.y[j];
        if (!Number.isFinite(v)) continue;
        const dist = Math.abs(j - i);
        const weight = Math.exp(-(dist * dist) / (2 * sigma * sigma));
        wSum += weight;
        vSum += v * weight;
      }
      if (wSum <= 0) continue;
      const avg = vSum / wSum;
      next.y[i] = current + (avg - current) * amount;
    }
    const changed = next.y.some((v, i) => v !== base.y[i]);
    if (!changed) return;
    setKnots(next);
    setKnotEdits((prev) => ({ ...prev, [partial.key]: next }));
    if (!smoothingActiveRef.current) {
      onRecordAction(partial.key, base, next, "smooth-selection");
      onCommitEdits(partial.key, next);
    }
  };
  const beginSmoothing = () => {
    if (!partial || partial.categories?.length) return;
    smoothingActiveRef.current = true;
    smoothBaseRef.current = knotEdits[partial.key] ?? knots;
  };
  const endSmoothing = () => {
    if (!partial || partial.categories?.length) return;
    if (!smoothingActiveRef.current) return;
    const base = smoothBaseRef.current ?? knotEdits[partial.key] ?? knots;
    const current = knotEdits[partial.key] ?? knots;
    const changed = current.y.some((v, i) => v !== base.y[i]);
    if (changed) {
      onRecordAction(partial.key, base, current, "smooth-selection");
      onCommitEdits(partial.key, current);
    }
    smoothingActiveRef.current = false;
    smoothBaseRef.current = null;
  };
  if (!partial) return null;

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <p className={styles.panelEyebrow}>Shape functions</p>
        <div className={styles.panelControlRow}>
          <div className={styles.featureHeaderRow}>
            <button
              type="button"
              className={styles.navButtonInline}
              onClick={() => setActivePartialIdx((prev) => (prev - 1 + result.partials.length) % result.partials.length)}
              aria-label="Previous feature"
            >
              ‹
            </button>
            <select
              className={styles.featureSelect}
              value={activePartialIdx}
              onChange={(event) => setActivePartialIdx(Number(event.target.value))}
              aria-label="Feature"
            >
              {result.partials.map((p, idx) => (
                <option key={p.key} value={idx}>
                  {p.categories && p.categories.length ? "Cat • " : "Cont • "}
                  {p.label || p.key || `x${idx + 1}`}
                </option>
              ))}
            </select>
            <button
              type="button"
              className={styles.navButtonInline}
              onClick={() => setActivePartialIdx((prev) => (prev + 1) % result.partials.length)}
              aria-label="Next feature"
            >
              ›
            </button>
          </div>
          <div className={styles.panelActions}>
            <div className={styles.panelToggle}>
              <button
                type="button"
                className={`${styles.panelToggleButton} ${interactionMode === "select" ? styles.panelToggleButtonActive : ""}`}
                onClick={() => setInteractionMode("select")}
              >
                ◎
              </button>
              <button
                type="button"
                className={`${styles.panelToggleButton} ${interactionMode === "zoom" ? styles.panelToggleButtonActive : ""}`}
                onClick={() => setInteractionMode("zoom")}
              >
                ⛶
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.actionsScroll}>
        <div className={styles.actionsGroup}>
          <div className={styles.actionsStack}>
            <button
              className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
              type="button"
              disabled={selectedKnots.length === 0}
              aria-label="Align selection"
              onClick={() => {
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
              }}
            >
              ≡
            </button>
              {!partial.categories?.length ? (
                <button
                  className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                  type="button"
                  disabled={selectedKnots.length < 2}
                  aria-label="Interpolate line"
                  onClick={() => {
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
                }}
                >
                  ↔
                </button>
              ) : null}
              {partial.categories?.length ? (
                <button
                  className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                  type="button"
                  disabled={selectedKnots.length === 0}
                  aria-label="Set to zero"
                  onClick={() => {
                  if (!selectedKnots.length) return;
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
                }}
                >
                  0
                </button>
              ) : null}
            <div className={styles.actionsRow}>
              {!partial.categories?.length ? (
                <>
                  <button
                    className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                    type="button"
                    disabled={selectedKnots.length === 0}
                    aria-label="Monotonic increase"
                    onClick={() => {
                      applyMonotonic("increasing");
                    }}
                  >
                    ↑
                  </button>
                  <button
                    className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                    type="button"
                    disabled={selectedKnots.length === 0}
                    aria-label="Monotonic decrease"
                    onClick={() => {
                      applyMonotonic("decreasing");
                    }}
                  >
                    ↓
                  </button>
                </>
              ) : null}
            </div>
            {!partial.categories?.length ? (
              <div className={styles.actionControl}>
                <span className={styles.actionControlLabel}>Smooth amount: {smoothAmount.toFixed(2)}</span>
                <input
                  className={styles.actionRange}
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={smoothAmount}
                  onPointerDown={beginSmoothing}
                  onPointerUp={endSmoothing}
                  onPointerLeave={endSmoothing}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    setSmoothAmount(next);
                    smoothSelection(next);
                  }}
                  aria-label="Smooth amount"
                />
              </div>
            ) : null}
            {!partial.categories?.length ? (
              <div className={styles.actionControl}>
                <span className={styles.actionControlLabel}>Smooth range: {smoothingRangeMax}</span>
                <input
                  className={styles.actionRange}
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={smoothingRangeMax}
                  onChange={(event) => setSmoothingRangeMax(Number(event.target.value))}
                  aria-label="Smooth range"
                />
              </div>
            ) : null}
            {!partial.categories?.length ? (
              <div className={styles.actionControl}>
                <span className={styles.actionControlLabel}>Neighbors: {smoothingNeighbors}</span>
                <input
                  className={styles.actionRange}
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={smoothingNeighbors}
                  onChange={(event) => setSmoothingNeighbors(Number(event.target.value))}
                  aria-label="Smoothing neighbors"
                />
              </div>
            ) : null}
            {!partial.categories?.length ? (
              <div className={styles.actionControl}>
                <span className={styles.actionControlLabel}>Smoothing rate: {smoothingRate.toFixed(2)}</span>
                <input
                  className={styles.actionRange}
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={smoothingRate}
                  onChange={(event) => setSmoothingRate(Number(event.target.value))}
                  aria-label="Smoothing rate"
                />
              </div>
            ) : null}
            {!partial.categories?.length ? (
              <div className={styles.actionControl}>
                <span className={styles.actionControlLabel}>Step/sec: {smoothingStepPerSec.toFixed(2)}</span>
                <input
                  className={styles.actionRange}
                  type="range"
                  min={1}
                  max={500}
                  step={1}
                  value={smoothingStepPerSec}
                  onChange={(event) => setSmoothingStepPerSec(Number(event.target.value))}
                  aria-label="Smoothing step per second"
                />
              </div>
            ) : null}
            {!partial.categories?.length ? (
              <button
                className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                type="button"
                aria-label={smoothingMode ? "Disable smoothing mode" : "Enable smoothing mode"}
                onClick={() => setSmoothingMode((prev) => !prev)}
              >
                {smoothingMode ? "◎" : "○"}
              </button>
            ) : null}
            {!partial.categories?.length ? (
              <button
                className={`${styles.actionButton} ${styles.actionButtonSquare} ${styles.actionIconButton}`}
                type="button"
                disabled={selectedKnots.length < 2}
                aria-label="Add points between"
                onClick={() => {
                  addPointsInSelection();
                }}
              >
                ＋
              </button>
            ) : null}
          </div>
        </div>
      </div>
      {(() => {
        const p = result.partials[activePartialIdx];
        if (!p) return null;
        const label = p.label || p.key || `x${activePartialIdx + 1}`;
        const baseKnots = knotEdits[p.key] ?? baselineKnots[p.key] ?? { x: p.editableX ?? [], y: p.editableY ?? [] };
        if (p.categories && p.categories.length) {
          const catRange = catRanges[p.key];
                return (
          <CategoricalShapePlot
            categories={p.categories}
            knots={baseKnots}
            baseline={baselineKnots[p.key]}
            title={label}
            fixedRange={catRange}
            interactionMode={interactionMode}
            selectedIdxs={selectedKnots}
            onSelect={(nextSel) => setSelectedKnots(nextSel)}
            onValueChange={(idxSel, value) => {
              const current = knotEdits[p.key] ?? knots;
              const next = { x: [...current.x], y: [...current.y] };
              if (idxSel < 0 || idxSel >= next.y.length) return;
              if (next.y[idxSel] === value) return;
              next.y[idxSel] = value;
              catPendingRef.current = next;
              setKnots(next);
              setKnotEdits((prev) => ({ ...prev, [p.key]: next }));
              setSelectedKnots([idxSel]);
            }}
            onMultiValueChange={(indices, values) => {
              const current = knotEdits[p.key] ?? knots;
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
              setKnotEdits((prev) => ({ ...prev, [p.key]: next }));
              setSelectedKnots(filtered);
            }}
            onDragStart={() => {
              const current = knotEdits[p.key] ?? knots;
              catDragStartRef.current = { x: [...current.x], y: [...current.y] };
            }}
              onDragEnd={() => {
                const start = catDragStartRef.current;
                const current = catPendingRef.current ?? knotEdits[p.key] ?? knots;
                if (start) {
                  onRecordAction(p.key, start, current, "cat-edit");
                }
                onCommitEdits(p.key, current);
                catDragStartRef.current = null;
                catPendingRef.current = null;
              }}
            />
          );
        }

        return (
          <VisxShapeEditor
            knots={baseKnots}
            baseline={baselineKnots[p.key]}
            scatterX={p.scatterX}
            featureKey={p.key}
            interactionMode={interactionMode}
            selected={selectedKnots}
            onSelectionChange={(nextSel) => setSelectedKnots(nextSel)}
            onKnotChange={handleKnotChange}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            title={label}
            smoothingMode={smoothingMode}
            smoothAmount={smoothAmount}
            smoothingRangeMax={smoothingRangeMax}
            smoothingNeighbors={smoothingNeighbors}
            smoothingRate={smoothingRate}
            smoothingStepPerSec={smoothingStepPerSec}
          />
        );
      })()}
    </div>
  );
}
