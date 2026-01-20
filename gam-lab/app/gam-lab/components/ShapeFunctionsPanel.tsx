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
  featureImportances?: Record<string, number>;
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
  featureImportances,
}: Props) {
  const catDragStartRef = useRef<KnotSet | null>(null);
  const catPendingRef = useRef<KnotSet | null>(null);
  const dragStartRef = useRef<KnotSet | null>(null);
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
              {result.partials.map((p, idx) => {
                const importance = featureImportances?.[p.key];
                const importanceLabel = Number.isFinite(importance) ? ` - ${importance!.toFixed(3)}` : "";
                return (
                  <option key={p.key} value={idx}>
                    {p.categories && p.categories.length ? "Cat • " : "Cont • "}
                    {p.label || p.key || `x${idx + 1}`}
                    {importanceLabel}
                  </option>
                );
              })}
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
          />
        );
      })()}
    </div>
  );
}
