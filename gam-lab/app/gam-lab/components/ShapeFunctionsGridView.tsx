import styles from "../page.module.css";
import { KnotSet, ShapeFunction } from "../types";
import { InteractionHeatmap } from "./InteractionHeatmap";

type Props = {
  shapes: ShapeFunction[];
  baselineKnots: Record<string, KnotSet>;
  knotEdits: Record<string, KnotSet>;
  onSelectFeature: (idx: number) => void;
  featureImportance: Record<string, number>;
};

type XYPoint = { x: number; y: number };

const CHART_W = 280;
const CHART_H = 120;
const PAD = { top: 8, right: 12, bottom: 20, left: 20 };

const scaleLinear = (value: number, inMin: number, inMax: number, outMin: number, outMax: number) => {
  if (!Number.isFinite(value)) return outMin;
  if (inMax === inMin) return (outMin + outMax) / 2;
  const t = (value - inMin) / (inMax - inMin);
  return outMin + t * (outMax - outMin);
};

const toPoints = (knots: KnotSet): XYPoint[] =>
  knots.x
    .map((x, idx) => ({ x, y: knots.y[idx] ?? 0 }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);

const minMax = (values: number[], fallbackMin: number, fallbackMax: number) => {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return { min: fallbackMin, max: fallbackMax };
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) { min -= 1; max += 1; }
  return { min, max };
};

const fmtVal = (v: number) => {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
};

export default function ShapeFunctionsGridView({ shapes, baselineKnots, knotEdits, onSelectFeature, featureImportance }: Props) {
  const interactionAbsMax = shapes.reduce((max, s) => {
    if (!s.editableZ) return max;
    const m = s.editableZ.flat().reduce((m2, v) => Math.max(m2, Math.abs(v)), 0);
    return Math.max(max, m);
  }, 1e-9);

  const sortedIndices = shapes
    .map((_, idx) => idx)
    .sort((a, b) => (featureImportance[shapes[b].key] ?? 0) - (featureImportance[shapes[a].key] ?? 0));

  return (
    <div className={styles.gridView}>
      {sortedIndices.map((idx) => {
        const shape = shapes[idx];
        const current =
          knotEdits[shape.key] ??
          baselineKnots[shape.key] ?? {
            x: shape.editableX ?? [],
            y: shape.editableY ?? [],
          };
        const baseline = baselineKnots[shape.key] ?? current;
        const title = shape.label || shape.key || `x${idx + 1}`;
        const importance = featureImportance[shape.key] ?? 0;

        const cardHeader = (
          <div className={styles.gridCardHeader}>
            <span className={styles.gridCardTitle}>{title}</span>
            <span className={styles.gridImportanceLabel}>{(importance * 100).toFixed(1)}%</span>
          </div>
        );

        if (shape.editableZ) {
          return (
            <button key={shape.key} type="button" className={styles.gridCard} onClick={() => onSelectFeature(idx)}>
              {cardHeader}
              <InteractionHeatmap shape={shape} width={CHART_W} height={CHART_H + 20} absMax={interactionAbsMax} />
            </button>
          );
        }

        if (shape.categories && shape.categories.length) {
          const categories = shape.categories;
          const currentY = categories.map((_, i) => current.y[i] ?? 0);
          const baselineY = categories.map((_, i) => baseline.y[i] ?? 0);
          const domain = minMax([...currentY, ...baselineY, 0], -1, 1);
          const innerW = CHART_W - PAD.left - PAD.right;
          const innerH = CHART_H - PAD.top - PAD.bottom;
          const barW = innerW / Math.max(1, categories.length);
          const yToPx = (y: number) => scaleLinear(y, domain.min, domain.max, PAD.top + innerH, PAD.top);
          const zeroY = yToPx(0);

          return (
            <button key={shape.key} type="button" className={styles.gridCard} onClick={() => onSelectFeature(idx)}>
              {cardHeader}
              <svg width={CHART_W} height={CHART_H} className={styles.gridChart} aria-label={title}>
                <line x1={PAD.left} x2={CHART_W - PAD.right} y1={zeroY} y2={zeroY} className={styles.gridZeroLine} />
                {categories.map((_, i) => {
                  const x = PAD.left + i * barW;
                  const baseVal = baselineY[i];
                  const curVal = currentY[i];
                  const baseTop = Math.min(yToPx(baseVal), zeroY);
                  const baseHeight = Math.abs(yToPx(baseVal) - zeroY);
                  const curTop = Math.min(yToPx(curVal), zeroY);
                  const curHeight = Math.abs(yToPx(curVal) - zeroY);
                  return (
                    <g key={`${shape.key}-${i}`}>
                      <rect x={x + barW * 0.18} y={baseTop} width={barW * 0.64} height={baseHeight} className={styles.gridBaselineBar} />
                      <rect x={x + barW * 0.28} y={curTop} width={barW * 0.44} height={curHeight} className={styles.gridCurrentBar} />
                    </g>
                  );
                })}
              </svg>
              <div className={styles.gridXRange}>
                <span>{categories.length} categories</span>
              </div>
            </button>
          );
        }

        const curPoints = toPoints(current);
        const basePoints = toPoints(baseline);
        const xDomain = minMax([...curPoints.map((p) => p.x), ...basePoints.map((p) => p.x)], 0, 1);
        const yDomain = minMax([...curPoints.map((p) => p.y), ...basePoints.map((p) => p.y), 0], -1, 1);
        const xToPx = (x: number) => scaleLinear(x, xDomain.min, xDomain.max, PAD.left, CHART_W - PAD.right);
        const yToPx = (y: number) => scaleLinear(y, yDomain.min, yDomain.max, CHART_H - PAD.bottom, PAD.top);
        const toPolyline = (points: XYPoint[]) => points.map((p) => `${xToPx(p.x)},${yToPx(p.y)}`).join(" ");

        return (
          <button key={shape.key} type="button" className={styles.gridCard} onClick={() => onSelectFeature(idx)}>
            {cardHeader}
            <svg width={CHART_W} height={CHART_H} className={styles.gridChart} aria-label={title}>
              <line x1={PAD.left} x2={CHART_W - PAD.right} y1={yToPx(0)} y2={yToPx(0)} className={styles.gridZeroLine} />
              {basePoints.length > 1 ? (
                <polyline points={toPolyline(basePoints)} fill="none" className={styles.gridBaselineLine} />
              ) : null}
              {curPoints.length > 1 ? (
                <polyline points={toPolyline(curPoints)} fill="none" className={styles.gridCurrentLine} />
              ) : null}
            </svg>
            <div className={styles.gridXRange}>
              <span>{fmtVal(xDomain.min)}</span>
              <span>{fmtVal(xDomain.max)}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
