import styles from "../page.module.css";
import { Models, TrainResponse } from "../types";

type Props = {
  result: TrainResponse;
  models: Models;
};

export default function PredictionFitPlot({ result, models }: Props) {
  if (result.task === "classification") {
    const preds = models.editedModel.preds;
    const labels = result.y;
    const threshold = 0.5;
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    labels.forEach((label, idx) => {
      const pred = preds[idx];
      if (!Number.isFinite(pred)) return;
      const actual = label >= 0.5;
      const predicted = pred >= threshold;
      if (actual && predicted) tp += 1;
      else if (!actual && !predicted) tn += 1;
      else if (!actual && predicted) fp += 1;
      else fn += 1;
    });
    return (
      <div className={styles.summaryPlot}>
        <p className={styles.panelEyebrow}>Prediction fit</p>
        <h3 className={styles.panelTitle}>Confusion matrix</h3>
        <svg width="100%" viewBox="0 0 700 320" preserveAspectRatio="xMidYMid meet">
          {(() => {
            const width = 700;
            const height = 320;
            const pad = { top: 26, right: 24, bottom: 36, left: 120 };
            const cellSize = 110;
            const gridW = cellSize * 2;
            const gridH = cellSize * 2;
            const originX = pad.left;
            const originY = pad.top + 10;
            const maxVal = Math.max(tp, tn, fp, fn, 1);
            const shade = (v: number) => {
              const alpha = 0.1 + 0.75 * (v / maxVal);
              return `rgba(14,165,233,${alpha.toFixed(3)})`;
            };
            const cells = [
              { label: "TN", value: tn, x: originX, y: originY },
              { label: "FP", value: fp, x: originX + cellSize, y: originY },
              { label: "FN", value: fn, x: originX, y: originY + cellSize },
              { label: "TP", value: tp, x: originX + cellSize, y: originY + cellSize },
            ];
            return (
              <>
                <rect x={0} y={0} width={width} height={height} fill="#ffffff" stroke="#edf2f7" rx={12} ry={12} />
                <text x={originX + gridW / 2} y={height - 10} fontSize={11} fill="#6b7280" textAnchor="middle">
                  Predicted
                </text>
                <text x={24} y={originY + gridH / 2} fontSize={11} fill="#6b7280" transform={`rotate(-90, 24, ${originY + gridH / 2})`}>
                  Actual
                </text>
                <text x={originX + cellSize / 2} y={originY - 8} fontSize={10} fill="#6b7280" textAnchor="middle">
                  Pred 0
                </text>
                <text x={originX + cellSize + cellSize / 2} y={originY - 8} fontSize={10} fill="#6b7280" textAnchor="middle">
                  Pred 1
                </text>
                <text x={originX - 8} y={originY + cellSize / 2} fontSize={10} fill="#6b7280" textAnchor="end">
                  Actual 0
                </text>
                <text x={originX - 8} y={originY + cellSize + cellSize / 2} fontSize={10} fill="#6b7280" textAnchor="end">
                  Actual 1
                </text>
                {cells.map((cell) => (
                  <g key={cell.label}>
                    <rect x={cell.x} y={cell.y} width={cellSize} height={cellSize} fill={shade(cell.value)} stroke="#e5e7eb" />
                    <text x={cell.x + cellSize / 2} y={cell.y + cellSize / 2 - 6} fontSize={14} fill="#111827" textAnchor="middle">
                      {cell.label}
                    </text>
                    <text x={cell.x + cellSize / 2} y={cell.y + cellSize / 2 + 14} fontSize={16} fill="#111827" textAnchor="middle">
                      {cell.value}
                    </text>
                  </g>
                ))}
              </>
            );
          })()}
        </svg>
      </div>
    );
  }

  return (
    <div className={styles.summaryPlot}>
      <p className={styles.panelEyebrow}>Prediction fit</p>
      <h3 className={styles.panelTitle}>Observed vs predicted</h3>
      <svg width="100%" viewBox="0 0 700 320" preserveAspectRatio="xMidYMid meet">
        {(() => {
          const width = 700;
          const height = 320;
          const pad = { top: 12, right: 12, bottom: 38, left: 46 };
          const usableW = width - pad.left - pad.right;
          const usableH = height - pad.top - pad.bottom;
          const trainY = result.y;
          const trainPreds = models.editedModel.preds;
          const domainVals = [
            ...trainY,
            ...trainPreds.filter((v) => Number.isFinite(v)),
          ];
          const minVal = Math.min(...domainVals);
          const maxVal = Math.max(...domainVals);
          const span = maxVal - minVal || 1;
          const xScale = (v: number) => pad.left + ((v - minVal) / span) * usableW;
          const yScale = (v: number) => pad.top + usableH - ((v - minVal) / span) * usableH;
          const tickCount = 4;
          const ticks = Array.from({ length: tickCount }, (_, i) => minVal + (i / (tickCount - 1)) * span);
          const formatTick = (v: number) => {
            const abs = Math.abs(v);
            if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
            if (abs >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
            return v.toFixed(2);
          };
          return (
            <>
              <rect x={0} y={0} width={width} height={height} fill="#ffffff" stroke="#edf2f7" rx={12} ry={12} />
              {ticks.map((t, idx) => {
                const x = xScale(t);
                const y = yScale(t);
                return (
                  <g key={`tick-${idx}`}>
                    <line x1={x} x2={x} y1={pad.top} y2={pad.top + usableH} stroke="#e5e7eb" strokeDasharray="2 4" />
                    <line x1={pad.left} x2={pad.left + usableW} y1={y} y2={y} stroke="#e5e7eb" strokeDasharray="2 4" />
                    <text x={x} y={pad.top + usableH + 14} fontSize={9} fill="#6b7280" textAnchor="middle">
                      {formatTick(t)}
                    </text>
                    <text x={pad.left - 6} y={y + 3} fontSize={9} fill="#6b7280" textAnchor="end">
                      {formatTick(t)}
                    </text>
                  </g>
                );
              })}
              <line x1={pad.left} y1={pad.top + usableH} x2={pad.left + usableW} y2={pad.top + usableH} stroke="#d4d4d8" />
              <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + usableH} stroke="#d4d4d8" />
              <text x={pad.left + usableW / 2} y={height - 8} fontSize={11} fill="#6b7280" textAnchor="middle">
                Observed y
              </text>
              <text x={6} y={pad.top + usableH / 2} fontSize={11} fill="#6b7280" transform={`rotate(-90, 6, ${pad.top + usableH / 2})`}>
                Predicted y
              </text>
              <line x1={xScale(minVal)} y1={yScale(minVal)} x2={xScale(maxVal)} y2={yScale(maxVal)} stroke="#e5e7eb" strokeDasharray="4 4" />
              {trainY.map((obs, idx) => {
                const pred = trainPreds[idx];
                if (!Number.isFinite(pred)) return null;
                return <circle key={`train-${idx}`} cx={xScale(obs)} cy={yScale(pred)} r={3.2} fill="rgba(14,165,233,0.75)" />;
              })}
              <g transform={`translate(${pad.left + 8}, ${pad.top + 10})`}>
                <rect x={0} y={0} width={9} height={9} rx={2} ry={2} fill="rgba(14,165,233,0.75)" />
                <text x={14} y={8} fontSize={9} fill="#6b7280">
                  Train
                </text>
              </g>
            </>
          );
        })()}
      </svg>
    </div>
  );
}
