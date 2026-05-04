"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listSavedModels, loadSavedModel } from "../lib/savedModelApi";
import { listModels, loadModel, trainModel, TrainRequest } from "../lib/modelApi";
import { TrainResponse, ShapeFunction, MetricSummary } from "../types";
import styles from "./compare.module.css";

// ─── Chart constants ──────────────────────────────────────────
const W = 280;
const H = 160;
const PAD = { top: 8, right: 8, bottom: 24, left: 36 };

const DATASETS = [
  { id: "bike_hourly", label: "Bike sharing (hourly)" },
  { id: "mimic4_mean_100_full", label: "MIMIC-IV mortality" },
];

// ─── Slot state ───────────────────────────────────────────────
type SlotMode = "saved" | "train";

type TrainConfig = {
  dataset: string;
  modelType: "igann" | "igann_interactive";
  nEstimators: number;
  boostRate: number;
  seed: number;
  centerShapes: boolean;
  sampleSize: number;
};

const DEFAULT_TRAIN_CONFIG: TrainConfig = {
  dataset: "bike_hourly",
  modelType: "igann_interactive",
  nEstimators: 100,
  boostRate: 0.1,
  seed: 3,
  centerShapes: true,
  sampleSize: 1000,
};

type SlotState = {
  mode: SlotMode;
  savedName: string;
  trainConfig: TrainConfig;
  model: TrainResponse | null;
  status: "idle" | "loading" | "error";
  error: string | null;
};

const defaultSlot = (): SlotState => ({
  mode: "saved",
  savedName: "",
  trainConfig: { ...DEFAULT_TRAIN_CONFIG },
  model: null,
  status: "idle",
  error: null,
});

// ─── Helpers ──────────────────────────────────────────────────
function scaleLinear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (v: number) => (d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0));
}

function fmt(v: number | null | undefined, digits = 3) {
  if (v == null) return "—";
  return v.toFixed(digits);
}

// ─── Shape chart ──────────────────────────────────────────────
function ShapeCompareChart({ a, b }: { a: ShapeFunction | null; b: ShapeFunction | null }) {
  const axA = a?.editableX ?? [];
  const ayA = a?.editableY ?? [];
  const axB = b?.editableX ?? [];
  const ayB = b?.editableY ?? [];
  const allX = [...axA, ...axB];
  const allY = [...ayA, ...ayB];
  if (allX.length === 0) return <div className={styles.noData}>no data</div>;

  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xS = scaleLinear([xMin, xMax], [PAD.left, W - PAD.right]);
  const yS = scaleLinear([yMin, yMax], [H - PAD.bottom, PAD.top]);
  const y0 = yS(Math.max(yMin, Math.min(yMax, 0)));
  const toPoints = (xs: number[], ys: number[]) => xs.map((x, i) => `${xS(x)},${yS(ys[i])}`).join(" ");
  const xTicks = [xMin, (xMin + xMax) / 2, xMax];
  const yTicks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <svg width={W} height={H} className={styles.chart}>
      <line x1={PAD.left} x2={W - PAD.right} y1={y0} y2={y0} className={styles.zeroLine} />
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} className={styles.axis} />
      <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} className={styles.axis} />
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 4} x2={PAD.left} y1={yS(v)} y2={yS(v)} className={styles.tick} />
          <text x={PAD.left - 6} y={yS(v) + 4} className={styles.tickLabel} textAnchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      {xTicks.map((v, i) => (
        <g key={i}>
          <line x1={xS(v)} x2={xS(v)} y1={H - PAD.bottom} y2={H - PAD.bottom + 4} className={styles.tick} />
          <text x={xS(v)} y={H - PAD.bottom + 14} className={styles.tickLabel} textAnchor="middle">{v.toFixed(1)}</text>
        </g>
      ))}
      {axA.length > 0 && <polyline points={toPoints(axA, ayA)} className={styles.lineA} fill="none" />}
      {axB.length > 0 && <polyline points={toPoints(axB, ayB)} className={styles.lineB} fill="none" />}
    </svg>
  );
}

function CategoricalCompareChart({ a, b }: { a: ShapeFunction | null; b: ShapeFunction | null }) {
  const catsA = a?.categories ?? [], catsB = b?.categories ?? [];
  const allCats = Array.from(new Set([...catsA, ...catsB]));
  const ysA = a?.editableY ?? [], ysB = b?.editableY ?? [];
  const mapA = Object.fromEntries(catsA.map((c, i) => [c, ysA[i] ?? 0]));
  const mapB = Object.fromEntries(catsB.map((c, i) => [c, ysB[i] ?? 0]));
  const allY = [...ysA, ...ysB];
  if (allCats.length === 0) return <div className={styles.noData}>no data</div>;

  const yMin = Math.min(...allY, 0), yMax = Math.max(...allY, 0);
  const barW = Math.max(4, (W - PAD.left - PAD.right) / allCats.length / 2 - 2);
  const yS = scaleLinear([yMin, yMax], [H - PAD.bottom, PAD.top]);
  const y0 = yS(0);
  const step = (W - PAD.left - PAD.right) / allCats.length;

  return (
    <svg width={W} height={H} className={styles.chart}>
      <line x1={PAD.left} x2={W - PAD.right} y1={y0} y2={y0} className={styles.zeroLine} />
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} className={styles.axis} />
      <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} className={styles.axis} />
      {allCats.map((cat, i) => {
        const cx = PAD.left + step * i + step / 2;
        const vA = mapA[cat] ?? 0, vB = mapB[cat] ?? 0;
        return (
          <g key={cat}>
            <rect x={cx - barW - 1} y={Math.min(y0, yS(vA))} width={barW} height={Math.abs(y0 - yS(vA))} className={styles.barA} />
            <rect x={cx + 1} y={Math.min(y0, yS(vB))} width={barW} height={Math.abs(y0 - yS(vB))} className={styles.barB} />
            <text x={cx} y={H - PAD.bottom + 14} className={styles.tickLabel} textAnchor="middle">
              {cat.length > 6 ? cat.slice(0, 5) + "…" : cat}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Metrics ──────────────────────────────────────────────────
function MetricRow({ label, a, b, lowerBetter }: { label: string; a?: number | null; b?: number | null; lowerBetter?: boolean }) {
  const diff = a != null && b != null ? b - a : null;
  const good = diff != null && (lowerBetter ? diff < 0 : diff > 0);
  const bad  = diff != null && (lowerBetter ? diff > 0 : diff < 0);
  return (
    <tr>
      <td className={styles.metricLabel}>{label}</td>
      <td className={styles.metricVal}>{fmt(a)}</td>
      <td className={styles.metricVal}>{fmt(b)}</td>
      <td className={`${styles.metricVal} ${good ? styles.up : bad ? styles.down : ""}`}>
        {diff != null ? (diff > 0 ? "+" : "") + fmt(diff) : "—"}
      </td>
    </tr>
  );
}

function MetricsTable({ a, b, task }: { a: MetricSummary | null; b: MetricSummary | null; task: string | null }) {
  return (
    <table className={styles.metricsTable}>
      <thead>
        <tr>
          <th>Metric</th>
          <th className={styles.colA}>A</th>
          <th className={styles.colB}>B</th>
          <th>Δ (B−A)</th>
        </tr>
      </thead>
      <tbody>
        {task === "regression" ? (
          <>
            <MetricRow label="RMSE" a={a?.rmse} b={b?.rmse} lowerBetter />
            <MetricRow label="MAE"  a={a?.mae}  b={b?.mae}  lowerBetter />
            <MetricRow label="R²"   a={a?.r2}   b={b?.r2} />
          </>
        ) : (
          <>
            <MetricRow label="Accuracy"  a={a?.acc}       b={b?.acc} />
            <MetricRow label="F1"        a={a?.f1}        b={b?.f1} />
            <MetricRow label="Precision" a={a?.precision} b={b?.precision} />
            <MetricRow label="Recall"    a={a?.recall}    b={b?.recall} />
          </>
        )}
      </tbody>
    </table>
  );
}

// ─── Train form ───────────────────────────────────────────────
function TrainForm({
  config,
  onChange,
  onTrain,
  loading,
}: {
  config: TrainConfig;
  onChange: (c: TrainConfig) => void;
  onTrain: () => void;
  loading: boolean;
}) {
  const set = <K extends keyof TrainConfig>(k: K, v: TrainConfig[K]) => onChange({ ...config, [k]: v });
  return (
    <div className={styles.trainForm}>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>Dataset</label>
        <select className={styles.formSelect} value={config.dataset} onChange={(e) => set("dataset", e.target.value)}>
          {DATASETS.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
        </select>
      </div>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>Model type</label>
        <select className={styles.formSelect} value={config.modelType} onChange={(e) => set("modelType", e.target.value as "igann" | "igann_interactive")}>
          <option value="igann_interactive">igann_interactive</option>
          <option value="igann">igann</option>
        </select>
      </div>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>n_estimators</label>
        <input className={styles.formInput} type="number" min={10} max={500} value={config.nEstimators}
          onChange={(e) => set("nEstimators", Number(e.target.value))} />
      </div>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>boost_rate</label>
        <input className={styles.formInput} type="number" min={0.001} max={1} step={0.01} value={config.boostRate}
          onChange={(e) => set("boostRate", Number(e.target.value))} />
      </div>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>seed</label>
        <input className={styles.formInput} type="number" value={config.seed}
          onChange={(e) => set("seed", Number(e.target.value))} />
      </div>
      <div className={styles.formRow}>
        <label className={styles.formLabel}>center shapes</label>
        <input type="checkbox" checked={config.centerShapes} onChange={(e) => set("centerShapes", e.target.checked)} />
      </div>
      {config.dataset === "mimic4_mean_100_full" && (
        <div className={styles.formRow}>
          <label className={styles.formLabel}>sample size</label>
          <input className={styles.formInput} type="number" min={100} max={65350} step={100}
            value={config.sampleSize ?? 1000} onChange={(e) => set("sampleSize", Number(e.target.value))} />
        </div>
      )}
      <button className={styles.trainBtn} onClick={onTrain} disabled={loading}>
        {loading ? "Training…" : "Train"}
      </button>
    </div>
  );
}

// ─── Slot panel ───────────────────────────────────────────────
function SlotPanel({
  label,
  colorClass,
  slot,
  savedNames,
  pretrainedNames,
  onChange,
  onTrain,
  onLoad,
}: {
  label: string;
  colorClass: string;
  slot: SlotState;
  savedNames: string[];
  pretrainedNames: string[];
  onChange: (s: SlotState) => void;
  onTrain: () => void;
  onLoad: (name: string, kind: "saved" | "pretrained") => void;
}) {
  return (
    <div className={`${styles.slotPanel} ${colorClass}`}>
      <div className={styles.slotHeader}>
        <span className={styles.slotLabel}>{label}</span>
        <div className={styles.modeToggle}>
          <button
            className={`${styles.modeBtn} ${slot.mode === "saved" ? styles.modeBtnActive : ""}`}
            onClick={() => onChange({ ...slot, mode: "saved" })}
          >
            Load
          </button>
          <button
            className={`${styles.modeBtn} ${slot.mode === "train" ? styles.modeBtnActive : ""}`}
            onClick={() => onChange({ ...slot, mode: "train" })}
          >
            Train
          </button>
        </div>
      </div>

      {slot.mode === "saved" && (
        <div className={styles.loadSection}>
          {savedNames.length > 0 && (
            <div className={styles.formRow}>
              <label className={styles.formLabel}>Saved</label>
              <select className={styles.formSelect} value={slot.savedName}
                onChange={(e) => { onChange({ ...slot, savedName: e.target.value }); onLoad(e.target.value, "saved"); }}>
                <option value="">— select —</option>
                {savedNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          )}
          {pretrainedNames.length > 0 && (
            <div className={styles.formRow}>
              <label className={styles.formLabel}>Pre-trained</label>
              <select className={styles.formSelect} value={slot.savedName}
                onChange={(e) => { onChange({ ...slot, savedName: e.target.value }); onLoad(e.target.value, "pretrained"); }}>
                <option value="">— select —</option>
                {pretrainedNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          )}
        </div>
      )}

      {slot.mode === "train" && (
        <TrainForm
          config={slot.trainConfig}
          onChange={(c) => onChange({ ...slot, trainConfig: c })}
          onTrain={onTrain}
          loading={slot.status === "loading"}
        />
      )}

      {slot.status === "loading" && <div className={styles.slotStatus}>Loading…</div>}
      {slot.status === "error" && <div className={styles.slotError}>{slot.error}</div>}
      {slot.model && (
        <div className={styles.slotLoaded}>
          ✓ {slot.model.model.dataset} · {slot.model.model.model_type} · {slot.model.version.shapes.length} features
        </div>
      )}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────
export default function ComparePage() {
  const [savedNames, setSavedNames]         = useState<string[]>([]);
  const [pretrainedNames, setPretrainedNames] = useState<string[]>([]);
  const [slotA, setSlotA] = useState<SlotState>(defaultSlot);
  const [slotB, setSlotB] = useState<SlotState>(defaultSlot);

  useEffect(() => {
    listSavedModels().then(setSavedNames).catch(() => {});
    listModels().then(setPretrainedNames).catch(() => {});
  }, []);

  const loadIntoSlot = async (
    set: (s: SlotState) => void,
    current: SlotState,
    name: string,
    kind: "saved" | "pretrained",
  ) => {
    if (!name) return;
    set({ ...current, status: "loading", error: null });
    try {
      const model = kind === "saved" ? await loadSavedModel(name) : await loadModel(name);
      set({ ...current, savedName: name, model, status: "idle", error: null });
    } catch (e) {
      set({ ...current, status: "error", error: String(e) });
    }
  };

  const trainIntoSlot = async (set: (s: SlotState) => void, current: SlotState) => {
    set({ ...current, status: "loading", error: null });
    const c = current.trainConfig;
    const req: TrainRequest = {
      dataset: c.dataset,
      model_type: c.modelType,
      center_shapes: c.centerShapes,
      seed: c.seed,
      n_estimators: c.nEstimators,
      boost_rate: c.boostRate,
      init_reg: 1,
      elm_alpha: 1,
      early_stopping: 50,
      n_hid: 10,
      scale_y: true,
      points: 250,
      sample_size: c.dataset === "mimic4_mean_100_full" ? c.sampleSize : undefined,
    };
    try {
      const model = await trainModel(req);
      set({ ...current, model, status: "idle", error: null });
    } catch (e) {
      set({ ...current, status: "error", error: String(e) });
    }
  };

  const mA = slotA.model, mB = slotB.model;
  const shapesA = mA?.version.shapes ?? [];
  const shapesB = mB?.version.shapes ?? [];
  const allKeys = Array.from(new Set([...shapesA.map((s) => s.key), ...shapesB.map((s) => s.key)]));
  const byKeyA = Object.fromEntries(shapesA.map((s) => [s.key, s]));
  const byKeyB = Object.fromEntries(shapesB.map((s) => [s.key, s]));
  const task = mA?.model.task ?? mB?.model.task ?? null;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link href="/gam-lab/train" className={styles.back}>← Train</Link>
        <h1 className={styles.title}>Model Comparison</h1>
      </header>

      <div className={styles.slots}>
        <SlotPanel
          label="Model A"
          colorClass={styles.slotA}
          slot={slotA}
          savedNames={savedNames}
          pretrainedNames={pretrainedNames}
          onChange={setSlotA}
          onTrain={() => trainIntoSlot(setSlotA, slotA)}
          onLoad={(name, kind) => loadIntoSlot(setSlotA, slotA, name, kind)}
        />
        <SlotPanel
          label="Model B"
          colorClass={styles.slotB}
          slot={slotB}
          savedNames={savedNames}
          pretrainedNames={pretrainedNames}
          onChange={setSlotB}
          onTrain={() => trainIntoSlot(setSlotB, slotB)}
          onLoad={(name, kind) => loadIntoSlot(setSlotB, slotB, name, kind)}
        />
      </div>

      {(mA || mB) && (
        <>
          <div className={styles.legend}>
            <span className={styles.dotA} /><span>Model A</span>
            <span className={styles.dotB} /><span>Model B</span>
          </div>

          <div className={styles.metricsRow}>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>Train Metrics</h2>
              <MetricsTable a={mA?.version.trainMetrics ?? null} b={mB?.version.trainMetrics ?? null} task={task} />
            </section>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>Test Metrics</h2>
              <MetricsTable a={mA?.version.testMetrics ?? null} b={mB?.version.testMetrics ?? null} task={task} />
            </section>
          </div>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Shape Functions</h2>
            <div className={styles.grid}>
              {allKeys.map((key) => {
                const sa = byKeyA[key] ?? null;
                const sb = byKeyB[key] ?? null;
                const label = sa?.label ?? sb?.label ?? key;
                const isCat = (sa?.categories ?? sb?.categories) != null;
                return (
                  <div key={key} className={styles.card}>
                    <div className={styles.cardTitle}>{label}</div>
                    {isCat
                      ? <CategoricalCompareChart a={sa} b={sb} />
                      : <ShapeCompareChart a={sa} b={sb} />}
                    {(!sa || !sb) && (
                      <div className={styles.missing}>{!sa ? "only in B" : "only in A"}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        </>
      )}

      {!mA && !mB && (
        <div className={styles.empty}>
          Load a saved model or train a new one in each slot to compare.
        </div>
      )}
    </div>
  );
}
