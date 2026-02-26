"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";

export default function Home() {
  const router = useRouter();
  const [mode, setMode] = useState<"model" | "train">("model");
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [dataset, setDataset] = useState("bike_hourly");
  const [modelType, setModelType] = useState<"igann" | "igann_interactive">("igann");
  const [centerShapes, setCenterShapes] = useState(false);
  const [points, setPoints] = useState(250);
  const [seed, setSeed] = useState(3);
  const [nEstimators, setNEstimators] = useState(100);
  const [boostRate, setBoostRate] = useState(0.1);
  const [initReg, setInitReg] = useState(1);
  const [elmAlpha, setElmAlpha] = useState(1);
  const [earlyStopping, setEarlyStopping] = useState(50);
  const [scaleY, setScaleY] = useState(true);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const [baseResp, savedResp] = await Promise.all([
          fetch("/api/models"),
          fetch("/api/saved-models"),
        ]);
        const basePayload = (await baseResp.json()) as { models?: string[] };
        const savedPayload = (await savedResp.json()) as { models?: string[] };
        const isValidName = (name: string | null | undefined) => Boolean(name && name !== "undefined");
        const base = (basePayload.models ?? []).filter(isValidName).map((name) => `model:${name}`);
        const saved = (savedPayload.models ?? []).filter(isValidName).map((name) => `saved:${name}`);
        const all = [...base, ...saved];
        setModels(all);
        setSelectedModel(all[0] ?? "");
      } catch (error) {
        console.warn("Failed to load model list.", error);
      }
    };
    loadModels();
  }, []);

  const handleLaunch = () => {
    if (mode === "train") {
      const params = new URLSearchParams({
        train: "1",
        dataset,
        model_type: modelType,
        center_shapes: centerShapes.toString(),
        points: points.toString(),
        seed: seed.toString(),
        n_estimators: nEstimators.toString(),
        boost_rate: boostRate.toString(),
        init_reg: initReg.toString(),
        elm_alpha: elmAlpha.toString(),
        early_stopping: earlyStopping.toString(),
        scale_y: scaleY.toString(),
      });
      router.push(`/gam-lab?${params.toString()}`);
      return;
    }
    if (!selectedModel) return;
    const params = new URLSearchParams({ model: selectedModel });
    router.push(`/gam-lab?${params.toString()}`);
  };

  const handleUpload = async (file: File) => {
    try {
      setUploadStatus(null);
      const text = await file.text();
      const payload = JSON.parse(text);
      const baseName = file.name.replace(/\.json$/i, "") || "uploaded-model";
      const response = await fetch("/api/saved-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: baseName, payload }),
      });
      if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
      const params = new URLSearchParams({ model: `saved:${baseName}` });
      router.push(`/gam-lab?${params.toString()}`);
    } catch (error) {
      console.warn("Failed to upload model.", error);
      setUploadStatus("Upload failed. Please check the JSON format.");
    }
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <div className={styles.hero}>
          <span className={styles.eyebrow}>GAM Lab</span>
          <h1 className={styles.title}>Shape functions, made tangible.</h1>
          <p className={styles.subtitle}>
            Train interpretable models and sculpt their effects with a hands-on editor. Designed for fast exploration,
            clean comparisons, and a crisp audit trail.
          </p>
          <div className={styles.selectorCard}>
            <div className={styles.switchRow}>
              <button
                type="button"
                className={`${styles.switchButton} ${mode === "model" ? styles.switchButtonActive : ""}`}
                onClick={() => setMode("model")}
              >
                Load model
              </button>
              <button
                type="button"
                className={`${styles.switchButton} ${mode === "train" ? styles.switchButtonActive : ""}`}
                onClick={() => setMode("train")}
              >
                Train new
              </button>
            </div>
            {mode === "model" ? (
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Model</span>
                <select
                  className={styles.select}
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  {models.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <div className={styles.formGrid}>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Dataset</span>
                  <select
                    className={styles.select}
                    value={dataset}
                    onChange={(event) => setDataset(event.target.value)}
                  >
                    <option value="bike_hourly">Bike sharing (hourly)</option>
                    <option value="adult_income">Adult income</option>
                    <option value="breast_cancer">Breast cancer (Wisconsin)</option>
                  </select>
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Model</span>
                  <select
                    className={styles.select}
                    value={modelType}
                    onChange={(event) => setModelType(event.target.value as "igann" | "igann_interactive")}
                  >
                    <option value="igann">IGANN</option>
                    <option value="igann_interactive">IGANN interactive</option>
                  </select>
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Center shapes</span>
                  <label className={styles.toggleLabel}>
                    <input
                      className={styles.toggleInput}
                      type="checkbox"
                      checked={centerShapes}
                      onChange={(event) => setCenterShapes(event.target.checked)}
                    />
                    <span className={styles.toggleTrack}>
                      <span className={styles.toggleThumb} />
                    </span>
                    <span className={styles.toggleText}>Enforce E[fj(Xj)] = 0</span>
                  </label>
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Points</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="1"
                    min="2"
                    max="250"
                    value={points}
                    onChange={(event) => setPoints(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Seed</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="1"
                    min="0"
                    max="9999"
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Estimators</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="1"
                    min="10"
                    max="500"
                    value={nEstimators}
                    onChange={(event) => setNEstimators(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Boost rate</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="1"
                    value={boostRate}
                    onChange={(event) => setBoostRate(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Init reg</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="10"
                    value={initReg}
                    onChange={(event) => setInitReg(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>ELM alpha</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="10"
                    value={elmAlpha}
                    onChange={(event) => setElmAlpha(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Early stopping</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="1"
                    min="5"
                    max="200"
                    value={earlyStopping}
                    onChange={(event) => setEarlyStopping(Number(event.target.value))}
                  />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Scale target</span>
                  <label className={styles.toggleLabel}>
                    <input
                      className={styles.toggleInput}
                      type="checkbox"
                      checked={scaleY}
                      onChange={(event) => setScaleY(event.target.checked)}
                    />
                    <span className={styles.toggleTrack}>
                      <span className={styles.toggleThumb} />
                    </span>
                    <span className={styles.toggleText}>Normalize y for training</span>
                  </label>
                </label>
              </div>
            )}
            <div className={styles.uploadBlock}>
              <div
                className={`${styles.dropzone} ${isDragging ? styles.dropzoneActive : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  const file = event.dataTransfer.files?.[0];
                  if (file) handleUpload(file);
                }}
              >
                <span className={styles.dropzoneTitle}>Drop a model JSON here</span>
                <span className={styles.dropzoneHint}>or upload from your device</span>
                <label className={styles.uploadButton}>
                  Upload JSON
                  <input
                    className={styles.uploadInput}
                    type="file"
                    accept="application/json"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) handleUpload(file);
                    }}
                  />
                </label>
              </div>
              {uploadStatus ? <p className={styles.uploadStatus}>{uploadStatus}</p> : null}
            </div>
            <button type="button" className={styles.primary} onClick={handleLaunch}>
              Open GAM Lab
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
