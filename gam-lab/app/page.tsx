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
  const [points, setPoints] = useState(10);
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
        points: points.toString(),
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
                  <span className={styles.fieldLabel}>Points</span>
                  <input
                    className={styles.input}
                    type="number"
                    step="1"
                    min="2"
                    max="200"
                    value={points}
                    onChange={(event) => setPoints(Number(event.target.value))}
                  />
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
