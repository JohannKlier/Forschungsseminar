"use client";

import styles from "./InspectorPanel.module.css";

type SettingsControl = {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  format?: (value: number) => string;
};

type SettingsConfig = {
  description: string;
  controls: SettingsControl[];
};

type InspectorPanelProps = {
  isOpen: boolean;
  onOpen: () => void;
  onClose: () => void;
  title: string;
  config?: SettingsConfig;
  values?: Record<string, number>;
  onSettingChange?: (controlId: string, value: number) => void;
};

export default function InspectorPanel({ isOpen, onOpen, onClose, title, config, values, onSettingChange }: InspectorPanelProps) {
  const renderControls = () => {
    if (!config) {
      return <p className={styles.empty}>No quick settings exposed for this module.</p>;
    }

    return (
      <>
        <p className={styles.description}>{config.description}</p>
        <div className={styles.grid}>
          {config.controls.map((control) => {
            const value = values?.[control.id] ?? control.defaultValue;
            const formatted = control.format ? control.format(value) : value.toString();
            return (
              <label key={control.id} className={styles.control}>
                <span className={styles.controlLabel}>
                  {control.label}: <strong>{formatted}</strong>
                </span>
                <input
                  type="range"
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  value={value}
                  onChange={(event) => onSettingChange?.(control.id, Number(event.target.value))}
                />
              </label>
            );
          })}
        </div>
      </>
    );
  };

  return (
    <>
      <aside className={`${styles.panel} ${isOpen ? styles.panelOpen : ""}`} aria-hidden={!isOpen}>
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Inspector</p>
            <h3 className={styles.title}>{title}</h3>
          </div>
          <button type="button" className={styles.closeButton} onClick={onClose}>
            Close
          </button>
        </header>
        {renderControls()}
      </aside>
      <button
        type="button"
        className={`${styles.tab} ${isOpen ? styles.tabOpen : ""}`}
        onClick={isOpen ? onClose : onOpen}
        aria-label={isOpen ? "Close inspector" : "Open inspector"}
      >
        {isOpen ? "→" : "←"}
      </button>
    </>
  );
}
