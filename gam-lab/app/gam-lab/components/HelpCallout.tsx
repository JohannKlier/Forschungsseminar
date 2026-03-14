import type { HelpStep } from "../hooks/useHelpTour";
import styles from "../page.module.css";

interface HelpCalloutProps {
  target: HelpStep["target"];
  currentStep: HelpStep;
  stepIndex: number;
  totalSteps: number;
  isFirstStep: boolean;
  isLastStep: boolean;
  isStepActive: (target: HelpStep["target"]) => boolean;
  onClose: () => void;
  onNext: () => void;
  onPrev: () => void;
}

export default function HelpCallout({
  target,
  currentStep,
  stepIndex,
  totalSteps,
  isFirstStep,
  isLastStep,
  isStepActive,
  onClose,
  onNext,
  onPrev,
}: HelpCalloutProps) {
  if (!isStepActive(target)) return null;

  return (
    <div
      className={`${styles.helpCallout} ${styles[currentStep.placementClass]}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby="dashboard-help-title"
    >
      <div className={styles.helpHeader}>
        <div>
          <p className={styles.helpEyebrow}>{currentStep.eyebrow}</p>
          <h2 id="dashboard-help-title" className={styles.helpTitle}>
            {currentStep.title}
          </h2>
        </div>
        <button
          type="button"
          className={styles.helpCloseButton}
          onClick={onClose}
          aria-label="Close help"
        >
          Close
        </button>
      </div>
      <div className={styles.helpBody}>
        <div className={styles.helpProgressRow}>
          <p className={styles.helpProgressLabel}>
            Page {stepIndex + 1} of {totalSteps}
          </p>
          <div className={styles.helpProgressDots} aria-hidden="true">
            {Array.from({ length: totalSteps }, (_, index) => (
              <span
                key={index}
                className={`${styles.helpProgressDot} ${index === stepIndex ? styles.helpProgressDotActive : ""}`}
              />
            ))}
          </div>
        </div>
        <section className={styles.helpSpotlight}>
          <p className={styles.helpLead}>{currentStep.text}</p>
          <ul className={styles.helpList}>
            {currentStep.bullets.map((bullet) => (
              <li key={bullet} className={styles.helpListItem}>
                {bullet}
              </li>
            ))}
          </ul>
        </section>
      </div>
      <div className={styles.helpFooter}>
        <button
          type="button"
          className={styles.helpNavButton}
          onClick={onPrev}
          disabled={isFirstStep}
        >
          Previous
        </button>
        <button
          type="button"
          className={styles.helpNavButtonPrimary}
          onClick={onNext}
        >
          {isLastStep ? "Start editing" : "Next"}
        </button>
      </div>
    </div>
  );
}
