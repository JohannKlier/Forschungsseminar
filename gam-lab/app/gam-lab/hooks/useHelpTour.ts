import { useCallback, useEffect, useState } from "react";

const helpSteps = [
  {
    eyebrow: "Step 1",
    target: "shapePanel",
    placementClass: "helpCalloutPanelRight",
    title: "Shape Functions Panel",
    text: "This is the main workspace for changing the model. Each shape shows how one feature contributes to the prediction.",
    bullets: [
      "Select a feature and inspect its current shape.",
      "Use the tools to drag points, smooth curves, or apply constraints.",
      "Switch between single and grid view to focus or compare.",
      "Undo, redo, and save your edits at the bottom.",
    ],
  },
  {
    eyebrow: "Step 2",
    target: "sidebar",
    placementClass: "helpCalloutSidebarLeft",
    title: "Sidebar",
    text: "The sidebar gives you supporting context and tracks your work.",
    bullets: [
      "The Edit tab shows statistics for the selected feature.",
      "The History tab logs every action so you can review your changes.",
      "Switch between tabs while iterating on the same feature.",
    ],
  },
] as const;

export type HelpStep = (typeof helpSteps)[number];

export function useHelpTour(setSidebarTab: (tab: "edit" | "history" | "features") => void) {
  const [showHelp, setShowHelp] = useState(false);
  const [helpStepIndex, setHelpStepIndex] = useState(0);

  const currentStep = helpSteps[helpStepIndex];
  const isFirstStep = helpStepIndex === 0;
  const isLastStep = helpStepIndex === helpSteps.length - 1;

  // Lock scroll and listen for Escape
  useEffect(() => {
    if (!showHelp) return;

    const previousBodyOverflow = document.body.style.overflow;
    const previousHtmlOverflow = document.documentElement.style.overflow;
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowHelp(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousBodyOverflow;
      document.documentElement.style.overflow = previousHtmlOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [showHelp]);

  // Sync sidebar tab when the sidebar step is active
  useEffect(() => {
    if (!showHelp) return;
    if (currentStep.target !== "sidebar") return;
    setSidebarTab("edit");
  }, [showHelp, currentStep, setSidebarTab]);

  const openHelp = useCallback(() => {
    setHelpStepIndex(0);
    setShowHelp(true);
  }, []);

  const closeHelp = useCallback(() => {
    setShowHelp(false);
  }, []);

  const goNext = useCallback(() => {
    if (helpStepIndex >= helpSteps.length - 1) {
      setShowHelp(false);
      return;
    }
    setHelpStepIndex((i) => Math.min(helpSteps.length - 1, i + 1));
  }, [helpStepIndex]);

  const goPrev = useCallback(() => {
    setHelpStepIndex((i) => Math.max(0, i - 1));
  }, []);

  const isStepActive = useCallback(
    (target: HelpStep["target"]) => showHelp && currentStep.target === target,
    [showHelp, currentStep],
  );

  return {
    showHelp,
    currentStep,
    stepIndex: helpStepIndex,
    totalSteps: helpSteps.length,
    isFirstStep,
    isLastStep,
    openHelp,
    closeHelp,
    goNext,
    goPrev,
    isStepActive,
  };
}
