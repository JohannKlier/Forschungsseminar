import { Dispatch, SetStateAction, useState } from "react";
import type { DragCurve, SmoothingAlgorithm } from "../components/VisxShapeEditor";

export type ToolSettings = {
  activeContinuousTool: "drag" | "smooth";
  setActiveContinuousTool: Dispatch<SetStateAction<"drag" | "smooth">>;
  dragFalloffRadius: number;
  setDragFalloffRadius: Dispatch<SetStateAction<number>>;
  dragRangeBoost: number;
  setDragRangeBoost: Dispatch<SetStateAction<number>>;
  dragCurve: DragCurve;
  setDragCurve: Dispatch<SetStateAction<DragCurve>>;
  smoothAmount: number;
  setSmoothAmount: Dispatch<SetStateAction<number>>;
  smoothingRangeMax: number;
  setSmoothingRangeMax: Dispatch<SetStateAction<number>>;
  smoothingSpeed: number;
  setSmoothingSpeed: Dispatch<SetStateAction<number>>;
  smoothingAlgorithm: SmoothingAlgorithm;
  setSmoothingAlgorithm: Dispatch<SetStateAction<SmoothingAlgorithm>>;
};

export const useToolSettings = (): ToolSettings => {
  const [activeContinuousTool, setActiveContinuousTool] = useState<"drag" | "smooth">("drag");
  const [dragFalloffRadius, setDragFalloffRadius] = useState(4);
  const [dragRangeBoost, setDragRangeBoost] = useState(1);
  const [dragCurve, setDragCurve] = useState<DragCurve>("adaptive");
  const [smoothAmount, setSmoothAmount] = useState(0.5);
  const [smoothingRangeMax, setSmoothingRangeMax] = useState(32);
  const [smoothingSpeed, setSmoothingSpeed] = useState(1);
  const [smoothingAlgorithm, setSmoothingAlgorithm] = useState<SmoothingAlgorithm>("tikhonov");

  return {
    activeContinuousTool,
    setActiveContinuousTool,
    dragFalloffRadius,
    setDragFalloffRadius,
    dragRangeBoost,
    setDragRangeBoost,
    dragCurve,
    setDragCurve,
    smoothAmount,
    setSmoothAmount,
    smoothingRangeMax,
    setSmoothingRangeMax,
    smoothingSpeed,
    setSmoothingSpeed,
    smoothingAlgorithm,
    setSmoothingAlgorithm,
  };
};
