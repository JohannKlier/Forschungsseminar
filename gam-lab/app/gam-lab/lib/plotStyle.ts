import { type BaseType, type Selection } from "d3-selection";

export const AXIS_TICK_LINE_STROKE = "#e2e8f0";
export const AXIS_TICK_LINE_DASH = "3 3";
export const AXIS_TICK_TEXT_FILL = "#475569";
export const AXIS_TICK_TEXT_SIZE = 10;
export const ZERO_LINE_STROKE = "#cbd5e1";
export const ZERO_LINE_DASH = "4 4";

export const HIST_FILL = "rgba(14,165,233,0.12)";
export const HIST_STROKE = "rgba(14,165,233,0.25)";
export const HIST_STROKE_WIDTH = 0.5;
export const HIST_LABEL_FILL = "#64748b";
export const HIST_LABEL_SIZE = 10;

export function applyCommonAxisStyles<T extends BaseType, D, P extends BaseType, PD>(root: Selection<T, D, P, PD>) {
  root.selectAll(".domain").attr("stroke", "none");
  root.selectAll(".tick line").attr("stroke", AXIS_TICK_LINE_STROKE).attr("stroke-dasharray", AXIS_TICK_LINE_DASH);
  root.selectAll(".tick text").attr("fill", AXIS_TICK_TEXT_FILL).attr("font-size", AXIS_TICK_TEXT_SIZE);
}

export function styleDensityBars<T extends BaseType, D, P extends BaseType, PD>(selection: Selection<T, D, P, PD>) {
  selection.attr("fill", HIST_FILL).attr("stroke", HIST_STROKE).attr("stroke-width", HIST_STROKE_WIDTH);
}

export function styleDensityLabel<T extends BaseType, D, P extends BaseType, PD>(selection: Selection<T, D, P, PD>) {
  selection.attr("fill", HIST_LABEL_FILL).attr("font-size", HIST_LABEL_SIZE);
}
