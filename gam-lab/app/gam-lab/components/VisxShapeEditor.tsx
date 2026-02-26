import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { select, type Selection } from "d3-selection";
import { scaleLinear } from "d3-scale";
import { axisBottom, axisLeft } from "d3-axis";
import { line as d3Line, curveLinear } from "d3-shape";
import { drag as d3Drag } from "d3-drag";
import { brush as d3Brush } from "d3-brush";
import { zoom, zoomIdentity, type ZoomTransform } from "d3-zoom";
import styles from "../page.module.css";
import { applyBrushSelection, applyClickSelection, resolveDragSelection } from "../lib/selection";
import { smoothSeriesGaussianReflect } from "../lib/smoothing";

type Props = {
  knots: { x: number[]; y: number[] };
  baseline?: { x: number[]; y: number[] };
  scatterX: number[];
  onKnotChange: (next: { x: number[]; y: number[] }) => void;
  onDragStart?: (snapshot: { x: number[]; y: number[] }) => void;
  onDragEnd?: (snapshot: { x: number[]; y: number[] }) => void;
  onSmoothEnd?: (start: { x: number[]; y: number[] }, end: { x: number[]; y: number[] }) => void;
  title: string;
  selected: number[];
  onSelectionChange: (indices: number[]) => void;
  featureKey: string;
  interactionMode: "select" | "zoom";
  dragFalloffRadius?: number;
  smoothingMode?: boolean;
  smoothAmount?: number;
  smoothingRangeMax?: number;
};

type KnotDatum = { x: number; y: number; idx: number };

const PADDING = { top: 16, right: 16, bottom: 72, left: 56 };
const HEIGHT = 560;
const SHOW_KNOT_MARKERS = false;

export default function VisxShapeEditor({
  knots,
  baseline,
  scatterX,
  onKnotChange,
  onDragStart,
  onDragEnd,
  onSmoothEnd,
  title,
  selected,
  onSelectionChange,
  featureKey,
  interactionMode,
  dragFalloffRadius = 4,
  smoothingMode = false,
  smoothAmount = 0.5,
  smoothingRangeMax = 32,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [width, setWidth] = useState(0);
  const yDomainRef = useRef<{ min: number; max: number } | null>(null);
  const lastFeatureRef = useRef<string | null>(null);
  const baselineSigRef = useRef<string>("");
  const dragTargetsRef = useRef<number[]>([]);
  const isDraggingRef = useRef(false);
  const dragStartMapRef = useRef<number[]>([]);
  const dragStartYRef = useRef<number | null>(null);
  const dragStartXRef = useRef<number | null>(null);
  const dragWeightsRef = useRef<number[]>([]);
  const dragWeightsMaxRef = useRef(0);
  const smoothWeightsRef = useRef<Record<number, number>>({});
  const smoothHoverIdxRef = useRef<number | null>(null);
  const smoothingDragActiveRef = useRef(false);
  const smoothingRafRef = useRef<number | null>(null);
  const smoothingLastTsRef = useRef<number | null>(null);
  const smoothingTargetIdxRef = useRef<number | null>(null);
  const smoothingBaseRef = useRef<{ x: number[]; y: number[] } | null>(null);
  const smoothingPreviewRef = useRef<{ x: number[]; y: number[] } | null>(null);
  const pendingSelectionRef = useRef<number[] | null>(null);
  const selectedRef = useRef<number[]>(selected);
  const dragBehaviorRef = useRef<ReturnType<typeof d3Drag<SVGCircleElement, { idx: number }>> | null>(null);
  const zoomRef = useRef<ZoomTransform | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const pendingDragRef = useRef<{ x: number[]; y: number[] } | null>(null);
  const onSelectionChangeRef = useRef(onSelectionChange);
  const onKnotChangeRef = useRef(onKnotChange);
  const onDragStartRef = useRef(onDragStart);
  const onDragEndRef = useRef(onDragEnd);
  const onSmoothEndRef = useRef(onSmoothEnd);
  const knotsRef = useRef(knots);

  // Precompute the x-density histogram used for the bottom overlay.
  const histogram = useMemo(() => {
    const vals = Array.isArray(scatterX) ? scatterX : [];
    let minVal = Number.POSITIVE_INFINITY;
    let maxVal = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < vals.length; i += 1) {
      const v = vals[i];
      if (!Number.isFinite(v)) continue;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
      return { bins: [0], binStart: 0, binWidth: 1 };
    }
    const span = maxVal - minVal || 1;
    const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
    const rangeBins = Math.round(span * 10);
    const sizeBins = Math.round(Math.sqrt(vals.length) * 2);
    const binCount = clamp(Math.min(rangeBins, sizeBins), 12, 80);
    const binWidth = span / binCount;
    const binStart = minVal;
    const bins = Array.from({ length: binCount }, () => 0);
    for (let i = 0; i < vals.length; i += 1) {
      const v = vals[i];
      if (!Number.isFinite(v)) continue;
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor((v - binStart) / binWidth)));
      bins[idx] += 1;
    }
    return { bins, binStart, binWidth };
  }, [scatterX]);

  // Track container width so SVG can reflow on resize.
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry?.contentRect?.width) {
        setWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Keep mutable callback/data refs in sync so D3 handlers always read latest props.
  useEffect(() => {
    onSelectionChangeRef.current = onSelectionChange;
    onKnotChangeRef.current = onKnotChange;
    onDragStartRef.current = onDragStart;
    onDragEndRef.current = onDragEnd;
    onSmoothEndRef.current = onSmoothEnd;
    knotsRef.current = knots;
  }, [onSelectionChange, onKnotChange, onDragStart, onDragEnd, onSmoothEnd]);

  // Sync selection visuals and selection bounding box when selected indices change.
  useEffect(() => {
    selectedRef.current = selected;
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const selectedSet = new Set(selected);
    const svg = select(svgEl);
    svg.selectAll("rect.selection-box").remove();
    const content = svg.select<SVGGElement>("g.shape-root");
    if (content.empty()) return;
    svg
      .selectAll<SVGCircleElement, any>("circle.knot")
      .attr("fill", (d: any) => (SHOW_KNOT_MARKERS ? (selectedSet.has(d.idx) ? "#0b6fa6" : "#0ea5e9") : "transparent"))
      .attr("stroke", SHOW_KNOT_MARKERS ? "#0b172a" : "none")
      .attr("stroke-width", (d: any) => (SHOW_KNOT_MARKERS ? (selectedSet.has(d.idx) ? 1.5 : 1) : 0))
      .attr("r", (d: any) => (SHOW_KNOT_MARKERS ? (selectedSet.has(d.idx) ? 6 : 5) : 0));
    if (!isDraggingRef.current && !(smoothingMode && smoothHoverIdxRef.current != null)) {
      svg
        .selectAll<SVGCircleElement, any>("circle.knot")
        .attr("fill", (d: any) => (SHOW_KNOT_MARKERS ? (selectedSet.has(d.idx) ? "#0b6fa6" : "#0ea5e9") : "transparent"));
    }
    const selectedNodes = svg
      .selectAll<SVGCircleElement, any>("circle.knot")
      .filter((d: any) => selectedSet.has(d.idx))
      .nodes();
    if (selectedNodes.length > 1) {
      const dragBehaviour = dragBehaviorRef.current;
      const xs = selectedNodes.map((node) => Number(node.getAttribute("cx") ?? 0));
      const ys = selectedNodes.map((node) => Number(node.getAttribute("cy") ?? 0));
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const minBoxHeight = 24;
      const pad = 10;
      const rawHeight = Math.max(0, maxY - minY);
      const extra = Math.max(0, minBoxHeight - rawHeight);
      const yPad = pad + extra / 2;
      const boxSel = content
        .selectAll<SVGRectElement, null>("rect.selection-box")
        .data([null])
        .join("rect")
        .classed("selection-box", true)
        .classed("drag-handle", true)
        .attr("x", minX - pad)
        .attr("y", minY - yPad)
        .attr("width", Math.max(0, maxX - minX + pad * 2))
        .attr("height", Math.max(minBoxHeight, rawHeight + yPad * 2))
        .style("cursor", "grab")
        .style("fill", "rgba(15, 23, 42, 0.04)")
        .style("stroke", "rgba(15, 23, 42, 0.35)")
        .style("stroke-dasharray", "4 4")
        .style("pointer-events", "all");
      if (dragBehaviour) boxSel.call(dragBehaviour as any);
    } else {
      content.selectAll("rect.selection-box").remove();
    }
  }, [selected, smoothingMode]);

  // Main D3 scene lifecycle: scales, layers, interactions, smoothing, and cursor states.
  useLayoutEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    if (width <= 0) return;
    const svg = select(svgEl);

    const usableWidth = Math.max(200, width - PADDING.left - PADDING.right);
    const usableHeight = HEIGHT - PADDING.top - PADDING.bottom;

    const baselineSig = baseline?.y?.join(",") ?? "";
    if (lastFeatureRef.current !== featureKey || baselineSigRef.current !== baselineSig) {
      yDomainRef.current = null;
      lastFeatureRef.current = featureKey;
      baselineSigRef.current = baselineSig;
    }

    const allX = [...knots.x, ...(baseline?.x ?? [])];
    const allY = [...knots.y, ...(baseline?.y ?? [])];
    const xBounds = (() => {
      let minX = allX.length ? Math.min(...allX) : 0;
      let maxX = allX.length ? Math.max(...allX) : 1;
      if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
        minX = 0;
        maxX = 1;
      }
      if (minX === maxX) {
        minX -= 1;
        maxX += 1;
      }
      const xPad = (maxX - minX || 1) * 0.05;
      return { min: minX - xPad, max: maxX + xPad };
    })();
    const nextYBounds = (() => {
      let minY = allY.length ? Math.min(...allY, 0) : -1;
      let maxY = allY.length ? Math.max(...allY, 0) : 1;
      if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
        minY = -1;
        maxY = 1;
      }
      minY = Math.min(minY, 0);
      if (minY === maxY) {
        minY -= 1;
        maxY += 1;
      }
      const span = maxY - minY || 1;
      const yPad = span * 0.15;
      const extraBottom = Math.max(span * 0.1, Math.abs(maxY) * 0.08, 0.5);
      return { min: minY - yPad - extraBottom, max: maxY + yPad };
    })();
    if (!yDomainRef.current) {
      yDomainRef.current = nextYBounds;
    }

    const baseXScale = scaleLinear().domain([xBounds.min, xBounds.max]).range([PADDING.left, PADDING.left + usableWidth]);
    const baseYScale = scaleLinear()
      .domain([yDomainRef.current.min, yDomainRef.current.max])
      .range([PADDING.top + usableHeight, PADDING.top]);

    const transform = zoomRef.current ?? zoomIdentity;
    const xScale = transform.rescaleX(baseXScale);
    const yScale = transform.rescaleY(baseYScale);

    const lineGen = d3Line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(curveLinear);

    const findNearestKnotIdxAtClientX = (clientX: number): number | null => {
      if (!sortedKnots.length) return null;
      const svgRect = svgEl.getBoundingClientRect();
      const localX = Math.min(PADDING.left + usableWidth, Math.max(PADDING.left, clientX - svgRect.left));
      const xVal = xScale.invert(localX);
      const nearest = sortedKnots.reduce(
        (best, item) => {
          const dist = Math.abs(item.x - xVal);
          return dist < best.dist ? { idx: item.idx, dist } : best;
        },
        { idx: sortedKnots[0].idx, dist: Number.POSITIVE_INFINITY }
      );
      return nearest.idx;
    };

    const computeSmoothedTarget = (centerIdx: number, dynamicAmount: number) => {
      const live = knotsRef.current ?? knots;
      // Use live values while dragging so smoothing follows the evolving curve,
      // instead of repeatedly pulling toward the initial press snapshot.
      const base = { x: live.x, y: live.y };
      const amount = Math.max(0, Math.min(1, dynamicAmount));
      const radius = Math.max(1, Math.round(smoothingRangeMax * Math.max(0.1, amount)));
      const minIdx = Math.max(0, centerIdx - radius);
      const maxIdx = Math.min(base.y.length - 1, centerIdx + radius);
      if (minIdx >= maxIdx) return null;

      const gaussianRadius = Math.max(1, Math.round(radius * 1.25));
      const sigma = Math.max(1e-3, gaussianRadius / 2);
      const smoothed = smoothSeriesGaussianReflect(base.y, gaussianRadius, sigma).slice(minIdx, maxIdx + 1);
      const radiusSafe = Math.max(1, radius);

      const next = { x: [...base.x], y: [...base.y] };
      const deltas: Record<number, number> = {};
      let maxDelta = 0;
      for (let i = minIdx; i <= maxIdx; i += 1) {
        const current = base.y[i];
        const target = smoothed[i - minIdx];
        if (!Number.isFinite(current) || !Number.isFinite(target)) continue;
        const dist = Math.abs(i - centerIdx);
        // Apply smoothing across the full range with a soft center-weighted taper.
        const t = Math.min(1, dist / radiusSafe);
        const influence = Math.pow(0.5 * (1 + Math.cos(Math.PI * t)), 2);
        let nextVal = current + (target - current) * influence;

        // Bound each update inside a small local envelope to avoid creating
        // new opposite extrema or jagged spikes while smoothing.
        const envStart = Math.max(minIdx, i - 2);
        const envEnd = Math.min(maxIdx, i + 2);
        let envMin = Number.POSITIVE_INFINITY;
        let envMax = Number.NEGATIVE_INFINITY;
        for (let j = envStart; j <= envEnd; j += 1) {
          const v = base.y[j];
          if (!Number.isFinite(v)) continue;
          if (v < envMin) envMin = v;
          if (v > envMax) envMax = v;
        }
        if (Number.isFinite(envMin) && Number.isFinite(envMax)) {
          nextVal = Math.max(envMin, Math.min(envMax, nextVal));
        }

        next.y[i] = nextVal;
        const delta = Math.abs(nextVal - current);
        deltas[i] = delta;
        if (delta > maxDelta) maxDelta = delta;
      }

      const weights: Record<number, number> = {};
      if (maxDelta > 0) {
        Object.entries(deltas).forEach(([key, value]) => {
          weights[Number(key)] = Math.min(1, value / maxDelta);
        });
      }
      return { next, weights, minIdx, maxIdx };
    };

    const applySmoothingHover = (hoverIdx: number | null) => {
      if (!smoothingMode) return;
      smoothHoverIdxRef.current = hoverIdx;
      if (hoverIdx == null) {
        smoothWeightsRef.current = {};
      } else {
        const target = computeSmoothedTarget(hoverIdx, smoothAmount);
        smoothWeightsRef.current = target?.weights ?? {};
      }
      updateKnotFill();
    };

    const applySmoothingStep = (dtSec: number) => {
      if (!smoothingMode || !smoothingDragActiveRef.current) return;
      const centerIdx = smoothingTargetIdxRef.current;
      if (centerIdx == null) return;
      const target = computeSmoothedTarget(centerIdx, smoothAmount);
      if (!target) return;
      const live = knotsRef.current ?? knots;
      const next = { x: [...live.x], y: [...live.y] };
      const dynamicAmount = Math.max(0, Math.min(1, smoothAmount));
      const ratePerSec = 0.6 + dynamicAmount * 2.0;
      const alpha = Math.max(0.005, Math.min(0.08, 1 - Math.exp(-ratePerSec * Math.min(0.033, dtSec))));
      const deltas: Record<number, number> = {};
      let maxDelta = 0;
      for (let i = target.minIdx; i <= target.maxIdx; i += 1) {
        const current = live.y[i];
        const targetVal = target.next.y[i];
        if (!Number.isFinite(current) || !Number.isFinite(targetVal)) continue;
        const nextVal = current + (targetVal - current) * alpha;
        next.y[i] = nextVal;
        const delta = Math.abs(nextVal - current);
        deltas[i] = delta;
        if (delta > maxDelta) maxDelta = delta;
      }
      const stepWeights: Record<number, number> = {};
      if (maxDelta > 0) {
        Object.entries(deltas).forEach(([key, value]) => {
          stepWeights[Number(key)] = Math.min(1, value / maxDelta);
        });
      } else {
        Object.assign(stepWeights, target.weights);
      }
      knotsRef.current = next;
      smoothingPreviewRef.current = next;
      smoothWeightsRef.current = stepWeights;
      applyDragPreview(next);
    };

    svg.attr("width", width).attr("height", HEIGHT);
    const content = svg.selectAll<SVGGElement, null>("g.shape-root").data([null]).join("g").classed("shape-root", true);
    content
      .selectAll<SVGRectElement, null>("rect.shape-bg")
      .data([null])
      .join("rect")
      .classed("shape-bg", true)
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", HEIGHT)
      .attr("fill", "#f9fafb")
      .attr("stroke", "none");

    const xAxis = axisBottom(xScale)
      .ticks(10)
      .tickSize(6)
      .tickPadding(8)
      .tickSizeOuter(0);
    const yAxis = axisLeft(yScale).ticks(7).tickSize(-usableWidth).tickSizeOuter(0);
    content
      .selectAll<SVGGElement, null>("g.x-axis")
      .data([null])
      .join("g")
      .classed("x-axis", true)
      .attr("transform", `translate(0, ${PADDING.top + usableHeight})`)
      .call(xAxis);
    content
      .selectAll<SVGGElement, null>("g.y-axis")
      .data([null])
      .join("g")
      .classed("y-axis", true)
      .attr("transform", `translate(${PADDING.left}, 0)`)
      .call(yAxis);
    content.selectAll(".tick line").attr("stroke", "#e2e8f0").attr("stroke-dasharray", "3 3");
    content.selectAll(".tick text").attr("fill", "#475569").attr("font-size", 10);

    content
      .selectAll<SVGLineElement, null>("line.zero-line")
      .data([null])
      .join("line")
      .classed("zero-line", true)
      .attr("x1", PADDING.left)
      .attr("x2", PADDING.left + usableWidth)
      .attr("y1", yScale(0))
      .attr("y2", yScale(0))
      .attr("stroke", "#cbd5e1")
      .attr("stroke-dasharray", "4 4");

    const sortedKnots = knots.x.map((x, idx) => ({ x, y: knots.y[idx] ?? 0, idx })).sort((a, b) => a.x - b.x);
    const avail = Math.max(8, PADDING.bottom - 8);
    const histMax = Math.min(avail, usableHeight * 0.18);
    const histBase = PADDING.top + usableHeight;

    content
      .selectAll<SVGLineElement, null>("line.y-axis-extend")
      .data([null])
      .join("line")
      .classed("y-axis-extend", true)
      .attr("x1", PADDING.left)
      .attr("x2", PADDING.left)
      .attr("y1", PADDING.top)
      .attr("y2", histBase + histMax)
      .attr("stroke", "#0f172a");

    // Histogram of raw x values (overlaid at bottom of plot)
    const { bins, binStart, binWidth } = histogram;
    let maxBin = 1;
    for (let i = 0; i < bins.length; i += 1) {
      if (bins[i] > maxBin) maxBin = bins[i];
    }
    const histData = bins.map((count, i) => ({ count, i }));
    const histSel = content
      .selectAll<SVGRectElement, any>("rect.hist-bar")
      .data(histData, (d: any) => d.i)
      .join(
        (enter) => enter.append("rect").classed("hist-bar", true),
        (update) => update,
        (exit) => exit.remove()
      );
    histSel
      .attr("x", (d) => {
        const x0 = binStart + d.i * binWidth;
        return xScale(x0);
      })
      .attr("y", () => histBase)
      .attr("width", (d) => {
        const x0 = binStart + d.i * binWidth;
        const x1 = x0 + binWidth;
        const px0 = xScale(x0);
        const px1 = xScale(x1);
        return Math.max(0, px1 - px0);
      })
      .attr("height", (d) => (d.count / maxBin) * histMax)
      .attr("fill", "rgba(14,165,233,0.12)")
      .attr("stroke", "rgba(14,165,233,0.25)")
      .attr("stroke-width", 0.5);

    content
      .selectAll<SVGTextElement, null>("text.density-label")
      .data([null])
      .join("text")
      .classed("density-label", true)
      .attr("x", PADDING.left)
      .attr("y", HEIGHT - 10)
      .attr("fill", "#64748b")
      .attr("font-size", 10)
      .text("Density");

    const baseData =
      baseline?.x?.length && baseline.y?.length
        ? baseline.x.map((x, idx) => ({ x, y: baseline.y[idx] ?? 0 })).sort((a, b) => a.x - b.x)
        : [];
    content
      .selectAll<SVGPathElement, any>("path.baseline-path")
      .data(baseData.length ? [baseData] : [])
      .join(
        (enter) => enter.append("path").classed("baseline-path", true),
        (update) => update,
        (exit) => exit.remove()
      )
      .attr("fill", "none")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2.2)
      .attr("stroke-dasharray", "6 4")
      .attr("d", lineGen);
    content
      .selectAll<SVGPathElement, any>("path.shape-path")
      .data([sortedKnots])
      .join(
        (enter) => enter.append("path").classed("shape-path", true),
        (update) => update,
        (exit) => exit.remove()
      )
      .attr("fill", "none")
      .attr("stroke", "#0ea5e9")
      .attr("stroke-width", 3)
      .attr("d", lineGen);

    const baseColor = [14, 165, 233];
    const highlightColor = [225, 29, 72];
    const blendColor = (t: number) => {
      if (!Number.isFinite(t)) return "rgb(14, 165, 233)";
      const clamped = Math.max(0, Math.min(1, t));
      const r = Math.round(baseColor[0] + (highlightColor[0] - baseColor[0]) * clamped);
      const g = Math.round(baseColor[1] + (highlightColor[1] - baseColor[1]) * clamped);
      const b = Math.round(baseColor[2] + (highlightColor[2] - baseColor[2]) * clamped);
      return `rgb(${r}, ${g}, ${b})`;
    };
    const updatePathHighlight = () => {
      const isDrag = isDraggingRef.current;
      const smoothWeights = smoothingMode && smoothHoverIdxRef.current != null ? smoothWeightsRef.current : {};
      const maxWeight = isDrag
        ? dragWeightsMaxRef.current
        : Math.max(0, ...Object.values(smoothWeights).map((v) => (Number.isFinite(v) ? v : 0)));
      const shapePath = content.selectAll<SVGPathElement, KnotDatum[]>("path.shape-path");
      if (maxWeight <= 0 || sortedKnots.length < 2) {
        shapePath.attr("stroke", blendColor(0)).attr("stroke-width", 3);
        return;
      }

      const gradientId = `shape-highlight-${featureKey.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
      const defs = svg.selectAll<SVGDefsElement, null>("defs.shape-defs").data([null]).join("defs").classed("shape-defs", true);
      const gradient = defs
        .selectAll<SVGLinearGradientElement, null>(`linearGradient#${gradientId}`)
        .data([null])
        .join("linearGradient")
        .attr("id", gradientId)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", PADDING.left)
        .attr("x2", PADDING.left + usableWidth)
        .attr("y1", 0)
        .attr("y2", 0);

      gradient
        .selectAll<SVGStopElement, KnotDatum>("stop")
        .data(sortedKnots)
        .join("stop")
        .attr("offset", (_, i) => `${(i / Math.max(1, sortedKnots.length - 1)) * 100}%`)
        .attr("stop-color", (d) => {
          const raw = isDrag ? dragWeightsRef.current[d.idx] : smoothWeights[d.idx];
          const w = Number.isFinite(raw) ? raw : 0;
          return blendColor(w);
        });

      shapePath.attr("stroke", `url(#${gradientId})`).attr("stroke-width", 3.6);
    };
    const updateKnotFill = (selection?: Selection<SVGCircleElement, any, SVGGElement, unknown>) => {
      const selectedSet = new Set(selectedRef.current);
      const sel = selection ?? content.selectAll<SVGCircleElement, any>("circle.knot");
      if (!SHOW_KNOT_MARKERS) {
        sel.attr("fill", "transparent").attr("stroke", "none").attr("stroke-width", 0).attr("r", 0);
        updatePathHighlight();
        return;
      }
      sel.attr("fill", (d: any) => {
        const idx = d.idx as number;
        if (isDraggingRef.current) {
          const raw = dragWeightsRef.current[idx];
          const weight = Number.isFinite(raw) ? (raw as number) : 0;
          return blendColor(weight);
        }
        if (smoothingMode && smoothHoverIdxRef.current != null) {
          const raw = smoothWeightsRef.current[idx];
          const weight = Number.isFinite(raw) ? (raw as number) : 0;
          if (weight > 0) return blendColor(weight);
        }
        return selectedSet.has(idx) ? "#0c4a6e" : "#0ea5e9";
      });
      updatePathHighlight();
    };
    const applyDragPreview = (next: { x: number[]; y: number[] }) => {
      const nextSorted = sortedKnots.map((d) => ({ ...d, y: next.y[d.idx] ?? d.y }));
      content.selectAll<SVGPathElement, any>("path.shape-path").attr("d", lineGen(nextSorted) ?? "");
      content
        .selectAll<SVGCircleElement, any>("circle.knot")
        .attr("cy", (d: any) => yScale(next.y[d.idx] ?? d.y));
      updateKnotFill();
      const selectedSet = new Set(selectedRef.current);
      const selectedKnots = nextSorted.filter((d) => selectedSet.has(d.idx));
      if (selectedKnots.length > 1) {
        const minBoxHeight = 24;
        const pad = 10;
        const xs = selectedKnots.map((d) => xScale(d.x));
        const ys = selectedKnots.map((d) => yScale(d.y));
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const rawHeight = Math.max(0, maxY - minY);
        const extra = Math.max(0, minBoxHeight - rawHeight);
        const yPad = pad + extra / 2;
        content
          .selectAll<SVGRectElement, null>("rect.selection-box")
          .data([null])
          .join("rect")
          .classed("selection-box", true)
          .classed("drag-handle", true)
          .attr("x", minX - pad)
          .attr("y", minY - yPad)
          .attr("width", Math.max(0, maxX - minX + pad * 2))
          .attr("height", Math.max(minBoxHeight, rawHeight + yPad * 2))
          .style("cursor", "grab")
          .style("fill", "rgba(15, 23, 42, 0.04)")
          .style("stroke", "rgba(15, 23, 42, 0.35)")
          .style("stroke-dasharray", "4 4")
          .style("pointer-events", "all");
      }
    };
    const applySelectionPreview = () => {
      content
        .selectAll<SVGCircleElement, any>("circle.knot")
        .attr("stroke", "#0b172a")
        .attr("stroke-width", 1);
      updateKnotFill();
    };

    const DRAG_FALLOFF_RADIUS = Math.max(0, dragFalloffRadius);
    const dragBehaviour = d3Drag<SVGCircleElement, { idx: number }>()
      .filter(() => !smoothingMode)
      .on("start", (event, d) => {
        isDraggingRef.current = true;
        const liveKnots = knotsRef.current ?? knots;
        const multi = event.sourceEvent.shiftKey || event.sourceEvent.metaKey || event.sourceEvent.ctrlKey;
        const nextSelection = d && d.idx != null
          ? resolveDragSelection({
              current: selectedRef.current,
              idx: d.idx,
              multi,
              mode: "contiguous",
            }).next
          : selectedRef.current;
        pendingSelectionRef.current = nextSelection;
        applySelectionPreview();
        if (d && d.idx != null) {
          dragTargetsRef.current = nextSelection.length ? nextSelection : [d.idx];
        } else {
          dragTargetsRef.current = nextSelection;
        }
        const svgRect = svgEl.getBoundingClientRect();
        const localY = Math.min(PADDING.top + usableHeight, Math.max(PADDING.top, event.sourceEvent.clientY - svgRect.top));
        dragStartYRef.current = yScale.invert(localY);
        dragStartXRef.current = event.sourceEvent.clientX;
        dragStartMapRef.current = [...liveKnots.y];
        if (dragWeightsRef.current.length !== liveKnots.y.length) {
          dragWeightsRef.current = Array.from({ length: liveKnots.y.length }, () => 0);
        } else {
          dragWeightsRef.current.fill(0);
        }
        dragWeightsMaxRef.current = 0;
        pendingDragRef.current = { x: liveKnots.x, y: [...liveKnots.y] };
        if (onDragStartRef.current) onDragStartRef.current({ x: [...liveKnots.x], y: [...liveKnots.y] });
      })
      .on("drag", (event, d) => {
        const svgRect = svgEl.getBoundingClientRect();
        const localY = Math.min(PADDING.top + usableHeight, Math.max(PADDING.top, event.sourceEvent.clientY - svgRect.top));
        const newY = yScale.invert(localY);
        const dragStartX = dragStartXRef.current;
        const dxPx = dragStartX == null ? 0 : event.sourceEvent.clientX - dragStartX;
        const liveKnots = knotsRef.current ?? knots;
        const targets = dragTargetsRef.current.length
          ? dragTargetsRef.current
          : d && d.idx != null
            ? [d.idx]
            : [];
        if (!targets.length) return;
        const startMap = dragStartMapRef.current;
        const dragStartY = dragStartYRef.current;
        const delta = dragStartY == null ? 0 : newY - dragStartY;
        const sortedTargets = [...targets].sort((a, b) => a - b);
        const avgSpacingPx = (() => {
          if (sortedKnots.length < 2) return 24;
          let total = 0;
          for (let i = 1; i < sortedKnots.length; i += 1) {
            total += Math.abs(xScale(sortedKnots[i].x) - xScale(sortedKnots[i - 1].x));
          }
          return Math.max(6, total / (sortedKnots.length - 1));
        })();
        const radiusBoost = avgSpacingPx > 0 ? dxPx / avgSpacingPx : 0;
        const dynamicRadius = Math.max(0, Math.min(60, DRAG_FALLOFF_RADIUS + radiusBoost));
        const dynamicSigma = dynamicRadius > 0 ? dynamicRadius / 2 : 1;
        const nearestDistance = (idx: number) => {
          if (!sortedTargets.length) return Number.POSITIVE_INFINITY;
          let best = Math.abs(sortedTargets[0] - idx);
          for (let i = 1; i < sortedTargets.length; i += 1) {
            const dist = Math.abs(sortedTargets[i] - idx);
            if (dist < best) best = dist;
            if (best === 0) break;
          }
          return best;
        };
        const pending = pendingDragRef.current ?? { x: liveKnots.x, y: [...liveKnots.y] };
        pending.x = liveKnots.x;
        if (pending.y.length !== liveKnots.y.length) {
          pending.y = [...liveKnots.y];
        }
        const nextY = pending.y;
        const weights = dragWeightsRef.current;
        const fade = Math.max(1, dynamicSigma);
        let maxWeight = 0;
        for (let idx = 0; idx < nextY.length; idx += 1) {
          const base = startMap[idx] ?? liveKnots.y[idx];
          const dist = nearestDistance(idx);
          if (!Number.isFinite(dist)) {
            weights[idx] = 0;
            nextY[idx] = base;
            continue;
          }
          if (dist === 0) {
            weights[idx] = 1;
            nextY[idx] = base + delta;
            maxWeight = 1;
            continue;
          }
          if (dynamicRadius === 0) {
            weights[idx] = 0;
            nextY[idx] = base;
            continue;
          }
          if (dist > dynamicRadius + fade) {
            weights[idx] = 0;
            nextY[idx] = base;
            continue;
          }
          const weight = Math.exp(-(dist * dist) / (2 * dynamicSigma * dynamicSigma));
          const tapered =
            dist > dynamicRadius
              ? weight * 0.5 * (1 + Math.cos(Math.PI * (dist - dynamicRadius) / fade))
              : weight;
          weights[idx] = tapered;
          if (tapered > maxWeight) maxWeight = tapered;
          nextY[idx] = base + delta * tapered;
        }
        dragWeightsMaxRef.current = maxWeight;
        pendingDragRef.current = pending;
        if (rafIdRef.current == null) {
          rafIdRef.current = window.requestAnimationFrame(() => {
            rafIdRef.current = null;
            if (pendingDragRef.current) {
              applyDragPreview(pendingDragRef.current);
            }
          });
        }
      })
      .on("end", () => {
        dragTargetsRef.current = [];
        isDraggingRef.current = false;
        dragStartMapRef.current = [];
        dragStartYRef.current = null;
        dragStartXRef.current = null;
        dragWeightsRef.current.fill(0);
        dragWeightsMaxRef.current = 0;
        if (pendingSelectionRef.current) {
          onSelectionChangeRef.current(pendingSelectionRef.current);
          pendingSelectionRef.current = null;
        }
        if (rafIdRef.current != null) {
          window.cancelAnimationFrame(rafIdRef.current);
          rafIdRef.current = null;
        }
        const pending = pendingDragRef.current;
        if (pending) {
          const committed = { x: [...pending.x], y: [...pending.y] };
          knotsRef.current = committed;
          applyDragPreview(committed);
          onKnotChangeRef.current(committed);
        }
        pendingDragRef.current = null;
        if (onDragEndRef.current) {
          const liveKnots = knotsRef.current ?? knots;
          onDragEndRef.current({ x: [...liveKnots.x], y: [...liveKnots.y] });
        }
      });
    dragBehaviorRef.current = dragBehaviour;

    if (interactionMode === "select") {
      const brushLayer = content
        .selectAll<SVGGElement, null>("g.brush-layer")
        .data([null])
        .join("g")
        .classed("brush-layer", true);
      const brushBehavior = d3Brush()
        .extent([
          [PADDING.left, PADDING.top],
          [PADDING.left + usableWidth, PADDING.top + usableHeight],
        ])
        .filter((event) => {
          const target = event.target as Element | null;
          return !isDraggingRef.current && !target?.closest(".drag-handle");
        })
        .on("start", () => {
          if (!isDraggingRef.current) return;
        })
        .on("end", (event) => {
          if (!event.selection) {
            if (!event.sourceEvent) return;
            const isClick = event.sourceEvent.type === "click";
            if (isClick) onSelectionChangeRef.current([]);
            return;
          }
          const [[x0], [x1]] = event.selection as [[number, number], [number, number]];
          const minX = Math.min(x0, x1);
          const maxX = Math.max(x0, x1);
          const selectedIndices = sortedKnots
            .filter((d) => {
              const px = xScale(d.x);
              return px >= minX && px <= maxX;
            })
            .map((d) => d.idx);
          const multi = event.sourceEvent?.shiftKey || event.sourceEvent?.metaKey || event.sourceEvent?.ctrlKey;
          if (multi) {
            const next = applyBrushSelection({
              current: selectedRef.current,
              selected: selectedIndices,
              multi: true,
              mode: "contiguous",
            });
            onSelectionChangeRef.current(next);
          } else {
            const next = applyBrushSelection({
              current: selectedRef.current,
              selected: selectedIndices,
              multi: false,
              mode: "contiguous",
            });
            onSelectionChangeRef.current(next);
          }
          brushLayer.call(brushBehavior.move as any, null);
        });
      brushLayer.call(brushBehavior as any);
      svg.on(".zoom", null);
    } else {
      content.selectAll("g.brush-layer").remove();
      const zoomBehavior = zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 6])
        .filter((event) => {
          const target = event.target as Element | null;
          return !isDraggingRef.current && !target?.closest(".drag-handle");
        })
        .on("zoom", (event) => {
          zoomRef.current = event.transform;
          const zx = event.transform.rescaleX(baseXScale);
          const zy = event.transform.rescaleY(baseYScale);
          const lineGenZoom = d3Line<{ x: number; y: number }>()
            .x((d) => zx(d.x))
            .y((d) => zy(d.y))
            .curve(curveLinear);
          content.selectAll<SVGGElement, null>("g.x-axis").call(axisBottom(zx).ticks(10).tickSize(6).tickPadding(8).tickSizeOuter(0) as any);
          content.selectAll<SVGGElement, null>("g.y-axis").call(axisLeft(zy).ticks(7).tickSize(-usableWidth).tickSizeOuter(0) as any);
          content.selectAll(".tick line").attr("stroke", "#e2e8f0").attr("stroke-dasharray", "3 3");
          content.selectAll(".tick text").attr("fill", "#475569").attr("font-size", 10);
          content
            .selectAll<SVGLineElement, null>("line.zero-line")
            .attr("y1", zy(0))
            .attr("y2", zy(0));
          content
            .selectAll<SVGRectElement, any>("rect.hist-bar")
            .attr("x", (d: any) => zx(binStart + d.i * binWidth))
            .attr("width", (d: any) => {
              const x0 = binStart + d.i * binWidth;
              const x1 = x0 + binWidth;
              return Math.max(0, zx(x1) - zx(x0));
            });
          content
            .selectAll<SVGPathElement, any>("path.baseline-path")
            .attr("d", lineGenZoom as any);
          content
            .selectAll<SVGPathElement, any>("path.shape-path")
            .attr("d", lineGenZoom as any);
          content
            .selectAll<SVGCircleElement, any>("circle.knot")
            .attr("cx", (d: any) => zx(d.x))
            .attr("cy", (d: any) => zy(d.y));
        });
      svg.call(zoomBehavior as any);
      svg.call(zoomBehavior.transform as any, zoomRef.current ?? zoomIdentity);
    }

    const knotsSel = content.selectAll<SVGCircleElement, any>("circle.knot").data(sortedKnots, (d: any) => d.idx);
    const hitSize = 20;
    content
      .selectAll<SVGRectElement, any>("rect.knot-hitbox")
      .data(sortedKnots, (d: any) => d.idx)
      .join(
        (enter) =>
          enter
            .append("rect")
            .classed("knot-hitbox", true)
            .classed("drag-handle", true)
            .attr("width", hitSize)
            .attr("height", hitSize)
            .style("cursor", "grab")
            .style("fill", "transparent")
            .on("click", (event, d) => {
              event.stopPropagation();
              const multi = event.shiftKey || event.metaKey || event.ctrlKey;
              const nextSelection = (() => {
                if (multi) {
                  const current = selectedRef.current;
                  return applyClickSelection({ current, idx: d.idx, multi: true, mode: "contiguous" });
                }
                const current = selectedRef.current;
                return applyClickSelection({ current, idx: d.idx, multi: false, mode: "contiguous" });
              })();
              onSelectionChangeRef.current(nextSelection);
            })
            .on("mouseenter", (_, d) => applySmoothingHover(d.idx))
            .on("mouseleave", () => applySmoothingHover(null))
            .call(dragBehaviour as any),
        (update) => update,
        (exit) => exit.remove()
      )
      .attr("x", (d) => xScale(d.x) - hitSize / 2)
      .attr("y", (d) => yScale(d.y) - hitSize / 2);

    knotsSel
      .join(
        (enter) =>
          enter
            .append("circle")
            .classed("knot", true)
            .classed("drag-handle", true)
            .attr("r", SHOW_KNOT_MARKERS ? 5 : 0)
            .attr("fill", SHOW_KNOT_MARKERS ? "#0ea5e9" : "transparent")
            .style("cursor", "grab")
            .on("click", (event, d) => {
              event.stopPropagation();
              const multi = event.shiftKey || event.metaKey || event.ctrlKey;
              const nextSelection = (() => {
                if (multi) {
                  const current = selectedRef.current;
                  return applyClickSelection({ current, idx: d.idx, multi: true, mode: "contiguous" });
                }
                const current = selectedRef.current;
                return applyClickSelection({ current, idx: d.idx, multi: false, mode: "contiguous" });
              })();
              onSelectionChangeRef.current(nextSelection);
            })
            .call(dragBehaviour as any),
        (update) => update,
        (exit) => exit.remove()
      )
      .on("mouseenter", (_, d) => applySmoothingHover(d.idx))
      .on("mouseleave", () => applySmoothingHover(null))
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("stroke", SHOW_KNOT_MARKERS ? "#0b172a" : "none")
      .attr("stroke-width", SHOW_KNOT_MARKERS ? 1 : 0)
      .attr("r", SHOW_KNOT_MARKERS ? 5 : 0);

    svg.on("click", () => onSelectionChangeRef.current([]));
    svg.on("mouseleave", () => applySmoothingHover(null));

    const smoothingLayer = content
      .selectAll<SVGRectElement, null>("rect.smooth-layer")
      .data([null])
      .join("rect")
      .classed("smooth-layer", true)
      .attr("x", PADDING.left)
      .attr("y", PADDING.top)
      .attr("width", usableWidth)
      .attr("height", usableHeight)
      .style("fill", "transparent")
      .style("pointer-events", smoothingMode ? "all" : "none")
      .style("cursor", smoothingMode ? (smoothingDragActiveRef.current ? "grabbing" : "grab") : "default");

    const smoothingDrag = d3Drag<SVGRectElement, null>()
      .filter((event) => {
        const target = event.target as Element | null;
        return smoothingMode && !isDraggingRef.current && !target?.closest(".drag-handle");
      })
      .on("start", (event) => {
        smoothingDragActiveRef.current = true;
        smoothingLayer.style("cursor", "grabbing");
        svg.style("cursor", "grabbing");
        const liveKnots = knotsRef.current ?? knots;
        smoothingLastTsRef.current = null;
        smoothingBaseRef.current = { x: [...liveKnots.x], y: [...liveKnots.y] };
        smoothingPreviewRef.current = null;
        const nearestIdx = findNearestKnotIdxAtClientX(event.sourceEvent.clientX);
        smoothingTargetIdxRef.current = nearestIdx;
        applySmoothingHover(nearestIdx);
        applySmoothingStep(1 / 60);
        if (smoothingRafRef.current == null) {
          const tick = (ts: number) => {
            smoothingRafRef.current = null;
            if (!smoothingDragActiveRef.current) return;
            const last = smoothingLastTsRef.current ?? ts;
            const dtSec = Math.max(0, (ts - last) / 1000);
            smoothingLastTsRef.current = ts;
            applySmoothingStep(dtSec);
            smoothingRafRef.current = window.requestAnimationFrame(tick);
          };
          smoothingRafRef.current = window.requestAnimationFrame(tick);
        }
      })
      .on("drag", (event) => {
        const nearestIdx = findNearestKnotIdxAtClientX(event.sourceEvent.clientX);
        smoothingTargetIdxRef.current = nearestIdx;
        applySmoothingHover(nearestIdx);
        applySmoothingStep(1 / 60);
      })
      .on("end", () => {
        smoothingDragActiveRef.current = false;
        const idleCursor = smoothingMode ? "grab" : "default";
        smoothingLayer.style("cursor", idleCursor);
        svg.style("cursor", idleCursor);
        smoothingTargetIdxRef.current = null;
        smoothingLastTsRef.current = null;
        if (smoothingRafRef.current != null) {
          window.cancelAnimationFrame(smoothingRafRef.current);
          smoothingRafRef.current = null;
        }
        const base = smoothingBaseRef.current;
        const preview = smoothingPreviewRef.current;
        if (preview) {
          knotsRef.current = preview;
          onKnotChangeRef.current(preview);
          if (base && onSmoothEndRef.current) {
            onSmoothEndRef.current(base, preview);
          }
        }
        smoothingBaseRef.current = null;
        smoothingPreviewRef.current = null;
        applySmoothingHover(null);
      });

    smoothingLayer.call(smoothingDrag as any);
    smoothingLayer
      .on("mousemove", (event) => {
        if (!smoothingMode || smoothingDragActiveRef.current) return;
        const nearestIdx = findNearestKnotIdxAtClientX(event.clientX);
        smoothingTargetIdxRef.current = nearestIdx;
        applySmoothingHover(nearestIdx);
      })
      .on("mouseleave", () => applySmoothingHover(null));
    smoothingLayer.raise();

    svg.style("cursor", smoothingMode ? (smoothingDragActiveRef.current ? "grabbing" : "grab") : "default");
  }, [
    knots,
    baseline,
    width,
    featureKey,
    title,
    histogram,
    interactionMode,
    dragFalloffRadius,
    smoothingMode,
    smoothAmount,
    smoothingRangeMax,
    selected,
  ]);

  return (
    <div ref={containerRef} className={styles.chartFrame} style={{ width: "100%" }}>
      <svg ref={svgRef} aria-label={title} />
    </div>
  );
}
