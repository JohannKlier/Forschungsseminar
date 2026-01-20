import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { pointer, select } from "d3-selection";
import { scaleBand, scaleLinear } from "d3-scale";
import { axisLeft, axisBottom } from "d3-axis";
import { drag as d3Drag } from "d3-drag";
import { brushX } from "d3-brush";
import { zoom, zoomIdentity, type ZoomTransform } from "d3-zoom";
import styles from "../page.module.css";
import { KnotSet } from "../types";
import { applyBrushSelection, applyClickSelection, resolveDragSelection } from "../lib/selection";

type Props = {
  categories: string[];
  knots: KnotSet;
  baseline?: KnotSet;
  title: string;
  fixedRange?: { min: number; max: number };
  selectedIdxs: number[];
  onSelect: (next: number[]) => void;
  onValueChange: (idx: number, value: number) => void;
  onMultiValueChange: (indices: number[], values: Record<number, number>) => void;
  onDragStart: () => void;
  onDragEnd: () => void;
  interactionMode: "select" | "zoom";
};

export default function CategoricalShapePlot({
  categories,
  knots,
  baseline,
  title,
  fixedRange,
  selectedIdxs,
  onSelect,
  onValueChange,
  onMultiValueChange,
  onDragStart,
  onDragEnd,
  interactionMode,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const isDraggingRef = useRef(false);
  const onSelectRef = useRef(onSelect);
  const onValueChangeRef = useRef(onValueChange);
  const onMultiValueChangeRef = useRef(onMultiValueChange);
  const onDragStartRef = useRef(onDragStart);
  const onDragEndRef = useRef(onDragEnd);
  const selectedIdxsRef = useRef<number[]>(selectedIdxs);
  const pendingSelectionRef = useRef<number[] | null>(null);
  const zoomRef = useRef<ZoomTransform | null>(null);
  const dragTargetsRef = useRef<number[]>([]);
  const pendingDragRef = useRef<{ idx: number; value: number; indices: number[] } | null>(null);
  const dragStartValsRef = useRef<Record<number, number>>({});
  const dragOffsetRef = useRef<number>(0);
  const knotsRef = useRef<KnotSet>(knots);
  const clipIdRef = useRef(`cat-clip-${Math.random().toString(36).slice(2, 9)}`);
  const [width, setWidth] = useState(0);
  const height = 560;
  const pad = { top: 16, right: 16, bottom: 52, left: 56 };
  const usableH = height - pad.top - pad.bottom;
  const computedRange = (() => {
    const vals = (knots.y ?? []).filter((v) => Number.isFinite(v));
    const fallback = { min: -1, max: 1 };
    if (!vals.length) return fallback;
    const minY = Math.min(...vals);
    const maxY = Math.max(...vals);
    const span = maxY - minY || Math.max(Math.abs(minY), Math.abs(maxY), 1);
    const padSpan = span * 0.08;
    return { min: minY - padSpan, max: maxY + padSpan };
  })();
  const yRange = fixedRange ?? computedRange;

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width;
      if (w) setWidth(w);
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    onSelectRef.current = onSelect;
    onValueChangeRef.current = onValueChange;
    onMultiValueChangeRef.current = onMultiValueChange;
    onDragStartRef.current = onDragStart;
    onDragEndRef.current = onDragEnd;
  }, [onSelect, onValueChange, onMultiValueChange, onDragStart, onDragEnd]);

  useEffect(() => {
    selectedIdxsRef.current = selectedIdxs;
  }, [selectedIdxs]);

  useEffect(() => {
    knotsRef.current = knots;
  }, [knots]);

  const usableW = width - pad.left - pad.right;
  useLayoutEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    if (width <= 0) return;
    const svg = select(svgEl);
    const { min, max } = yRange;
    const baseYScale = scaleLinear().domain([min, max]).range([pad.top + usableH, pad.top]);
    const xScaleD3 = scaleBand<string>().domain(categories).range([pad.left, pad.left + usableW]).padding(0.15);
    const transform = zoomRef.current ?? zoomIdentity;
    const yScale = transform.rescaleY(baseYScale);
    const getYScale = () => (zoomRef.current ?? zoomIdentity).rescaleY(baseYScale);

    svg.attr("width", width).attr("height", height);
    const root = svg.selectAll<SVGGElement, null>("g.cat-root").data([null]).join("g").classed("cat-root", true);
    root
      .selectAll<SVGRectElement, null>("rect.cat-bg")
      .data([null])
      .join("rect")
      .classed("cat-bg", true)
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#f9fafb")
      .attr("stroke", "none");

    const defs = root.selectAll("defs").data([null]).join("defs");
    defs
      .selectAll("clipPath")
      .data([clipIdRef.current])
      .join("clipPath")
      .attr("id", clipIdRef.current)
      .selectAll("rect")
      .data([null])
      .join("rect")
      .attr("x", pad.left)
      .attr("y", pad.top)
      .attr("width", usableW)
      .attr("height", usableH);

    const yAxis = axisLeft(yScale).ticks(6).tickSize(-usableW).tickFormat((d) => `${(d as number).toFixed(1)}`);
    const tickValues = categories;
    const formatLabel = (label: string) => (label.length > 14 ? `${label.slice(0, 12)}…` : label);
    const xAxis = axisBottom(xScaleD3)
      .tickValues(tickValues)
      .tickSize(6)
      .tickPadding(8)
      .tickSizeOuter(0)
      .tickFormat((d) => formatLabel(String(d)));
    root
      .selectAll<SVGGElement, null>("g.y-axis")
      .data([null])
      .join("g")
      .classed("y-axis", true)
      .attr("transform", `translate(${pad.left},0)`)
      .call(yAxis as any)
      .selectAll("text")
      .attr("fill", "#475569")
      .attr("font-size", 10);
    root
      .selectAll<SVGGElement, null>("g.x-axis")
      .data([null])
      .join("g")
      .classed("x-axis", true)
      .attr("transform", `translate(0, ${pad.top + usableH})`)
      .call(xAxis as any)
      .selectAll("text")
      .attr("fill", "#475569")
      .attr("font-size", 10)
      .attr("text-anchor", "end")
      .attr("transform", "rotate(-30)")
      .attr("dx", "-0.4em")
      .attr("dy", "0.6em");
    root.selectAll(".domain").attr("stroke", "none");
    root.selectAll(".tick line").attr("stroke", "#e2e8f0").attr("stroke-dasharray", "3 3");

    root
      .selectAll<SVGLineElement, null>("line.zero-line")
      .data([null])
      .join("line")
      .classed("zero-line", true)
      .attr("x1", pad.left)
      .attr("x2", pad.left + usableW)
      .attr("y1", yScale(0))
      .attr("y2", yScale(0))
      .attr("stroke", "#cbd5e1")
      .attr("stroke-dasharray", "4 4");

    const valFromEvent = (event: any) => {
      const svgRect = svgEl.getBoundingClientRect();
      const clientY = event?.sourceEvent?.clientY ?? event?.sourceEvent?.touches?.[0]?.clientY ?? event?.y ?? 0;
      const localY = Math.min(pad.top + usableH, Math.max(pad.top, clientY - svgRect.top));
      return getYScale().invert(localY);
    };
    const updatePreview = (indices: number[], values: Record<number, number>) => {
      const previewScale = getYScale();
      const bars = root.selectAll<SVGGElement, null>("g.cat-bars");
      bars
        .selectAll<SVGRectElement, any>("rect.cat-bar")
        .filter((d: any) => indices.includes(d.idx))
        .attr("y", (d: any) => {
          const val = values[d.idx] ?? 0;
          const y0 = previewScale(Math.max(0, val));
          const yBase = previewScale(Math.min(0, val));
          return Math.min(y0, yBase);
        })
        .attr("height", (d: any) => {
          const val = values[d.idx] ?? 0;
          const y0 = previewScale(Math.max(0, val));
          const yBase = previewScale(Math.min(0, val));
          return Math.abs(y0 - yBase);
        });
      bars
        .selectAll<SVGCircleElement, any>("circle.cat-dot")
        .filter((d: any) => indices.includes(d.idx))
        .attr("cy", (d: any) => previewScale(values[d.idx] ?? 0));
    };
    const applySelectionPreview = (nextSelection: number[]) => {
      const selectedSet = new Set(nextSelection);
      const bars = root.selectAll<SVGGElement, null>("g.cat-bars");
      bars
        .selectAll<SVGRectElement, any>("rect.cat-bar")
        .attr("fill", (d: any) => (selectedSet.has(d.idx) ? "rgba(14,165,233,0.95)" : "rgba(14,165,233,0.7)"))
        .attr("stroke", (d: any) => (selectedSet.has(d.idx) ? "#0c4a6e" : "#0b172a"));
      bars
        .selectAll<SVGCircleElement, any>("circle.cat-dot")
        .attr("fill", (d: any) => (selectedSet.has(d.idx) ? "#0c4a6e" : "#0ea5e9"))
        .attr("stroke", "#0b172a")
        .attr("stroke-width", 1);
    };

    const dragBehaviour = d3Drag<SVGRectElement | SVGCircleElement, { idx: number }>()
      .on("start", (event, d) => {
        isDraggingRef.current = true;
        onDragStartRef.current();
        const multi = event.sourceEvent?.shiftKey || event.sourceEvent?.metaKey || event.sourceEvent?.ctrlKey;
        const { next, targets } = resolveDragSelection({
          current: selectedIdxsRef.current,
          idx: d.idx,
          multi: Boolean(multi),
          mode: "free",
        });
        pendingSelectionRef.current = next;
        applySelectionPreview(next);
        const val = valFromEvent(event);
        dragTargetsRef.current = targets;
        const startVals: Record<number, number> = {};
        const currentKnots = knotsRef.current;
        targets.forEach((idx) => {
          startVals[idx] = currentKnots.y[idx] ?? 0;
        });
        dragStartValsRef.current = startVals;
        const baseVal = startVals[d.idx] ?? 0;
        dragOffsetRef.current = baseVal - val;
        pendingDragRef.current = { idx: d.idx, value: val, indices: targets };
      })
      .on("drag", (event, d) => {
        const val = valFromEvent(event);
        const indices = dragTargetsRef.current.length ? dragTargetsRef.current : [d.idx];
        pendingDragRef.current = { idx: d.idx, value: val, indices };
        const baseVal = dragStartValsRef.current[d.idx] ?? 0;
        const desired = val + dragOffsetRef.current;
        const delta = desired - baseVal;
        const nextVals: Record<number, number> = {};
        indices.forEach((idx) => {
          const base = dragStartValsRef.current[idx] ?? 0;
          nextVals[idx] = base + delta;
        });
        updatePreview(indices, nextVals);
      })
      .on("end", () => {
        const pending = pendingDragRef.current;
        if (pending) {
          const selected = dragTargetsRef.current.length ? dragTargetsRef.current : selectedIdxsRef.current;
          if (selected.length > 1 && selected.includes(pending.idx)) {
            const baseVal = dragStartValsRef.current[pending.idx] ?? 0;
            const desired = pending.value + dragOffsetRef.current;
            const delta = desired - baseVal;
            const nextVals: Record<number, number> = {};
            selected.forEach((idx) => {
              const base = dragStartValsRef.current[idx] ?? 0;
              nextVals[idx] = base + delta;
            });
            onMultiValueChangeRef.current(selected, nextVals);
            const nextY = [...knotsRef.current.y];
            selected.forEach((idx) => {
              if (nextVals[idx] != null) nextY[idx] = nextVals[idx];
            });
            knotsRef.current = { x: [...knotsRef.current.x], y: nextY };
          } else {
            const baseVal = dragStartValsRef.current[pending.idx] ?? 0;
            const desired = pending.value + dragOffsetRef.current;
            const delta = desired - baseVal;
            const nextVal = baseVal + delta;
            onValueChangeRef.current(pending.idx, nextVal);
            const nextY = [...knotsRef.current.y];
            nextY[pending.idx] = nextVal;
            knotsRef.current = { x: [...knotsRef.current.x], y: nextY };
          }
        }
        pendingDragRef.current = null;
        dragStartValsRef.current = {};
        dragOffsetRef.current = 0;
        dragTargetsRef.current = [];
        if (pendingSelectionRef.current) {
          onSelectRef.current(pendingSelectionRef.current);
          pendingSelectionRef.current = null;
        }
        onDragEndRef.current();
        isDraggingRef.current = false;
      });

    if (interactionMode === "select") {
      const brushLayer = root
        .selectAll<SVGGElement, null>("g.cat-brush")
        .data([null])
        .join("g")
        .classed("cat-brush", true);
      const brush = brushX()
        .extent([
          [pad.left, pad.top],
          [pad.left + usableW, pad.top + usableH],
        ])
        .filter((event) => {
          const target = event.target as Element | null;
          return !isDraggingRef.current && !target?.closest(".drag-handle");
        })
        .on("end", (event) => {
          if (!event.selection) {
            if (!event.sourceEvent) return;
            const isClick = event.sourceEvent.type === "click";
            if (isClick) onSelectRef.current([]);
            return;
          }
          const [x0, x1] = event.selection as [number, number];
          const minX = Math.min(x0, x1);
          const maxX = Math.max(x0, x1);
          const selected = categories
            .map((cat, idx) => {
              const center = (xScaleD3(cat) ?? pad.left) + xScaleD3.bandwidth() / 2;
              return center >= minX && center <= maxX ? idx : null;
            })
            .filter((idx): idx is number => idx != null);
          const multi = event.sourceEvent?.shiftKey || event.sourceEvent?.metaKey || event.sourceEvent?.ctrlKey;
          if (multi) {
            const next = applyBrushSelection({
              current: selectedIdxsRef.current,
              selected,
              multi: true,
              mode: "free",
            });
            onSelectRef.current(next);
          } else {
            const next = applyBrushSelection({
              current: selectedIdxsRef.current,
              selected,
              multi: false,
              mode: "free",
            });
            onSelectRef.current(next);
          }
          brushLayer.call(brush.move as any, null);
        });
      brushLayer.call(brush as any);
    } else {
      root.selectAll("g.cat-brush").remove();
    }

    const bars = root
      .selectAll<SVGGElement, null>("g.cat-bars")
      .data([null])
      .join("g")
      .classed("cat-bars", true)
      .attr("clip-path", `url(#${clipIdRef.current})`);

    const barData = categories.map((cat, idx) => ({ cat, idx, val: knots.y[idx] ?? 0 }));
    const baseData = baseline ? categories.map((cat, idx) => ({ cat, idx, val: baseline.y[idx] ?? 0 })) : [];
    const baseSel = bars
      .selectAll<SVGRectElement, any>("rect.cat-bar-base")
      .data(baseData, (d: any) => d.idx)
      .join(
        (enter) =>
          enter
            .append("rect")
            .classed("cat-bar-base", true)
            .style("pointer-events", "none"),
        (update) => update,
        (exit) => exit.remove()
      );
    baseSel
      .attr("x", (d) => ((xScaleD3(d.cat) ?? pad.left) + xScaleD3.bandwidth() * 0.15))
      .attr("width", xScaleD3.bandwidth() * 0.7)
      .attr("fill", "rgba(148,163,184,0.12)")
      .attr("stroke", "rgba(148,163,184,0.6)")
      .attr("stroke-width", 1)
      .attr("y", (d) => {
        const y0 = yScale(Math.max(0, d.val));
        const yBase = yScale(Math.min(0, d.val));
        return Math.min(y0, yBase);
      })
      .attr("height", (d) => {
        const y0 = yScale(Math.max(0, d.val));
        const yBase = yScale(Math.min(0, d.val));
        return Math.abs(y0 - yBase);
      });
    const rectSel = bars
      .selectAll<SVGRectElement, any>("rect.cat-bar")
      .data(barData, (d: any) => d.idx)
      .join(
        (enter) =>
          enter
            .append("rect")
            .classed("cat-bar", true)
            .classed("drag-handle", true)
            .style("cursor", "grab")
            .on("click", (event, d) => {
              const multi = event.shiftKey || event.metaKey || event.ctrlKey;
              const current = selectedIdxsRef.current;
              const next = applyClickSelection({ current, idx: d.idx, multi: Boolean(multi), mode: "free" });
              onSelectRef.current(next);
            })
            .call(dragBehaviour as any),
        (update) => update,
        (exit) => exit.remove()
      );
    rectSel
      .attr("x", (d) => ((xScaleD3(d.cat) ?? pad.left) + xScaleD3.bandwidth() * 0.15))
      .attr("width", xScaleD3.bandwidth() * 0.7)
      .attr("fill", (d) => (selectedIdxs.includes(d.idx) ? "rgba(14,165,233,0.95)" : "rgba(14,165,233,0.7)"))
      .attr("stroke", (d) => (selectedIdxs.includes(d.idx) ? "#0c4a6e" : "#0b172a"));
    rectSel
      .attr("y", (d) => {
        const y0 = yScale(Math.max(0, d.val));
        const yBase = yScale(Math.min(0, d.val));
        return Math.min(y0, yBase);
      })
      .attr("height", (d) => {
        const y0 = yScale(Math.max(0, d.val));
        const yBase = yScale(Math.min(0, d.val));
        return Math.abs(y0 - yBase);
      });

    const dotSel = bars
      .selectAll<SVGCircleElement, any>("circle.cat-dot")
      .data(barData, (d: any) => d.idx)
      .join(
        (enter) =>
          enter
            .append("circle")
            .classed("cat-dot", true)
            .classed("drag-handle", true)
            .attr("r", 6)
            .style("cursor", "grab")
            .on("click", (event, d) => {
              const multi = event.shiftKey || event.metaKey || event.ctrlKey;
              const current = selectedIdxsRef.current;
              const next = applyClickSelection({ current, idx: d.idx, multi: Boolean(multi), mode: "free" });
              onSelectRef.current(next);
            })
            .call(dragBehaviour as any),
        (update) => update,
        (exit) => exit.remove()
      );
    dotSel
      .attr("cx", (d) => ((xScaleD3(d.cat) ?? pad.left) + xScaleD3.bandwidth() * 0.15 + xScaleD3.bandwidth() * 0.35))
      .attr("fill", (d) => (selectedIdxs.includes(d.idx) ? "#0c4a6e" : "#0ea5e9"))
      .attr("stroke", "#0b172a")
      .attr("stroke-width", 1);
    dotSel.attr("cy", (d) => yScale(d.val));

    bars.raise();
    bars.selectAll<SVGElement, unknown>(".drag-handle").raise();

    if (interactionMode === "zoom") {
      const zoomBehavior = zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 6])
        .filter((event) => {
          const target = event.target as Element | null;
          return !isDraggingRef.current && !target?.closest(".drag-handle");
        })
        .on("zoom", (event) => {
          zoomRef.current = event.transform;
          const zy = event.transform.rescaleY(baseYScale);
          root.selectAll<SVGGElement, null>("g.y-axis").call(axisLeft(zy).ticks(6).tickSize(-usableW).tickSizeOuter(0) as any);
          root.selectAll(".tick line").attr("stroke", "#e2e8f0").attr("stroke-dasharray", "3 3");
          root.selectAll(".tick text").attr("fill", "#475569").attr("font-size", 10);
          root
            .selectAll<SVGLineElement, null>("line.zero-line")
            .attr("y1", zy(0))
            .attr("y2", zy(0));
          bars
            .selectAll<SVGRectElement, any>("rect.cat-bar")
            .attr("y", (d: any) => {
              const y0 = zy(Math.max(0, d.val));
              const yBase = zy(Math.min(0, d.val));
              return Math.min(y0, yBase);
            })
            .attr("height", (d: any) => {
              const y0 = zy(Math.max(0, d.val));
              const yBase = zy(Math.min(0, d.val));
              return Math.abs(y0 - yBase);
            });
          bars
            .selectAll<SVGRectElement, any>("rect.cat-bar-base")
            .attr("y", (d: any) => {
              const y0 = zy(Math.max(0, d.val));
              const yBase = zy(Math.min(0, d.val));
              return Math.min(y0, yBase);
            })
            .attr("height", (d: any) => {
              const y0 = zy(Math.max(0, d.val));
              const yBase = zy(Math.min(0, d.val));
              return Math.abs(y0 - yBase);
            });
          bars.selectAll<SVGCircleElement, any>("circle.cat-dot").attr("cy", (d: any) => zy(d.val));
        });
      svg.call(zoomBehavior as any);
      svg.call(zoomBehavior.transform as any, zoomRef.current ?? zoomIdentity);
    } else {
      svg.on(".zoom", null);
    }
  }, [categories, knots, selectedIdxs, yRange.min, yRange.max, width, usableH, usableW, pad.top, pad.left, interactionMode]);

  return (
    <div ref={containerRef} className={styles.chartFrame}>
      <svg ref={svgRef} aria-label={title} />
    </div>
  );
}
