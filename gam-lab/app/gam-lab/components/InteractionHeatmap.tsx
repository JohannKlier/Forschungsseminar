"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as d3 from "d3";
import styles from "../page.module.css";
import { ShapeFunction } from "../types";

type Props = {
  shape: ShapeFunction; // must have editableZ set
  /** If omitted the component fills its container via ResizeObserver (ShapeFunctions panel). */
  width?: number;
  /** Defaults to 560, matching VisxShapeEditor / CategoricalShapePlot. Heights < 90 render as compact thumbnails. */
  height?: number;
};

const HEIGHT = 560;
const COLOR = d3.scaleDiverging(d3.interpolateRdBu);
const FMT = d3.format(".3~g");

/** Bilinear interpolation of Z values at fractional grid position (u, v). */
function bilinearZ(z: number[][], u: number, v: number): number {
  const rows = z.length;
  const cols = z[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return 0;
  const x0 = Math.max(0, Math.min(cols - 2, Math.floor(u)));
  const y0 = Math.max(0, Math.min(rows - 2, Math.floor(v)));
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const tx = u - x0;
  const ty = v - y0;
  return (
    (z[y0]?.[x0] ?? 0) * (1 - tx) * (1 - ty) +
    (z[y0]?.[x1] ?? 0) * tx * (1 - ty) +
    (z[y1]?.[x0] ?? 0) * (1 - tx) * ty +
    (z[y1]?.[x1] ?? 0) * tx * ty
  );
}

export const InteractionHeatmap = ({ shape, width: widthProp, height = HEIGHT }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [measuredWidth, setMeasuredWidth] = useState(0);

  useEffect(() => {
    if (widthProp !== undefined) return;
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width;
      if (w) setMeasuredWidth(w);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [widthProp]);

  const width = widthProp ?? measuredWidth;
  const isCompact = height < 90;

  useLayoutEffect(() => {
    if (!width) return;
    const { gridX, gridX2, editableZ, xCategories, yCategories } = shape;
    if (!editableZ?.length) return;

    const showColorBar = !isCompact && width >= 280;
    const MARGIN = isCompact
      ? { top: 2, right: 2, bottom: 2, left: 2 }
      : { top: 14, right: showColorBar ? 58 : 14, bottom: 50, left: 58 };

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const g = svg
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const allZ = editableZ.flat();
    const absMax = Math.max(1e-9, d3.max(allZ.map(Math.abs)) ?? 1e-9);
    COLOR.domain([-absMax, 0, absMax]);

    const xIsCat = !gridX?.length && !!xCategories?.length;
    const yIsCat = !gridX2?.length && !!yCategories?.length;

    // ── x-axis scale ─────────────────────────────────────────────────────────
    let xScale: d3.ScaleBand<string> | d3.ScaleLinear<number, number>;
    let cellW: (xi: number) => number;
    let xPos: (xi: number) => number;

    if (xIsCat && xCategories) {
      const band = d3.scaleBand().domain(xCategories).range([0, innerW]).padding(0.04);
      xScale = band;
      cellW = () => band.bandwidth();
      xPos = (xi) => band(xCategories[xi]) ?? 0;
    } else if (gridX?.length) {
      const lin = d3.scaleLinear().domain([gridX[0], gridX[gridX.length - 1]]).range([0, innerW]);
      xScale = lin;
      const step = gridX.length > 1 ? gridX[1] - gridX[0] : 1;
      const pxStep = lin(gridX[0] + step) - lin(gridX[0]);
      cellW = () => pxStep;
      xPos = (xi) => lin(gridX[xi]) - pxStep / 2;
    } else {
      return;
    }

    // ── y-axis scale ─────────────────────────────────────────────────────────
    let yScale: d3.ScaleBand<string> | d3.ScaleLinear<number, number>;
    let cellH: (yi: number) => number;
    let yPos: (yi: number) => number;

    if (yIsCat && yCategories) {
      const band = d3.scaleBand().domain(yCategories).range([0, innerH]).padding(0.04);
      yScale = band;
      cellH = () => band.bandwidth();
      yPos = (yi) => band(yCategories[yi]) ?? 0;
    } else if (gridX2?.length) {
      const lin = d3.scaleLinear().domain([gridX2[0], gridX2[gridX2.length - 1]]).range([0, innerH]);
      yScale = lin;
      const step = gridX2.length > 1 ? gridX2[1] - gridX2[0] : 1;
      const pxStep = Math.abs(lin(gridX2[0] + step) - lin(gridX2[0]));
      cellH = () => pxStep;
      yPos = (yi) => lin(gridX2[yi]) - pxStep / 2;
    } else {
      return;
    }

    // ── draw heatmap cells ────────────────────────────────────────────────────
    const pw = Math.round(innerW);
    const ph = Math.round(innerH);
    const nCols = editableZ[0]?.length ?? 0;
    const nRows = editableZ.length;

    if (xIsCat && yIsCat) {
      // Cat × cat: no interpolation makes sense — discrete SVG rects
      editableZ.forEach((row, yi) => {
        row.forEach((val, xi) => {
          g.append("rect")
            .attr("x", xPos(xi))
            .attr("y", yPos(yi))
            .attr("width", Math.max(1, cellW(xi)))
            .attr("height", Math.max(1, cellH(yi)))
            .attr("fill", COLOR(val));
        });
      });
    } else {
      // Any continuous axis → canvas rendering
      const canvas = document.createElement("canvas");
      canvas.width = pw;
      canvas.height = ph;
      const ctx = canvas.getContext("2d");
      if (ctx && nCols > 0 && nRows > 0) {
        const imgData = ctx.createImageData(pw, ph);

        const setPixel = (px: number, py: number, zVal: number) => {
          const c = d3.rgb(COLOR(zVal) as string);
          const idx = (py * pw + px) * 4;
          imgData.data[idx] = Math.round(c.r);
          imgData.data[idx + 1] = Math.round(c.g);
          imgData.data[idx + 2] = Math.round(c.b);
          imgData.data[idx + 3] = 255;
        };

        if (!xIsCat && !yIsCat) {
          // Cont × cont: full bilinear interpolation per pixel
          for (let py = 0; py < ph; py++) {
            for (let px = 0; px < pw; px++) {
              const u = pw > 1 ? (px / (pw - 1)) * (nCols - 1) : 0;
              const v = ph > 1 ? (py / (ph - 1)) * (nRows - 1) : 0;
              setPixel(px, py, bilinearZ(editableZ, u, v));
            }
          }
        } else if (!xIsCat && yIsCat && yCategories) {
          // Cont × cat: smooth gradient along x within each category stripe
          for (let yi = 0; yi < yCategories.length; yi++) {
            const rowTop = Math.round(yPos(yi));
            const rowBot = Math.round(yPos(yi) + cellH(yi));
            for (let py = Math.max(0, rowTop); py < Math.min(ph, rowBot); py++) {
              for (let px = 0; px < pw; px++) {
                const u = pw > 1 ? (px / (pw - 1)) * (nCols - 1) : 0;
                const x0 = Math.max(0, Math.min(nCols - 2, Math.floor(u)));
                const zVal = (editableZ[yi]?.[x0] ?? 0) * (1 - (u - x0)) + (editableZ[yi]?.[x0 + 1] ?? 0) * (u - x0);
                setPixel(px, py, zVal);
              }
            }
          }
        } else if (xIsCat && !yIsCat && xCategories) {
          // Cat × cont: smooth gradient along y within each category column
          for (let xi = 0; xi < xCategories.length; xi++) {
            const colLeft = Math.round(xPos(xi));
            const colRight = Math.round(xPos(xi) + cellW(xi));
            for (let py = 0; py < ph; py++) {
              const v = ph > 1 ? (py / (ph - 1)) * (nRows - 1) : 0;
              const y0 = Math.max(0, Math.min(nRows - 2, Math.floor(v)));
              const zVal = (editableZ[y0]?.[xi] ?? 0) * (1 - (v - y0)) + (editableZ[y0 + 1]?.[xi] ?? 0) * (v - y0);
              for (let px = Math.max(0, colLeft); px < Math.min(pw, colRight); px++) {
                setPixel(px, py, zVal);
              }
            }
          }
        }

        ctx.putImageData(imgData, 0, 0);
        g.append("image")
          .attr("x", 0).attr("y", 0)
          .attr("width", innerW).attr("height", innerH)
          .attr("href", canvas.toDataURL());
      }
    }

    if (isCompact) return;

    // ── axes ─────────────────────────────────────────────────────────────────
    const isBandX = "bandwidth" in xScale;
    const isBandY = "bandwidth" in yScale;
    const xTicks = Math.max(3, Math.min(6, Math.floor(innerW / 55)));
    const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 45)));

    const xAxisG = g.append("g").attr("transform", `translate(0,${innerH})`);
    const yAxisG = g.append("g");

    const styleAxis = (sel: d3.Selection<SVGGElement, unknown, null, undefined>) => {
      sel.select(".domain").attr("stroke", "rgba(0,0,0,0.2)");
      sel.selectAll("line").attr("stroke", "rgba(0,0,0,0.15)");
      sel.selectAll("text").attr("fill", "#444").attr("font-size", 10);
    };

    if (isBandX) {
      const maxLen = Math.max(...(xCategories ?? []).map((c) => c.length));
      const rotate = (xCategories?.length ?? 0) > 4 || maxLen > 5;
      xAxisG.call(d3.axisBottom(xScale as d3.ScaleBand<string>).tickSize(3));
      if (rotate) {
        xAxisG
          .selectAll("text")
          .attr("transform", "rotate(-40)")
          .attr("text-anchor", "end")
          .attr("dy", "0.35em")
          .attr("dx", "-0.4em");
      }
    } else {
      xAxisG.call(
        d3.axisBottom(xScale as d3.ScaleLinear<number, number>)
          .ticks(xTicks)
          .tickFormat(FMT as (n: d3.NumberValue) => string)
          .tickSize(3),
      );
    }

    if (isBandY) {
      yAxisG.call(d3.axisLeft(yScale as d3.ScaleBand<string>).tickSize(3));
    } else {
      yAxisG.call(
        d3.axisLeft(yScale as d3.ScaleLinear<number, number>)
          .ticks(yTicks)
          .tickFormat(FMT as (n: d3.NumberValue) => string)
          .tickSize(3),
      );
    }

    styleAxis(xAxisG);
    styleAxis(yAxisG);

    // ── axis labels ──────────────────────────────────────────────────────────
    const xLabel = shape.label.split(" × ")[0] ?? shape.label;
    const yLabel = shape.label2 ?? shape.label.split(" × ")[1] ?? "";
    const rotatedX = isBandX && (xCategories?.length ?? 0) > 4;

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", rotatedX ? innerH + 44 : innerH + 36)
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("fill", "#444")
      .text(xLabel);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -(MARGIN.left - 10))
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("fill", "#444")
      .text(yLabel);

    // ── color scale bar ───────────────────────────────────────────────────────
    if (showColorBar) {
      const barX = innerW + 14;
      const barW = 9;
      const barH = innerH;

      // Draw bar as a canvas too for smooth gradient
      const barCanvas = document.createElement("canvas");
      barCanvas.width = barW;
      barCanvas.height = Math.round(barH);
      const bctx = barCanvas.getContext("2d");
      if (bctx) {
        const bImg = bctx.createImageData(barW, Math.round(barH));
        for (let by = 0; by < Math.round(barH); by++) {
          const t = by / (Math.round(barH) - 1);
          const val = absMax * (1 - t * 2);
          const c = d3.rgb(COLOR(val) as string);
          for (let bx = 0; bx < barW; bx++) {
            const idx = (by * barW + bx) * 4;
            bImg.data[idx] = Math.round(c.r);
            bImg.data[idx + 1] = Math.round(c.g);
            bImg.data[idx + 2] = Math.round(c.b);
            bImg.data[idx + 3] = 255;
          }
        }
        bctx.putImageData(bImg, 0, 0);
        g.append("image")
          .attr("x", barX).attr("y", 0)
          .attr("width", barW).attr("height", barH)
          .attr("href", barCanvas.toDataURL());
      }

      const barScale = d3.scaleLinear().domain([absMax, -absMax]).range([0, innerH]);
      const barAxisG = g.append("g").attr("transform", `translate(${barX + barW}, 0)`);
      barAxisG.call(
        d3.axisRight(barScale)
          .ticks(4)
          .tickFormat(FMT as (n: d3.NumberValue) => string)
          .tickSize(3),
      );
      styleAxis(barAxisG);
    }
  }, [shape, width, height, isCompact]);

  if (widthProp !== undefined) {
    return <svg ref={svgRef} />;
  }

  return (
    <div ref={containerRef} className={styles.chartFrame}>
      <svg ref={svgRef} />
    </div>
  );
};
