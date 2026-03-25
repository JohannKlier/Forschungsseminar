"use client";

import { useLayoutEffect, useRef } from "react";
import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import styles from "../page.module.css";

export default function FeatureMiniHistogram({
  bins,
  selectedBins = [],
}: {
  bins: number[];
  selectedBins?: number[];
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useLayoutEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const width = 100;
    const height = 48;
    const gap = 1;
    const data = bins.map((count, index) => ({
      index,
      count,
      selectedCount: selectedBins[index] ?? 0,
    }));
    const maxBin = Math.max(...bins, 1);
    const maxSelectedBin = Math.max(...selectedBins, 1);
    const barWidth = data.length ? Math.max(0, width / data.length - gap) : 0;
    const xScale = scaleLinear().domain([0, Math.max(data.length, 1)]).range([0, width]);
    const yScale = scaleLinear().domain([0, maxBin]).range([0, height]);
    const selectedYScale = scaleLinear().domain([0, maxSelectedBin]).range([0, height]);

    const svg = select(svgEl);
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "none");

    svg
      .selectAll<SVGRectElement, (typeof data)[number]>("rect.bin")
      .data(data, (d) => String(d.index))
      .join("rect")
      .attr("class", "bin")
      .attr("x", (d) => xScale(d.index))
      .attr("y", (d) => height - yScale(d.count))
      .attr("width", barWidth)
      .attr("height", (d) => yScale(d.count))
      .attr("fill", "rgba(149, 184, 220, 0.62)");

    svg
      .selectAll<SVGRectElement, (typeof data)[number]>("rect.bin-selected")
      .data(data.filter((d) => d.selectedCount > 0), (d) => String(d.index))
      .join("rect")
      .attr("class", "bin-selected")
      .attr("x", (d) => xScale(d.index))
      .attr("y", (d) => height - selectedYScale(d.selectedCount))
      .attr("width", barWidth)
      .attr("height", (d) => selectedYScale(d.selectedCount))
      .attr("fill", "rgba(244, 191, 117, 0.72)");
  }, [bins, selectedBins]);

  return <svg ref={svgRef} className={styles.featureDistributionHistogramSvg} aria-hidden="true" />;
}
