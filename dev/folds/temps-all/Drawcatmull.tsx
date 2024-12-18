import { useEffect, useRef } from "react";
import * as d3 from "d3";

const Drawtopend = () => {
  const x = document.documentElement.clientHeight;
  const y = document.documentElement.clientWidth;

  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);

    // Random data points for the line
    const data = Array.from({ length: 10 }, (_, i) => ({
      x: i * 50 + 50,
      y: Math.random() * 300 + 50,
    }));

    console.log(data);

    // Create a Catmull-Rom line generator
    const lineGenerator = d3
      .line<{ x: number; y: number }>()
      .curve(d3.curveCatmullRom.alpha(0.5))
      .x((d) => d.x)
      .y((d) => d.y);

    // Draw the Catmull-Rom line
    svg
      .append("path")
      .datum(data)
      .attr("d", lineGenerator)
      .attr("fill", "none")
      .attr("stroke", "blue")
      .attr("stroke-width", 2);

    console.log(document.documentElement.clientWidth + " and such");

    return () => {
      // Remove the generated SVG elements when unmounting
      svg.selectAll("*").remove();
    };
  }, []);

  return <svg ref={svgRef} width={500} height={500}></svg>;
};
