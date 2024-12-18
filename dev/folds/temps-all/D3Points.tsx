import React from "react";
import { useRef, useEffect } from "react";
import * as d3 from "d3";
// import { data } from "autoprefixer";

const data = [
  { x: 20, y: 30 },
  { x: 50, y: 60 },
  { x: 80, y: 90 },
  { x: 110, y: 30 },
  { x: 140, y: 60 },
  { x: 170, y: 90 },
  { x: 200, y: 30 },
  { x: 230, y: 60 },
  { x: 260, y: 90 },
  { x: 290, y: 30 },
];

const DrawPoints = () => {
  const svgRef = useRef(null);
  useEffect(() => {
    // D3 code to draw dots
    const svg = d3.select(svgRef.current);

    // const line = d3
    //   .line()
    //   .x((d  ) => d.x)
    //   .y((d) => d.y)
    //   .curve(d3.curveCatmullRom.alpha(0.5)); // You can choose a different curve type here

    // svg
    //   .append("path")
    //   .datum(data)
    //   .attr("d", line)
    //   .attr("fill", "none")
    //   .attr("stroke", "blue")
    //   .attr("stroke-width", 2);

    svg
      .selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", 5) // Radius of the dots
      .attr("fill", "red"); // Dot color

    // Cleanup
    return () => {
      // Remove D3.js elements and event listeners if necessary
      svg.selectAll("*").remove();
    };
  }, [data]);

  return (
    <div>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default DrawPoints;
