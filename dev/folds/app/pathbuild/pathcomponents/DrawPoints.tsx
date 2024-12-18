import { useRef, useEffect } from "react";
import * as d3 from "d3";
import {
  Rand_n_points_determiner,
  rdm_point_selector,
} from "../pathFunctions/pathfuncs";

const  Drawtopend= ()=> {
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

    console.log(data)
    // alert(data);

    // Create a Catmull-Rom line generator
    const lineGenerator = d3.line<{ x: number; y: number }>()
      .curve(d3.curveCatmullRom.alpha(0.5))
      .x(d => d.x)
      .y(d => d.y);

    // Draw the Catmull-Rom line
    svg.append('path')
      .datum(data)
      .attr('d', lineGenerator)
      .attr('fill', 'none')
      .attr('stroke', 'grey')
      .attr('stroke-width', 2);

      console.log(document.documentElement.clientWidth + ' and such')

      return () => {
        // Remove the generated SVG elements when unmounting
        svg.selectAll('*').remove();
      };
  }, []);

  return (<svg ref={svgRef} width={500} height={500}></svg>);
}

// @ts-check
const DrawPoints = () => {
  // @dev under it to be executed in original paper
  // alert('If these page is comming in unsupported size , use browser normally');
  console.log(document.documentElement.clientWidth + ' and such')

  return (
    <Drawtopend />
  );
};

interface catmullinter {
  data:{x: number, y: number} [] ;
}

export default DrawPoints;

const DrawExact = () => { 
  const svgRef = useRef< SVGSVGElement | null > (null) ;

  useEffect(() => {
    const svg = d3.select(svgRef.current) ;

    let dtss : catmullinter ;
    dtss = {
      data: [
        { x: 1, y: 2 },
        { x: 3, y: 4 },
        { x: 5, y: 6 },
      ],
    };

    const catmullLine = d3.line<{x: number ; y: number}>()
    .x(d=> d.x)
    .y(d=> d.y)
    .curve(d3.curveCatmullRom)

    svg.append('path').datum(dtss.data).attr('d' , catmullLine(dtss.data)).attr('fill', 'none')
    .attr('stroke', 'blue')
    .attr('stroke-width', 2);
  })

  return (
    <div>
      <svg  ref= {svgRef} width={760} height={770} ></svg>
    </div>
  )
  
}

function pointer( n? : number ) {
  if (n!==null) {   n = 10 ;     }
  let xn = [  ] , yn = [ ] ;
  const xr = document.documentElement.clientWidth ;
  const yr = document.documentElement.clientHeight ;

  const xp = (xr/10) + (xr%10) ;
  const yp = (yr/10) + (yr%10) ;

  for (let i = 0; i < n; i++) {
    const xi = xp * i ;
    const yi = yp * i ;

   let rd ;
   do{ 
    rd = Math.random() ;
   }while(!(rd >= 0.1) ) 
   let rdy ;
   do{ 
    rdy = Math.random() ;
   }while(!(rdy >= 0.1) ) 

   const xj = ((xp/rd) + (xp % rd)) 
   const yj = ((yp/rdy) + (yp % rdy)) 
   xn.push(xj)
   yn.push(yj)
  }
}