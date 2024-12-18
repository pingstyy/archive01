interface nested {
    x: number;
    y: number;
   }
  
   type num_arr = nested [] ;

   function left_side (point : num_arr) {

   }

   function generate_num( n: number){
    const xr = document.documentElement.clientWidth ;
    const yr = document.documentElement.clientHeight ;

    var pints: num_arr = [];
    for (var i = 0; i < n; i++) { 
        pints[i].x =  Math.floor(xr/ Math.random() * 10) ;
        pints[i].y =  Math.floor(xr/ Math.random() * 10) ;
       
     }
     console.log(pints)
     return pints ;
   }

   generate_num(5 ) ;

  type num_Arr = [number , number ] ;

  function find_n_m() {
    let pts : num_Arr [] ;
    
  }

  let coord : Map< number , nested > = new Map();
