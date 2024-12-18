type num_Arr =  number[][] ;
//   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
export function Rand_n_points_determiner() {
    let rdm_var: number = Math.floor(Math.random() * (10));
    for (var i = 0; i <= 25; i++) {
        rdm_var = Math.floor(Math.random() * (10));
        if (rdm_var > 5) {
            break;
        }
    }
    return rdm_var;
}

export function rdm_point_selector(n: number, x_range: number, y_range: number) {
    let points: num_Arr = [];
    // abscissa & ordinate qualifier 
    for (var i = 0; i < n; i++) {
        points[i][0] = (x_range / Math.floor(Math.random() * 100)) +
            (x_range % Math.floor(Math.random() * 100));
        points[i][1] = (y_range / Math.floor(Math.random() * 100)) +
            (y_range % Math.floor(Math.random() * 100));
    }
    return points;
}

function two_way_builder(points: num_Arr , amps : number) {
    let left_end : num_Arr = [] , right_end : num_Arr = [] ;
    

}