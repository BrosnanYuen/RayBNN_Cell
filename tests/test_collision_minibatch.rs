#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_sphere_cell_collision_minibatch() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);






}
