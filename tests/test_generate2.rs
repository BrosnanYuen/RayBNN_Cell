#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_hidden_generate2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);






}
