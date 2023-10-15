use arrayfire;

use std::collections::HashMap;


use crate::Util::Math::set_diag;

const TWO_F64: f64 = 2.0;

const NEURON_RAD_FACTOR: f64 = 1.1;

const HIGH_F64: f64 = f64::INFINITY;

const ONEHALF_F64: f64 = 0.5;

const TARGET_DENSITY: f64 = 3500.0;






/*
Creates input neurons on the surface of a sphere for 2D images of size (Nx,Ny) 

Inputs
sphere_rad:   3D Sphere Radius
Nx:           Image X dimension size
Ny:           Image Y dimension size

Outputs:
The 3D position of neurons on the surface of a 3D sphere

*/

pub fn create_spaced_input_neuron_on_sphere<Z: arrayfire::FloatingPoint > (
	sphere_rad: f64,
	Nx: u64,
	Ny: u64,

	) -> arrayfire::Array<Z>
	{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let ONEHALF = arrayfire::constant::<f64>(ONEHALF_F64,single_dims).cast::<Z>();




	let gen_dims = arrayfire::Dim4::new(&[1,Nx,1,1]);
	let rep_dims = arrayfire::Dim4::new(&[Ny,1,1,1]);

	let mut theta = arrayfire::iota::<f64>(gen_dims,rep_dims)+one;
	theta = theta/((Nx+1) as f64);

	theta = TWO*(theta-ONEHALF);
	theta = arrayfire::acos(&theta);


	let gen_dims = arrayfire::Dim4::new(&[Ny,1,1,1]);
	let rep_dims = arrayfire::Dim4::new(&[1,Nx,1,1]);

	let mut phi = arrayfire::iota::<f64>(gen_dims,rep_dims)+one;
	phi = phi/((Ny+1) as f64);

	phi = phi*TWO*std::f64::consts::PI;


	let mut x = sphere_rad*arrayfire::sin(&theta)*arrayfire::cos(&phi);
	let mut y = sphere_rad*arrayfire::sin(&theta)*arrayfire::sin(&phi);
	let mut z = sphere_rad*arrayfire::cos(&theta);

	x = arrayfire::flat(&x);
	y = arrayfire::flat(&y);
	z = arrayfire::flat(&z);


	arrayfire::join_many(1, vec![&x,&y,&z])
}





