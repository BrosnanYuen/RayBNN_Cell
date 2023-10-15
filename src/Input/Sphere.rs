use arrayfire;

use std::collections::HashMap;


use crate::Util::Math::set_diag;

const TWO_F64: f64 = 2.0;

const NEURON_RAD_FACTOR: f64 = 1.1;

const HIGH_F64: f64 = f64::INFINITY;

const ONEHALF_F64: f64 = 0.5;

const TARGET_DENSITY: f64 = 3500.0;

const ONE_F64: f64 = 1.0;




/*
Creates input neurons on the surface of a sphere for 2D images of size (Nx,Ny) 

Inputs
sphere_rad:   3D Sphere Radius
Nx:           Image X dimension size
Ny:           Image Y dimension size

Outputs:
The 3D position of neurons on the surface of a 3D sphere

*/

pub fn create_spaced_input_neuron_on_sphere<Z: arrayfire::FloatingPoint<UnaryOutType = Z> > (
	sphere_rad: f64,
	Nx: u64,
	Ny: u64,

	) -> arrayfire::Array<Z>
	{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let ONEHALF = arrayfire::constant::<f64>(ONEHALF_F64,single_dims).cast::<Z>();

	let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();

	let TWO_PI = arrayfire::constant::<f64>(TWO_F64*std::f64::consts::PI,single_dims).cast::<Z>();


	let Nx_Z = arrayfire::constant::<u64>((Nx+1),single_dims).cast::<Z>();




	let gen_dims = arrayfire::Dim4::new(&[1,Nx,1,1]);
	let rep_dims = arrayfire::Dim4::new(&[Ny,1,1,1]);

	let mut theta = arrayfire::iota::<Z>(gen_dims,rep_dims)+ONE.clone();
	theta = theta/Nx_Z;

	theta = TWO*(theta-ONEHALF);
	theta = arrayfire::acos(&theta);


	let gen_dims = arrayfire::Dim4::new(&[Ny,1,1,1]);
	let rep_dims = arrayfire::Dim4::new(&[1,Nx,1,1]);

	let Ny_Z = arrayfire::constant::<u64>((Ny+1),single_dims).cast::<Z>();



	let mut phi = arrayfire::iota::<Z>(gen_dims,rep_dims)+ONE;
	phi = phi/Ny_Z;

	phi = phi*TWO_PI;


	let sphere_rad_Z = arrayfire::constant::<f64>(sphere_rad,single_dims).cast::<Z>();



	let mut x = arrayfire::sin(&theta)*arrayfire::cos(&phi);
	let mut y = arrayfire::sin(&theta)*arrayfire::sin(&phi);
	let mut z = arrayfire::cos(&theta);

	x = arrayfire::flat(&x);
	y = arrayfire::flat(&y);
	z = arrayfire::flat(&z);


	sphere_rad_Z*arrayfire::join_many(1, vec![&x,&y,&z])
}





