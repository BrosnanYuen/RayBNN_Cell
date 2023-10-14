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




	let neuron_size: u64 = 51000;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 3;
	let active_size: u64 = 500000;
	let space_dims: u64 = 3;
	let sim_steps: u64 = 1;
	let mut batch_size: u64 = 105;
	let neuron_rad = 0.1;

	let mut netdata: clusterdiffeq::neural::network_f32::network_metadata_type = clusterdiffeq::neural::network_f32::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: sim_steps,
		batch_size: batch_size,
		del_unused_neuron: true,

		time_step: 0.3,
		nratio: 0.5,
		neuron_std: 0.3,
		sphere_rad: 30.0,
		neuron_rad: neuron_rad,
		con_rad: 0.6,
		init_prob: 0.5,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 0.01
	};

	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);



	
	let mut H = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut A = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut B = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut C = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut D = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut E = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);




	let mut WValues = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut WRowIdxCOO = arrayfire::constant::<i32>(0,temp_dims);
	let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);


    let start = Instant::now();

	clusterdiffeq::physics::initial_f32::spherical_structV3(
		&netdata,
		&mut glia_pos,
		&mut neuron_pos
	);

	let duration = start.elapsed();

    println!("Time elapsed in expensive_function() is: {:?}", duration);

	println!("glia_pos.dims()[0] {}",glia_pos.dims()[0]);
	println!("neuron_pos.dims()[0] {}",neuron_pos.dims()[0]);

	
	assert_eq!(glia_pos.dims()[1],space_dims);
	assert_eq!(neuron_pos.dims()[1],space_dims);

	let total_obj = arrayfire::join(0, &glia_pos, &neuron_pos);
	drop(neuron_pos);
	drop(glia_pos);

	let mut active_size = total_obj.dims()[0];
	assert!(active_size >= 200000);


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let neuron_sq = 4.0*neuron_rad*neuron_rad;

	for i in 0u64..active_size
	{
		let select_pos = arrayfire::row(&total_obj,i as i64);

		let mut dist = arrayfire::sub(&select_pos,&total_obj, true);
		let mut magsq = arrayfire::pow(&dist,&two,false);
		let mut magsq = arrayfire::sum(&magsq,1);


		let insert = arrayfire::constant::<f32>(1000000.0,single_dims);

		arrayfire::set_row(&mut magsq, &insert, i as i64);

		let (m0,_) = arrayfire::min_all::<f32>(&magsq);

		//println!("{} dist {}",i, m0);
		assert!(m0 > neuron_sq);
	}




}
