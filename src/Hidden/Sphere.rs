use arrayfire;





pub fn get_inside_idx_cubeV2<Z: arrayfire::FloatingPoint>(
	pos: &arrayfire::Array<Z>
	, cube_size: f64
	, pivot: &Vec<f64>)
	-> arrayfire::Array<u32>
{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let pivot_pos = pivot.clone();
	let space_dims = pivot_pos.len();


	//let mut negative_range = pivot_pos[0].clone();
	//let mut positive_range = negative_range + cube_size;

	let mut negative_range = arrayfire::constant::<f64>(pivot_pos[0].clone(),single_dims).cast::<Z>();
	let mut positive_range = arrayfire::constant::<f64>(pivot_pos[0].clone() + cube_size,single_dims).cast::<Z>();


	let mut axis = arrayfire::col(pos,0);

	let mut cmp1 = arrayfire::lt(&axis, &positive_range, false);
	let mut cmp2 = arrayfire::lt(&negative_range,  &axis, false);
	cmp1 = arrayfire::and(&cmp1,&cmp2, false);

	for idx in 1..space_dims
	{
		//negative_range = pivot_pos[idx].clone();
		//positive_range = negative_range + cube_size;


		negative_range = arrayfire::constant::<f64>(pivot_pos[idx].clone(),single_dims).cast::<Z>();
		positive_range = arrayfire::constant::<f64>(pivot_pos[idx].clone() + cube_size,single_dims).cast::<Z>();
	

	
		axis = arrayfire::col(pos,idx as i64);

		cmp2 = arrayfire::lt(&axis, &positive_range, false);
		cmp1 = arrayfire::and(&cmp1,&cmp2, false);
		cmp2 = arrayfire::lt(&negative_range,  &axis, false);
		cmp1 = arrayfire::and(&cmp1,&cmp2, false);
	
	}

	arrayfire::locate(&cmp1)
}







