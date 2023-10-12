use arrayfire;






pub fn set_diag(
	magsq_matrix: &mut arrayfire::Array<f64>,
	val: f64
)
{
	let pos_num = magsq_matrix.dims()[0];

	let magsq_dims = magsq_matrix.dims();

	let N_dims = arrayfire::Dim4::new(&[pos_num,1,1,1]);
	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let idx = (pos_num+1)*arrayfire::iota::<i32>(N_dims,repeat_dims);
	*magsq_matrix  = arrayfire::flat(magsq_matrix);


	let large_vec = arrayfire::constant::<f64>(val, arrayfire::Dim4::new(&[pos_num,1,1,1]));

	let mut idxrs = arrayfire::Indexer::default();
	idxrs.set_index(&idx, 0, None);
	arrayfire::assign_gen(magsq_matrix, &idxrs, &large_vec);

	*magsq_matrix  = arrayfire::moddims(magsq_matrix, magsq_dims);


}




