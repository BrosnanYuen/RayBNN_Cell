use arrayfire;






pub fn set_diag<Z: arrayfire::FloatingPoint + arrayfire::ConstGenerator<OutType = Z>>(
	magsq_matrix: &mut arrayfire::Array<Z>,
	val: Z
)
{
	let pos_num = magsq_matrix.dims()[0];

	let magsq_dims = magsq_matrix.dims();

	let N_dims = arrayfire::Dim4::new(&[pos_num,1,1,1]);
	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let idx = (pos_num+1)*arrayfire::iota::<i32>(N_dims,repeat_dims);
	*magsq_matrix  = arrayfire::flat(magsq_matrix);


	let large_vec = arrayfire::constant(val, arrayfire::Dim4::new(&[pos_num,1,1,1]));

	let mut idxrs = arrayfire::Indexer::default();
	idxrs.set_index(&idx, 0, None);
	arrayfire::assign_gen(magsq_matrix, &idxrs, &large_vec);

	*magsq_matrix  = arrayfire::moddims(magsq_matrix, magsq_dims);


}





pub fn matrix_dist(
	pos_vec: &arrayfire::Array<f64>,
	dist_matrix: &mut arrayfire::Array<f64>,
	magsq_matrix: &mut arrayfire::Array<f64>
)
{






	let mut p1 = pos_vec.clone();


	p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));



	

	*dist_matrix = arrayfire::sub(&p1, pos_vec, true);



	*magsq_matrix = arrayfire::pow(dist_matrix,&two,false);

	*magsq_matrix = arrayfire::sum(magsq_matrix,1);
}




