use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, ViewRepr};
use nshare::{IntoNalgebra, IntoNdarray1};
use numpy::{Ix1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use log::*;

use crate::{progress::ProgressHandle, sparse::CSRMatrix};

#[pyfunction]
pub(super) fn train_implicit_matrix<'py>(
    py: Python<'py>,
    matrix: PyArrowType<ArrayData>,
    this: Bound<'py, PyArray2<f32>>,
    other: Bound<'py, PyArray2<f32>>,
    otor: Bound<'py, PyArray2<f32>>,
    progress: Bound<'py, PyAny>,
) -> PyResult<f32> {
    let matrix_ref = make_array(matrix.0);
    let matrix: CSRMatrix<i32> = CSRMatrix::from_arrow(matrix_ref)?;

    let mut this_py = this.readwrite();
    let mut this = this_py.as_array_mut();

    let other_py = other.readonly();
    let other = other_py.as_array();

    let otor_py = otor.readonly();
    let otor = otor_py.as_array();

    let progress = ProgressHandle::from_input(progress);
    debug!(
        "beginning implicit ALS training half with {} rows",
        other.nrows()
    );
    let frob: f32 = py.allow_threads(|| {
        this.outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .map(|(i, row)| {
                let f = train_row_solve(&matrix, i, row, &other, &otor);
                progress.tick();
                f
            })
            .sum()
    });

    Ok(frob.sqrt())
}

fn train_row_solve(
    matrix: &CSRMatrix<i32>,
    row_num: usize,
    mut row_data: ArrayBase<ViewRepr<&mut f32>, Ix1>,
    other: &ArrayView2<f32>,
    otor: &ArrayView2<f32>,
) -> f32 {
    let cols = matrix.row_cols(row_num);
    let vals = matrix.row_vals(row_num);

    if cols.len() == 0 {
        return 0.0;
    }

    let cols: Vec<_> = cols.iter().map(|c| *c as usize).collect();
    let mut vals: Array1<_> = vals.iter().map(|f| *f).collect();

    let nd = row_data.len();

    let o_picked = other.select(Axis(0), &cols);

    let mt = o_picked.t();
    let mtl = &mt * &vals;
    let mtm = mtl.dot(&o_picked);
    assert_eq!(mtm.shape(), &[nd, nd]);

    let a = otor + &mtm;
    vals += 1.0;
    let y = mt.dot(&vals);

    let a = a.into_nalgebra();
    let y = y.into_nalgebra();
    let cholesky = a.cholesky().expect("matrix is not positive definite");

    let soln = cholesky.solve(&y);
    let soln = soln.into_ndarray1();
    let deltas = &soln - &row_data;
    row_data.assign(&soln);

    deltas.dot(&deltas)
}
