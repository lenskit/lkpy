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

use crate::sparse::CSRMatrix;

const RR_EPOCHS: usize = 2;

#[pyfunction]
pub(super) fn train_explicit_matrix_cd<'py>(
    matrix: PyArrowType<ArrayData>,
    this: Bound<'py, PyArray2<f32>>,
    other: Bound<'py, PyArray2<f32>>,
    reg: f32,
) -> PyResult<f32> {
    let matrix_ref = make_array(matrix.0);
    let matrix: CSRMatrix<i32> = CSRMatrix::from_arrow(matrix_ref)?;

    let mut this_py = this.readwrite();
    let mut this = this_py.as_array_mut();

    let other_py = other.readonly();
    let other = other_py.as_array();

    debug!(
        "beginning explicit ALS training half with {} rows",
        other.nrows()
    );

    let frob: f32 = this
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .map(|(i, row)| train_row_solve(&matrix, i, row, &other, reg))
        .sum();

    Ok(frob.sqrt())
}

fn train_row_cd(
    matrix: &CSRMatrix<i32>,
    row_num: usize,
    mut row_data: ArrayBase<ViewRepr<&mut f32>, Ix1>,
    other: &ArrayView2<f32>,
    reg: f32,
) -> f32 {
    let cols = matrix.row_cols(row_num);
    let vals = matrix.row_vals(row_num);
    let col_us: Vec<_> = cols.iter().map(|c| *c as usize).collect();

    let nd = row_data.len();

    let o_picked = other.select(Axis(0), &col_us);
    let mut resid = Array1::from_iter(vals.iter().map(|f| *f));
    let scores = o_picked.dot(&row_data);
    resid -= &scores;

    let mut deltas = Array1::zeros(nd);

    for _e in 0..RR_EPOCHS {
        for d in 0..nd {
            let dvec = o_picked.column(d);
            let num = dvec.dot(&resid) - reg * row_data[d];
            let denom = dvec.dot(&dvec) + reg;
            let dw = num / denom;
            deltas[d] += dw;
            row_data[d] += dw;
            resid.scaled_add(-dw, &dvec);
        }
    }

    deltas.dot(&deltas)
}

fn train_row_solve(
    matrix: &CSRMatrix<i32>,
    row_num: usize,
    mut row_data: ArrayBase<ViewRepr<&mut f32>, Ix1>,
    other: &ArrayView2<f32>,
    reg: f32,
) -> f32 {
    let cols = matrix.row_cols(row_num);
    let vals = matrix.row_vals(row_num);

    if cols.len() == 0 {
        return 0.0;
    }

    let col_us: Vec<_> = cols.iter().map(|c| *c as usize).collect();
    let vals: Array1<_> = vals.iter().map(|f| *f).collect();

    let nd = row_data.len();

    let o_picked = other.select(Axis(0), &col_us);

    let mt = o_picked.t();
    let mut mtm = mt.dot(&o_picked);
    assert_eq!(mtm.shape(), &[nd, nd]);
    for i in 0..nd {
        mtm[[i, i]] += reg * cols.len() as f32;
    }

    let v = mt.dot(&vals);
    assert_eq!(v.shape(), &[nd]);

    let mtm = mtm.into_nalgebra();
    let v = v.into_nalgebra();
    let cholesky = mtm.cholesky().expect("matrix is not positive definite");

    let soln = cholesky.solve(&v);
    let soln = soln.into_ndarray1();
    let deltas = &soln - &row_data;
    row_data.assign(&soln);

    deltas.dot(&deltas)
}
