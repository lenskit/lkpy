use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, ViewRepr};
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

    let frob = this
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .map(|(i, row)| train_row_cd(&matrix, i, row, &other, reg))
        .sum();

    Ok(frob)
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
