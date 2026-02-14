// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, ViewRepr};
use numpy::{Ix1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use log::*;

use crate::{
    als::solve::POSV,
    progress::ProgressHandle,
    sparse::{CSRMatrix, CSR},
};

#[pyfunction]
pub(super) fn train_explicit_matrix<'py>(
    py: Python<'py>,
    matrix: PyArrowType<ArrayData>,
    this: Bound<'py, PyArray2<f32>>,
    other: Bound<'py, PyArray2<f32>>,
    reg: f32,
    progress: Bound<'py, PyAny>,
) -> PyResult<f32> {
    let solver = POSV::load(py)?;
    let matrix_ref = make_array(matrix.0);
    let matrix: CSRMatrix<i32> = CSRMatrix::from_arrow(matrix_ref)?;

    let mut this_py = this.readwrite();
    let mut this = this_py.as_array_mut();

    let other_py = other.readonly();
    let other = other_py.as_array();

    let mut progress = ProgressHandle::from_input(progress);
    debug!(
        "beginning explicit ALS training half with {} rows",
        other.nrows()
    );

    let frob: f32 = py.detach(|| {
        this.outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .map(|(i, row)| {
                let f = train_row_solve(&solver, &matrix, i, row, &other, reg);
                progress.tick();
                f
            })
            .sum()
    });
    progress.shutdown()?;

    Ok(frob.sqrt())
}

fn train_row_solve(
    solver: &POSV,
    matrix: &CSRMatrix<i32>,
    row_num: usize,
    mut row_data: ArrayBase<ViewRepr<&mut f32>, Ix1>,
    other: &ArrayView2<f32>,
    reg: f32,
) -> f32 {
    let cols = matrix.row_cols(row_num);
    let vals = matrix.row_vals(row_num);

    if cols.len() == 0 {
        row_data.fill(0.0);
        return 0.0;
    }

    let cols: Vec<_> = cols.iter().map(|c| *c as usize).collect();
    let vals: Array1<_> = vals.iter().map(|f| *f).collect();

    let nd = row_data.len();

    let o_picked = other.select(Axis(0), &cols);

    let mt = o_picked.t();
    let mut mtm = mt.dot(&o_picked);
    assert_eq!(mtm.shape(), &[nd, nd]);
    for i in 0..nd {
        mtm[[i, i]] += reg * cols.len() as f32;
    }

    let v = mt.dot(&vals);
    assert_eq!(v.shape(), &[nd]);

    let soln = solver.solve(&mut mtm, &v).expect("LAPACK error");

    let deltas = &soln - &row_data;
    row_data.assign(&soln);

    deltas.dot(&deltas)
}
