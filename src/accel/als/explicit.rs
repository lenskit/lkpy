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
use pyo3::{prelude::*, IntoPyObjectExt};
use rayon::prelude::*;

use rayon_cancel::CancelAdapter;

use crate::{
    als::solve::POSV,
    parallel::maybe_fuse,
    sparse::{CSRMatrix, CSR},
    tasks::{AccelTask, AccelTaskImpl, IterCancel},
};

struct ExplicitTrainTask {
    solver: POSV,
    matrix: CSRMatrix<i32>,
    this: Py<PyArray2<f32>>,
    other: Py<PyArray2<f32>>,
    reg: f32,
}

#[pyfunction]
pub(super) fn train_explicit_matrix<'py>(
    py: Python<'py>,
    matrix: PyArrowType<ArrayData>,
    this: Py<PyArray2<f32>>,
    other: Py<PyArray2<f32>>,
    reg: f32,
) -> PyResult<AccelTask> {
    let solver = POSV::load(py)?;
    let matrix_ref = make_array(matrix.0);
    let matrix: CSRMatrix<i32> = CSRMatrix::from_arrow(matrix_ref)?;

    Ok(AccelTask::wrap(ExplicitTrainTask {
        solver,
        matrix,
        this,
        other,
        reg,
    }))
}

impl AccelTaskImpl for ExplicitTrainTask {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>> {
        let mut this_py = self.this.bind(py).readwrite();
        let mut this = this_py.as_array_mut();

        let other_py = self.other.bind(py).readonly();
        let other = other_py.as_array();

        let iter = maybe_fuse(this.outer_iter_mut().into_par_iter()).enumerate();
        let adapter = CancelAdapter::new(iter);
        task.set_cancel(IterCancel::from_adapter(&adapter));

        let frob: f32 = py.detach(move || {
            adapter
                .map(|(i, row)| {
                    let f = train_row_solve(&self.solver, &self.matrix, i, row, &other, self.reg);
                    f
                })
                .sum::<f32>()
                .sqrt()
        });

        frob.into_bound_py_any(py)
    }
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
