// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{ArrayData, make_array},
    pyarrow::PyArrowType,
};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, ViewRepr};
use numpy::{Ix1, PyArray2, PyArrayMethods};
use pyo3::{IntoPyObjectExt, exceptions::PyRuntimeError, prelude::*};
use rayon::prelude::*;

use log::*;
use rayon_cancel::CancelAdapter;

use crate::{
    als::solve::{POSV, SolveError},
    parallel::maybe_fuse,
    sparse::{CSR, CSRMatrix},
    tasks::{AccelTask, AccelTaskImpl, IterCancel},
};

struct ImplicitTrainTask {
    solver: POSV,
    matrix: CSRMatrix<i32>,
    this: Py<PyArray2<f32>>,
    other: Py<PyArray2<f32>>,
    otor: Py<PyArray2<f32>>,
}

#[pyfunction]
pub(super) fn train_implicit_matrix<'py>(
    py: Python<'py>,
    matrix: PyArrowType<ArrayData>,
    this: Py<PyArray2<f32>>,
    other: Py<PyArray2<f32>>,
    otor: Py<PyArray2<f32>>,
) -> PyResult<AccelTask> {
    let solver = POSV::load(py)?;
    let matrix_ref = make_array(matrix.0);
    let matrix: CSRMatrix<i32> = CSRMatrix::from_arrow(matrix_ref)?;

    Ok(AccelTask::wrap(ImplicitTrainTask {
        solver,
        matrix,
        this,
        other,
        otor,
    }))
}

impl AccelTaskImpl for ImplicitTrainTask {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>> {
        let mut this_py = self.this.bind(py).readwrite();
        let mut this = this_py.as_array_mut();

        let other_py = self.other.bind(py).readonly();
        let other = other_py.as_array();

        let otor_py = self.otor.bind(py).readonly();
        let otor = otor_py.as_array();

        debug!(
            "beginning implicit ALS training half with {} rows",
            other.nrows()
        );

        let iter = maybe_fuse(this.outer_iter_mut().into_par_iter()).enumerate();
        let adapter = CancelAdapter::new(iter);
        task.set_cancel(IterCancel::from_adapter(&adapter));

        let frob: PyResult<f32> = py.detach(move || {
            Ok(adapter
                .map(|(i, row)| train_row_solve(&self.solver, &self.matrix, i, row, &other, &otor))
                .try_reduce(|| 0.0f32, |v1, v2| Ok(v1 + v2))
                .map_err(|e| PyRuntimeError::new_err(format!("ALS solve error: {}", e)))?
                .sqrt())
        });

        frob?.into_bound_py_any(py)
    }
}

fn train_row_solve(
    solver: &POSV,
    matrix: &CSRMatrix<i32>,
    row_num: usize,
    mut row_data: ArrayBase<ViewRepr<&mut f32>, Ix1>,
    other: &ArrayView2<f32>,
    otor: &ArrayView2<f32>,
) -> Result<f32, SolveError> {
    let cols = matrix.row_cols(row_num);
    let vals = matrix.row_vals(row_num);

    if cols.len() == 0 {
        row_data.fill(0.0);
        return Ok(0.0);
    }

    let cols: Vec<_> = cols.iter().map(|c| *c as usize).collect();
    let mut vals: Array1<_> = vals.iter().map(|f| *f).collect();

    let nd = row_data.len();

    let o_picked = other.select(Axis(0), &cols);

    let mt = o_picked.t();
    let mtl = &mt * &vals;
    let mtm = mtl.dot(&o_picked);
    assert_eq!(mtm.shape(), &[nd, nd]);

    let mut a = otor + &mtm;
    vals += 1.0;
    let y = mt.dot(&vals);

    let soln = solver.solve(&mut a, &y)?;

    let deltas = &soln - &row_data;
    row_data.assign(&soln);

    Ok(deltas.dot(&deltas))
}
