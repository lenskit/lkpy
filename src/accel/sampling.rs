// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accelerated sampling support.
use std::mem;

use log::*;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::data::RowColumnSet;

/// Efficient-ish negative sampling.
#[pyclass]
pub struct NegativeSampler {
    rc_set: Py<RowColumnSet>,
    rows: Py<PyArray1<i32>>,
    negatives: Array2<i32>,
    remaining: Vec<(u16, u16)>,
}

#[pymethods]
impl NegativeSampler {
    /// Construct a new negative sampler.
    #[new]
    fn new<'py>(
        rc_set: Bound<'py, RowColumnSet>,
        rows: Bound<'py, PyArray1<i32>>,
        tgt_n: usize,
    ) -> PyResult<Self> {
        let n_rows = rows.readonly().as_array().len();
        let rows = rows.unbind();

        if n_rows > u16::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "too many rows: {} > 2^16",
                n_rows
            )));
        }
        if tgt_n > u16::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "too many targets: {} > 2^16",
                tgt_n
            )));
        }
        debug!(
            "creating sampler for {} negatives for {} rows",
            tgt_n, n_rows
        );

        Ok(NegativeSampler {
            rc_set: rc_set.unbind(),
            rows,
            negatives: Array2::from_elem((n_rows, tgt_n), -1),
            remaining: (0..n_rows as u16)
                .flat_map(|r| (0..tgt_n as u16).map(move |c| (r, c)))
                .collect(),
        })
    }

    fn num_remaining(&self) -> usize {
        self.remaining.len()
    }

    fn accumulate<'py>(
        &mut self,
        py: Python<'py>,
        items: Bound<'py, PyArray1<i32>>,
        force: bool,
    ) -> PyResult<()> {
        if self.negatives.is_empty() {
            return Err(PyRuntimeError::new_err(
                "sampler already finished".to_string(),
            ));
        }
        let rows_py = self.rows.bind_borrowed(py).readonly();
        let rows = rows_py.as_array();
        let items_py = items.readonly();
        let items = items_py.as_array();
        let rcs_ref = self.rc_set.borrow(py);

        let nr = self.remaining.len();
        let mut remaining = Vec::with_capacity(nr);

        for i in 0..nr {
            let (pr, pc) = self.remaining[i];
            let ri = pr as usize;
            let ci = pc as usize;

            let item = items[i];
            let row = rows[ri];
            if force || !rcs_ref.contains_pair(row, item) {
                self.negatives[(ri, ci)] = item;
            } else {
                remaining.push((pr, pc));
            }
        }

        mem::swap(&mut remaining, &mut self.remaining);

        Ok(())
    }

    fn result<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let array = mem::take(&mut self.negatives);
        let array = PyArray2::from_owned_array(py, array);
        Ok(array)
    }
}
