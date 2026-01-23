// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accelerated sampling support.
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::data::CoordinateTable;

/// Sample negative columns for given rows from a coordinate table.
#[pyfunction]
#[pyo3(signature=(coords, rows, n_cols, *, n=1, max_attempts=10, pop_weighted=false, seed))]
pub fn sample_negatives<'py>(
    py: Python<'py>,
    coords: &CoordinateTable,
    rows: Bound<'py, PyArray1<i32>>,
    n_cols: i32,
    n: usize,
    max_attempts: i32,
    pop_weighted: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let mut rng = Pcg64::seed_from_u64(seed);

    let rows_py = rows.readonly();
    let rows = rows_py.as_array();
    let n_rows = rows.len();

    let mut result = Array2::uninit((n_rows, n));

    for rep in 0..n {
        for i in 0..n_rows {
            let row = rows[i];
            let mut attempts = 0;
            loop {
                let c = if pop_weighted {
                    let i = rng.random_range(0..coords.len());
                    coords.get(1, i)
                } else {
                    rng.random_range(0..n_cols)
                };
                let pair = [row, c];
                if coords.lookup(&pair).is_none() || attempts >= max_attempts {
                    result[(i, rep)].write(c);
                    break;
                } else {
                    attempts += 1
                }
            }
        }
    }

    let result = unsafe { result.assume_init() };

    Ok(PyArray2::from_owned_array(py, result))
}
