// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accelerated sampling support.
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::data::CoordinateTable;

/// Sample negative columns for given rows from a coordinate table.
#[pyfunction]
#[pyo3(signature=(coords, rows, n_cols, *, max_attempts=10, pop_weighted=false, seed))]
pub fn sample_negatives<'py>(
    py: Python<'py>,
    coords: &CoordinateTable,
    rows: Bound<'py, PyArray1<i32>>,
    n_cols: i32,
    max_attempts: i32,
    pop_weighted: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut rng = Pcg64::seed_from_u64(seed);

    let n = rows.len()?;
    let rows_py = rows.readonly();
    let rows = rows_py.as_array();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
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
                result.push(c);
                break;
            } else {
                attempts += 1
            }
        }
    }

    Ok(PyArray1::from_vec(py, result))
}
