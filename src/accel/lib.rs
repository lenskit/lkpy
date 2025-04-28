// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

mod als;
mod data;
mod funksvd;
mod knn;
mod progress;
mod sampling;
mod sparse;
mod types;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    knn::register_knn(m)?;
    als::register_als(m)?;
    data::register_data(m)?;

    m.add_class::<funksvd::FunkSVDTrainer>()?;
    m.add_class::<sampling::NegativeSampler>()?;
    m.add_function(wrap_pyfunction!(init_accel_pool, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::sparse_row_debug, m)?)?;

    Ok(())
}

#[pyfunction]
fn init_accel_pool(n_threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Rayon initialization error"))
}
