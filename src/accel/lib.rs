// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::prelude::*;

mod als;
mod arrow;
mod atomic;
mod data;
mod funksvd;
mod indirect_hashing;
mod knn;
mod parallel;
mod progress;
mod slim;
mod sparse;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    knn::register_knn(m)?;
    als::register_als(m)?;
    data::register_data(m)?;
    slim::register_slim(m)?;

    m.add_class::<funksvd::FunkSVDTrainer>()?;
    m.add_function(wrap_pyfunction!(parallel::init_accel_pool, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::sparse_row_debug_type, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::sparse_structure_debug_large, m)?)?;

    Ok(())
}
