// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
use pyo3::prelude::*;

mod als;
mod arrow;
mod cython;
mod data;
mod errors;
mod funksvd;
mod indirect;
mod knn;
mod logging;
mod parallel;
mod sparse;
mod tasks;
mod util;

/// Layout for LensKit accelerator Python module.
#[pymodule(gil_used = false, name = "_accel")]
mod accel {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::als::pymod as als;
    #[pymodule_export]
    use super::data::pymod as data;
    #[pymodule_export]
    use super::knn::pymod as knn;

    #[pymodule_export]
    use super::funksvd::FunkSVDTrainer;
    #[pymodule_export]
    use super::logging::AccelLogListener;
    #[pymodule_export]
    use super::parallel::NestedAccelPool;
    #[pymodule_export]
    use super::util::atomic::AtomicInt;

    #[pymodule_export]
    use super::arrow::arrow_type;
    #[pymodule_export]
    use super::parallel::init_accel_pool;
    #[pymodule_export]
    use super::parallel::thread_count;
    #[pymodule_export]
    use super::sparse::sparse_row_debug_type;
    #[pymodule_export]
    use super::sparse::sparse_structure_debug_large;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::logging::init_rust_logger()
            .map_err(|_| PyRuntimeError::new_err("failed to install Rust logger"))?;

        let helper = m.py().import("lenskit.logging._accel")?;
        helper.call_method("ensure_accel_logging", (), None)?;
        Ok(())
    }
}
