// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use log::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[cfg(debug_assertions)]
use rayon::iter::PanicFuse;
use rayon::{ThreadPool, ThreadPoolBuilder, current_num_threads, iter::ParallelIterator};

#[pyfunction]
pub fn init_accel_pool(n_threads: usize) -> PyResult<()> {
    debug!(
        "initializing global accelerator thread pool with {} threads",
        n_threads
    );
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Rayon initialization error"))
}

#[pyfunction]
pub fn thread_count() -> PyResult<usize> {
    Ok(current_num_threads())
}

#[cfg(not(debug_assertions))]
pub fn maybe_fuse<I: ParallelIterator>(iter: I) -> I {
    iter
}

#[cfg(debug_assertions)]
pub fn maybe_fuse<I: ParallelIterator>(iter: I) -> PanicFuse<I> {
    iter.panic_fuse()
}

#[pyclass]
pub struct NestedAccelPool {
    pool: Option<Arc<ThreadPool>>,
}

#[pymethods]
impl NestedAccelPool {
    #[new]
    fn create(n_threads: usize) -> PyResult<NestedAccelPool> {
        debug!("creating nested pool with {} threads", n_threads);
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not make thread pool: {}", e)))?;
        Ok(NestedAccelPool {
            pool: Some(Arc::new(pool)),
        })
    }

    fn shutdown(&mut self) -> PyResult<()> {
        debug!("shutting down nested pool");
        self.pool = None;
        Ok(())
    }
}

impl NestedAccelPool {
    pub fn get_pool(&self) -> Option<Arc<ThreadPool>> {
        self.pool.clone()
    }
}
