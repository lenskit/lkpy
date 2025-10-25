// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use log::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[cfg(feature = "fuse-parallel")]
use rayon::iter::PanicFuse;
use rayon::{current_num_threads, iter::ParallelIterator, ThreadPoolBuilder};

#[pyfunction]
pub fn init_accel_pool(n_threads: usize) -> PyResult<()> {
    debug!(
        "initializing accelerator thread pool with {} threads",
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

#[cfg(not(feature = "fuse-parallel"))]
pub fn maybe_fuse<I: ParallelIterator>(iter: I) -> I {
    iter
}

#[cfg(feature = "fuse-parallel")]
pub fn maybe_fuse<I: ParallelIterator>(iter: I) -> PanicFuse<I> {
    iter.panic_fuse()
}
