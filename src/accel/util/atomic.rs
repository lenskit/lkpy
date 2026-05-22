// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

#[pyclass]
pub struct AtomicInt {
    inner: AtomicI64,
}

#[pymethods]
impl AtomicInt {
    #[new]
    #[pyo3(signature=(*, initial=0))]
    fn new(initial: i64) -> AtomicInt {
        AtomicInt {
            inner: AtomicI64::new(initial),
        }
    }

    fn load(&self) -> i64 {
        self.inner.load(Ordering::Relaxed)
    }

    fn store(&self, x: i64) {
        self.inner.store(x, Ordering::Relaxed);
    }

    #[pyo3(signature=(*, incr=1))]
    fn fetch_add(&self, incr: i64) -> i64 {
        self.inner.fetch_add(incr, Ordering::Relaxed)
    }
}
