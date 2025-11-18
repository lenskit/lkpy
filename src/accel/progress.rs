// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::{
    ptr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    time::Instant,
};

use pyo3::{intern, prelude::*, types::PyDict};

const UPDATE_SECS: f64 = 0.2;

struct UpdateState {
    count: usize,
    time: f64,
    rate: f64,
}

/// Thin Rust wrapper around a LensKit progress bar.
///
/// This method applies internal throttling to reduce the number of calls
/// to the Python progress bar.
pub(crate) struct ProgressHandle {
    pb: Option<Py<PyAny>>,
    start: Instant,
    count: AtomicUsize,
    last_update: AtomicPtr<UpdateState>,
}

impl ProgressHandle {
    pub fn from_input<'py>(maybe_pb: Bound<'py, PyAny>) -> Self {
        let pb = if maybe_pb.is_none() {
            None
        } else {
            Some(maybe_pb.unbind())
        };
        Self::new(pb)
    }

    pub fn new(pb: Option<Py<PyAny>>) -> Self {
        ProgressHandle {
            pb,
            count: AtomicUsize::new(0),
            start: Instant::now(),
            last_update: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub fn tick(&self) {
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;

        let lu_ptr = self.last_update.load(Ordering::Relaxed);

        let last_update = if lu_ptr.is_null() {
            None
        } else {
            unsafe { Some(&*lu_ptr) }
        };

        let thresh = if let Some(lu) = last_update {
            // bail early if the rate estimate says we don't need to update
            let n = (count - lu.count) as f64;
            if n / lu.rate < UPDATE_SECS * 0.95 {
                return;
            }

            lu.time
        } else {
            0.0
        };

        let time = self.start.elapsed().as_secs_f64();
        // bail if we haven't been running long enough
        if time < thresh + UPDATE_SECS {
            return;
        }

        // we're ready to set the time!
        let update = UpdateState {
            count,
            time,
            rate: count as f64 / time,
        };
        let update_ptr = Box::leak(Box::new(update));
        match self.last_update.compare_exchange_weak(
            lu_ptr,
            update_ptr,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.refresh(count);
                // free the old state if needed
                if !lu_ptr.is_null() {
                    let _ = unsafe { Box::from_raw(lu_ptr) };
                }
            }
            Err(_) => {
                // someone else beat us to the update, free our state and go on
                let _ = unsafe { Box::from_raw(update_ptr) };
            }
        }
    }

    fn refresh(&self, count: usize) {
        Python::attach(|py| {
            py.check_signals()?;
            if let Some(pb) = &self.pb {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "completed"), count)?;
                pb.call_method(py, intern!(py, "update"), (), Some(&kwargs))?;
            }
            Ok::<(), PyErr>(())
        })
        .expect("progress update failed")
    }
}
