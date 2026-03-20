// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::thread;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{intern, prelude::*, types::PyDict};
use rayon::iter::ParallelIterator;
use rayon_cancel::CancelAdapter;

const UPDATE_TIMEOUT: Duration = Duration::from_millis(200);

/// Thin Rust wrapper around a LensKit progress bar.
///
/// This method applies internal throttling to reduce the number of calls
/// to the Python progress bar.
pub struct ProgressHandle {
    pb: Option<Py<PyAny>>,
    count: usize,
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
        ProgressHandle { pb, count: 0 }
    }

    pub fn tick<'py>(&self, py: Python<'py>) {
        self.advance(py, 1);
    }

    pub fn advance<'py>(&self, py: Python<'py>, n: usize) {
        self.update(py, self.count + n);
    }

    pub fn update<'py>(&self, py: Python<'py>, complete: usize) -> PyResult<()> {
        if let Some(pb) = &self.pb {
            let pb = pb.bind(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "completed"), complete)?;
            pb.call_method(intern!(py, "update"), (), Some(&kwargs))?;
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Process an iterator, with progress, thread-detach, and interrupt checks.
    pub fn process_iter<'py, I, R, F>(&self, py: Python<'py>, iter: I, proc: F) -> PyResult<R>
    where
        I: ParallelIterator + Send,
        R: Send,
        F: FnOnce(CancelAdapter<I>) -> PyResult<R> + Send,
    {
        let adapter = CancelAdapter::new(iter);
        let counter = adapter.counter();
        let cancel = adapter.canceller();
        let caller = thread::current();

        thread::scope(move |scope| {
            let handle = scope.spawn(move || {
                let result = proc(adapter);
                caller.unpark();
                result
            });

            while !handle.is_finished() {
                py.detach(|| thread::park_timeout(UPDATE_TIMEOUT));
                if let Err(e) = py.check_signals() {
                    cancel.cancel();
                    return Err(e);
                }
                let n = counter.get();
                self.update(py, n);
            }

            match handle.join() {
                Ok(r) => r,
                Err(_) => Err(PyRuntimeError::new_err("worker thread panicked")),
            }
        })
    }
}
