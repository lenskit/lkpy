// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use log::*;
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

    /// Advance the progress bar by the specified amount.
    pub fn advance<'py>(&self, py: Python<'py>, n: usize) -> PyResult<()> {
        self.update(py, self.count + n)
    }

    /// Update the current completed total of the progress bar.
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
                    debug!("caught Python signal, cancelling");
                    cancel.cancel();
                    return Err(e);
                }
                let n = counter.get();
                if let Err(e) = self.update(py, n) {
                    cancel.cancel();
                    return Err(e);
                }
            }

            debug!("waiting for thread to join");
            match handle.join() {
                Ok(r) => r,
                Err(_) => Err(PyRuntimeError::new_err("worker thread panicked")),
            }
        })
    }

    /// Process an iterator, with progress, thread-detach, and interrupt checks.
    pub fn process_iter_with_counter<'py, I, R, F>(
        &self,
        py: Python<'py>,
        iter: I,
        proc: F,
    ) -> PyResult<R>
    where
        I: ParallelIterator + Send,
        R: Send,
        F: FnOnce(CancelAdapter<I>, &AtomicUsize) -> PyResult<R> + Send,
    {
        let adapter = CancelAdapter::new(iter);
        let cancel = adapter.canceller();
        let caller = thread::current();
        let rc = AtomicUsize::new(0);

        py.detach(move || {
            thread::scope(|scope| {
                let handle = scope.spawn(|| {
                    let result = proc(adapter, &rc);
                    debug!("iteration finished, unparking caller");
                    caller.unpark();
                    result
                });

                let rv = Python::attach(|py| {
                    while !handle.is_finished() {
                        py.detach(|| thread::park_timeout(UPDATE_TIMEOUT));
                        if let Err(e) = py.check_signals() {
                            debug!("caught Python signal, cancelling");
                            cancel.cancel();
                            return Err(e);
                        }
                        let n = rc.load(Ordering::Relaxed);
                        if let Err(e) = self.update(py, n) {
                            debug!("failed to update progress bar, cancelling");
                            cancel.cancel();
                            return Err(e);
                        }
                    }
                    Ok(())
                });

                debug!("waiting for thread to join");
                match handle.join() {
                    Ok(r) => rv.map(|_| r).flatten(),
                    Err(_) => Err(PyRuntimeError::new_err("worker thread panicked")),
                }
            })
        })
    }
}
