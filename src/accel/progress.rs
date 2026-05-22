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
use rayon_cancel::{CancelAdapter, CountHandle};

const UPDATE_TIMEOUT: Duration = Duration::from_millis(200);

/// Thin Rust wrapper around a LensKit progress bar.
///
/// This method applies internal throttling to reduce the number of calls
/// to the Python progress bar.
pub struct ProgressHandle {
    pb: Option<Py<PyAny>>,
    count: usize,
}

trait ProgressCounter {
    fn get_total(&self) -> usize;
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

        self.run_with_progress(py, &counter, || cancel.cancel(), || proc(adapter))
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
        let rc = AtomicUsize::new(0);

        self.run_with_progress(py, &rc, || cancel.cancel(), || proc(adapter, &rc))
    }

    /// run a thread in the background, monitoring progress and issuing cancels.
    fn run_with_progress<'py, R, C, X, F>(
        &self,
        py: Python<'py>,
        counter: C,
        cancel: X,
        proc: F,
    ) -> PyResult<R>
    where
        R: Send,
        C: ProgressCounter + Send + Sync,
        X: Fn() -> () + Send,
        F: FnOnce() -> PyResult<R> + Send,
    {
        let caller = thread::current();

        py.detach(move || {
            thread::scope(move |scope| {
                let handle = scope.spawn(move || {
                    let result = proc();
                    caller.unpark();
                    result
                });

                let mut err = Ok(());

                while err.is_ok() && !handle.is_finished() {
                    thread::park_timeout(UPDATE_TIMEOUT);
                    err = Python::attach(|py| {
                        if let Err(e) = py.check_signals() {
                            debug!("caught Python signal, cancelling");
                            cancel();
                            return Err(e);
                        }
                        let n = counter.get_total();
                        if let Err(e) = self.update(py, n) {
                            cancel();
                            return Err(e);
                        }
                        Ok(())
                    });
                }

                debug!("waiting for thread to join");
                match handle.join() {
                    Ok(Err(e)) => {
                        if let Err(e2) = err {
                            error!("status update failed: {}", e2);
                        }
                        Err(e)
                    }
                    Ok(Ok(r)) => err.map(|_| r),
                    Err(_) => Err(PyRuntimeError::new_err("worker thread panicked")),
                }
            })
        })
    }
}

impl ProgressCounter for &CountHandle {
    fn get_total(&self) -> usize {
        self.get()
    }
}

impl ProgressCounter for &AtomicUsize {
    fn get_total(&self) -> usize {
        self.load(Ordering::Relaxed)
    }
}
