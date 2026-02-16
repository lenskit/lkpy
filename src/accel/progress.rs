// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{spawn, JoinHandle};
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{intern, prelude::*, types::PyDict};

const UPDATE_MS: u64 = 200;

struct ProgressThreadState {
    running: bool,
    last_count: usize,
}

struct ProgressData {
    pb: Py<PyAny>,
    count: AtomicUsize,
    state: Mutex<ProgressThreadState>,
    condition: Condvar,
}

/// Thin Rust wrapper around a LensKit progress bar.
///
/// This method applies internal throttling to reduce the number of calls
/// to the Python progress bar.
pub(crate) struct ProgressHandle {
    data: Option<Arc<ProgressData>>,
    handle: Option<JoinHandle<()>>,
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
        pb.map(|pb| {
            let data = Arc::new(ProgressData {
                pb,
                count: AtomicUsize::new(0),
                state: Mutex::new(ProgressThreadState {
                    running: true,
                    last_count: 0,
                }),
                condition: Condvar::new(),
            });
            let d2 = data.clone();
            let handle = spawn(move || d2.background_update());
            ProgressHandle {
                data: Some(data),
                handle: Some(handle),
            }
        })
        .unwrap_or_else(Self::null)
    }

    pub fn null() -> Self {
        ProgressHandle {
            data: None,
            handle: None,
        }
    }

    pub fn tick(&self) {
        self.advance(1);
    }

    pub fn advance(&self, n: usize) {
        if let Some(data) = &self.data {
            data.count.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Force an update of the progress bar.
    pub fn flush(&self) {
        if let Some(data) = &self.data {
            data.ping();
        }
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        if let Some(data) = self.data.take() {
            data.shutdown();
        }
        if let Some(h) = self.handle.take() {
            h.join()
                .map_err(|_e| PyRuntimeError::new_err(format!("progress thread panicked")))
        } else {
            Ok(())
        }
    }
}

impl Clone for ProgressHandle {
    fn clone(&self) -> Self {
        ProgressHandle {
            data: self.data.clone(),
            handle: None,
        }
    }
}

impl Drop for ProgressHandle {
    fn drop(&mut self) {
        self.shutdown().expect("backend panicked")
    }
}

impl ProgressData {
    fn background_update(&self) {
        let timeout = Duration::from_millis(UPDATE_MS);
        let mut state = self.state.lock().expect("poisoned lock");
        while state.running {
            // wait to be notified, or for timeout
            let (lock, _res) = self
                .condition
                .wait_timeout(state, timeout)
                .expect("poisoned lock");
            state = lock;
            let count = self.count.load(Ordering::Relaxed);
            if count > state.last_count {
                Python::attach(|py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item(intern!(py, "completed"), count)?;
                    self.pb
                        .call_method(py, intern!(py, "update"), (), Some(&kwargs))?;
                    Ok::<(), PyErr>(())
                })
                .expect("progress update failed")
            }
            state.last_count = count;
        }
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().expect("poisoned lock");
        state.running = false;
        self.condition.notify_all();
    }

    fn ping(&self) {
        self.condition.notify_all();
    }
}
