// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::RwLock;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use pyo3::{intern, prelude::*, types::PyDict};

const UPDATE_SECS: f64 = 0.2;

#[derive(Clone, Copy)]
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
    last_update: RwLock<Option<UpdateState>>,
}

pub(crate) struct ThreadLocalProgressHandle<'parent> {
    parent: &'parent ProgressHandle,
    last_sync: Instant,
    since_last_sync: usize,
    rate: f64,
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
            last_update: RwLock::new(None),
        }
    }

    pub fn thread_local(&self) -> ThreadLocalProgressHandle<'_> {
        ThreadLocalProgressHandle {
            parent: self,
            last_sync: Instant::now(),
            since_last_sync: 0,
            rate: 1000.0,
        }
    }

    pub fn tick(&self) {
        self.advance(1)
    }

    pub fn advance(&self, n: usize) {
        let count = self.count.fetch_add(n, Ordering::Relaxed) + n;

        let last_update = {
            let lock = self.last_update.read().expect("poisoned lock");
            *lock
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

        self.update(count, time);
    }

    /// Force an update of the progress bar.
    pub fn flush(&self) {
        let count = self.count.load(Ordering::Relaxed);
        let sent = (*self.last_update.read().expect("poisoned lock"))
            .map(|s| s.count)
            .unwrap_or_default();
        if count > sent {
            let time = self.start.elapsed().as_secs_f64();
            self.update(self.count.load(Ordering::Relaxed), time);
        }
    }

    fn update(&self, count: usize, time: f64) {
        // we're ready to set the time! if someone else is writing, do nothing, they've handled it
        if let Ok(mut lock) = self.last_update.try_write() {
            *lock = Some(UpdateState {
                count,
                time,
                rate: count as f64 / time,
            });
            self.refresh(count);
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

impl Drop for ProgressHandle {
    fn drop(&mut self) {
        self.flush();
    }
}

impl<'a> ThreadLocalProgressHandle<'a> {
    #[allow(dead_code)]
    pub fn tick(&mut self) {
        self.advance(1)
    }

    pub fn advance(&mut self, n: usize) {
        self.since_last_sync += n;

        // check rate for fast bail
        if self.since_last_sync as f64 / self.rate >= UPDATE_SECS * 0.95 {
            let now = Instant::now();
            let secs = now.saturating_duration_since(self.last_sync).as_secs_f64();
            self.rate = self.since_last_sync as f64 / secs;
            if secs >= UPDATE_SECS {
                self.parent.advance(self.since_last_sync);
                self.since_last_sync = 0;
                self.last_sync = now;
            }
        }
    }
}

impl<'a> Drop for ThreadLocalProgressHandle<'a> {
    fn drop(&mut self) {
        if self.since_last_sync > 0 {
            self.parent.advance(self.since_last_sync);
        }
    }
}
