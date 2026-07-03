// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for monitored accelerator tasks.

use std::{panic::catch_unwind, sync::Mutex};

use pyo3::{IntoPyObjectExt, exceptions::PyRuntimeError, prelude::*, types::PyNone};
mod atomic;
mod progress;

pub use atomic::AtomicCancel;
use log::*;
pub use progress::IterCancel;

use crate::parallel::NestedAccelPool;

/// Trait for the Rust side of the accelerator task interface.
pub trait AccelTaskImpl: Sync + Send + 'static {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>>;
}

/// Trait for cancellation support in task backends.
pub trait CancelImpl: Sync + Send + 'static {
    fn cancel(&self);
    fn current_progress(&self) -> Option<(u64, Option<u64>)>;
}

/// Struct for the Python side of the accelerator task interface.
#[pyclass]
pub struct AccelTask {
    task: Box<dyn AccelTaskImpl>,
    cancel: Mutex<Option<Box<dyn CancelImpl>>>,
}

impl AccelTask {
    pub fn wrap<T: AccelTaskImpl>(task: T) -> AccelTask {
        AccelTask {
            task: Box::new(task),
            cancel: Mutex::new(None),
        }
    }

    pub(crate) fn set_cancel<C: CancelImpl>(&self, cancel: C) {
        let mut lock = self.cancel.lock().expect("lock poisoned");
        *lock = Some(Box::new(cancel))
    }
}

impl<T: AccelTaskImpl> From<T> for AccelTask {
    fn from(value: T) -> Self {
        AccelTask {
            task: Box::new(value),
            cancel: Mutex::new(None),
        }
    }
}

#[pymethods]
impl AccelTask {
    #[pyo3(signature=(*,pool=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        pool: Option<PyRef<'py, NestedAccelPool>>,
    ) -> PyResult<Py<PyAny>> {
        let pool = pool
            .map(|p| {
                let ip = p.get_pool();
                if ip.is_none() {
                    warn!("attempted to invoke accelerator task with shut-down thread pool")
                }
                ip
            })
            .flatten();

        if let Some(pool) = pool {
            py.detach(|| {
                pool.install(|| Python::attach(|py| self.task.invoke(py, self).map(Bound::unbind)))
            })
        } else {
            trace!("running task on current thread");
            Ok(self.task.invoke(py, self)?.unbind())
        }
    }

    fn cancel(&self) {
        let lock = self.cancel.lock().expect("lock poisoned");
        if let Some(cancel) = &*lock {
            cancel.cancel();
        }
    }

    fn current_progress<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let lock = self.cancel.lock().expect("lock poisoned");
        if let Some(cancel) = &*lock {
            match cancel.current_progress() {
                Some((x, None)) => Ok(x.into_pyobject(py)?.as_any().clone()),
                x => Ok(x.into_pyobject(py)?.as_any().clone()),
            }
        } else {
            PyNone::get(py).into_bound_py_any(py)
        }
    }
}
