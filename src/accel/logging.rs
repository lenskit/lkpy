// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::str::FromStr;
use std::sync::Arc;

use arc_swap::{ArcSwap, ArcSwapOption};
use log::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;

static LOG: PyLogger = PyLogger {
    channel: ArcSwapOption::const_empty(),
};

#[derive(Debug, Clone, Serialize)]
struct LogMsg {
    level: Level,
    logger: String,
    message: String,
}

/// Inner structure for handling log messages.
struct LogMeet {
    channel: crossbeam_channel::Sender<LogMsg>,
    filter: ArcSwap<LevelFilter>,
}

/// Rust logging implementation.
struct PyLogger {
    channel: ArcSwapOption<LogMeet>,
}

#[pyclass(module = "lenskit._accel")]
pub struct AccelLogListener {
    channel: crossbeam_channel::Receiver<LogMsg>,
}

impl PyLogger {
    fn install(&self, meet: LogMeet) {
        self.channel.store(Some(Arc::new(meet)));
    }
}

impl Log for PyLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        if let Some(meet) = &*self.channel.load() {
            let filt = meet.filter.load();
            metadata.level() <= **filt
        } else {
            false
        }
    }

    fn log(&self, record: &Record) {
        if let Some(meet) = self.channel.load_full() {
            let msg = LogMsg {
                level: record.level(),
                logger: record.target().to_string(),
                message: record.args().to_string(),
            };
            let _ = meet.channel.send(msg);
        }
    }

    fn flush(&self) {
        // do nothing
    }
}

#[pymethods]
impl AccelLogListener {
    #[new]
    fn new() -> PyResult<AccelLogListener> {
        let level = ArcSwap::from_pointee(LevelFilter::Debug);
        let (send, recv) = crossbeam_channel::unbounded();
        LOG.install(LogMeet {
            channel: send,
            filter: level,
        });
        Ok(AccelLogListener { channel: recv })
    }

    /// Get the next message, if any.
    fn get_message<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let res = py.detach(|| self.channel.recv());
        match res {
            Ok(m) => {
                let dict = PyDict::new(py);
                dict.set_item("level", m.level.to_string())?;
                dict.set_item(
                    "logger",
                    m.logger
                        .replace("lenskit_accel", "lenskit._accel")
                        .replace("::", "."),
                )?;
                dict.set_item("message", m.message)?;
                Ok(Some(dict))
            }
            Err(_) => Ok(None),
        }
    }

    fn update_level(&self, level: &str) -> PyResult<()> {
        let filter = LevelFilter::from_str(level)
            .map_err(|_| PyValueError::new_err(format!("unsupported level {}", level)))?;
        set_max_level(filter);

        if let Some(meet) = &*LOG.channel.load() {
            meet.filter.store(Arc::new(filter));
        }

        Ok(())
    }
}

/// Initialize the Rust logging backend.  This does **not** install a Python logger.
pub fn init_rust_logger() -> Result<(), SetLoggerError> {
    set_logger(&LOG)
}
