// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::fmt::Display;

use pyo3::{PyErr, PyTypeInfo};

#[macro_export]
macro_rules! ok_or_pyerr {
    ($err:expr, $ety:ty, $($arg:expr),*) => {
        $err.ok_or_else(|| <$ety>::new_err(format!($($arg),*)))
    };
}

/// Trait for Pythonizing foreign errors.
///
/// This is implemented for [Result], prefixing the error message with the given
/// string, and for [Option], using the message as the error message.
pub trait PythonizeError {
    type Output;

    /// Convert this error to a Python exception, with the given prefix.
    fn pythonize_err<S: Display, PyE: PyTypeInfo>(self, msg: S) -> Self::Output;
}

impl<T, E> PythonizeError for Result<T, E>
where
    E: Display,
{
    type Output = Result<T, PyErr>;

    fn pythonize_err<S: Display, PyE: PyTypeInfo>(self, msg: S) -> Self::Output {
        self.map_err(|e| PyErr::new::<PyE, _>(format!("{}: {}", msg, e)))
    }
}

impl<T> PythonizeError for Option<T> {
    type Output = Result<T, PyErr>;

    fn pythonize_err<S: Display, E: PyTypeInfo>(self, msg: S) -> Self::Output {
        self.ok_or_else(|| PyErr::new::<E, _>(msg.to_string()))
    }
}
