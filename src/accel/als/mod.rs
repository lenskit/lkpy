// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

mod explicit;
mod implicit;
mod solve;

use pyo3::prelude::*;

/// Register the lenskit._accel.als module
pub fn register_als(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let als = PyModule::new(parent.py(), "als")?;
    parent.add_submodule(&als)?;

    als.add_function(wrap_pyfunction!(explicit::train_explicit_matrix, &als)?)?;
    als.add_function(wrap_pyfunction!(implicit::train_implicit_matrix, &als)?)?;

    Ok(())
}
