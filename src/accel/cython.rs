// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Cython FFI interface (primarily for accessing SciPy BLAS & LAPACK).
use std::ffi::c_void;

use pyo3::{prelude::*, types::PyCapsule};

/// Look up a Cython function pointer from a PYX CAPI capsule.
pub fn resolve_ffi_funptr<'py>(
    py: Python<'py>,
    module: &str,
    name: &str,
) -> PyResult<*const c_void> {
    let pymod = py.import(module)?;
    let ffi = pymod.getattr("__pyx_capi__")?;
    let cap = ffi.get_item(name)?;
    let cap: Bound<'_, PyCapsule> = cap.extract()?;
    Ok(cap.pointer())
}
