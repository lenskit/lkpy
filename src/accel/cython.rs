// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Cython FFI interface (primarily for accessing SciPy BLAS & LAPACK).
use std::{
    ffi::{c_void, CString},
    ptr::NonNull,
};

use pyo3::{prelude::*, types::PyCapsule};

/// Look up a Cython function pointer from a PYX CAPI capsule.
pub fn resolve_ffi_funptr<'py>(
    py: Python<'py>,
    module: &str,
    name: &str,
    check_name: &str,
) -> PyResult<NonNull<c_void>> {
    let pymod = py.import(module)?;
    let ffi = pymod.getattr("__pyx_capi__")?;
    let cap = ffi.get_item(name)?;
    let cap: Bound<'_, PyCapsule> = cap.extract()?;

    let check_name = CString::new(check_name)?;

    // get the capsule name for debugging
    let name = cap
        .name()?
        .map(|n| unsafe { n.as_cstr().to_str().map(|s| s.to_owned()) });
    let name = name.transpose()?;
    match cap.pointer_checked(Some(&check_name)) {
        Err(e) => {
            e.add_note(
                py,
                format!("capsule name was: {}", name.unwrap_or("<unnamed>".into())),
            )?;
            Err(e)
        }
        Ok(x) => Ok(x),
    }
}
