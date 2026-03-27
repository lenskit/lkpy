// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::borrow::Cow;
use std::ffi::c_int;
use std::mem::transmute_copy;

use ndarray::{Array1, Array2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

use crate::cython::resolve_ffi_funptr;

type LapackSPOSV = unsafe extern "C" fn(
    uplo: *const u8,
    n: *const c_int,
    nrhs: *const c_int,
    a: *mut f32,
    lda: *const c_int,
    b: *mut f32,
    ldb: *const c_int,
    info: *mut c_int,
) -> ();

#[derive(Error, Debug)]
pub enum SolveError {
    #[error("invalid matrix/vector size: {0}")]
    InvalidSize(Cow<'static, str>),
    #[error("illegal BLAS argument {0}")]
    IllegalValue(i32),
    #[error("array minor {0} is not positive")]
    NotPositive(i32),
}

/// Wrapper for LAPACK solver functions.
#[derive(Clone, Copy)]
pub struct POSV {
    lapack_fn: LapackSPOSV,
}

impl POSV {
    pub fn load<'py>(py: Python<'py>) -> PyResult<POSV> {
        let ptr = resolve_ffi_funptr(py, "scipy.linalg.cython_lapack", "sposv")?;
        Ok(POSV {
            // ugly but it's how we translate void* to a function pointer
            lapack_fn: unsafe { transmute_copy(&ptr) },
        })
    }

    /// Solve a linear system.
    ///
    /// This uses LAPACK `sposv` to solve the linear system. `matrix` must be square,
    /// symmetric, and positive definite.  After solving, it will contain the
    /// Cholesky decomposition.
    pub fn solve(
        &self,
        matrix: &mut Array2<f32>,
        vector: &Array1<f32>,
    ) -> Result<Array1<f32>, SolveError> {
        let mshape = matrix.shape();
        let n = mshape[0];
        if mshape[1] != n {
            return Err(SolveError::InvalidSize("matrix must be square".into()));
        }
        if vector.len() != n {
            return Err(SolveError::InvalidSize(
                format!("matrix dim {} != vector dim {}", n, vector.len()).into(),
            ));
        }

        let n = n as i32;
        let uplo = b'U';
        let nrhs = 1;
        let mut info: c_int = 0;
        let mut soln = vector.clone();
        unsafe {
            (self.lapack_fn)(
                &uplo,
                &n,
                &nrhs,
                matrix.as_mut_ptr(),
                &n,
                soln.as_mut_ptr(),
                &n,
                &mut info,
            );
        }

        if info == 0 {
            Ok(soln)
        } else if info < 0 {
            Err(SolveError::NotPositive(-info - 1))
        } else {
            Err(SolveError::IllegalValue(info))
        }
    }
}

impl Into<PyErr> for SolveError {
    fn into(self) -> PyErr {
        PyRuntimeError::new_err(format!("LAPACK error: {}", self))
    }
}
