// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::ffi::c_int;

use ndarray::{Array1, Array2};
use pyo3::{exceptions::PyRuntimeError, PyErr};
use thiserror::Error;

unsafe extern "C" {
    fn scipy_sposv_(
        uplo: *const u8,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        b: *mut f32,
        ldb: *const c_int,
        info: *mut c_int,
    );
}

#[derive(Error, Debug)]
pub enum SolveError {
    #[error("illegal value")]
    IllegalValue,
    #[error("array is not positive")]
    NotPositive,
}

pub fn sposv(matrix: &mut Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, SolveError> {
    let mshape = matrix.shape();
    assert_eq!(mshape[0], mshape[1]);
    assert_eq!(vector.len(), mshape[0]);
    let uplo = b'U';
    let n = mshape[1] as i32;
    let nrhs = 1;
    let mut info: c_int = 0;
    let mut soln = vector.clone();
    unsafe {
        scipy_sposv_(
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
        Err(SolveError::NotPositive)
    } else {
        Err(SolveError::IllegalValue)
    }
}

impl Into<PyErr> for SolveError {
    fn into(self) -> PyErr {
        PyRuntimeError::new_err(format!("LAPACK error: {}", self))
    }
}
