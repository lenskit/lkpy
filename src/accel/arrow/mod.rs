// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Utilities and types for working with Arrow data.
pub mod dispatch;
pub mod lists;
pub mod types;

use arrow::array::Array;
use arrow::array::ArrayData;
use arrow::array::ArrowPrimitiveType;
use arrow::array::PrimitiveArray;
use arrow::array::downcast_array;
use arrow::array::make_array;
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
pub use types::SparseIndexListType;
pub use types::SparseIndexType;
pub use types::SparseRowType;

use crate::ok_or_pyerr;

#[pyfunction]
pub fn arrow_type(arr: PyArrowType<ArrayData>) -> String {
    let arr = make_array(arr.0);
    format!("{:?}", arr.data_type())
}

pub fn checked_array_ref<'array, T: Array + 'static>(
    name: &str,
    tstr: &str,
    array: &'array dyn Array,
) -> PyResult<&'array T> {
    ok_or_pyerr!(
        array.as_any().downcast_ref(),
        PyTypeError,
        "invalid {} type {}, expected {}",
        name,
        array.data_type(),
        tstr
    )
}

pub fn checked_array<'array, E: ArrowPrimitiveType + 'static>(
    name: &str,
    array: &'array dyn Array,
) -> PyResult<PrimitiveArray<E>> {
    if array.data_type().equals_datatype(&E::DATA_TYPE) {
        Ok(downcast_array(array))
    } else {
        Err(PyTypeError::new_err(format!(
            "invalid {} type {}, expected {}",
            name,
            array.data_type(),
            E::DATA_TYPE
        )))
    }
}
