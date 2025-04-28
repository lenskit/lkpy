// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse matrix support.

use log::*;
use pyo3::prelude::*;

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use pyo3::exceptions::PyTypeError;
use pyo3::PyResult;

mod consumer;
mod index;
mod index_list;
mod matrix;
mod row;

pub use consumer::ArrowCSRConsumer;
pub use index::SparseIndexType;
pub use index_list::SparseIndexListType;
pub use matrix::{CSRMatrix, CSRStructure, CSR};
pub use row::SparseRowType;

/// Test function to make sure we can convert sparse rows.
#[pyfunction]
pub(crate) fn sparse_row_debug(array: PyArrowType<ArrayData>) -> PyResult<(String, usize, usize)> {
    let array = make_array(array.0);
    debug!("building matrix with {} rows", array.len());
    debug!("array data type: {}", array.data_type());

    array
        .data_type()
        .try_into()
        .map(|rt: SparseRowType| {
            debug!(
                "got {} x {} matrix with {} values",
                array.len(),
                rt.dimension(),
                rt.value_type
            );
            (format!("{:?}", rt), array.len(), rt.dimension())
        })
        .or_else(|e| {
            debug!("row matrix failed: {}", e);
            array.data_type().try_into().map(|rt: SparseIndexListType| {
                debug!(
                    "got {} x {} matrix without values",
                    array.len(),
                    rt.dimension(),
                );
                (format!("{:?}", rt), array.len(), rt.dimension())
            })
        })
        .map_err(|e| PyTypeError::new_err(format!("arrow error: {}", e)))
}
