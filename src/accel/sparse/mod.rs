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
mod coo;
mod csr;

pub use crate::arrow::SparseIndexListType;
pub use crate::arrow::SparseIndexType;
pub use crate::arrow::SparseRowType;
pub use consumer::ArrowCSRConsumer;
pub use coo::{COOMatrix, COOMatrixBuilder};
pub use csr::{csr_structure, CSRMatrix, CSRStructure, IxVar, CSR};

/// Test function to make sure we can convert sparse rows.
#[pyfunction]
pub(crate) fn sparse_row_debug_type(
    array: PyArrowType<ArrayData>,
) -> PyResult<(String, usize, usize)> {
    let array = make_array(array.0);
    debug!("building matrix with {} rows", array.len());
    debug!("array data type: {}", array.data_type());

    array
        .data_type()
        .try_into()
        .map(|rt: SparseRowType| {
            debug!(
                "got {} x {} matrix with {} indices and {} values",
                array.len(),
                rt.dimension(),
                rt.offset_type,
                rt.value_type
            );
            (format!("{:?}", rt), array.len(), rt.dimension())
        })
        .or_else(|e| {
            debug!("row matrix failed: {}", e);
            array.data_type().try_into().map(|rt: SparseIndexListType| {
                debug!(
                    "got {} x {} matrix with {} indices and no values",
                    array.len(),
                    rt.dimension(),
                    rt.offset_type,
                );
                (format!("{:?}", rt), array.len(), rt.dimension())
            })
        })
        .map_err(|e| PyTypeError::new_err(format!("arrow error: {}", e)))
}

/// Test function to make sure we can convert sparse rows to large.
#[pyfunction]
pub(crate) fn sparse_structure_debug_large(
    array: PyArrowType<ArrayData>,
) -> PyResult<(usize, usize, usize)> {
    let array = make_array(array.0);
    debug!("extracting matrix with {} rows", array.len());
    debug!("array data type: {}", array.data_type());
    let csr = CSRStructure::<i64>::from_arrow(array)?;
    assert_eq!(csr.len(), csr.n_rows);
    Ok((csr.len(), csr.n_cols, csr.nnz()))
}
