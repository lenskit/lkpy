// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use arrow_schema::DataType;
use pyo3::prelude::*;

use crate::sparse::CSRStructure;

#[pyfunction]
pub fn transpose_csr(arr: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let array = make_array(arr.0);
}
