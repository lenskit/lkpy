// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{array::ArrayData, pyarrow::PyArrowType};
use pyo3::prelude::*;

#[pyfunction]
pub fn transpose_csr(arr: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    todo!()
}
