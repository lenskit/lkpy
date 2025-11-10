// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Array scattering support.
use arrow::{array::ArrayData, pyarrow::PyArrowType};
use pyo3::prelude::*;

#[pyfunction]
fn scatter_array(
    dst: PyArrowType<ArrayData>,
    idx: PyArrowType<ArrayData>,
    src: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    todo!();
}

#[pyfunction]
fn scatter_array_empty(
    dst_size: usize,
    idx: PyArrowType<ArrayData>,
    src: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    todo!();
}
