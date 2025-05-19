// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{make_array, Array, ArrayData, BooleanBuilder, Int32Array},
    pyarrow::PyArrowType,
};
use pyo3::prelude::*;

use crate::types::checked_array;

/// Efficiently create a negative mask array from an array of indices.
#[pyfunction]
pub(super) fn negative_mask(
    n: usize,
    indices: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let indices = make_array(indices.0);
    let indices: Int32Array = checked_array("indices", &indices)?;
    let mut indices: Vec<i32> = indices.into_iter().flatten().collect();
    indices.sort_unstable();

    let mut mask = BooleanBuilder::with_capacity(n);
    let mut j = 0;
    for i in 0..n {
        let old_j = j;
        // loop, to handle duplicate indices
        while i == indices[j] as usize {
            j += 1;
        }
        if old_j == j {
            // no match
            mask.append_value(true);
        } else {
            // matched, skip this position
            mask.append_value(false);
        }
    }

    let mask = mask.finish();
    Ok(mask.into_data().into())
}
