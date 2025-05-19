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
    let mut last_i = 0;
    for i in 0..n {
        let old_j = j;
        // loop, to handle duplicate indices
        while j < indices.len() && i == indices[j] as usize {
            j += 1;
        }
        // this is optimized for the common case: many fewer to remove
        if old_j != j {
            // this index matches to-remove. step carefully!

            // append `true` for everything since the write position
            if i > last_i {
                mask.append_n(i - last_i, true);
            }
            // append `false` for this item
            mask.append_value(false);
            // update the write position
            last_i = i + 1;
        }
    }
    // fill in write position
    if n > last_i {
        mask.append_n(n - last_i, true);
    }

    let mask = mask.finish();
    Ok(mask.into_data().into())
}
