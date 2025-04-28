// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{Int32Array, RecordBatch},
    pyarrow::PyArrowType,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::types::checked_array;

/// Check if two columns of a table are properly-sorted COO.
#[pyfunction]
pub(super) fn is_sorted_coo<'py>(
    data: Vec<PyArrowType<RecordBatch>>,
    c1: &'py str,
    c2: &'py str,
) -> PyResult<bool> {
    let mut last = None;
    for PyArrowType(batch) in data {
        let col1 = batch
            .column_by_name(c1)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column: {}", c1)))?;
        let col2 = batch
            .column_by_name(c2)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column: {}", c2)))?;

        let col1: Int32Array = checked_array(c1, col1)?;
        let col2: Int32Array = checked_array(c2, col2)?;

        for i in 0..col1.len() {
            let v1 = col1.value(i);
            let v2 = col2.value(i);
            let k = (v1, v2);
            if let Some(lk) = last {
                if k <= lk {
                    // found a key out-of-order, we're done
                    return Ok(false);
                }
            }
            last = Some(k);
        }
    }

    // got this far, we're sorted
    Ok(true)
}
