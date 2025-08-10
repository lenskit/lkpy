// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Data management accelerators.
use std::cmp::min;

use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
    row::{RowConverter, SortField},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

mod index;
mod rc_set;
mod selection;
mod sorting;

use index::IDIndex;
pub use rc_set::RowColumnSet;
use sha1::{Digest, Sha1};

/// Register the lenskit._accel.als module
pub fn register_data(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<RowColumnSet>()?;

    let data = PyModule::new(parent.py(), "data")?;
    parent.add_submodule(&data)?;

    data.add_class::<IDIndex>()?;

    data.add_function(wrap_pyfunction!(sorting::is_sorted_coo, &data)?)?;
    data.add_function(wrap_pyfunction!(sorting::argsort_descending, &data)?)?;
    data.add_function(wrap_pyfunction!(selection::negative_mask, &data)?)?;
    data.add_function(wrap_pyfunction!(hash_array, &data)?)?;

    Ok(())
}

#[pyfunction]
fn hash_array(arr: PyArrowType<ArrayData>) -> PyResult<String> {
    let arr = make_array(arr.0);
    let len = arr.len();
    let mut hash = Sha1::new();
    let mut start = 0;
    let rc = RowConverter::new(vec![SortField::new(arr.data_type().clone())])
        .map_err(|e| PyRuntimeError::new_err(format!("could not build row converter: {}", e)))?;
    while start < len {
        let bsize = min(2048, len - start);
        let slice = arr.slice(start, bsize);
        let rows = rc
            .convert_columns(&[slice])
            .map_err(|e| PyRuntimeError::new_err(format!("could not build rows: {}", e)))?;
        for row in rows.into_iter() {
            hash.update(row.data());
        }
        start += bsize;
    }

    Ok(hex::encode(&hash.finalize()))
}
