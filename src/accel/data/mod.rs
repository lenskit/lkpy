// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Data management accelerators.
use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use ndarray::ArrayD;
use pyo3::prelude::*;

mod index;
mod rc_set;
mod selection;
mod sorting;

use index::IDIndex;
pub use rc_set::RowColumnSet;

/// Register the lenskit._accel.als module
pub fn register_data(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<RowColumnSet>()?;

    let data = PyModule::new(parent.py(), "data")?;
    parent.add_submodule(&data)?;

    data.add_class::<IDIndex>()?;

    data.add_function(wrap_pyfunction!(sorting::is_sorted_coo, &data)?)?;
    data.add_function(wrap_pyfunction!(sorting::argsort_descending, &data)?)?;
    data.add_function(wrap_pyfunction!(selection::negative_mask, &data)?)?;

    Ok(())
}
