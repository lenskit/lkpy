// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Data management accelerators.
use std::cmp::min;

use arrow::{
    array::{ArrayData, make_array},
    pyarrow::PyArrowType,
    row::{RowConverter, SortField},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use sha1::{Digest, Sha1};

mod cooc;
mod coordinates;
mod index;
mod pairs;
mod pyon;
mod sampling;
mod scatter;
mod selection;
mod sorting;
mod transpose;

pub use coordinates::CoordinateTable;
pub use index::IDIndex;

#[pymodule(name = "data")]
pub mod pymod {
    #[pymodule_export]
    use super::CoordinateTable;
    #[pymodule_export]
    use super::IDIndex;

    #[pymodule_export]
    use super::cooc::count_cooc;
    #[pymodule_export]
    use super::cooc::dense_cooc;
    #[pymodule_export]
    use super::hash_array;
    #[pymodule_export]
    use super::pyon::pyon_loads;
    #[pymodule_export]
    use super::sampling::sample_negatives;
    #[pymodule_export]
    use super::scatter::scatter_array;
    #[pymodule_export]
    use super::scatter::scatter_array_empty;
    #[pymodule_export]
    use super::selection::negative_mask;
    #[pymodule_export]
    use super::sorting::argsort_descending;
    #[pymodule_export]
    use super::sorting::argtopn;
    #[pymodule_export]
    use super::sorting::is_sorted_coo;
    #[pymodule_export]
    use super::transpose::transpose_csr;
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

    Ok(hex::encode(hash.finalize()))
}
