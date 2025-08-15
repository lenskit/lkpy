// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::hash::{Hash, Hasher};

use arrow::array::{make_array, Array, ArrayData, AsArray, StringArray, StringBuilder};
use arrow::compute::cast;
use arrow::pyarrow::PyArrowType;
use arrow_schema::DataType;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyAnyMethods;
use rustc_hash::FxHasher;

use crate::indirect_hashing::{IndirectHashContent, IndirectSearcher};

/// ID array implementation for string IDs.
pub struct StringContentArray {
    array: StringArray,
}

pub struct StringSearch<'a> {
    this: &'a StringArray,
    other: StringArray,
}

impl StringContentArray {
    pub fn new(array: StringArray) -> Self {
        StringContentArray { array }
    }
}

impl IndirectHashContent for StringContentArray {
    type Searcher<'a> = StringSearch<'a>;

    fn hash_entry(&self, idx: u32) -> u64 {
        hash_array_idx(&self.array, idx as usize)
    }

    fn compare_entries(&self, i1: u32, i2: u32) -> bool {
        compare_array_idx(&self.array, i1 as usize, &self.array, i2 as usize)
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    fn create_searcher<'py, 'a>(
        &'a self,
        _py: pyo3::Python<'py>,
        val: pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<StringSearch<'a>> {
        let arr = if let Ok(val) = val.extract::<&str>() {
            let mut ab = StringBuilder::with_capacity(1, val.len() + 1);
            ab.append_value(val);
            ab.finish()
        } else if let Ok(PyArrowType(arr)) = val.extract::<PyArrowType<ArrayData>>() {
            let arr = make_array(arr);
            let arr = cast(&arr, &DataType::Utf8)
                .map_err(|e| PyTypeError::new_err(format!("error casting arrays: {}", e)))?;
            arr.as_string().clone()
        } else {
            return Err(PyTypeError::new_err(format!(
                "invalid value type {}",
                val.get_type()
            )));
        };

        Ok(StringSearch {
            this: &self.array,
            other: arr,
        })
    }
}

impl<'a> IndirectSearcher<'a> for StringSearch<'a> {
    fn len(&self) -> usize {
        self.other.len()
    }

    fn hash(&self, idx: usize) -> u64 {
        hash_array_idx(&self.other, idx)
    }

    fn compare_with_entry(&self, search_idx: usize, table_idx: u32) -> bool {
        compare_array_idx(self.this, table_idx as usize, &self.other, search_idx)
    }
}

fn hash_array_idx(arr: &StringArray, idx: usize) -> u64 {
    let mut h = FxHasher::default();
    let v = arr.value(idx);
    v.hash(&mut h);
    h.finish()
}

fn compare_array_idx(arr1: &StringArray, idx1: usize, arr2: &StringArray, idx2: usize) -> bool {
    let v1 = arr1.value(idx1);
    let v2 = arr2.value(idx2);
    v1 == v2
}
