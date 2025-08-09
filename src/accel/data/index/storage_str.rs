// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::hash::{Hash, Hasher};

use arrow::array::{Array, ArrayData, AsArray, StringArray, StringBuilder};
use arrow::compute::cast;
use arrow_schema::DataType;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyAnyMethods;
use rustc_hash::FxHasher;

use crate::data::index::storage::WrappedData;

use super::storage::IDArray;

/// ID array implementation for string IDs.
pub struct IDStringArray {
    array: StringArray,
}

struct StringWrapper<'a> {
    this: &'a StringArray,
    other: StringArray,
}

impl IDStringArray {
    pub fn new(array: StringArray) -> Self {
        IDStringArray { array }
    }
}

impl IDArray for IDStringArray {
    fn hash_entry(&self, idx: u32) -> u64 {
        hash_array_idx(&self.array, idx as usize)
    }

    fn compare_entries(&self, i1: u32, i2: u32) -> bool {
        compare_array_idx(&self.array, i1 as usize, &self.array, i2 as usize)
    }

    fn data(&self) -> ArrayData {
        self.array.clone().into_data()
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    fn wrap_value<'py, 'a>(
        &'a self,
        val: pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<Box<dyn WrappedData + 'a>> {
        let val: &str = val.extract()?;
        let mut ab = StringBuilder::with_capacity(1, val.len() + 1);
        ab.append_value(val);
        Ok(Box::new(StringWrapper {
            this: &self.array,
            other: ab.finish(),
        }))
    }

    fn wrap_array<'py, 'a>(
        &'a self,
        arr: arrow::array::ArrayRef,
    ) -> pyo3::PyResult<Box<dyn WrappedData + 'a>> {
        let arr = cast(&arr, &DataType::Utf8)
            .map_err(|e| PyTypeError::new_err(format!("error castting arrays: {}", e)))?;
        let arr = arr.as_string();
        Ok(Box::new(StringWrapper {
            this: &self.array,
            other: arr.clone(),
        }))
    }
}

impl<'a> WrappedData for StringWrapper<'a> {
    fn hash(&self, idx: usize) -> u64 {
        hash_array_idx(&self.other, idx)
    }

    fn compare_with_entry(&self, w_idx: usize, a_idx: u32) -> bool {
        compare_array_idx(self.this, a_idx as usize, &self.other, w_idx)
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
