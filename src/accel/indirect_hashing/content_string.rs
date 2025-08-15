// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::hash::{Hash, Hasher};

use arrow::array::{
    make_array, Array, ArrayData, AsArray, GenericStringArray, OffsetSizeTrait, StringArray,
    StringBuilder, StringViewArray,
};
use arrow::pyarrow::PyArrowType;
use arrow_schema::DataType;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyAnyMethods;
use rustc_hash::FxHasher;

use crate::indirect_hashing::{IndirectHashContent, IndirectSearcher};

/// Helper trait for accessing string arrays.
trait StringAccess {
    fn len(&self) -> usize;
    fn value(&self, idx: usize) -> &str;
}

/// ID array implementation for string IDs.
pub struct StringContentArray {
    array: StringArray,
}

pub struct StringSearch<'a> {
    this: &'a StringArray,
    other: Box<dyn StringAccess>,
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
        Array::len(&self.array)
    }

    fn create_searcher<'py, 'a>(
        &'a self,
        _py: pyo3::Python<'py>,
        val: pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<StringSearch<'a>> {
        let arr: Box<dyn StringAccess> = if let Ok(val) = val.extract::<&str>() {
            let mut ab = StringBuilder::with_capacity(1, val.len() + 1);
            ab.append_value(val);
            Box::new(ab.finish())
        } else if let Ok(PyArrowType(arr)) = val.extract::<PyArrowType<ArrayData>>() {
            let arr = make_array(arr);
            match arr.data_type() {
                DataType::Utf8 => Box::new(arr.as_string::<i32>().clone()),
                DataType::Utf8View => Box::new(arr.as_string_view().clone()),
                DataType::LargeUtf8 => Box::new(arr.as_string::<i64>().clone()),
                t => {
                    return Err(PyTypeError::new_err(format!(
                        "unsupported array type {:?}",
                        t
                    )))
                }
            }
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
        hash_array_idx(self.other.as_ref(), idx)
    }

    fn compare_with_entry(&self, search_idx: usize, table_idx: u32) -> bool {
        compare_array_idx(
            self.this,
            table_idx as usize,
            self.other.as_ref(),
            search_idx,
        )
    }
}

fn hash_array_idx(arr: &dyn StringAccess, idx: usize) -> u64 {
    let mut h = FxHasher::default();
    let v = arr.value(idx);
    v.hash(&mut h);
    h.finish()
}

fn compare_array_idx(
    arr1: &StringArray,
    idx1: usize,
    arr2: &dyn StringAccess,
    idx2: usize,
) -> bool {
    let v1 = arr1.value(idx1);
    let v2 = arr2.value(idx2);
    v1 == v2
}

impl<O> StringAccess for GenericStringArray<O>
where
    O: OffsetSizeTrait,
{
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn value(&self, idx: usize) -> &str {
        GenericStringArray::value(self, idx)
    }
}

impl StringAccess for StringViewArray {
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn value(&self, idx: usize) -> &str {
        StringViewArray::value(self, idx)
    }
}
