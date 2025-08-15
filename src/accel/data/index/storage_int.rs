// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::convert::Infallible;
use std::hash::{Hash, Hasher};

use arrow::array::{
    make_array, Array, ArrayData, ArrowPrimitiveType, AsArray, PrimitiveArray, PrimitiveBuilder,
};
use arrow::compute::cast;
use arrow::pyarrow::PyArrowType;
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyInt};
use rustc_hash::FxHasher;

use log::*;

use crate::indirect_hashing::{IndirectHashContent, IndirectSearcher};

/// Backend hash storage for primitive (integer) arrays.
pub struct PrimitiveIDArray<T: ArrowPrimitiveType> {
    array: PrimitiveArray<T>,
}

pub struct PrimSearcher<'a, T: ArrowPrimitiveType> {
    this: &'a PrimitiveArray<T>,
    other: PrimitiveArray<T>,
}

impl<T: ArrowPrimitiveType> PrimitiveIDArray<T> {
    pub fn new(array: PrimitiveArray<T>) -> Self {
        PrimitiveIDArray { array }
    }
}

impl<T: ArrowPrimitiveType> Default for PrimitiveIDArray<T> {
    fn default() -> Self {
        PrimitiveIDArray {
            array: PrimitiveBuilder::new().finish(),
        }
    }
}

impl<T> IndirectHashContent for PrimitiveIDArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash
        + for<'a> FromPyObject<'a>
        + for<'a> IntoPyObject<'a, Target = PyInt, Output = Bound<'a, PyInt>, Error = Infallible>,
{
    type Searcher<'a> = PrimSearcher<'a, T>;

    fn len(&self) -> usize {
        self.array.len()
    }

    fn hash_entry(&self, idx: u32) -> u64 {
        hash_array_idx(&self.array, idx as usize)
    }

    fn compare_entries(&self, i1: u32, i2: u32) -> bool {
        compare_array_idx(&self.array, i1 as usize, &self.array, i2 as usize)
    }

    fn create_searcher<'py, 'a>(
        &'a self,
        _py: Python<'py>,
        val: Bound<'py, PyAny>,
    ) -> PyResult<PrimSearcher<'a, T>> {
        let search_array = if let Ok(val) = val.extract::<T::Native>() {
            let mut ab = PrimitiveBuilder::new();
            ab.append_value(val);
            ab.finish()
        } else if let Ok(PyArrowType(arr)) = val.extract::<PyArrowType<ArrayData>>() {
            let arr = make_array(arr);
            // convert the array to our type
            trace!(
                "casting {} IDs from {} to {}",
                arr.len(),
                arr.data_type(),
                self.array.data_type()
            );
            let arr = cast(&arr, self.array.data_type())
                .map_err(|e| PyTypeError::new_err(format!("error castting arrays: {}", e)))?;
            let arr: &PrimitiveArray<T> = arr.as_primitive();
            arr.clone()
        } else {
            return Err(PyTypeError::new_err(format!(
                "invalid lookup type {}",
                val.get_type()
            )));
        };

        Ok(PrimSearcher {
            this: &self.array,
            other: search_array,
        })
    }
}

impl<'a, T> IndirectSearcher<'a> for PrimSearcher<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    fn len(&self) -> usize {
        self.other.len()
    }

    fn hash(&self, idx: usize) -> u64 {
        hash_array_idx(&self.other, idx)
    }

    fn compare_with_entry(&self, search_idx: usize, tbl_idx: u32) -> bool {
        compare_array_idx(self.this, tbl_idx as usize, &self.other, search_idx)
    }
}

fn hash_array_idx<T>(arr: &PrimitiveArray<T>, idx: usize) -> u64
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    let mut h = FxHasher::default();
    let v = arr.value(idx);
    v.hash(&mut h);
    h.finish()
}

fn compare_array_idx<T>(
    arr1: &PrimitiveArray<T>,
    idx1: usize,
    arr2: &PrimitiveArray<T>,
    idx2: usize,
) -> bool
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    let v1 = arr1.value(idx1);
    let v2 = arr2.value(idx2);
    v1 == v2
}
