// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::convert::Infallible;
use std::hash::{Hash, Hasher};

use arrow::array::{
    Array, ArrayData, ArrowPrimitiveType, AsArray, PrimitiveArray, PrimitiveBuilder,
};
use arrow::compute::cast;
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyInt};
use rustc_hash::FxHasher;

use log::*;

use super::storage::{IDArray, WrappedData};

/// ID array implementation for primitive (integer) IDs.
pub struct IDPrimArray<T: ArrowPrimitiveType> {
    array: PrimitiveArray<T>,
}

struct PrimWrapper<'a, T: ArrowPrimitiveType> {
    this: &'a PrimitiveArray<T>,
    other: PrimitiveArray<T>,
}

impl<T: ArrowPrimitiveType> IDPrimArray<T> {
    pub fn new(array: PrimitiveArray<T>) -> Self {
        IDPrimArray { array }
    }
}

impl<T> IDArray for IDPrimArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash
        + for<'a> FromPyObject<'a>
        + for<'a> IntoPyObject<'a, Target = PyInt, Output = Bound<'a, PyInt>, Error = Infallible>,
{
    fn hash_entry(&self, idx: u32) -> u64 {
        hash_array_idx(&self.array, idx as usize)
    }

    fn compare_entries(&self, i1: u32, i2: u32) -> bool {
        compare_array_idx(&self.array, i1 as usize, &self.array, i2 as usize)
    }

    fn data(&self) -> ArrayData {
        self.array.to_data()
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    fn wrap_value<'py, 'a>(
        &'a self,
        val: Bound<'py, PyAny>,
    ) -> PyResult<Box<dyn WrappedData + 'a>> {
        let val: T::Native = val.extract()?;
        let mut ab = PrimitiveBuilder::new();
        ab.append_value(val);
        Ok(Box::new(PrimWrapper {
            this: &self.array,
            other: ab.finish(),
        }))
    }

    fn wrap_array<'py, 'a>(
        &'a self,
        arr: arrow::array::ArrayRef,
    ) -> PyResult<Box<dyn WrappedData + 'a>> {
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
        Ok(Box::new(PrimWrapper {
            this: &self.array,
            other: arr.clone(),
        }))
    }
}

impl<'a, T> WrappedData for PrimWrapper<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    fn hash(&self, idx: usize) -> u64 {
        hash_array_idx(&self.other, idx)
    }

    fn compare_with_entry(&self, w_idx: usize, a_idx: u32) -> bool {
        compare_array_idx(self.this, a_idx as usize, &self.other, w_idx)
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
