// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Traits for supporting dynamic ID storage and value comparison.
use arrow::array::{ArrayData, ArrayRef};
use pyo3::prelude::*;

/// Internal object for dispatch to typed ID storage.
pub trait IDArray: Sync + Send {
    fn wrap_value<'py, 'a>(&'a self, val: Bound<'py, PyAny>)
        -> PyResult<Box<dyn WrappedData + 'a>>;
    fn wrap_array<'py, 'a>(&'a self, arr: ArrayRef) -> PyResult<Box<dyn WrappedData + 'a>>;

    fn hash_entry(&self, idx: u32) -> u64;
    fn compare_entries(&self, i1: u32, i2: u32) -> bool;

    fn data(&self) -> ArrayData;
    fn len(&self) -> usize {
        self.data().len()
    }
}

/// Wrap values and arrays for hashing and comparison with entries.
pub trait WrappedData {
    fn hash(&self, idx: usize) -> u64;
    fn compare_with_entry(&self, w_idx: usize, a_idx: u32) -> bool;
}
