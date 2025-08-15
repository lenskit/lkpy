// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::{exceptions::PyValueError, prelude::*};

use thiserror::Error;

pub mod content_int;
pub mod content_string;
mod table;

pub use table::{IndirectHashTable, PositionLookup};

/// Errors that can be raised by the hashing abstraction.
#[derive(Error, Debug)]
pub enum HashError {
    #[error("duplicate entry found at index {0}")]
    Duplicate(u32),
}

/// Trait for storage for indirect hashtable details.
pub trait IndirectHashContent {
    type Searcher<'a>: IndirectSearcher<'a>
    where
        Self: 'a;

    /// Get the size of the hash content.
    fn len(&self) -> usize;

    /// Get the hash for a position in the indirect storage.
    fn hash_entry(&self, idx: u32) -> u64;

    /// Compare two positions in the indirect storage.
    fn compare_entries(&self, i1: u32, i2: u32) -> bool;

    /// Create a searcher for an input value.
    fn create_searcher<'py, 'a>(
        &'a self,
        py: Python<'py>,
        val: Bound<'py, PyAny>,
    ) -> PyResult<Self::Searcher<'a>>;
}

/// Trait for values to search for in an indirect hashtable.
pub trait IndirectSearcher<'a> {
    /// Get the number of elements in this searcher.
    fn len(&self) -> usize;
    /// Get the hash for the entry at a particular index.
    fn hash(&self, idx: usize) -> u64;
    /// Compare an entry in the searcher with an entry in the table.
    fn compare_with_entry(&self, search_idx: usize, tbl_idx: u32) -> bool;
}

impl From<HashError> for PyErr {
    fn from(value: HashError) -> Self {
        PyValueError::new_err(format!("{}", value))
    }
}
