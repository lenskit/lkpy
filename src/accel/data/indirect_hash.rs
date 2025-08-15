// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::array::{Int32Array, Int32Builder};
use hashbrown::{hash_table::Entry, HashTable};
use pyo3::{exceptions::PyValueError, prelude::*};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HashError {
    #[error("duplicate entry found at index {0}")]
    Duplicate(u32),
}

/// Hash table using indirect storage.
pub struct IndirectHashTable<C: IndirectHashContent> {
    table: HashTable<u32>,
    content: C,
    n_unique: usize,
}

/// Trait for looking up indirect content.
pub trait PositionLookup {
    /// Look up the position of a single value.
    fn lookup_value<'py>(&self, py: Python<'py>, val: Bound<'py, PyAny>) -> PyResult<Option<u32>>;
    /// Look up the position of an array of values.
    fn lookup_array<'py>(&self, py: Python<'py>, val: Bound<'py, PyAny>) -> PyResult<Int32Array>;
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

impl<C> Default for IndirectHashTable<C>
where
    C: IndirectHashContent + Default,
{
    fn default() -> Self {
        IndirectHashTable {
            table: HashTable::new(),
            content: C::default(),
            n_unique: 0,
        }
    }
}

impl<C: IndirectHashContent> IndirectHashTable<C> {
    /// Create a table from unique content, failing if it is non-unique.
    pub fn from_unique(content: C) -> Result<Self, HashError> {
        Self::create(content, true)
    }

    /// Create a table from content that may have repeats.
    #[allow(dead_code)]
    pub fn from_content(content: C) -> Self {
        // since duplicates are only error, this is infallible
        Self::create(content, false).expect("unexpected construction error")
    }

    /// Get the number of unique elements in this hash.
    #[allow(dead_code)]
    pub fn n_unique(&self) -> usize {
        self.n_unique
    }

    fn create(content: C, unique: bool) -> Result<Self, HashError> {
        let n = content.len();
        let mut table = HashTable::with_capacity(n);
        let mut n_unique = 0;
        for i in 0..n {
            let i = i as u32;
            let hash = content.hash_entry(i);
            let e = table.entry(
                hash,
                |jr| content.compare_entries(i, *jr),
                |jr| content.hash_entry(*jr),
            );
            if let Entry::Occupied(_) = &e {
                if unique {
                    return Err(HashError::Duplicate(i));
                }
            } else {
                n_unique += 1;
            }
            e.insert(i);
        }

        Ok(IndirectHashTable {
            table,
            content,
            n_unique,
        })
    }
}

impl<C: IndirectHashContent> PositionLookup for IndirectHashTable<C> {
    fn lookup_value<'py>(&self, py: Python<'py>, val: Bound<'py, PyAny>) -> PyResult<Option<u32>> {
        let search = self.content.create_searcher(py, val)?;
        if search.len() != 1 {
            return Err(PyValueError::new_err(
                "lookup requires a single input value",
            ));
        }
        let hash = search.hash(0);
        let res = self
            .table
            .find(hash, |jr| search.compare_with_entry(0, *jr));
        Ok(res.map(|ir| *ir))
    }

    fn lookup_array<'py>(&self, py: Python<'py>, val: Bound<'py, PyAny>) -> PyResult<Int32Array> {
        let search = self.content.create_searcher(py, val)?;
        let n = search.len();
        let mut rb = Int32Builder::with_capacity(n);
        // TODO: parallelize index lookup
        for i in 0..n {
            let hash = search.hash(i);
            if let Some(ir) = self
                .table
                .find(hash, |jr| search.compare_with_entry(i, *jr))
            {
                rb.append_value(*ir as i32);
            } else {
                rb.append_null();
            }
        }
        Ok(rb.finish())
    }
}

impl From<HashError> for PyErr {
    fn from(value: HashError) -> Self {
        PyValueError::new_err(format!("{}", value))
    }
}
