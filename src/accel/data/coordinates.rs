// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Coordinates in a sparse matrix or tensor.

use std::hash::{Hash, Hasher};

use arrow::array::{array, AsArray, Int32Array, RecordBatch};
use hashbrown::{hash_table::Entry, HashTable};
use log::*;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use rustc_hash::FxHasher;

use crate::indirect_hashing::{IndirectHashContent, IndirectSearcher};

/// A table of sparse matrix or tensor coordinates.
#[pyclass]
#[derive(Clone)]
pub struct CoordinateTable {
    chunks: Vec<Vec<Int32Array>>,
    n_dims: usize,
    n_rows: usize,
    n_unique: usize,
    index: HashTable<(u32, u32)>,
}

#[pymethods]
impl CoordinateTable {
    #[new]
    /// Construct a new, empty coordinate table.
    fn new(dims: usize) -> Self {
        CoordinateTable {
            chunks: Vec::new(),
            n_dims: dims,
            n_rows: 0,
            n_unique: 0,
            index: HashTable::new(),
        }
    }

    /// Copy a table (reusing original storage).
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Extend a table with more records.
    fn extend<'py>(
        &mut self,
        py: Python<'py>,
        coords: Bound<'py, PyAny>,
    ) -> PyResult<(usize, usize)> {
        todo!()
    }

    /// Get the number of dimensions in which we store coordinates.
    fn dimensions(&self) -> usize {
        self.n_dims
    }

    /// Get the number of unique coordinates in the table.
    fn unique_count(&self) -> usize {
        self.n_unique
    }

    /// Get the length of the coordinate table.
    fn __len__(&self) -> usize {
        self.len()
    }

    /// Query if the index contains the specified coordinates.
    fn contains(&self, coords: Vec<i32>) -> bool {
        todo!()
    }
}

impl CoordinateTable {
    fn len(&self) -> usize {
        self.n_rows
    }

    fn add_record_batch(&mut self, records: &RecordBatch) -> PyResult<(usize, usize)> {
        if records.num_columns() != self.dimensions() {
            return Err(PyValueError::new_err(format!(
                "invalid batch size of {} columns (expected {})",
                records.num_columns(),
                self.dimensions()
            )));
        }

        let mut cols: Vec<Int32Array> = Vec::with_capacity(records.num_columns());
        for col in records.columns() {
            if let Some(arr) = col.as_primitive_opt() {
                cols.push(arr.clone())
            } else {
                return Err(PyTypeError::new_err(format!(
                    "invalid column type {}",
                    col.data_type()
                )));
            }
        }

        self.add_arrays(records.num_rows(), cols)
    }

    /// Add coordinates from a slice of arrays.
    fn add_arrays(&mut self, n: usize, arrays: Vec<Int32Array>) -> PyResult<(usize, usize)> {
        let ci = self.chunks.len() as u32;
        self.chunks.push(arrays);
        let mut added = 0;
        let mut uq_added = 0;
        for i in 0..n {
            let i = i as u32;
            let hash = hash_entry(&self.chunks, ci, i);
            let e = self.index.entry(
                hash,
                |(c2, r2)| compare_entries(&self.chunks, (ci, i), (*c2, *r2)),
                |(c2, r2)| hash_entry(&self.chunks, *c2, *r2),
            );
            added += 1;
            self.n_rows += 1;
            if let Entry::Vacant(_) = &e {
                uq_added += 1;
                self.n_unique += 1;
            }
            e.insert((ci, i));
        }

        Ok((added, uq_added))
    }
}

fn hash_entry(chunks: &[Vec<Int32Array>], ci: u32, ri: u32) -> u64 {
    let chunk = &chunks[ci as usize];
    hash_chunk_entry(&chunk, ri)
}

fn hash_chunk_entry(chunk: &[Int32Array], ri: u32) -> u64 {
    let mut hash = FxHasher::default();
    for arr in chunk {
        arr.value(ri as usize).hash(&mut hash);
    }
    hash.finish()
}

fn compare_entries(chunks: &[Vec<Int32Array>], i1: (u32, u32), i2: (u32, u32)) -> bool {
    let (c1, r1) = i1;
    let (c2, r2) = i2;
    let chunk1 = &chunks[c1 as usize];
    let chunk2 = &chunks[c2 as usize];

    compare_chunk_entries(chunk1, r1, chunk2, r2)
}

fn compare_chunk_entries(chunk1: &[Int32Array], i1: u32, chunk2: &[Int32Array], i2: u32) -> bool {
    for (a1, a2) in chunk1.iter().zip(chunk2.iter()) {
        if a1.value(i1 as usize) != a2.value(i2 as usize) {
            return false;
        }
    }

    true
}
