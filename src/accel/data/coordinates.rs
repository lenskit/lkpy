// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Coordinates in a sparse matrix or tensor.

use std::hash::{Hash, Hasher};

use arrow::{
    array::{AsArray, Int32Array, RecordBatch},
    pyarrow::PyArrowType,
};
use hashbrown::{hash_table::Entry, HashTable};
use log::*;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use rustc_hash::FxHasher;

/// Two-element indices for chunked lookup.
#[derive(Debug, Clone, Copy)]
pub struct ChunkIndex {
    /// The chunk number.
    chunk: u32,
    /// The index within a chunk.
    item: u32,
}

/// A table of sparse matrix or tensor coordinates.
#[pyclass]
#[derive(Clone)]
pub struct CoordinateTable {
    chunks: Vec<Vec<Int32Array>>,
    offsets: Vec<usize>,
    n_dims: u32,
    n_rows: usize,
    n_unique: usize,
    index: HashTable<ChunkIndex>,
}

#[pymethods]
impl CoordinateTable {
    #[new]
    /// Construct a new, empty coordinate table.
    fn new(dims: u32) -> PyResult<CoordinateTable> {
        if dims < 1 {
            Err(PyValueError::new_err("must have at least 1 dimension"))
        } else {
            Ok(CoordinateTable {
                chunks: Vec::new(),
                offsets: vec![0],
                n_dims: dims,
                n_rows: 0,
                n_unique: 0,
                index: HashTable::new(),
            })
        }
    }

    /// Copy a table (reusing original storage).
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Extend a table with more records.
    fn extend<'py>(&mut self, coords: Bound<'py, PyAny>) -> PyResult<(usize, usize)> {
        if let Ok(PyArrowType(rb)) = coords.extract::<PyArrowType<RecordBatch>>() {
            debug!("extending coordinate with batch of {}", rb.num_rows());
            self.add_record_batch(&rb)
        } else if let Ok(batches) = coords.extract::<Vec<PyArrowType<RecordBatch>>>() {
            debug!("adding {} record batches", batches.len());
            let mut tot = 0;
            let mut tot_uq = 0;
            for PyArrowType(batch) in batches {
                let (n, nuq) = self.add_record_batch(&batch)?;
                tot += n;
                tot_uq += nuq;
            }
            Ok((tot, tot_uq))
        } else {
            Err(PyTypeError::new_err(format!(
                "invalid coordinate type {}",
                coords.get_type()
            )))
        }
    }

    /// Get the number of dimensions in which we store coordinates.
    fn dimensions(&self) -> u32 {
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
    #[pyo3(signature = (*coords))]
    fn contains(&self, coords: Vec<i32>) -> bool {
        self.find(coords).is_some()
    }

    /// Query if the index contains the specified coordinates.
    fn contains_pair(&self, r: i32, c: i32) -> bool {
        let coords = [r, c];
        self.lookup(&coords)
            .map(|cx| self.global_index(&cx))
            .is_some()
    }

    /// Query if the index contains the specified coordinates.
    #[pyo3(signature = (*coords))]
    fn find(&self, coords: Vec<i32>) -> Option<usize> {
        self.lookup(&coords).map(|cx| self.global_index(&cx))
    }
}

impl CoordinateTable {
    pub fn len(&self) -> usize {
        self.n_rows
    }

    pub fn global_index(&self, cx: &ChunkIndex) -> usize {
        let base = self.offsets[cx.chunk_index()];
        base + cx.item_index()
    }

    fn add_record_batch(&mut self, records: &RecordBatch) -> PyResult<(usize, usize)> {
        if records.num_columns() as u32 != self.dimensions() {
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
        let chunk = self.chunks.len() as u32;
        self.offsets
            .push(self.offsets[chunk as usize] + arrays[0].len());
        self.chunks.push(arrays);
        let mut added = 0;
        let mut uq_added = 0;
        for i in 0..n {
            let i = i as u32;
            let cx = ChunkIndex::new(chunk, i);
            let hash = hash_entry(&self.chunks, &cx);
            let e = self.index.entry(
                hash,
                |j| compare_entries(&self.chunks, &cx, j),
                |j| hash_entry(&self.chunks, j),
            );
            added += 1;
            self.n_rows += 1;
            if let Entry::Vacant(_) = &e {
                uq_added += 1;
                self.n_unique += 1;
            }
            e.insert(cx);
        }

        Ok((added, uq_added))
    }

    pub fn lookup(&self, coords: &[i32]) -> Option<ChunkIndex> {
        let hash = hash_coords(coords);
        self.index
            .find(hash, |cx| compare_coords(&self.chunks, cx, coords))
            .copied()
    }

    pub fn get(&self, dim: usize, entry: usize) -> i32 {
        let chunk = match self.offsets.binary_search(&entry) {
            // Exact match: offset points to beginning, so will be first element of chunk
            Ok(c) => c,
            // Non-exact match: index of the _next_ item, because we would
            // insert item after the offset.
            Err(c) => c - 1,
        };
        let row = entry - self.offsets[chunk];
        self.chunks[chunk][dim].value(row)
    }
}

impl ChunkIndex {
    #[inline]
    fn new(chunk: u32, item: u32) -> Self {
        ChunkIndex { chunk, item }
    }

    #[inline]
    fn chunk_index(&self) -> usize {
        self.chunk as usize
    }

    #[inline]
    fn item_index(&self) -> usize {
        self.item as usize
    }
}

fn hash_entry(chunks: &[Vec<Int32Array>], ix: &ChunkIndex) -> u64 {
    let chunk = &chunks[ix.chunk_index()];
    hash_chunk_entry(&chunk, ix.item)
}

fn hash_chunk_entry(chunk: &[Int32Array], ri: u32) -> u64 {
    let mut hash = FxHasher::default();
    for arr in chunk {
        arr.value(ri as usize).hash(&mut hash);
    }
    hash.finish()
}

fn hash_coords(coords: &[i32]) -> u64 {
    let mut hash = FxHasher::default();
    for c in coords {
        c.hash(&mut hash);
    }
    hash.finish()
}

fn compare_entries(chunks: &[Vec<Int32Array>], i1: &ChunkIndex, i2: &ChunkIndex) -> bool {
    let chunk1 = &chunks[i1.chunk_index()];
    let chunk2 = &chunks[i2.chunk_index()];

    compare_chunk_entries(chunk1, i1.item, chunk2, i2.item)
}

fn compare_chunk_entries(chunk1: &[Int32Array], i1: u32, chunk2: &[Int32Array], i2: u32) -> bool {
    if chunk1.len() != chunk2.len() {
        return false;
    }

    for (a1, a2) in chunk1.iter().zip(chunk2.iter()) {
        if a1.value(i1 as usize) != a2.value(i2 as usize) {
            return false;
        }
    }

    true
}

fn compare_coords(chunks: &[Vec<Int32Array>], idx: &ChunkIndex, coords: &[i32]) -> bool {
    let chunk = &chunks[idx.chunk_index()];
    if chunk.len() != coords.len() {
        return false;
    }

    for (c1, c2) in chunk.iter().zip(coords.iter()) {
        if c1.value(idx.item_index()) != *c2 {
            return false;
        }
    }

    true
}
