// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::Arc;
use std::{mem::transmute, sync::atomic::Ordering};

use arrow::array::Int32Array;
use arrow::datatypes::Int32Type;
use log::*;

use crate::{
    data::pairs::{ConcurrentPairCounter, PairCounter},
    sparse::{COOMatrix, COOMatrixBuilder},
};

/// Accumulate symmetric pair counts.
pub struct SymmetricPairCounter {
    n_items: usize,
    data: Vec<AtomicI32>,
    diag: Option<Vec<AtomicI32>>,
    nnz: AtomicUsize,
}

impl SymmetricPairCounter {
    pub fn with_diagonal(n: usize, diagonal: bool) -> Self {
        let cap = arith_tot(n);
        SymmetricPairCounter {
            n_items: n,
            // SAFETY: i32 and AtomicI32 have the same layout
            data: unsafe { transmute(vec![0i32; cap]) },
            diag: if diagonal {
                unsafe { transmute(vec![0i32; n]) }
            } else {
                None
            },
            nnz: AtomicUsize::new(0),
        }
    }
}

impl PairCounter for SymmetricPairCounter {
    fn create(n: usize) -> SymmetricPairCounter {
        Self::with_diagonal(n, false)
    }

    fn record(&mut self, row: i32, col: i32) {
        self.crecord(row, col)
    }

    fn nnz(&self) -> usize {
        self.nnz.load(Ordering::Relaxed)
    }

    fn finish(mut self) -> Vec<COOMatrix<Int32Type, Int32Type>> {
        let nnz = self.nnz();
        debug!(
            "creating matrix with {} symmetric co-occurrance counts",
            nnz
        );
        let mut bld = COOMatrixBuilder::with_capacity(nnz);
        let mut idx = 0;
        // SAFETY: i32 and AtomicI32 have identical layouts
        let data: Vec<i32> = unsafe { transmute(self.data) };
        for i in 0..self.n_items {
            let row = i as i32;
            for j in i..self.n_items {
                let col = j as i32;
                let val = data[idx];
                if val > 0 {
                    bld.add_entry(row, col, val);
                }
                idx += 1;
            }
        }
        let coo = bld.finish();
        let coo2 = coo.transpose();
        if let Some(diag) = self.diag.take() {
            // SAFETY: i32 and AtomicI32 have the same layout
            let diag: Vec<i32> = unsafe { transmute(diag) };
            let n = self.n_items as i32;
            let ptr = Int32Array::from_iter_values(0..n);
            let ptr = Arc::new(ptr);
            let cood = COOMatrix {
                row: ptr.clone(),
                col: ptr,
                val: Arc::new(diag.into()),
            };
            vec![cood, coo, coo2]
        } else {
            vec![coo, coo2]
        }
    }
}

impl ConcurrentPairCounter for SymmetricPairCounter {
    fn crecord(&self, row: i32, col: i32) {
        if row == col {
            if let Some(diag) = &self.diag {
                diag[row as usize].fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let (row, col) = self.order_coords(row, col);
            let idx = self.compute_index(row, col);
            let old = self.data[idx].fetch_add(1, Ordering::Relaxed);
            if old == 0 {
                self.nnz.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl SymmetricPairCounter {
    fn compute_index(&self, row: i32, col: i32) -> usize {
        debug_assert!(row >= 0);
        debug_assert!(col >= row);
        let row = row as usize;
        let col = col as usize;

        // total array size
        let total = self.data.len();
        // array capacity used by current & later rows
        let remaining = arith_tot(self.n_items - row);
        // difference is the array capacity used by earlier rows
        let base = total - remaining;
        base + col - row
    }

    /// Validate and order coordinates. Guarantees the smaller coordinate is
    /// first, and that the coordinates are in-bounds.
    fn order_coords(&self, row: i32, col: i32) -> (i32, i32) {
        assert!(row >= 0, "negative row index {}", row);
        assert!(col >= 0, "negative column index {}", col);
        assert!(
            (row as usize) < self.n_items,
            "row index {} exceeds size {}",
            row,
            self.n_items
        );
        assert!(
            (col as usize) < self.n_items,
            "column index {} exceeds size {}",
            col,
            self.n_items
        );

        if row <= col {
            (row, col)
        } else {
            (col, row)
        }
    }
}

/// Compute the total of an arithmetic series.
fn arith_tot(n: usize) -> usize {
    let num = n * (n + 1);
    num >> 1
}

#[test]
fn test_arith_zero() {
    assert_eq!(arith_tot(0), 0);
}

#[test]
fn test_arith_one() {
    assert_eq!(arith_tot(1), 1);
}

#[test]
fn test_arith_two() {
    assert_eq!(arith_tot(2), 3);
}

#[test]
fn test_arith_big() {
    let n = 250;
    let tot = (1..=n).sum::<usize>();
    assert_eq!(arith_tot(n), tot);
}

#[test]
fn test_first_row() {
    let acc = SymmetricPairCounter::create(1000);
    let i = acc.compute_index(0, 0);
    assert_eq!(i, 0);
    let i = acc.compute_index(0, 999);
    assert_eq!(i, 999);
}

#[test]
fn test_second_row() {
    let acc = SymmetricPairCounter::create(1000);
    let i = acc.compute_index(1, 1);
    assert_eq!(i, 1000);
    let i = acc.compute_index(1, 999);
    assert_eq!(i, 1998);
}

#[test]
fn test_last_element() {
    let acc = SymmetricPairCounter::create(1000);
    let i = acc.compute_index(999, 999);
    assert_eq!(i, acc.data.len() - 1);
}

#[test]
fn test_diagonal() {
    let acc = SymmetricPairCounter::create(1000);
    let n = 1000;
    let mut pos = 0;
    for row in 0..n {
        let i = acc.compute_index(row as i32, row as i32);
        assert_eq!(i, pos);
        pos += n - row;
    }
}

#[test]
fn test_last_col() {
    let acc = SymmetricPairCounter::create(1000);
    let n = 1000;
    let mut pos = 0;
    for row in 0..n {
        let i = acc.compute_index(row as i32, 999);
        assert_eq!(i, pos + 999 - row);
        pos += n - row;
    }
}
