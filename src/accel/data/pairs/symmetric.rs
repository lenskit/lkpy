// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::{mem::transmute, sync::atomic::Ordering};

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
    nnz: AtomicUsize,
}

impl PairCounter for SymmetricPairCounter {
    fn create(n: usize) -> SymmetricPairCounter {
        let cap = arith_tot(n);
        SymmetricPairCounter {
            n_items: n,
            // SAFETY: i32 and AtomicI32 have the same layout
            data: unsafe { transmute(vec![0i32; cap]) },
            nnz: AtomicUsize::new(0),
        }
    }

    fn record(&mut self, row: i32, col: i32) {
        self.crecord(row, col)
    }

    fn nnz(&self) -> usize {
        self.nnz.load(Ordering::Relaxed)
    }

    fn finish(self) -> COOMatrix<Int32Type, Int32Type> {
        let nnz = self.nnz();
        debug!(
            "creating matrix with {} symmetric co-occurrance counts",
            nnz
        );
        let mut bld = COOMatrixBuilder::with_capacity(nnz * 2);
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
                    bld.add_entry(col, row, val);
                }
                idx += 1;
            }
        }
        bld.finish()
    }
}

impl ConcurrentPairCounter for SymmetricPairCounter {
    fn crecord(&self, row: i32, col: i32) {
        let (row, col) = order_coords(row, col);
        let idx = self.compute_index(row, col);
        let old = self.data[idx].fetch_add(1, Ordering::Relaxed);
        if old == 0 {
            self.nnz.fetch_add(1, Ordering::Relaxed);
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
}

fn order_coords(row: i32, col: i32) -> (i32, i32) {
    if row < 0 {
        panic!("negative row index");
    }
    if col < 0 {
        panic!("negative column index");
    }
    if row <= col {
        (row, col)
    } else {
        (col, row)
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
