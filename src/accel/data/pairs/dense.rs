// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::{mem::transmute, sync::atomic::Ordering};

use log::*;
use ndarray::{Array1, Array2};

use crate::data::pairs::{ConcurrentPairCounter, PairCounter};

/// Accumulate symmetric pair counts.
pub struct DensePairCounter {
    n_items: usize,
    diagonal: bool,
    symmetric: bool,
    data: Vec<AtomicU32>,
    nnz: AtomicUsize,
}

impl DensePairCounter {
    pub fn with_diagonal(n: usize, diagonal: bool) -> Self {
        DensePairCounter {
            n_items: n,
            diagonal,
            symmetric: true,
            // SAFETY: u32 and AtomicU32 have the same layout
            data: unsafe { transmute(vec![0u32; n * n]) },
            nnz: AtomicUsize::new(0),
        }
    }
}

impl PairCounter for DensePairCounter {
    type Output = Array2<f32>;

    fn create(n: usize) -> DensePairCounter {
        Self::with_diagonal(n, false)
    }

    fn record(&mut self, row: i32, col: i32) {
        self.crecord(row, col)
    }

    fn nnz(&self) -> usize {
        self.nnz.load(Ordering::Relaxed)
    }

    fn finish(self) -> Array2<f32> {
        let nnz = self.nnz();
        debug!(
            "finalizing dense matrix with {} symmetric co-occurrance counts",
            nnz
        );
        // SAFETY: u32 and AtomicU32 have the same layout
        let mut data: Vec<u32> = unsafe { transmute(self.data) };

        // translate everything to float, and store u32 reprs
        for b in data.as_mut_slice() {
            *b = (*b as f32).to_bits()
        }

        // SAFETY: u32 and f32 have same layout, and we've set up the bits
        let data: Vec<f32> = unsafe { transmute(data) };

        let arr = Array1::from_vec(data);
        let mat = arr
            .into_shape_with_order((self.n_items, self.n_items))
            .expect("array reshape failed");
        mat
    }
}

impl ConcurrentPairCounter for DensePairCounter {
    fn crecord(&self, row: i32, col: i32) {
        if row == col {
            if self.diagonal {
                self.record_entry(row, col);
            }
        } else {
            self.record_entry(row, col);
            if self.symmetric {
                self.record_entry(col, row);
            }
        }
    }
}

impl DensePairCounter {
    fn record_entry(&self, row: i32, col: i32) {
        let idx = self.data_index(row, col);
        let old = self.data[idx].fetch_add(1, Ordering::Relaxed);
        if old == 0 {
            self.nnz.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn data_index(&self, row: i32, col: i32) -> usize {
        let row = row as usize;
        let col = col as usize;
        row * self.n_items + col
    }
}
