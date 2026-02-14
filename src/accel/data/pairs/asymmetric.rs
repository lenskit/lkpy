// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use arrow::datatypes::Int32Type;
use log::*;
use rustc_hash::FxBuildHasher;

use crate::{
    data::pairs::PairCounter,
    sparse::{COOMatrix, COOMatrixBuilder},
};

/// Accumulate asymmetric pair counts.
pub struct AsymmetricPairCounter {
    rows: Vec<HashMap<i32, i32, FxBuildHasher>>,
    total: usize,
}

impl PairCounter for AsymmetricPairCounter {
    fn create(n: usize) -> AsymmetricPairCounter {
        AsymmetricPairCounter {
            rows: vec![HashMap::with_hasher(FxBuildHasher); n],
            total: 0,
        }
    }

    fn record(&mut self, row: i32, col: i32) {
        if row < 0 {
            panic!("negative row index")
        }
        if col < 0 {
            panic!("negative column index")
        }

        let row = &mut self.rows[row as usize];
        *row.entry(col).or_default() += 1;
        self.total += 1;
    }

    fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    fn finish(self) -> Vec<COOMatrix<Int32Type, Int32Type>> {
        // compute the # of rows
        let size = self.nnz();
        debug!(
            "creating matrix with {} asymmetric co-occurrance counts",
            size
        );
        let mut bld = COOMatrixBuilder::with_capacity(size);
        for (i, row) in self.rows.into_iter().enumerate() {
            let ri = i as i32;
            for (ci, count) in row.into_iter() {
                bld.add_entry(ri, ci, count);
            }
        }
        vec![bld.finish()]
    }
}
