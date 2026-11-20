// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::HashSet;

use arrow::array::{BooleanArray, BooleanBuilder, Int32Array, downcast_array};
use arrow::error::Result as ArrowResult;

use crate::data::dedup::DedupMemory;

pub(super) struct PairMemory {
    row_col: String,
    col_col: String,
    table: HashSet<(i32, i32)>,
}

impl PairMemory {
    pub fn new(fields: Vec<String>) -> Self {
        PairMemory {
            // FIXME don't clone
            row_col: fields[0].clone(),
            col_col: fields[1].clone(),
            table: HashSet::with_capacity(100_000),
        }
    }
}

impl DedupMemory for PairMemory {
    fn len(&self) -> usize {
        self.table.len()
    }

    fn batch_unique_mask(
        &mut self,
        batch: &arrow::array::RecordBatch,
    ) -> ArrowResult<BooleanArray> {
        let mut result = BooleanBuilder::with_capacity(batch.num_rows());

        // FIXME: proper error handling
        let ra = downcast_array::<Int32Array>(batch.column_by_name(&self.row_col).unwrap());
        let ca = downcast_array::<Int32Array>(batch.column_by_name(&self.col_col).unwrap());
        for i in 0..batch.num_rows() {
            let r = ra.value(i);
            let c = ca.value(i);
            result.append_value(self.table.insert((r, c)));
        }
        Ok(result.finish())
    }
}
