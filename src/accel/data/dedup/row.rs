// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::HashSet;
use std::hash::Hash;

use arrow::array::{BooleanArray, BooleanBuilder};
use arrow::error::Result as ArrowResult;
use arrow::row::{OwnedRow, Row};
use arrow::{
    datatypes::Schema,
    row::{RowConverter, SortField},
};
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

use super::DedupDirection;
use super::DedupMemory;

/// [DedupMemory] implementation that uses row strings to de-duplicate rows.
///
/// This implementation effectively keeps a second copy of the entire input data
/// in memory, but does not suffer from (unlikely) hash collisions.
pub(super) struct RowMemory<K: RowKey> {
    converter: RowConverter,
    table: HashSet<K>,
}

pub(super) trait RowKey: Hash + Eq + Send + Sync {
    fn from_row(row: &Row<'_>) -> Self;
}

/// Create a [DedupMemory] using full row strings.
pub(super) fn row_memory(schema: &Schema) -> ArrowResult<Box<dyn DedupMemory>> {
    Ok(Box::new(RowMemory::<OwnedRow>::new(schema)?))
}

/// Create a [DedupMemory] using 128-bit row hashes.
pub(super) fn row_hash_memory(schema: &Schema) -> ArrowResult<Box<dyn DedupMemory>> {
    Ok(Box::new(RowMemory::<u128>::new(schema)?))
}

/// Create a [DedupMemory] using 64-bit row hashes.
pub(super) fn row_short_hash_memory(schema: &Schema) -> ArrowResult<Box<dyn DedupMemory>> {
    Ok(Box::new(RowMemory::<u64>::new(schema)?))
}

impl<K: RowKey> RowMemory<K> {
    fn new(schema: &Schema) -> ArrowResult<Self> {
        let mut fields = vec![];
        for field in schema.fields() {
            fields.push(SortField::new(field.data_type().clone()))
        }
        Ok(RowMemory {
            converter: RowConverter::new(fields)?,
            table: HashSet::with_capacity(100_000),
        })
    }
}

impl<K: RowKey> DedupMemory for RowMemory<K> {
    fn len(&self) -> usize {
        self.table.len()
    }

    fn batch_unique_mask(
        &mut self,
        batch: &arrow::array::RecordBatch,
        _dir: DedupDirection,
    ) -> ArrowResult<BooleanArray> {
        let cols = batch.columns();
        let mut result = BooleanBuilder::with_capacity(batch.num_rows());
        for row in self.converter.convert_columns(cols)?.into_iter() {
            let key = K::from_row(&row);
            result.append_value(self.table.insert(key))
        }
        Ok(result.finish())
    }
}

impl RowKey for OwnedRow {
    fn from_row(row: &Row<'_>) -> Self {
        row.owned()
    }
}

impl RowKey for u128 {
    fn from_row(row: &Row<'_>) -> Self {
        xxh3_128(row.data())
    }
}

impl RowKey for u64 {
    fn from_row(row: &Row<'_>) -> Self {
        xxh3_64(row.data())
    }
}
