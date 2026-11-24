// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::borrow::Cow;
use std::collections::HashSet;
use std::hash::Hash;

use arrow::array::{ArrayRef, BooleanArray, BooleanBuilder};
use arrow::error::Result as ArrowResult;
use arrow::row::{OwnedRow, Row};
use arrow::{
    datatypes::Schema,
    row::{RowConverter, SortField},
};
use arrow_schema::ArrowError;
use pyo3::PyResult;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

use super::DedupDirection;
use super::DedupMemory;

/// [DedupMemory] implementation that uses row strings to de-duplicate rows.
///
/// This implementation effectively keeps a second copy of the entire input data
/// in memory, but does not suffer from (unlikely) hash collisions.
pub(super) struct RowMemory<K: RowKey> {
    converter: RowConverter,
    columns: Option<Vec<String>>,
    table: HashSet<K>,
}

pub(super) trait RowKey: Hash + Eq + Send + Sync {
    fn from_row(row: &Row<'_>) -> Self;
}

/// Builder to create row memories.
pub(super) struct RowMemoryBuilder<'s> {
    schema: &'s Schema,
    columns: Option<Vec<String>>,
}

impl<'s> RowMemoryBuilder<'s> {
    pub fn new(schema: &'s Schema) -> Self {
        Self {
            schema,
            columns: None,
        }
    }

    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Build a [`DedupMemory`] with the named method.
    pub fn build_named(self, name: &str) -> PyResult<Box<dyn DedupMemory>> {
        match name {
            "full" => Ok(Box::new(self.build_exact().map_err(|e| {
                PyRuntimeError::new_err(format!("failed to create memory: {}", e))
            })?)),
            "hash" => Ok(Box::new(self.build_hash().map_err(|e| {
                PyRuntimeError::new_err(format!("failed to create memory: {}", e))
            })?)),
            "short-hash" => Ok(Box::new(self.build_short_hash().map_err(|e| {
                PyRuntimeError::new_err(format!("failed to create memory: {}", e))
            })?)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown dedup comparison method: {}",
                name
            ))),
        }
    }

    pub fn build_exact(self) -> Result<RowMemory<OwnedRow>, ArrowError> {
        RowMemory::new(self.schema, self.columns)
    }

    pub fn build_hash(self) -> Result<RowMemory<u128>, ArrowError> {
        RowMemory::new(self.schema, self.columns)
    }

    pub fn build_short_hash(self) -> Result<RowMemory<u64>, ArrowError> {
        RowMemory::new(self.schema, self.columns)
    }
}

impl<K: RowKey> RowMemory<K> {
    fn new(schema: &Schema, columns: Option<Vec<String>>) -> ArrowResult<Self> {
        if let Some(cols) = columns {
            Self::with_columns(schema, cols)
        } else {
            Self::without_columns(schema)
        }
    }

    fn without_columns(schema: &Schema) -> ArrowResult<Self> {
        let mut fields = vec![];
        for field in schema.fields() {
            fields.push(SortField::new(field.data_type().clone()))
        }
        Ok(RowMemory {
            converter: RowConverter::new(fields)?,
            columns: None,
            table: HashSet::with_capacity(100_000),
        })
    }

    fn with_columns(schema: &Schema, columns: Vec<String>) -> ArrowResult<Self> {
        let mut fields = Vec::with_capacity(columns.len());
        let mut col_owned = Vec::with_capacity(columns.len());
        for name in columns {
            let name = name.as_ref();
            let field = schema.field_with_name(name)?;
            fields.push(SortField::new(field.data_type().clone()));
            col_owned.push(name.to_string());
        }
        Ok(RowMemory {
            converter: RowConverter::new(fields)?,
            columns: Some(col_owned),
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
        let cols: Cow<'_, [ArrayRef]> = if let Some(wanted) = &self.columns {
            let mut cols = Vec::with_capacity(wanted.len());
            for cn in wanted {
                cols.push(
                    batch
                        .column_by_name(cn)
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError(format!("missing column: {}", cn))
                        })?
                        .clone(),
                );
            }
            cols.into()
        } else {
            batch.columns().into()
        };
        let mut result = BooleanBuilder::with_capacity(batch.num_rows());
        for row in self.converter.convert_columns(&cols)?.into_iter() {
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
