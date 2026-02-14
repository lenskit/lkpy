// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse coordinate arrays.

use std::sync::Arc;

use arrow::{
    array::{ArrowPrimitiveType, OffsetSizeTrait, PrimitiveArray, PrimitiveBuilder, RecordBatch},
    datatypes::Int32Type,
};
use arrow_schema::{ArrowError, DataType, Field, SchemaBuilder};

/// Representation of coordinate sparse matrices.
pub struct COOMatrix<V, Ix = Int32Type>
where
    V: ArrowPrimitiveType,
    Ix: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    pub row: Arc<PrimitiveArray<Ix>>,
    pub col: Arc<PrimitiveArray<Ix>>,
    pub val: Arc<PrimitiveArray<V>>,
}

/// Builder for coordinate sparse matrices.
pub struct COOMatrixBuilder<V, Ix = Int32Type>
where
    V: ArrowPrimitiveType,
    Ix: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    pub row: PrimitiveBuilder<Ix>,
    pub col: PrimitiveBuilder<Ix>,
    pub val: PrimitiveBuilder<V>,
}

impl<V, Ix> COOMatrix<V, Ix>
where
    V: ArrowPrimitiveType,
    Ix: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    /// Transpose this matrix (swap rows and columns).
    pub fn transpose(&self) -> COOMatrix<V, Ix> {
        COOMatrix {
            row: self.col.clone(),
            col: self.row.clone(),
            val: self.val.clone(),
        }
    }

    /// Create a record batch from this matrix's contents.
    pub fn record_batch(&self, value_name: &str) -> Result<RecordBatch, ArrowError> {
        let mut schema = SchemaBuilder::new();
        schema.push(Field::new("row", DataType::Int32, false));
        schema.push(Field::new("col", DataType::Int32, false));
        schema.push(Field::new(value_name, DataType::Int32, false));
        let schema = schema.finish();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![self.row.clone(), self.col.clone(), self.val.clone()],
        )
    }
}

impl<V, Ix> COOMatrixBuilder<V, Ix>
where
    V: ArrowPrimitiveType,
    Ix: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    /// Initialize a builder with a specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        COOMatrixBuilder {
            row: PrimitiveBuilder::with_capacity(cap),
            col: PrimitiveBuilder::with_capacity(cap),
            val: PrimitiveBuilder::with_capacity(cap),
        }
    }

    pub fn add_entry(&mut self, row: Ix::Native, col: Ix::Native, val: V::Native) {
        self.row.append_value(row);
        self.col.append_value(col);
        self.val.append_value(val);
    }

    /// Build the final COO matrix from this builder.
    pub fn finish(mut self) -> COOMatrix<V, Ix> {
        COOMatrix {
            row: self.row.finish().into(),
            col: self.col.finish().into(),
            val: self.val.finish().into(),
        }
    }
}
