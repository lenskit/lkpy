// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse coordinate arrays.

use arrow::{
    array::{ArrowPrimitiveType, OffsetSizeTrait, PrimitiveArray, PrimitiveBuilder},
    datatypes::Int32Type,
};

pub struct COOMatrix<V, Ix = Int32Type>
where
    V: ArrowPrimitiveType,
    Ix: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    pub row: PrimitiveArray<Ix>,
    pub col: PrimitiveArray<Ix>,
    pub val: PrimitiveArray<V>,
}

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
            row: self.row.finish(),
            col: self.col.finish(),
            val: self.val.finish(),
        }
    }
}
