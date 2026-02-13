// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::datatypes::Int32Type;

use crate::sparse::COOMatrix;

mod asymmetric;
mod symmetric;

pub(super) use asymmetric::AsymmetricPairCounter;
pub(super) use symmetric::SymmetricPairCounter;

/// Trait for accmulating counts of item pairs.
pub(super) trait PairCounter {
    fn create(n_items: usize) -> Self;

    /// Record an instance of an item pair.
    fn record(&mut self, row: i32, col: i32);

    /// Count the number of nonzero co-occurrance counts.
    fn nnz(&self) -> usize;

    /// Finish the counting into a matrix.
    fn finish(self) -> COOMatrix<Int32Type, Int32Type>;
}
