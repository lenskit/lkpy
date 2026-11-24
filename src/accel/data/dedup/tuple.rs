// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::HashSet;
use std::hash::Hash;
use std::mem::MaybeUninit;

use arrow::array::{
    Array, AsArray, BooleanArray, BooleanBufferBuilder, BooleanBuilder, Int32Array, RecordBatch,
};
use arrow::datatypes::Int32Type;
use arrow::error::ArrowError;
use arrow::error::Result as ArrowResult;

use super::DedupDirection;
use super::DedupMemory;

/// Struct for efficiently storing de-duplication basted on row key tuples.
pub(super) struct TupleMemory<T, const N: usize>
where
    T: Hash + Eq + RowTuple<N>,
{
    fields: [String; N],
    table: HashSet<T>,
}

/// Trait to extract a row tuple from the position in arrays.
///
/// We parameterize by `N` so implementations can benefit from loop unrolling,
/// bound check elision, and other compile-time optimizations based on known
/// tuple sizes.  This might be premature optimization, but it does show a
/// very modest speedup on ML-25M.
pub(super) trait RowTuple<const N: usize> {
    fn from_row(arrays: &[Int32Array; N], row: usize) -> Self;
}

pub(super) fn tuple_memory(fields: &[String]) -> Option<Box<dyn DedupMemory>> {
    if fields.len() == 1 {
        Some(Box::new(TupleMemory::<i32, 1> {
            fields: [fields[0].clone()],
            table: HashSet::with_capacity(100_000),
        }))
    } else if fields.len() == 2 {
        Some(Box::new(TupleMemory::<(i32, i32), 2> {
            fields: [fields[0].clone(), fields[1].clone()],
            table: HashSet::with_capacity(100_000),
        }))
    } else if fields.len() == 3 {
        Some(Box::new(TupleMemory::<(i32, i32, i32), 3> {
            fields: [fields[0].clone(), fields[1].clone(), fields[2].clone()],
            table: HashSet::with_capacity(100_000),
        }))
    } else if fields.len() == 4 {
        Some(Box::new(TupleMemory::<(i32, i32, i32, i32), 4> {
            fields: [
                fields[0].clone(),
                fields[1].clone(),
                fields[2].clone(),
                fields[3].clone(),
            ],
            table: HashSet::with_capacity(100_000),
        }))
    } else {
        None
    }
}

impl<T, const N: usize> TupleMemory<T, N>
where
    T: Hash + Eq + RowTuple<N>,
{
    /// Extract the tuple arrays specified by this memory's field list from the record batch.
    fn extract_arrays(&self, batch: &RecordBatch) -> Result<[Int32Array; N], ArrowError> {
        let mut arrays = [const { MaybeUninit::uninit() }; N];
        for k in 0..N {
            let field = &self.fields[k];
            let arr = batch.column_by_name(field).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "record batch has no column named {}",
                    field
                ))
            })?;
            let arr = arr.as_primitive_opt::<Int32Type>().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "incorrect type for {}: expected Int32, found {}",
                    field,
                    arr.data_type()
                ))
            })?;
            arrays[k].write(arr.clone());
        }
        // SAFETY: we have just initialized all the arrays
        Ok(arrays.map(|a| unsafe { a.assume_init() }))
    }

    fn forward_unique_mask(
        &mut self,
        batch: &RecordBatch,
        arrays: &[Int32Array; N],
    ) -> ArrowResult<BooleanArray> {
        let mut result = BooleanBuilder::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let key = T::from_row(&arrays, i);
            result.append_value(self.table.insert(key));
        }
        Ok(result.finish())
    }

    fn reverse_unique_mask(
        &mut self,
        batch: &RecordBatch,
        arrays: &[Int32Array; N],
    ) -> ArrowResult<BooleanArray> {
        let mut buffer = BooleanBufferBuilder::new(batch.num_rows());
        buffer.append_n(batch.num_rows(), true);

        // walk through the array backwards
        for i in (0..batch.num_rows()).rev() {
            let key = T::from_row(&arrays, i);
            // clear the bit when the key is not inserted
            if !self.table.insert(key) {
                buffer.set_bit(i, false);
            }
        }

        // now we build our buffer!
        Ok(BooleanArray::new(buffer.build(), None))
    }
}

impl<T, const N: usize> DedupMemory for TupleMemory<T, N>
where
    T: Hash + Eq + RowTuple<N> + Send + Sync,
{
    fn len(&self) -> usize {
        self.table.len()
    }

    fn batch_unique_mask(
        &mut self,
        batch: &RecordBatch,
        direction: DedupDirection,
    ) -> ArrowResult<BooleanArray> {
        assert_eq!(self.fields.len(), N);
        let arrays = self.extract_arrays(batch)?;
        for a in &arrays {
            if a.null_count() > 0 {
                return Err(ArrowError::InvalidArgumentError(
                    "null row/column values are not supported".to_string(),
                ));
            }
        }
        match direction {
            DedupDirection::Forward => self.forward_unique_mask(batch, &arrays),
            DedupDirection::Reverse => self.reverse_unique_mask(batch, &arrays),
        }
    }
}

impl RowTuple<1> for i32 {
    fn from_row(arrays: &[Int32Array; 1], row: usize) -> Self {
        let v1 = arrays[0].value(row);
        v1
    }
}

impl RowTuple<2> for (i32, i32) {
    fn from_row(arrays: &[Int32Array; 2], row: usize) -> Self {
        let v1 = arrays[0].value(row);
        let v2 = arrays[1].value(row);
        (v1, v2)
    }
}

impl RowTuple<3> for (i32, i32, i32) {
    fn from_row(arrays: &[Int32Array; 3], row: usize) -> Self {
        let v1 = arrays[0].value(row);
        let v2 = arrays[1].value(row);
        let v3 = arrays[2].value(row);
        (v1, v2, v3)
    }
}

impl RowTuple<4> for (i32, i32, i32, i32) {
    fn from_row(arrays: &[Int32Array; 4], row: usize) -> Self {
        let v1 = arrays[0].value(row);
        let v2 = arrays[1].value(row);
        let v3 = arrays[2].value(row);
        let v4 = arrays[3].value(row);
        (v1, v2, v3, v4)
    }
}
