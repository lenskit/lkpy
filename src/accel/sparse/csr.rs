// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::fmt::Debug;
use std::sync::Arc;

use log::*;

use arrow::{
    array::{
        downcast_array, Array, ArrowPrimitiveType, GenericListArray, Int32Array, OffsetSizeTrait,
        PrimitiveArray, StructArray,
    },
    datatypes::{DataType, Float32Type},
};
use pyo3::exceptions::PyTypeError;
use pyo3::PyResult;

use crate::arrow::{lists::ExtractListArray, SparseIndexListType, SparseRowType};
use crate::ok_or_pyerr;

use super::SparseIndexType;

/// Variant type to wrap data with different index types.
#[derive(Clone)]
pub enum IxVar<T32, T64> {
    Ix32(T32),
    Ix64(T64),
}

/// A compressed sparse row matrix with only structure, not values.
pub struct CSRStructure<Ix: OffsetSizeTrait = i32> {
    pub n_rows: usize,
    pub n_cols: usize,
    array: GenericListArray<Ix>,
    pub col_inds: Int32Array,
}

/// A compressed sparse row matrix.
pub struct CSRMatrix<Ix: OffsetSizeTrait = i32, V: ArrowPrimitiveType = Float32Type> {
    pub n_rows: usize,
    pub n_cols: usize,
    array: GenericListArray<Ix>,
    pub col_inds: Int32Array,
    pub values: PrimitiveArray<V>,
}

/// Common methods for compressed sparse row matrices.
pub trait CSR<Ix: OffsetSizeTrait + TryInto<usize, Error: Debug> = i32> {
    /// Get the underlying Arrow array of rows.
    fn array(&self) -> &GenericListArray<Ix>;
    /// Get the underlying Arrow array of column indices.
    fn col_inds(&self) -> &Int32Array;

    /// Get the "length" (number of rows) in the matrix.
    fn len(&self) -> usize {
        self.array().len()
    }

    /// Get the number of observed values in the matrix.
    fn nnz(&self) -> usize {
        self.row_ptrs()[self.len()].try_into().unwrap()
    }

    /// Get the row pointers as a slice.
    fn row_ptrs(&self) -> &[Ix] {
        self.array().value_offsets()
    }

    /// Get the extent in the underlying arrays for a row in the matrix.
    fn extent(&self, row: usize) -> (usize, usize) {
        let off = self.row_ptrs();
        (
            off[row].try_into().unwrap(),
            off[row + 1].try_into().unwrap(),
        )
    }

    /// Get the column indices for a row in the matrix.
    fn row_cols(&self, row: usize) -> &[i32] {
        let (start, end) = self.extent(row);
        &self.col_inds().values()[start..end]
    }
}

/// Extract a CSR structure with either 32 or 64-bit indices.
pub fn csr_structure(
    array: Arc<dyn Array>,
) -> PyResult<IxVar<CSRStructure<i32>, CSRStructure<i64>>> {
    match array.data_type() {
        DataType::List(_) => Ok(IxVar::Ix32(CSRStructure::from_arrow(array)?)),
        DataType::LargeList(_) => Ok(IxVar::Ix64(CSRStructure::from_arrow(array)?)),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported CSR data type {}",
            array.data_type()
        ))),
    }
}

impl<Ix> CSRStructure<Ix>
where
    Ix: OffsetSizeTrait + TryInto<usize, Error: Debug>,
    GenericListArray<Ix>: ExtractListArray,
{
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>) -> PyResult<CSRStructure<Ix>> {
        let array: GenericListArray<Ix> = ok_or_pyerr!(
            GenericListArray::extract_list_array(&array),
            PyTypeError,
            "invalid array type {}, expected List or LargeList",
            array.data_type()
        )?;
        debug!("extracted array of type {}", array.data_type());
        let arr_type = SparseIndexListType::try_from(array.data_type())
            .map_err(|e| PyTypeError::new_err(format!("invalid array type: {:?}", e)))?;

        let col_inds: &Int32Array = ok_or_pyerr!(
            array.values().as_any().downcast_ref(),
            PyTypeError,
            "invalid element type {}, expected Int32",
            array.values().data_type()
        )?;
        let col_inds = downcast_array(&col_inds);

        Ok(CSRStructure {
            n_rows: array.len(),
            n_cols: arr_type.index_type.dimension(),
            array,
            col_inds,
        })
    }
}

impl<Ix: OffsetSizeTrait + TryInto<usize, Error: Debug>> CSR<Ix> for CSRStructure<Ix> {
    fn array(&self) -> &GenericListArray<Ix> {
        &self.array
    }
    fn col_inds(&self) -> &Int32Array {
        &self.col_inds
    }
}

impl<Ix, V> CSRMatrix<Ix, V>
where
    Ix: OffsetSizeTrait + TryInto<usize, Error: Debug>,
    V: ArrowPrimitiveType,
{
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>) -> PyResult<CSRMatrix<Ix, V>> {
        let sa: &GenericListArray<Ix> = ok_or_pyerr!(
            array.as_any().downcast_ref(),
            PyTypeError,
            "invalid array type {}, expected List or LargeList",
            array.data_type()
        )?;
        // type-check the columns
        let _arr_type = SparseRowType::try_from(array.data_type())
            .map_err(|e| PyTypeError::new_err(format!("invalid array type: {:?}", e)))?;

        let rows: &StructArray = ok_or_pyerr!(
            sa.values().as_any().downcast_ref(),
            PyTypeError,
            "invalid element type {}, expected Struct",
            sa.values().data_type()
        )?;

        let fields = rows.fields();
        assert_eq!(fields.len(), 2);

        let idx_f = &fields[0];
        let val_f = &fields[1];

        let idx_t: SparseIndexType = idx_f
            .try_extension_type()
            .map_err(|e| PyTypeError::new_err(format!("invalid index type: {}", e)))?;

        if val_f.data_type() != &V::DATA_TYPE {
            return Err(PyTypeError::new_err(format!(
                "invalid value column type {}, expected {}",
                val_f.data_type(),
                V::DATA_TYPE
            )));
        }

        Ok(CSRMatrix {
            n_rows: array.len(),
            n_cols: idx_t.dimension(),
            array: downcast_array(array.as_ref()),
            col_inds: downcast_array(rows.column(0)),
            values: downcast_array(rows.column(1)),
        })
    }

    /// Get the values for a row in the matrix.
    pub fn row_vals(&self, row: usize) -> &[V::Native] {
        let (start, end) = self.extent(row);
        &self.values.values()[start..end]
    }
}

impl<Ix, V> CSR<Ix> for CSRMatrix<Ix, V>
where
    Ix: OffsetSizeTrait + TryInto<usize, Error: Debug>,
    V: ArrowPrimitiveType,
{
    fn array(&self) -> &GenericListArray<Ix> {
        &self.array
    }
    fn col_inds(&self) -> &Int32Array {
        &self.col_inds
    }
}
