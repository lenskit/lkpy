use std::sync::Arc;

use pyo3::prelude::*;

use arrow::{
    array::{
        downcast_array, Array, Float32Array, GenericListArray, Int32Array, OffsetSizeTrait,
        StructArray,
    },
    buffer::ScalarBuffer,
    datatypes::DataType,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyResult;

use super::SparseIndexType;

pub struct CSRStructure<Ix: OffsetSizeTrait = i32> {
    pub n_rows: usize,
    pub n_cols: usize,
    array: GenericListArray<Ix>,
    pub col_inds: Int32Array,
}

pub struct CSRMatrix<Ix: OffsetSizeTrait = i32> {
    pub n_rows: usize,
    pub n_cols: usize,
    array: GenericListArray<Ix>,
    pub col_inds: Int32Array,
    pub values: Float32Array,
}

impl<Ix: OffsetSizeTrait> CSRStructure<Ix> {
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>) -> PyResult<CSRStructure<Ix>> {
        let sa: &GenericListArray<Ix> = array.as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid array type {}, expected List",
                array.data_type()
            ))
        })?;

        let field = match sa.data_type() {
            DataType::List(f) => f,
            DataType::LargeList(f) => f,
            _ => unreachable!("downcast and type are inconsistent"),
        };
        let rows: &Int32Array = sa.values().as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid element type {}, expected Struct",
                sa.values().data_type()
            ))
        })?;

        let idx_t: SparseIndexType = field
            .try_extension_type()
            .map_err(|e| PyTypeError::new_err(format!("invalid index type: {}", e)))?;

        Ok(CSRStructure {
            n_rows: array.len(),
            n_cols: idx_t.dimension(),
            array: downcast_array(array.as_ref()),
            col_inds: downcast_array(&rows),
        })
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn row_ptrs(&self) -> &[Ix] {
        self.array.value_offsets()
    }

    pub fn extent(&self, row: usize) -> (Ix, Ix) {
        let off = self.row_ptrs();
        (off[row], off[row + 1])
    }

    pub fn row_columns(&self, row: usize) -> ScalarBuffer<i32> {
        let (start, end) = self.extent(row);
        self.col_inds
            .values()
            .slice(start.as_usize(), (end - start).as_usize())
    }
}

impl<Ix: OffsetSizeTrait> CSRMatrix<Ix> {
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>) -> PyResult<CSRMatrix<Ix>> {
        let sa: &GenericListArray<Ix> = array.as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid array type {}, expected List",
                array.data_type()
            ))
        })?;

        let rows: &StructArray = sa.values().as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid element type {}, expected Struct",
                sa.values().data_type()
            ))
        })?;

        let fields = rows.fields();

        if fields.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "row entries must have 2 fields",
            ));
        }

        let idx_f = &fields[0];
        let val_f = &fields[1];

        if idx_f.name() != "index" {
            return Err(PyErr::new::<PyValueError, _>(
                "first row field must be 'index'",
            ));
        }
        if val_f.name() != "value" {
            return Err(PyErr::new::<PyValueError, _>(
                "first row field must be 'value'",
            ));
        }

        let idx_t: SparseIndexType = idx_f
            .try_extension_type()
            .map_err(|e| PyTypeError::new_err(format!("invalid index type: {}", e)))?;

        if val_f.data_type() != &DataType::Float32 {
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "invalid value column type {}, expected Float32",
                val_f.data_type()
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

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn row_ptrs(&self) -> &[Ix] {
        self.array.value_offsets()
    }

    pub fn extent(&self, row: usize) -> (Ix, Ix) {
        let off = self.row_ptrs();
        (off[row], off[row + 1])
    }
}
