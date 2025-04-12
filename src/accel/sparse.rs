//! Sparse matrix support.

use std::sync::Arc;

use pyo3::prelude::*;

use arrow::{
    array::{downcast_array, Array, Float32Array, Int32Array, ListArray, StructArray},
    datatypes::DataType,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyResult;

pub struct CSRMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub array: ListArray,
    pub col_inds: Int32Array,
    pub values: Float32Array,
}

impl CSRMatrix {
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>, nr: usize, nc: usize) -> PyResult<CSRMatrix> {
        let sa: &ListArray = array.as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid array type {}, expected List",
                array.data_type()
            ))
        })?;

        let rows: &StructArray = sa.values().as_any().downcast_ref().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>(format!(
                "invalid array type {}, expected Struct",
                sa.values().data_type()
            ))
        })?;

        let names = rows.column_names();
        if names.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "row entries must have 2 fields",
            ));
        }
        if names[0] != "index" {
            return Err(PyErr::new::<PyValueError, _>(
                "first row field must be 'index'",
            ));
        }
        if names[1] != "value" {
            return Err(PyErr::new::<PyValueError, _>(
                "first row field must be 'value'",
            ));
        }
        if *rows.column(0).data_type() != DataType::Int32 {
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "invalid index column type {}, expected Int32",
                rows.column(0).data_type()
            )));
        }
        if *rows.column(1).data_type() != DataType::Float32 {
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "invalid value column type {}, expected Float32",
                rows.column(0).data_type()
            )));
        }

        Ok(CSRMatrix {
            n_rows: nr,
            n_cols: nc,
            array: downcast_array(array.as_ref()),
            col_inds: downcast_array(rows.column(0)),
            values: downcast_array(rows.column(1)),
        })
    }

    pub fn row_ptrs(&self) -> &[i32] {
        self.array.value_offsets()
    }

    pub fn extent(&self, row: usize) -> (i32, i32) {
        let off = self.array.value_offsets();
        (off[row], off[row + 1])
    }
}
