//! Sparse matrix support.

use std::sync::Arc;

use log::*;
use pyo3::prelude::*;

use arrow::{
    array::{
        downcast_array, Array, ArrayData, Float32Array, GenericListArray, Int32Array,
        OffsetSizeTrait, StructArray,
    },
    datatypes::DataType,
    pyarrow::PyArrowType,
};
use arrow_schema::extension::ExtensionType;
use arrow_schema::ArrowError;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};

/// Arrow extension type for sparse rows.
#[derive(Debug)]
pub struct SparseRowType {
    value_type: DataType,
    meta: SparseRowMeta,
}

/// Metadata for sparse rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseRowMeta {
    /// The number of columns in the sparse row.
    pub dimension: usize,
}

impl SparseRowMeta {
    /// Create a new sparse row metadata object.
    pub fn create(dim: usize) -> SparseRowMeta {
        SparseRowMeta { dimension: dim }
    }
}

impl SparseRowType {
    /// Create a new sparse row extension for the given dimension.
    pub fn create(dim: usize) -> SparseRowType {
        SparseRowType {
            value_type: DataType::Float32,
            meta: SparseRowMeta { dimension: dim },
        }
    }

    fn check_type_compat(data_type: &DataType) -> Result<(), ArrowError> {
        let element = match data_type {
            DataType::List(f) => f.data_type(),
            DataType::LargeList(f) => f.data_type(),
            // DataType::ListView(f) => f.data_type(),
            // DataType::LargeListView(f) => f.data_type(),
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "unsupported data type {}",
                    data_type
                )))
            }
        };
        let fields = match element {
            DataType::Struct(fs) => fs,
            t => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "unsupported element type {}",
                    t
                )))
            }
        };

        if fields.len() != 2 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "entries must have 2 fields, found {}",
                fields.len()
            )));
        }

        let idx_f = fields.get(0).unwrap();
        let val_f = fields.get(1).unwrap();
        if idx_f.name() != "index" {
            return Err(ArrowError::InvalidArgumentError(format!(
                "first field must be 'index', found, found {}",
                idx_f.name()
            )));
        }
        if idx_f.data_type() != &DataType::Int32 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "index field must have type Int32, found {}",
                idx_f.data_type()
            )));
        }
        if val_f.name() != "value" {
            return Err(ArrowError::InvalidArgumentError(format!(
                "second field must be 'value', found, found {}",
                idx_f.name()
            )));
        }

        Ok(())
    }
}

impl ExtensionType for SparseRowType {
    type Metadata = SparseRowMeta;

    const NAME: &'static str = "lenskit.sparse_row";

    fn metadata(&self) -> &Self::Metadata {
        &self.meta
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(to_string(&self.meta).expect("metadata serialization failed"))
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        let meta_str = metadata
            .ok_or_else(|| ArrowError::SchemaError("sparse row requires metadata".into()))?;
        let meta: SparseRowMeta =
            from_str(meta_str).map_err(|e| ArrowError::JsonError(e.to_string()))?;
        Ok(meta)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        Self::check_type_compat(data_type)
    }

    fn try_new(
        data_type: &DataType,
        metadata: Self::Metadata,
    ) -> Result<Self, arrow_schema::ArrowError> {
        Self::check_type_compat(data_type)?;
        Ok(Self {
            value_type: data_type.clone(),
            meta: metadata,
        })
    }
}

pub struct CSRMatrix<Ix: OffsetSizeTrait = i32> {
    pub n_rows: usize,
    pub n_cols: usize,
    array: GenericListArray<Ix>,
    pub col_inds: Int32Array,
    pub values: Float32Array,
}

impl<Ix: OffsetSizeTrait> CSRMatrix<Ix> {
    /// Convert an Arrow structured array into a CSR matrix, checking for type errors.
    pub fn from_arrow(array: Arc<dyn Array>, nr: usize, nc: usize) -> PyResult<CSRMatrix<Ix>> {
        let sa: &GenericListArray<Ix> = array.as_any().downcast_ref().ok_or_else(|| {
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

/// Test function to make sure we can convert sparse rows.
#[pyfunction]
pub(crate) fn sparse_row_debug(
    array: PyArrowType<ArrayData>,
    dim: usize,
) -> PyResult<(String, usize, usize)> {
    let data = array.0;
    debug!("building matrix {}x{}", data.len(), dim);
    debug!("array data type: {}", data.data_type());
    let meta = SparseRowMeta::create(dim);
    let rt = SparseRowType::try_new(data.data_type(), meta)
        .map_err(|e| PyTypeError::new_err(format!("{}", e)))?;
    Ok((format!("{:?}", rt), data.len(), dim))
}
