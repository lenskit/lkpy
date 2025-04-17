//! Sparse matrix support.

use std::sync::Arc;

use log::*;
use pyo3::prelude::*;

use arrow::{
    array::{
        downcast_array, make_array, Array, ArrayData, Float32Array, GenericListArray, Int32Array,
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
    pub offset_type: DataType,
    pub index_type: SparseIndexType,
    pub value_type: DataType,
}

/// Arrow extension type for sparse row indices.
#[derive(Debug)]
pub struct SparseIndexType {
    meta: SparseMeta,
}

/// Metadata for sparse matrix rows.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SparseMeta {
    /// The number of columns in the sparse row.
    pub dimension: usize,
}

impl SparseMeta {
    /// Create a new sparse row metadata object.
    pub fn create(dim: usize) -> SparseMeta {
        SparseMeta { dimension: dim }
    }
}

impl SparseIndexType {
    /// Create a new sparse row extension for the given dimension.
    pub fn create(dim: usize) -> SparseIndexType {
        SparseIndexType {
            meta: SparseMeta { dimension: dim },
        }
    }

    /// Get the dimension of the sparse row indices.
    pub fn dimension(&self) -> usize {
        self.meta.dimension
    }
}

impl ExtensionType for SparseIndexType {
    type Metadata = SparseMeta;

    const NAME: &'static str = "lenskit.sparse_index";

    fn metadata(&self) -> &Self::Metadata {
        &self.meta
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(to_string(&self.meta).expect("metadata serialization failed"))
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        let meta_str = metadata
            .ok_or_else(|| ArrowError::SchemaError("sparse row requires metadata".into()))?;
        let meta: SparseMeta =
            from_str(meta_str).map_err(|e| ArrowError::JsonError(e.to_string()))?;
        Ok(meta)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        match data_type {
            DataType::Int32 => Ok(()),
            t => Err(ArrowError::InvalidArgumentError(format!(
                "expected Int32 indices, got {}",
                t
            ))),
        }
    }

    fn try_new(
        data_type: &DataType,
        metadata: Self::Metadata,
    ) -> Result<Self, arrow_schema::ArrowError> {
        match data_type {
            DataType::Int32 => Ok(Self { meta: metadata }),
            t => Err(ArrowError::InvalidArgumentError(format!(
                "expected Int32 indices, got {}",
                t
            ))),
        }
    }
}

impl SparseRowType {
    /// Create a new sparse row extension for the given dimension.
    pub fn create(dim: usize) -> SparseRowType {
        SparseRowType {
            offset_type: DataType::Int32,
            index_type: SparseIndexType::create(dim),
            value_type: DataType::Float32,
        }
    }

    /// Create a new sparse row extension for the given dimension with large offsets.
    pub fn create_large(dim: usize) -> SparseRowType {
        SparseRowType {
            offset_type: DataType::Int64,
            index_type: SparseIndexType::create(dim),
            value_type: DataType::Float32,
        }
    }

    pub fn dimension(&self) -> usize {
        self.index_type.meta.dimension
    }
}

impl TryFrom<&DataType> for SparseRowType {
    type Error = ArrowError;

    fn try_from(value: &DataType) -> Result<Self, Self::Error> {
        Self::try_new(&value, ())
    }
}

impl ExtensionType for SparseRowType {
    type Metadata = ();

    const NAME: &'static str = "lenskit.sparse_row";

    fn metadata(&self) -> &Self::Metadata {
        &()
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn deserialize_metadata(_metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        let srt = Self::try_new(data_type, ())?;
        if srt.value_type != self.value_type {
            return Err(ArrowError::InvalidArgumentError(format!(
                "value type mismatch: {} != {}",
                srt.value_type, self.value_type
            )));
        }

        if srt.dimension() != self.dimension() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "index dimension mismatch: {} != {}",
                srt.dimension(),
                self.dimension()
            )));
        }

        Ok(())
    }

    fn try_new(
        data_type: &DataType,
        _metadata: Self::Metadata,
    ) -> Result<Self, arrow_schema::ArrowError> {
        let (off_t, elt_t) = match data_type {
            DataType::List(f) => (DataType::Int32, f.data_type()),
            DataType::LargeList(f) => (DataType::Int64, f.data_type()),
            // DataType::ListView(f) => f.data_type(),
            // DataType::LargeListView(f) => f.data_type(),
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "unsupported data type {}",
                    data_type
                )))
            }
        };
        let fields = match elt_t {
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
        if idx_f.name() != "index" {
            return Err(ArrowError::InvalidArgumentError(format!(
                "first field must be 'index', found, found {}",
                idx_f.name()
            )));
        }
        let idx_t: SparseIndexType = idx_f.try_extension_type()?;

        let val_f = fields.get(1).unwrap();
        if val_f.name() != "value" {
            return Err(ArrowError::InvalidArgumentError(format!(
                "second field must be 'value', found, found {}",
                idx_f.name()
            )));
        }

        Ok(SparseRowType {
            offset_type: off_t,
            index_type: idx_t,
            value_type: val_f.data_type().clone(),
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

/// Test function to make sure we can convert sparse rows.
#[pyfunction]
pub(crate) fn sparse_row_debug(array: PyArrowType<ArrayData>) -> PyResult<(String, usize, usize)> {
    let array = make_array(array.0);
    debug!("building matrix with {} rows", array.len());
    debug!("array data type: {}", array.data_type());
    let rt: SparseRowType = array
        .data_type()
        .try_into()
        .map_err(|e: ArrowError| PyTypeError::new_err(e.to_string()))?;
    debug!("got {} x {} matrix", array.len(), rt.dimension());
    Ok((format!("{:?}", rt), array.len(), rt.dimension()))
}
