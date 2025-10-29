// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::datatypes::DataType;
use arrow_schema::extension::ExtensionType;
use arrow_schema::ArrowError;

use super::SparseIndexType;

/// Arrow extension type for sparse rows.
#[derive(Debug)]
pub struct SparseRowType {
    pub offset_type: DataType,
    #[allow(dead_code)]
    pub index_name: String,
    pub index_type: SparseIndexType,
    pub value_type: DataType,
}

impl SparseRowType {
    /// Create a new sparse row extension for the given dimension.
    #[cfg(false)]
    pub fn create(dim: usize) -> SparseRowType {
        SparseRowType {
            offset_type: DataType::Int32,
            index_name: "index".into(),
            index_type: SparseIndexType::create(dim),
            value_type: DataType::Float32,
        }
    }

    /// Create a new sparse row extension for the given dimension with large offsets.
    #[cfg(false)]
    pub fn create_large(dim: usize) -> SparseRowType {
        SparseRowType {
            offset_type: DataType::Int64,
            index_name: "index".into(),
            index_type: SparseIndexType::create(dim),
            value_type: DataType::Float32,
        }
    }

    pub fn dimension(&self) -> usize {
        self.index_type.dimension()
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
        let idx_name = idx_f.name();
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
            index_name: idx_name.clone(),
            index_type: idx_t,
            value_type: val_f.data_type().clone(),
        })
    }
}
