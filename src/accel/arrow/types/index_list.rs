// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::datatypes::DataType;
use arrow_schema::extension::ExtensionType;
use arrow_schema::ArrowError;

use super::SparseIndexType;

/// Arrow extension type for sparse index lists.
#[derive(Debug)]
pub struct SparseIndexListType {
    #[allow(dead_code)]
    pub offset_type: DataType,
    pub index_type: SparseIndexType,
}

impl SparseIndexListType {
    /// Create a new sparse row extension for the given dimension.
    #[allow(dead_code)]
    pub fn create(dim: usize) -> SparseIndexListType {
        SparseIndexListType {
            offset_type: DataType::Int32,
            index_type: SparseIndexType::create(dim),
        }
    }

    /// Create a new sparse row extension for the given dimension with large offsets.
    #[allow(dead_code)]
    pub fn create_large(dim: usize) -> SparseIndexListType {
        SparseIndexListType {
            offset_type: DataType::Int64,
            index_type: SparseIndexType::create(dim),
        }
    }

    pub fn dimension(&self) -> usize {
        self.index_type.dimension()
    }
}

impl TryFrom<&DataType> for SparseIndexListType {
    type Error = ArrowError;

    fn try_from(value: &DataType) -> Result<Self, Self::Error> {
        Self::try_new(&value, ())
    }
}

impl ExtensionType for SparseIndexListType {
    type Metadata = ();

    const NAME: &'static str = "lenskit.sparse_index_list";

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
        let (off_t, elt_f) = match data_type {
            DataType::List(f) => (DataType::Int32, f),
            DataType::LargeList(f) => (DataType::Int64, f),
            // DataType::ListView(f) => f.data_type(),
            // DataType::LargeListView(f) => f.data_type(),
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "unsupported data type {}",
                    data_type
                )))
            }
        };

        let idx_t: SparseIndexType = elt_f.try_extension_type()?;

        Ok(SparseIndexListType {
            offset_type: off_t,
            index_type: idx_t,
        })
    }
}
