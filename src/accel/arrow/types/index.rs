// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::datatypes::DataType;
use arrow_schema::extension::ExtensionType;
use arrow_schema::ArrowError;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};

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
    #[allow(dead_code)]
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
