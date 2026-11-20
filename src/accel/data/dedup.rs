// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Deduplication support.
use arrow::compute::filter_record_batch;
use arrow::datatypes::Schema;
use arrow::{
    array::{BooleanArray, RecordBatch},
    pyarrow::PyArrowType,
};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;

mod row;

/// Arrow table de-duplicator.
#[pyclass]
pub struct TableDedup {
    memory: Box<dyn DedupMemory>,
}

/// Trait for remembering rows for de-duplication.
trait DedupMemory: Send + Sync {
    /// The number of entries in the de-duplication cache.
    fn len(&self) -> usize;

    /// Create a mask array to de-duplicate a single batch.
    fn batch_unique_mask(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<BooleanArray, arrow::error::ArrowError>;
}

#[pymethods]
impl TableDedup {
    /// Construct a new table deduplicator.
    #[new]
    #[pyo3(signature=(schema, *, columns=None))]
    fn new<'py>(schema: PyArrowType<Schema>, columns: Option<Vec<String>>) -> PyResult<Self> {
        if columns.is_some() {
            Err(PyNotImplementedError::new_err(
                "column-based dedup not yet implemented",
            ))
        } else {
            let memory = row::RowMemory::new(&schema.0).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to create deduplicator: {}", e))
            })?;
            Ok(TableDedup {
                memory: Box::new(memory),
            })
        }
    }

    /// Get the number of unique values this deduplicator has seen.
    fn unique_values(&self) -> usize {
        self.memory.len()
    }

    /// Deduplicate a record batch.
    fn dedup_batch(
        &mut self,
        batch: PyArrowType<RecordBatch>,
    ) -> PyResult<PyArrowType<RecordBatch>> {
        let batch = batch.0;
        let mask = self
            .memory
            .batch_unique_mask(&batch)
            .map_err(|e| PyRuntimeError::new_err(format!("deduplication failed: {}", e)))?;
        if mask.false_count() > 0 {
            let filtered = filter_record_batch(&batch, &mask)
                .map_err(|e| PyRuntimeError::new_err(format!("deduplication failed: {}", e)))?;
            Ok(filtered.into())
        } else {
            // no duplicates, return the batch unmodified.
            Ok(batch.into())
        }
    }

    /// Count duplicate rows in a record batch, updating the deduplicator's
    /// memory and statistics.
    fn count_duplicates(&mut self, batch: PyArrowType<RecordBatch>) -> PyResult<usize> {
        let batch = batch.0;
        let mask = self
            .memory
            .batch_unique_mask(&batch)
            .map_err(|e| PyRuntimeError::new_err(format!("deduplication failed: {}", e)))?;
        Ok(mask.false_count())
    }
}
