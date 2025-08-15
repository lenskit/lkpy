// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Arrow-based ID index.

use std::sync::Arc;
use std::{convert::Infallible, hash::Hash};

use arrow_schema::DataType;
use pyo3::types::PyInt;
use pyo3::{
    exceptions::{PyOverflowError, PyTypeError},
    prelude::*,
};

use arrow::{
    array::{make_array, Array, ArrayData, ArrowPrimitiveType, AsArray, Int32Builder},
    datatypes::{Int16Type, Int32Type, Int64Type, UInt16Type, UInt32Type, UInt64Type},
    pyarrow::PyArrowType,
};

mod storage_int;
mod storage_str;

use storage_int::PrimitiveIDArray;
use storage_str::StringIDArray;

use crate::data::indirect_hash::{IndirectHashTable, PositionLookup};

/// Arrow-based ID index.
#[pyclass]
pub struct IDIndex {
    /// The array of IDs.
    ids: Arc<dyn Array>,

    /// Lookup table.
    index: Box<dyn PositionLookup + Sync + Send>,
}

/// Helper function for primitive array tables.
fn prim_tbl<T>(arr: &dyn Array) -> PyResult<Box<IndirectHashTable<PrimitiveIDArray<T>>>>
where
    T: ArrowPrimitiveType,
    T::Native: Hash
        + for<'a> FromPyObject<'a>
        + for<'a> IntoPyObject<'a, Target = PyInt, Output = Bound<'a, PyInt>, Error = Infallible>,
{
    let arr = PrimitiveIDArray::new(arr.as_primitive::<T>().clone());
    let tbl = IndirectHashTable::from_unique(arr)?;
    Ok(Box::new(tbl))
}

impl IDIndex {
    /// Create an empty ID index.
    fn empty() -> Self {
        IDIndex {
            ids: Arc::new(Int32Builder::new().finish()),
            index: Box::new(IndirectHashTable::<PrimitiveIDArray<Int32Type>>::default()),
        }
    }

    /// Create an ID index from an array of data.
    fn from_data(data: PyArrowType<ArrayData>) -> PyResult<Self> {
        let ids = make_array(data.0);
        let index: Box<dyn PositionLookup + Sync + Send> = match ids.data_type() {
            DataType::Null if ids.len() == 0 => return Ok(Self::empty()),
            DataType::Int16 => prim_tbl::<Int16Type>(&ids)?,
            DataType::UInt16 => prim_tbl::<UInt16Type>(&ids)?,
            DataType::Int32 => prim_tbl::<Int32Type>(&ids)?,
            DataType::UInt32 => prim_tbl::<UInt32Type>(&ids)?,
            DataType::Int64 => prim_tbl::<Int64Type>(&ids)?,
            DataType::UInt64 => prim_tbl::<UInt64Type>(&ids)?,
            DataType::Utf8 => Box::new(IndirectHashTable::from_unique(StringIDArray::new(
                ids.as_string().clone(),
            ))?),
            // TODO: add support for large strings, views, and binaries
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported ID type {}",
                    ids.data_type()
                )))
            }
        };

        Ok(IDIndex { ids, index })
    }
}

#[pymethods]
impl IDIndex {
    /// Construct a new ID index.
    ///
    /// The elements **must** be unique.
    #[new]
    fn new(entity_ids: Option<PyArrowType<ArrayData>>) -> PyResult<Self> {
        if let Some(data) = entity_ids {
            Self::from_data(data)
        } else {
            Ok(Self::empty())
        }
    }

    /// Look up a single index by ID.
    fn get_index<'py>(&self, py: Python<'py>, id: Bound<'py, PyAny>) -> PyResult<Option<u32>> {
        match self.index.lookup_value(py, id) {
            Ok(w) => Ok(w),
            Err(e)
                if e.is_instance_of::<PyTypeError>(py)
                    || e.is_instance_of::<PyOverflowError>(py) =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Look up multiple indexes by ID.
    fn get_indexes<'py>(
        &self,
        py: Python<'py>,
        ids: Bound<'py, PyAny>,
    ) -> PyResult<PyArrowType<ArrayData>> {
        self.index
            .lookup_array(py, ids)
            .map(|a| a.into_data().into())
    }

    /// Get the ID array.
    fn id_array(&self) -> PyArrowType<ArrayData> {
        self.ids.to_data().into()
    }

    /// Get the length of the array.
    fn __len__(&self) -> usize {
        self.ids.len()
    }
}
