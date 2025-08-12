// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Arrow-based ID index.

use arrow_schema::DataType;
use pyo3::{
    exceptions::{PyOverflowError, PyTypeError, PyValueError},
    prelude::*,
};

use arrow::{
    array::{make_array, Array, ArrayData, AsArray, Int32Builder, UInt32Builder},
    datatypes::{Int16Type, Int32Type, Int64Type, UInt16Type, UInt32Type, UInt64Type},
    pyarrow::PyArrowType,
};
use hashbrown::{hash_table::Entry, HashTable};
use log::*;

mod storage;
mod storage_int;
mod storage_str;

use storage::IDArray;
use storage_int::IDPrimArray;
use storage_str::IDStringArray;

/// Arrow-based ID index.
#[pyclass]
pub struct IDIndex {
    /// The array of IDs.
    ids: Box<dyn IDArray>,

    /// Table of indices.
    index: HashTable<u32>,
}

impl IDIndex {
    /// Create an empty ID index.
    fn empty() -> Self {
        IDIndex {
            ids: Box::new(IDPrimArray::new(Int32Builder::new().finish())),
            index: HashTable::new(),
        }
    }

    /// Create an ID index from an array of data.
    fn from_data(data: PyArrowType<ArrayData>) -> PyResult<Self> {
        let arr = make_array(data.0);
        let ids: Box<dyn IDArray> = match arr.data_type() {
            DataType::Null if arr.len() == 0 => return Ok(Self::empty()),
            DataType::Int16 => Box::new(IDPrimArray::new(arr.as_primitive::<Int16Type>().clone())),
            DataType::UInt16 => {
                Box::new(IDPrimArray::new(arr.as_primitive::<UInt16Type>().clone()))
            }
            DataType::Int32 => Box::new(IDPrimArray::new(arr.as_primitive::<Int32Type>().clone())),
            DataType::UInt32 => {
                Box::new(IDPrimArray::new(arr.as_primitive::<UInt32Type>().clone()))
            }
            DataType::Int64 => Box::new(IDPrimArray::new(arr.as_primitive::<Int64Type>().clone())),
            DataType::UInt64 => {
                Box::new(IDPrimArray::new(arr.as_primitive::<UInt64Type>().clone()))
            }
            DataType::Utf8 => Box::new(IDStringArray::new(arr.as_string().clone())),
            // TODO: add support for large strings, views, and binaries
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported ID type {}",
                    arr.data_type()
                )))
            }
        };

        let mut index = HashTable::with_capacity(ids.len());
        for i in 0..ids.len() {
            let i = i as u32;
            let hash = ids.hash_entry(i);
            let e = index.entry(
                hash,
                |jr| ids.compare_entries(i, *jr),
                |jr| ids.hash_entry(*jr),
            );
            if let Entry::Occupied(_) = &e {
                return Err(PyValueError::new_err(format!("duplicate IDs found")));
            }
            e.insert(i);
        }

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
        let wrap = match self.ids.wrap_value(id) {
            Ok(w) => w,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py)
                    || e.is_instance_of::<PyOverflowError>(py) =>
            {
                return Ok(None)
            }
            Err(e) => return Err(e),
        };
        let hash = wrap.hash(0);
        let res = self.index.find(hash, |jr| wrap.compare_with_entry(0, *jr));
        Ok(res.map(|ir| *ir))
    }

    /// Look up multiple indexes by ID.
    fn get_indexes(&self, ids: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
        let arr = make_array(ids.0);
        let n = arr.len();
        let wrap = self.ids.wrap_array(arr)?;
        let mut rb = Int32Builder::with_capacity(n);
        // TODO: parallelize index lookup
        for i in 0..n {
            let hash = wrap.hash(i);
            if let Some(ir) = self.index.find(hash, |jr| wrap.compare_with_entry(i, *jr)) {
                rb.append_value(*ir as i32);
            } else {
                rb.append_null();
            }
        }
        let numbers = rb.finish();
        trace!("resolved with {} nulls", numbers.null_count());
        trace!("validity mask: {:?}", numbers.nulls());
        Ok(numbers.into_data().into())
    }

    /// Get the ID array.
    fn id_array(&self) -> PyArrowType<ArrayData> {
        self.ids.data().into()
    }

    /// Get the length of the array.
    fn __len__(&self) -> usize {
        self.ids.len()
    }
}
