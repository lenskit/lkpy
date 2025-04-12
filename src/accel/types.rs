//! Support for type-checking.
use std::sync::Arc;

use pyo3::prelude::*;

use arrow::array::Array;
use pyo3::exceptions::PyTypeError;

pub(crate) fn checked_array_convert<'array, T: Array + 'static>(
    name: &str,
    tstr: &str,
    array: &'array dyn Array,
) -> PyResult<&'array T> {
    array.as_any().downcast_ref().ok_or_else(|| {
        PyErr::new::<PyTypeError, _>(format!(
            "invalid {} type {}, expected {}",
            name,
            array.data_type(),
            tstr
        ))
    })
}
