//! Arrow conversion utilities.

use std::sync::Arc;

use pyo3::prelude::*;

use arrow::array::Array;
use pyo3::exceptions::PyTypeError;

pub(crate) fn typed_array_ref<T: Array + 'static>(array: &Arc<dyn Array>) -> PyResult<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| PyErr::new::<PyTypeError, _>("invalid array type"))
}
