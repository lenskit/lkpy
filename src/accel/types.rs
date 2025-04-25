//! Support for type-checking.
use pyo3::prelude::*;

use arrow::array::{downcast_array, Array, ArrowPrimitiveType, PrimitiveArray};
use pyo3::exceptions::PyTypeError;

pub(crate) fn checked_array_ref<'array, T: Array + 'static>(
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

pub(crate) fn checked_array<'array, E: ArrowPrimitiveType + 'static>(
    name: &str,
    array: &'array dyn Array,
) -> PyResult<PrimitiveArray<E>> {
    if array.data_type().equals_datatype(&E::DATA_TYPE) {
        Ok(downcast_array(array))
    } else {
        Err(PyTypeError::new_err(format!(
            "invalid {} type {}, expected {}",
            name,
            array.data_type(),
            E::DATA_TYPE
        )))
    }
}
