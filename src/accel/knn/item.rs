use arrow::{
    array::{make_array, ArrayData, Int32Array, StructArray},
    pyarrow::PyArrowType,
};
use log::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{
    arrow::typed_array_ref,
    sparse::{BoundCSR, CSRMatrix},
};

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    ratings: PyArrowType<ArrayData>,
    shape: (usize, usize),
    min_sim: f64,
    save_nbrs: Option<i64>,
) -> PyResult<PyArrowType<ArrayData>> {
    let (nu, ni) = shape;

    // extract the data
    debug!("extracting rating data for {}x{} matrix", nu, ni);
    let ratings = make_array(ratings.0);
    let rview: &StructArray = typed_array_ref(&ratings)?;
    let users: &Int32Array = typed_array_ref(
        rview
            .column_by_name("user")
            .ok_or_else(|| PyErr::new::<PyValueError, _>("missing column 'user'"))?,
    )?;
    let items: &Int32Array = typed_array_ref(
        rview
            .column_by_name("item")
            .ok_or_else(|| PyErr::new::<PyValueError, _>("missing column 'item'"))?,
    )?;
    let values: &Int32Array = typed_array_ref(
        rview
            .column_by_name("value")
            .ok_or_else(|| PyErr::new::<PyValueError, _>("missing column 'value'"))?,
    )?;

    todo!()
}
