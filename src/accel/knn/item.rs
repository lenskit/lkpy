use arrow::{
    array::{make_array, ArrayData, Int32Array, StructArray},
    pyarrow::PyArrowType,
};
use log::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{arrow::typed_array_ref, sparse::CSRMatrix};

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    ui_ratings: PyArrowType<ArrayData>,
    iu_ratings: PyArrowType<ArrayData>,
    shape: (u32, u32),
    min_sim: f64,
    save_nbrs: Option<i64>,
) -> PyResult<PyArrowType<ArrayData>> {
    let (nu, ni) = shape;

    // extract the data
    debug!("preparing {}x{} matrix", nu, ni);
    let ui_mat = CSRMatrix::from_arrow(make_array(ui_ratings.0), nu, ni)?;
    let iu_mat = CSRMatrix::from_arrow(make_array(iu_ratings.0), ni, nu)?;

    todo!()
}
