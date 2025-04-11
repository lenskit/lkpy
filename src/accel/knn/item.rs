use log::*;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sparse::{BoundCSR, CSRMatrix};

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    iumat: &Bound<'py, CSRMatrix>,
    uimat: &Bound<'py, CSRMatrix>,
    min_sim: f64,
    save_nbrs: Option<i64>,
) -> PyResult<Bound<'py, CSRMatrix>> {
    let iumb = iumat.borrow().bind_csr(py);
    let uimb = uimat.borrow().bind_csr(py);

    todo!()
}
