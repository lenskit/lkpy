use pyo3::prelude::*;

mod knn;
mod sparse;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    knn::register_knn(m)?;

    m.add_function(wrap_pyfunction!(sparse::make_csr, m)?)?;

    Ok(())
}
