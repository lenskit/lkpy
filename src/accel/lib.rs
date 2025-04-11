use pyo3::prelude::*;

mod knn;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    knn::register_knn(m)?;

    Ok(())
}
