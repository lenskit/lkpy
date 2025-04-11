use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

mod arrow;
mod knn;
mod sparse;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    knn::register_knn(m)?;

    m.add_function(wrap_pyfunction!(init_accel_pool, m)?)?;

    Ok(())
}

#[pyfunction]
fn init_accel_pool(n_threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Rayon initialization error"))
}
