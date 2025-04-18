use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

mod data;
mod knn;
mod sampling;
mod sparse;
mod types;

/// Entry point for LensKit accelerator module.
#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    knn::register_knn(m)?;

    m.add_class::<sampling::NegativeSampler>()?;
    m.add_class::<data::RowColumnSet>()?;
    m.add_function(wrap_pyfunction!(init_accel_pool, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::sparse_row_debug, m)?)?;

    Ok(())
}

#[pyfunction]
fn init_accel_pool(n_threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Rayon initialization error"))
}
