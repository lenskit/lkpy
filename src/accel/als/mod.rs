mod explicit;
mod implicit;

use pyo3::prelude::*;

/// Register the lenskit._accel.als module
pub fn register_als(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let als = PyModule::new(parent.py(), "als")?;
    parent.add_submodule(&als)?;

    als.add_function(wrap_pyfunction!(explicit::train_explicit_matrix, &als)?)?;
    als.add_function(wrap_pyfunction!(implicit::train_implicit_matrix, &als)?)?;

    Ok(())
}
