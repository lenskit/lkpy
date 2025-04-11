use pyo3::prelude::*;

pub fn register_knn(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let knn = PyModule::new(parent.py(), "knn")?;
    parent.add_submodule(&knn)?;
    Ok(())
}
