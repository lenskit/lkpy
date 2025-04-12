use pyo3::prelude::*;

mod item;

/// Register the lenskit._accel.knn module
pub fn register_knn(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let knn = PyModule::new(parent.py(), "knn")?;
    parent.add_submodule(&knn)?;
    knn.add_function(wrap_pyfunction!(item::compute_similarities, &knn)?)?;
    knn.add_function(wrap_pyfunction!(item::score_explicit, &knn)?)?;
    knn.add_function(wrap_pyfunction!(item::score_implicit, &knn)?)?;
    Ok(())
}
