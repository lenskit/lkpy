use pyo3::prelude::*;

mod accum;
mod item_score;
mod item_train;
mod user_score;

/// Register the lenskit._accel.knn module
pub fn register_knn(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let knn = PyModule::new(parent.py(), "knn")?;
    parent.add_submodule(&knn)?;
    knn.add_function(wrap_pyfunction!(item_train::compute_similarities, &knn)?)?;
    knn.add_function(wrap_pyfunction!(item_score::score_explicit, &knn)?)?;
    knn.add_function(wrap_pyfunction!(item_score::score_implicit, &knn)?)?;
    knn.add_function(wrap_pyfunction!(
        user_score::user_score_items_explicit,
        &knn
    )?)?;
    knn.add_function(wrap_pyfunction!(
        user_score::user_score_items_implicit,
        &knn
    )?)?;
    Ok(())
}
