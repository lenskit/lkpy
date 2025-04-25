use arrow::{
    array::{make_array, ArrayData, Float32Array, Int32Array},
    pyarrow::PyArrowType,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::types::checked_array;

#[derive(FromPyObject, Clone, Debug)]
struct FunkSVDConfig {
    learning_rate: f64,
    regularization: f64,
    rating_min: f64,
    rating_max: f64,
}

#[derive(FromPyObject)]
struct FunkSVDTrainingInput {
    users: PyArrowType<ArrayData>,
    items: PyArrowType<ArrayData>,
    ratings: PyArrowType<ArrayData>,
}

struct FunkSVDTrainingData {
    users: Int32Array,
    items: Int32Array,
    ratings: Float32Array,
}

#[pyclass]
pub struct FunkSVDTrainer {
    config: FunkSVDConfig,
    data: FunkSVDTrainingData,
    user_features: Py<PyArray2<f32>>,
    item_features: Py<PyArray2<f32>>,
}

#[pymethods]
impl FunkSVDTrainer {
    /// Instantiate a new FunkSVD trainer.
    #[new]
    fn new<'py>(
        config: Bound<'py, PyAny>,
        data: FunkSVDTrainingInput,
        user_features: Bound<'py, PyArray2<f32>>,
        item_features: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Self> {
        let config = FunkSVDConfig::extract_bound(&config)?;

        Ok(FunkSVDTrainer {
            config,
            data: data.try_into()?,
            user_features: user_features.unbind(),
            item_features: item_features.unbind(),
        })
    }

    /// Train a FunkSVD feature for one epoch.
    fn feature_epoch<'py>(
        &self,
        py: Python<'py>,
        feature: usize,
        estimates: Bound<'py, PyArray1<f32>>,
        trail: f64,
    ) -> PyResult<f64> {
        let mut uf_ref = self.user_features.bind(py).readwrite();
        let mut uf_mat = uf_ref.as_array_mut();
        let mut if_ref = self.item_features.bind(py).readwrite();
        let mut if_mat = if_ref.as_array_mut();
        let est_ref = estimates.readonly();
        let est_vec = est_ref.as_array();

        let mut sse = 0.0;

        for s in 0..self.data.n_samples() {
            let user = self.data.users.value(s) as usize;
            let item = self.data.items.value(s) as usize;
            let rating = self.data.ratings.value(s) as f64;
            let ufv = uf_mat[[user, feature]] as f64;
            let ifv = if_mat[[item, feature]] as f64;

            let pred = est_vec[s] as f64 + ufv * ifv + trail;
            let pred = pred.clamp(self.config.rating_min, self.config.rating_max);

            let error = rating - pred;
            sse += error * error;

            let ufd = error * ifv - self.config.regularization * ufv;
            let ufd = ufd * self.config.learning_rate;
            let ifd = error * ufv - self.config.regularization * ifv;
            let ifd = ifd * self.config.learning_rate;
            uf_mat[[user, feature]] += ufd as f32;
            if_mat[[item, feature]] += ifd as f32;
        }

        Ok((sse / self.data.n_samples() as f64).sqrt())
    }
}

impl FunkSVDTrainingData {
    fn n_samples(&self) -> usize {
        self.users.len()
    }
}

impl TryInto<FunkSVDTrainingData> for FunkSVDTrainingInput {
    type Error = PyErr;

    fn try_into(self) -> PyResult<FunkSVDTrainingData> {
        Ok(FunkSVDTrainingData {
            users: checked_array("users", &make_array(self.users.0))?,
            items: checked_array("items", &make_array(self.items.0))?,
            ratings: checked_array("ratings", &make_array(self.ratings.0))?,
        })
    }
}
