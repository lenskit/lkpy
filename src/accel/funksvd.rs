// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{make_array, ArrayData, Float32Array, Int32Array},
    pyarrow::PyArrowType,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::arrow::checked_array;

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

/// Train FunkSVD models.
#[pyclass]
pub struct FunkSVDTrainer {
    config: FunkSVDConfig,
    data: FunkSVDTrainingData,
    user_embeddings: Py<PyArray2<f32>>,
    item_embeddings: Py<PyArray2<f32>>,
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
            user_embeddings: user_features.unbind(),
            item_embeddings: item_features.unbind(),
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
        let mut uf_ref = self.user_embeddings.bind(py).readwrite();
        let mut uf_mat = uf_ref.as_array_mut();
        let mut if_ref = self.item_embeddings.bind(py).readwrite();
        let mut if_mat = if_ref.as_array_mut();
        let est_ref = estimates.readonly();
        let est_vec = est_ref.as_array();

        let users = self.data.users.values();
        let items = self.data.items.values();
        let ratings = self.data.ratings.values();
        let n = self.data.n_samples();
        assert_eq!(users.len(), n);
        assert_eq!(items.len(), n);
        assert_eq!(ratings.len(), n);

        let lr = self.config.learning_rate;
        let reg = self.config.regularization;

        let mut uf_col = uf_mat.column_mut(feature);
        let mut if_col = if_mat.column_mut(feature);

        py.allow_threads(|| {
            let mut sse = 0.0;

            for s in 0..n {
                // SAFETY: we checked the slice lengths above
                let user = unsafe { *users.get_unchecked(s) as usize };
                let item = unsafe { *items.get_unchecked(s) as usize };
                let rating = unsafe { *ratings.get_unchecked(s) as f64 };

                // get references to user & item feature values so we only need to look up once
                let ufr = &mut uf_col[user];
                let ifr = &mut if_col[item];

                let ufv = *ufr as f64;
                let ifv = *ifr as f64;

                // compute the prediction and error
                let pred = est_vec[s] as f64 + ufv * ifv + trail;
                let pred = pred.clamp(self.config.rating_min, self.config.rating_max);
                let error = rating - pred;
                sse += error * error;

                // compute and apply gradient updates
                let ufd = error * ifv - reg * ufv;
                let ufd = ufd * lr;
                let ifd = error * ufv - reg * ifv;
                let ifd = ifd * lr;
                *ufr = (ufv + ufd) as f32;
                *ifr = (ifv + ifd) as f32;
            }

            Ok((sse / n as f64).sqrt())
        })
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
