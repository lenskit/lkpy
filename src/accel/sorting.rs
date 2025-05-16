// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use arrow::{
    array::{make_array, Array, ArrayData, Float32Array, Int32Array},
    pyarrow::PyArrowType,
};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use ordered_float::NotNan;
use pyo3::prelude::*;
use rayon::slice::ParallelSliceMut;

use crate::types::checked_array;

#[pyfunction]
pub(crate) fn argsort_f32<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let scores_py = scores.readonly();
    let scores = scores_py.as_array();

    let mut indices: Array1<i32> = (0..(scores.len() as i32)).collect();
    indices
        .as_slice_mut()
        .unwrap()
        .par_sort_unstable_by_key(|i| NotNan::new(-scores[*i as usize]).unwrap());

    Ok(indices.to_pyarray(py))
}

#[pyfunction]
pub(crate) fn argsort_f32_arrow<'py>(
    scores: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let scores = make_array(scores.0);
    let scores: Float32Array = checked_array("scores", &scores)?;
    let sbuf = scores.values();

    let mut indices: Vec<i32> = (0..(scores.len() as i32)).collect();
    indices.par_sort_unstable_by_key(|i| NotNan::new(-sbuf[*i as usize]).unwrap());

    let array = Int32Array::from(indices);
    Ok(array.into_data().into())
}
