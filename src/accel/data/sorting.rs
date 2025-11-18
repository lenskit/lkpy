// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrowPrimitiveType, Int32Array, PrimitiveArray, RecordBatch,
    },
    pyarrow::PyArrowType,
};
use ordered_float::{FloatCore, NotNan};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::slice::ParallelSliceMut;

use crate::match_array_type;
use crate::{arrow::checked_array, ok_or_pyerr};

const PAR_SORT_THRESHOLD: usize = 10_000;

/// Check if two columns of a table are properly-sorted COO.
#[pyfunction]
pub(super) fn is_sorted_coo<'py>(
    data: Vec<PyArrowType<RecordBatch>>,
    c1: &'py str,
    c2: &'py str,
) -> PyResult<bool> {
    let mut last = None;
    for PyArrowType(batch) in data {
        let col1 = ok_or_pyerr!(
            batch.column_by_name(c1),
            PyValueError,
            "unknown column: {}",
            c1
        )?;
        let col2 = ok_or_pyerr!(
            batch.column_by_name(c2),
            PyValueError,
            "unknown column: {}",
            c2
        )?;

        let col1: Int32Array = checked_array(c1, col1)?;
        let col2: Int32Array = checked_array(c2, col2)?;

        for i in 0..col1.len() {
            let v1 = col1.value(i);
            let v2 = col2.value(i);
            let k = (v1, v2);
            if let Some(lk) = last {
                if k <= lk {
                    // found a key out-of-order, we're done
                    return Ok(false);
                }
            }
            last = Some(k);
        }
    }

    // got this far, we're sorted
    Ok(true)
}

#[pyfunction]
pub(crate) fn argsort_descending<'py>(
    py: Python<'py>,
    scores: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let scores = make_array(scores.0);
    let array = py.detach(|| {
        let indices = match_array_type!(scores, {
            floating(arr) => argsort_float(arr),
            integer(arr) => argsort_int(arr),
        })?;

        PyResult::Ok(Int32Array::from(indices))
    })?;
    Ok(array.into_data().into())
}

fn argsort_float<T: ArrowPrimitiveType>(scores: &PrimitiveArray<T>) -> Vec<i32>
where
    T::Native: FloatCore,
{
    let sbuf = scores.values();

    let mut indices = Vec::with_capacity(scores.len());
    for (i, v) in scores.iter().enumerate() {
        if let Some(v) = v {
            if !v.is_nan() {
                indices.push(i as i32);
            }
        }
    }

    if scores.len() >= PAR_SORT_THRESHOLD {
        indices.par_sort_unstable_by_key(|i| Reverse(NotNan::new(sbuf[*i as usize]).unwrap()));
    } else {
        indices.sort_unstable_by_key(|i| Reverse(NotNan::new(sbuf[*i as usize]).unwrap()));
    }

    indices
}

fn argsort_int<T: ArrowPrimitiveType>(scores: &PrimitiveArray<T>) -> Vec<i32>
where
    T::Native: Ord,
{
    let sbuf = scores.values();

    let mut indices = Vec::with_capacity(scores.len());
    for (i, v) in scores.iter().enumerate() {
        if let Some(_v) = v {
            indices.push(i as i32);
        }
    }

    if scores.len() >= PAR_SORT_THRESHOLD {
        indices.par_sort_unstable_by_key(|i| Reverse(sbuf[*i as usize]));
    } else {
        indices.sort_unstable_by_key(|i| Reverse(sbuf[*i as usize]));
    }

    indices
}
