// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrowPrimitiveType, Float16Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, PrimitiveArray, RecordBatch, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    },
    pyarrow::PyArrowType,
};
use arrow_schema::DataType;
use ordered_float::{FloatCore, NotNan};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use rayon::slice::ParallelSliceMut;

use crate::types::checked_array;

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
        let col1 = batch
            .column_by_name(c1)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column: {}", c1)))?;
        let col2 = batch
            .column_by_name(c2)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column: {}", c2)))?;

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
    let array = py.allow_threads(|| {
        let indices = match scores.data_type() {
            DataType::Float16 => {
                argsort_float(scores.as_any().downcast_ref::<Float16Array>().unwrap())
            }
            DataType::Float32 => {
                argsort_float(scores.as_any().downcast_ref::<Float32Array>().unwrap())
            }
            DataType::Float64 => {
                argsort_float(scores.as_any().downcast_ref::<Float64Array>().unwrap())
            }
            DataType::Int8 => argsort_int(scores.as_any().downcast_ref::<Int8Array>().unwrap()),
            DataType::Int16 => argsort_int(scores.as_any().downcast_ref::<Int16Array>().unwrap()),
            DataType::Int32 => argsort_int(scores.as_any().downcast_ref::<Int32Array>().unwrap()),
            DataType::Int64 => argsort_int(scores.as_any().downcast_ref::<Int64Array>().unwrap()),
            DataType::UInt8 => argsort_int(scores.as_any().downcast_ref::<UInt8Array>().unwrap()),
            DataType::UInt16 => argsort_int(scores.as_any().downcast_ref::<UInt16Array>().unwrap()),
            DataType::UInt32 => argsort_int(scores.as_any().downcast_ref::<UInt32Array>().unwrap()),
            DataType::UInt64 => argsort_int(scores.as_any().downcast_ref::<UInt64Array>().unwrap()),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported type {}",
                    scores.data_type()
                )))
            }
        };

        Ok(Int32Array::from(indices))
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
