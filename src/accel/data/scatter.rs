// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Array scattering support.
use std::sync::Arc;

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrayRef, ArrowPrimitiveType, OffsetSizeTrait, PrimitiveArray,
    },
    compute::kernels::cast,
    datatypes::ArrowNativeType,
    pyarrow::PyArrowType,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::match_array_type;

#[pyfunction]
pub fn scatter_array(
    dst: PyArrowType<ArrayData>,
    idx: PyArrowType<ArrayData>,
    src: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let dst = make_array(dst.0);
    let idx = make_array(idx.0);
    let src = make_array(src.0);
    let src = cast(&src, dst.data_type())
        .map_err(|e| PyRuntimeError::new_err(format!("error casting src: {:?}", e)))?;
    let arr = match_array_type!(src, {
        numeric(src) => scatter_dst_arr(&dst, idx, src)
    })
    .flatten()?;
    Ok(arr.into_data().into())
}

#[pyfunction]
pub fn scatter_array_empty(
    dst_size: usize,
    idx: PyArrowType<ArrayData>,
    src: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let idx = make_array(idx.0);
    let src = make_array(src.0);
    let arr = match_array_type!(src, {
        numeric(src) => scatter_empty_stype(dst_size, idx, src)
    })
    .flatten()?;
    Ok(arr.into_data().into())
}

fn scatter_dst_arr<T: ArrowPrimitiveType>(
    dst: &dyn Array,
    idx: ArrayRef,
    src: &PrimitiveArray<T>,
) -> PyResult<ArrayRef> {
    let dst = dst.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let mut dst_valid = if let Some(nv) = dst.nulls() {
        nv.into_iter().collect()
    } else {
        vec![true; dst.len()]
    };
    let mut dst = dst.values().to_vec();
    match_array_type!(idx, {
        size(idx) => scatter_impl(&mut dst, &mut dst_valid, idx, src),
    })?;
    let arr = PrimitiveArray::<T>::new(dst.into(), Some(dst_valid.into()));
    Ok(Arc::new(arr))
}

fn scatter_empty_stype<T: ArrowPrimitiveType>(
    dst_size: usize,
    idx: ArrayRef,
    src: &PrimitiveArray<T>,
) -> PyResult<ArrayRef> {
    let mut dst = vec![T::Native::default(); dst_size];
    let mut dst_valid = vec![false; dst_size];
    match_array_type!(idx, {
        size(idx) => scatter_impl(&mut dst, &mut dst_valid, idx, src),
    })?;
    let arr = PrimitiveArray::<T>::new(dst.into(), Some(dst_valid.into()));
    Ok(Arc::new(arr))
}

fn scatter_impl<Ix, T>(
    dst: &mut Vec<T::Native>,
    dst_valid: &mut Vec<bool>,
    idx: &PrimitiveArray<Ix>,
    src: &PrimitiveArray<T>,
) where
    Ix: ArrowPrimitiveType,
    T: ArrowPrimitiveType,
    Ix::Native: OffsetSizeTrait,
{
    for (si, ix) in idx.iter().enumerate() {
        if let Some(i) = ix {
            let i = i.as_usize();
            let valid = src.is_valid(si);
            dst_valid[i] = valid;
            if valid {
                dst[i] = src.value(si);
            }
        }
    }
}
