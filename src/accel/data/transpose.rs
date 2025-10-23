// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::{fmt::Debug, ops::AddAssign};

use arrow::{
    array::{make_array, Array, ArrayData, ArrowPrimitiveType, OffsetSizeTrait, PrimitiveArray},
    datatypes::{Int32Type, Int64Type},
    pyarrow::PyArrowType,
};
use pyo3::prelude::*;

use crate::sparse::{csr_structure, CSRStructure, IxVar, CSR};

/// Transpose the structure of a CSR matrix.
#[pyfunction]
pub fn transpose_csr(
    arr: PyArrowType<ArrayData>,
    permute: bool,
) -> PyResult<(
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
    Option<PyArrowType<ArrayData>>,
)> {
    let array = make_array(arr.0);
    let csr = csr_structure(array)?;
    let (row_ptrs, col_inds, permutation) = match csr {
        IxVar::Ix32(csr) => transpose_structure::<Int32Type>(csr, permute),
        IxVar::Ix64(csr) => transpose_structure::<Int64Type>(csr, permute),
    };
    Ok((
        row_ptrs.into(),
        col_inds.into(),
        permutation.map(|p| p.into()),
    ))
}

/// Transpose the matrix structure.
fn transpose_structure<It>(
    csr: CSRStructure<It::Native>,
    permute: bool,
) -> (ArrayData, ArrayData, Option<ArrayData>)
where
    It: ArrowPrimitiveType,
    It::Native: OffsetSizeTrait
        + TryInto<usize, Error: Debug>
        + From<i32>
        + TryFrom<usize, Error: Debug>
        + AddAssign<It::Native>,
    PrimitiveArray<It>: From<Vec<It::Native>>,
{
    let nnz = csr.nnz();
    let mut row_ptrs = Vec::with_capacity(csr.n_cols + 1);
    row_ptrs.resize(csr.n_cols + 1, It::Native::from(0));
    let mut col_inds = Vec::with_capacity(nnz);
    col_inds.resize(nnz, 0);
    let mut permutation = if permute {
        let mut p = Vec::with_capacity(nnz);
        p.resize(nnz, It::Native::from(0));
        Some(p)
    } else {
        None
    };

    // step 1: count column values, placing counts in rps[c+1].
    for c in csr.col_inds().values() {
        let c = *c as usize;
        row_ptrs[c + 1] += It::Native::from(1);
    }

    // step 2: convert column counts into row offsets
    for i in 1..=csr.n_cols {
        let prev = row_ptrs[i - 1];
        row_ptrs[i] += prev;
    }

    // step 3: insert column indices and order indices into outputs
    let mut row_ips = row_ptrs.clone();
    let mut i = 0;
    let cols = csr.col_inds.values();
    for row in 0..csr.n_rows {
        let (sp, ep) = csr.extent(row);
        for ci in sp..ep {
            let cv = cols[ci] as usize;
            let pos: usize = row_ips[cv].try_into().expect("unexpected size error");
            col_inds[pos] = row as i32;
            if let Some(perm) = &mut permutation {
                perm[pos] = i.try_into().expect("unexpected size error");
            }
            row_ips[cv] += It::Native::from(1);
            i += 1;
        }
    }

    // now we're done, and the result is transposed!
    let rp_arr: PrimitiveArray<It> = row_ptrs.into();
    let ci_arr: PrimitiveArray<Int32Type> = col_inds.into();
    let perm_arr: Option<PrimitiveArray<It>> = permutation.map(|p| p.into());

    (
        rp_arr.into_data(),
        ci_arr.into_data(),
        perm_arr.map(|p| p.into_data()),
    )
}
