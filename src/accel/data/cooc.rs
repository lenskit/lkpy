// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for counting co-occurrences.

use std::sync::Arc;

use arrow::{
    array::{make_array, Array, ArrayData, Int32Array, RecordBatch},
    datatypes::Int32Type,
    pyarrow::PyArrowType,
};
use arrow_schema::{DataType, Field, SchemaBuilder};
use log::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    arrow::checked_array_ref,
    data::pairs::{
        AsymmetricPairCounter, ConcurrentPairCounter, PairCounter, SymmetricPairCounter,
    },
    progress::ProgressHandle,
    sparse::COOMatrix,
};

/// Count co-occurrances.
#[pyfunction]
pub fn count_cooc<'py>(
    py: Python<'py>,
    n_groups: usize,
    n_items: usize,
    groups: PyArrowType<ArrayData>,
    items: PyArrowType<ArrayData>,
    ordered: bool,
    progress: Bound<'py, PyAny>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let groups = make_array(groups.0);
    let items = make_array(items.0);
    let nrows = groups.len();
    if items.len() != nrows {
        return Err(PyValueError::new_err("array length mismatch"));
    }

    let pb = ProgressHandle::from_input(progress);

    let out = py.detach(move || {
        let groups = checked_array_ref::<Int32Array>("groups", "Int32", &groups)?;
        let items = checked_array_ref::<Int32Array>("items", "Int32", &items)?;

        if ordered {
            count_cooc_sequential::<AsymmetricPairCounter>(&pb, groups, items, n_groups, n_items)
        } else {
            count_cooc_parallel::<SymmetricPairCounter>(&pb, groups, items, n_groups, n_items)
        }
    })?;

    let mut schema = SchemaBuilder::new();
    schema.push(Field::new("row", DataType::Int32, false));
    schema.push(Field::new("col", DataType::Int32, false));
    schema.push(Field::new("count", DataType::Int32, false));
    let schema = schema.finish();
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(out.row), Arc::new(out.col), Arc::new(out.val)],
    )
    .map_err(|e| PyRuntimeError::new_err(format!("error assembling result array: {}", e)))?;

    Ok(batch.into())
}

fn count_cooc_sequential<PC: PairCounter>(
    pb: &ProgressHandle,
    groups: &Int32Array,
    items: &Int32Array,
    n_groups: usize,
    n_items: usize,
) -> PyResult<COOMatrix<Int32Type, Int32Type>> {
    let gvals = groups.values();
    let ivals = items.values();

    let g_ptrs = compute_group_pointers(n_groups, gvals)?;

    // TODO: parallelize this logic
    debug!("pass 2: counting groups");
    let mut counts = PC::create(n_items);
    for i in 0..n_groups {
        let start = g_ptrs[i];
        let end = g_ptrs[i + 1];
        let items = &ivals[start..end];
        count_items(&mut counts, items);
        pb.advance(items.len());
    }
    pb.flush();

    // assemble the result
    Ok(counts.finish())
}

fn count_cooc_parallel<PC: ConcurrentPairCounter>(
    pb: &ProgressHandle,
    groups: &Int32Array,
    items: &Int32Array,
    n_groups: usize,
    n_items: usize,
) -> PyResult<COOMatrix<Int32Type, Int32Type>> {
    let gvals = groups.values();
    let ivals = items.values();

    let g_ptrs = compute_group_pointers(n_groups, gvals)?;

    debug!("pass 2: counting groups");
    let counts = PC::create(n_items);
    (0..n_groups).into_par_iter().for_each_init(
        || pb.thread_local(),
        |lpb, i| {
            let start = g_ptrs[i];
            let end = g_ptrs[i + 1];
            let items = &ivals[start..end];
            let n = items.len();

            for i in 0..n {
                let ri = items[i as usize];
                for j in (i + 1)..n {
                    if i != j {
                        let ci = items[j as usize];
                        counts.crecord(ri, ci);
                    }
                }
            }

            lpb.advance(items.len());
        },
    );
    pb.flush();

    // assemble the result
    Ok(counts.finish())
}

fn compute_group_pointers(n_groups: usize, gvals: &[i32]) -> PyResult<Vec<usize>> {
    debug!("pass 1: counting group sizes");
    let mut group_sizes = vec![0; n_groups];
    let mut last = None;
    for g in gvals {
        if let Some(lg) = last {
            assert!(*g >= lg)
        };
        last = Some(*g);
        group_sizes[*g as usize] += 1;
    }

    // convert to row pointers
    debug!("pass 1.5: converting group sizes to pointers");
    let mut g_ptrs = Vec::with_capacity(n_groups + 1);
    g_ptrs.push(0);
    for i in 0..n_groups {
        g_ptrs.push(g_ptrs[i] + group_sizes[i]);
    }
    Ok(g_ptrs)
}

fn count_items<PC: PairCounter>(counts: &mut PC, items: &[i32]) {
    let n = items.len();
    for i in 0..n {
        let ri = items[i as usize];
        for j in (i + 1)..n {
            if i != j {
                let ci = items[j as usize];
                counts.record(ri, ci);
            }
        }
    }
}
