// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for counting co-occurrences.

use arrow::{
    array::{Array, ArrayData, Int32Array, make_array},
    datatypes::Int32Type,
    pyarrow::PyArrowType,
};
use log::*;
use numpy::ToPyArray;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    arrow::checked_array,
    data::pairs::{
        AsymmetricPairCounter, ConcurrentPairCounter, DensePairCounter, PairCounter,
        SymmetricPairCounter,
    },
    tasks::{AccelTask, AccelTaskImpl, AtomicCancel},
};

pub(super) struct COOCInput {
    n_groups: usize,
    n_items: usize,
    ordered: bool,
    diagonal: bool,
    groups: Int32Array,
    items: Int32Array,
}

struct SparseCOOCTask {
    input: COOCInput,
}

struct DenseCOOCTask {
    input: COOCInput,
}

/// Count co-occurrances.
#[pyfunction]
#[pyo3(signature=(n_groups, n_items, groups, items, *, ordered=false, diagonal=true))]
pub fn count_cooc(
    n_groups: usize,
    n_items: usize,
    groups: PyArrowType<ArrayData>,
    items: PyArrowType<ArrayData>,
    ordered: bool,
    diagonal: bool,
) -> PyResult<AccelTask> {
    let groups = make_array(groups.0);
    let items = make_array(items.0);
    let nrows = groups.len();
    if items.len() != nrows {
        return Err(PyValueError::new_err("array length mismatch"));
    }

    let groups = checked_array::<Int32Type>("groups", &groups)?;
    let items = checked_array::<Int32Type>("items", &items)?;

    Ok(AccelTask::wrap(SparseCOOCTask {
        input: COOCInput {
            n_groups,
            n_items,
            ordered,
            diagonal,
            groups,
            items,
        },
    }))
}

impl AccelTaskImpl for SparseCOOCTask {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>> {
        let cancel = AtomicCancel::new();
        task.set_cancel(cancel.clone());

        let out = py.detach(|| {
            if self.input.ordered {
                count_cooc_sequential::<AsymmetricPairCounter>(&self.input, &cancel)
            } else {
                let ctr =
                    SymmetricPairCounter::with_diagonal(self.input.n_items, self.input.diagonal);
                count_cooc_parallel(ctr, &self.input, &cancel)
            }
        })?;
        debug!(
            "finished counting {} co-occurrances",
            out.iter().map(|m| m.nnz()).sum::<usize>()
        );

        let mut batches: Vec<PyArrowType<_>> = Vec::with_capacity(out.len());
        for coo in out {
            batches.push(
                coo.record_batch("count")
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("error assembling result array: {}", e))
                    })?
                    .into(),
            );
        }

        batches.into_bound_py_any(py)
    }
}

#[pyfunction]
#[pyo3(signature=(n_groups, n_items, groups, items, *, diagonal=true))]
pub fn dense_cooc(
    n_groups: usize,
    n_items: usize,
    groups: PyArrowType<ArrayData>,
    items: PyArrowType<ArrayData>,
    diagonal: bool,
) -> PyResult<AccelTask> {
    let groups = make_array(groups.0);
    let items = make_array(items.0);
    let nrows = groups.len();
    if items.len() != nrows {
        return Err(PyValueError::new_err("array length mismatch"));
    }

    let groups = checked_array::<Int32Type>("groups", &groups)?;
    let items = checked_array::<Int32Type>("items", &items)?;

    Ok(AccelTask::wrap(DenseCOOCTask {
        input: COOCInput {
            n_groups,
            n_items,
            ordered: false,
            diagonal,
            groups,
            items,
        },
    }))
}

impl AccelTaskImpl for DenseCOOCTask {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>> {
        let cancel = AtomicCancel::new();
        task.set_cancel(cancel.clone());

        let out = py.detach(|| {
            let ctr = DensePairCounter::with_diagonal(self.input.n_items, self.input.diagonal);
            count_cooc_parallel(ctr, &self.input, &cancel)
        })?;
        debug!("finished counting co-occurrances");

        out.to_pyarray(py).into_bound_py_any(py)
    }
}

fn count_cooc_sequential<PC: PairCounter>(
    input: &COOCInput,
    cancel: &AtomicCancel,
) -> PyResult<PC::Output> {
    let gvals = input.groups.values();
    let ivals = input.items.values();

    let g_ptrs = compute_group_pointers(input.n_groups, gvals)?;

    // TODO: parallelize this logic
    debug!("pass 2: counting groups");
    let mut counts = PC::create(input.n_items);
    for i in 0..input.n_groups {
        let start = g_ptrs[i];
        let end = g_ptrs[i + 1];
        let items = &ivals[start..end];
        count_items(&mut counts, items);
        cancel.advance(items.len());
    }

    // assemble the result
    Ok(counts.finish())
}

fn count_cooc_parallel<PC: ConcurrentPairCounter>(
    counts: PC,
    input: &COOCInput,
    cancel: &AtomicCancel,
) -> PyResult<PC::Output> {
    let gvals = input.groups.values();
    let ivals = input.items.values();

    let g_ptrs = compute_group_pointers(input.n_groups, gvals)?;

    debug!("pass 2: counting groups");
    // TODO: fix progress update
    (0..input.n_groups).for_each(|i| {
        let start = g_ptrs[i];
        let end = g_ptrs[i + 1];
        let items = &ivals[start..end];
        let n = items.len();

        for i in 0..n {
            let ri = items[i as usize];
            for j in i..n {
                let ci = items[j as usize];
                counts.crecord(ri, ci);
            }
        }
        cancel.advance(items.len());
    });

    // assemble the result
    Ok(counts.finish())
}

fn compute_group_pointers(n_groups: usize, gvals: &[i32]) -> PyResult<Vec<usize>> {
    debug!("pass 1: counting sizes for {} groups", n_groups);
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
