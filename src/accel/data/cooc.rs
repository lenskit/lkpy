// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for counting co-occurrences.

use std::{collections::HashMap, sync::Arc};

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
use rustc_hash::FxBuildHasher;

use crate::{
    arrow::checked_array_ref,
    progress::ProgressHandle,
    sparse::{COOMatrix, COOMatrixBuilder},
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
        let mut counts = PairCountAccumulator::create(n_items, ordered);
        really_count_cooc(&mut counts, &pb, groups, items, n_groups)?;
        // assemble the result
        debug!(
            "assembling arrays for {} co-occurrence counts",
            counts.total
        );
        PyResult::Ok(counts.finish())
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

fn really_count_cooc(
    counts: &mut PairCountAccumulator,
    pb: &ProgressHandle,
    groups: Arc<dyn Array>,
    items: Arc<dyn Array>,
    n_groups: usize,
) -> PyResult<()> {
    let groups = checked_array_ref::<Int32Array>("groups", "Int32", &groups)?;
    let items = checked_array_ref::<Int32Array>("items", "Int32", &items)?;
    let gvals = groups.values();
    let ivals = items.values();

    // compute group sizes for CSR (input is sorted by group)
    let g_ptrs = compute_group_pointers(n_groups, gvals)?;

    // TODO: parallelize this logic
    debug!("pass 2: counting groups");
    for i in 0..n_groups {
        let start = g_ptrs[i];
        let end = g_ptrs[i + 1];
        let items = &ivals[start..end];
        count_items(counts, items);
        pb.advance(items.len());
    }

    Ok(())
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

fn count_items(counts: &mut PairCountAccumulator, items: &[i32]) {
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

/// Accumulator for counts.
struct PairCountAccumulator {
    rows: Vec<HashMap<i32, i32, FxBuildHasher>>,
    ordered: bool,
    total: usize,
}

impl PairCountAccumulator {
    fn create(n: usize, ordered: bool) -> PairCountAccumulator {
        PairCountAccumulator {
            rows: vec![HashMap::with_hasher(FxBuildHasher); n],
            ordered,
            total: 0,
        }
    }

    fn record(&mut self, row: i32, col: i32) {
        if row < 0 {
            panic!("negative row index")
        }
        if col < 0 {
            panic!("negative column index")
        }
        if self.ordered || row <= col {
            self._record_entry(row, col);
        } else {
            self._record_entry(col, row);
        }
    }

    fn _record_entry(&mut self, row: i32, col: i32) {
        let row = &mut self.rows[row as usize];
        *row.entry(col).or_default() += 1;
        self.total += 1;
    }

    fn finish(self) -> COOMatrix<Int32Type, Int32Type> {
        let size = if self.ordered {
            self.total
        } else {
            self.total * 2
        };
        let mut bld = COOMatrixBuilder::with_capacity(size);
        for (i, row) in self.rows.into_iter().enumerate() {
            let ri = i as i32;
            for (ci, count) in row.into_iter() {
                bld.add_entry(ri, ci, count);
                if !self.ordered {
                    bld.add_entry(ci, ri, count);
                }
            }
        }
        bld.finish()
    }
}
