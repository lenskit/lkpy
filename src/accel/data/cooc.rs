// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for counting co-occurrences.

use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{make_array, ArrayData, Int32Array, RecordBatch},
    datatypes::Int32Type,
    pyarrow::PyArrowType,
};
use arrow_schema::{DataType, Field, SchemaBuilder};
use log::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rustc_hash::FxBuildHasher;

use crate::{
    arrow::checked_array_ref,
    progress::ProgressHandle,
    sparse::{COOMatrix, COOMatrixBuilder},
};

/// Count co-occurrances.
#[pyfunction]
pub fn count_cooc<'py>(
    _py: Python<'py>,
    n_items: usize,
    groups: PyArrowType<ArrayData>,
    items: PyArrowType<ArrayData>,
    ordered: bool,
    progress: Bound<'py, PyAny>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let groups = make_array(groups.0);
    let groups = checked_array_ref::<Int32Array>("groups", "Int32", &groups)?;
    let items = make_array(items.0);
    let items = checked_array_ref::<Int32Array>("items", "Int32", &items)?;

    // TODO: parallelize this logic
    let mut counts = PairCountAccumulator::create(n_items);
    let mut cur_group = -1;
    let mut cur_items = Vec::new();

    let pb = ProgressHandle::from_input(progress);
    for (g, i) in groups.iter().zip(items) {
        if let (Some(g), Some(i)) = (g, i) {
            assert!(g >= 0);
            if g != cur_group {
                count_items(&mut counts, &cur_items);
                cur_group = g;
                cur_items.clear();
            }
            cur_items.push(i);
            pb.tick();
        }
    }

    // final count
    count_items(&mut counts, &cur_items);

    // assemble the result
    debug!(
        "assembling arrays for {} co-occurrence counts",
        counts.total
    );

    let out = if ordered {
        counts.finish_ordered()
    } else {
        counts.finish_symmetric()
    };

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
    total: usize,
}

impl PairCountAccumulator {
    fn create(n: usize) -> PairCountAccumulator {
        PairCountAccumulator {
            rows: vec![HashMap::with_hasher(FxBuildHasher); n],
            total: 0,
        }
    }

    fn record(&mut self, row: i32, col: i32) {
        if row < 0 {
            panic!("negative row index")
        }
        let row = &mut self.rows[row as usize];
        *row.entry(col).or_default() += 1;
        self.total += 1;
    }

    fn finish_ordered(self) -> COOMatrix<Int32Type, Int32Type> {
        let mut bld = COOMatrixBuilder::with_capacity(self.total);
        for (i, row) in self.rows.into_iter().enumerate() {
            let ri = i as i32;
            for (ci, count) in row.into_iter() {
                bld.add_entry(ri, ci, count);
            }
        }
        bld.finish()
    }

    fn finish_symmetric(self) -> COOMatrix<Int32Type, Int32Type> {
        let mut bld = COOMatrixBuilder::with_capacity(self.total * 2);
        for (i, row) in self.rows.into_iter().enumerate() {
            let ri = i as i32;
            for (ci, count) in row.into_iter() {
                bld.add_entry(ri, ci, count);
                bld.add_entry(ci, ri, count);
            }
        }
        bld.finish()
    }
}
