// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for counting co-occurrences.

use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{make_array, ArrayData, Int32Array, Int32Builder, RecordBatch},
    pyarrow::PyArrowType,
};
use arrow_schema::{DataType, Field, SchemaBuilder};
use log::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rustc_hash::FxBuildHasher;

use crate::{arrow::checked_array_ref, progress::ProgressHandle};

/// Count co-occurrances.
#[pyfunction]
pub fn count_cooc<'py>(
    _py: Python<'py>,
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
    let mut counts = HashMap::with_hasher(FxBuildHasher);
    let mut cur_group = -1;
    let mut cur_items = Vec::new();

    let pb = ProgressHandle::from_input(progress);
    for (g, i) in groups.iter().zip(items) {
        if let (Some(g), Some(i)) = (g, i) {
            assert!(g >= 0);
            if g != cur_group {
                count_items(&mut counts, &cur_items, ordered);
                cur_group = g;
                cur_items.clear();
            }
            cur_items.push(i);
            pb.tick();
        }
    }

    // final count
    count_items(&mut counts, &cur_items, ordered);

    // assemble the result
    debug!(
        "assembling arrays for {} co-occurrence counts",
        counts.len()
    );
    let mut out_rows = Int32Builder::with_capacity(counts.len());
    let mut out_cols = Int32Builder::with_capacity(counts.len());
    let mut out_counts = Int32Builder::with_capacity(counts.len());

    for ((i1, i2), c) in counts.into_iter() {
        out_rows.append_value(i1);
        out_cols.append_value(i2);
        out_counts.append_value(c);
    }

    let mut schema = SchemaBuilder::new();
    schema.push(Field::new("row", DataType::Int32, false));
    schema.push(Field::new("col", DataType::Int32, false));
    schema.push(Field::new("count", DataType::Int32, false));
    let schema = schema.finish();
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(out_rows.finish()),
            Arc::new(out_cols.finish()),
            Arc::new(out_counts.finish()),
        ],
    )
    .map_err(|e| PyRuntimeError::new_err(format!("error assembling result array: {}", e)))?;

    Ok(batch.into())
}

fn count_items(counts: &mut HashMap<(i32, i32), i32, FxBuildHasher>, items: &[i32], ordered: bool) {
    let n = items.len();
    for i in 0..n {
        let start = if ordered { i + 1 } else { 0 };
        if start < n {
            for j in start..n {
                if i != j {
                    let ri = items[i as usize];
                    let ci = items[j as usize];
                    *counts.entry((ri, ci)).or_default() += 1;
                }
            }
        }
    }
}
