// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::{prelude::*, IntoPyObjectExt};
use rayon::prelude::*;
use rayon_cancel::CancelAdapter;

use crate::{
    sparse::{ArrowCSRConsumer, CSRMatrix, CSR},
    tasks::{AccelTask, AccelTaskImpl, IterCancel},
};

struct ItemSimTask {
    ui_ratings: CSRMatrix,
    iu_ratings: CSRMatrix,
    n_items: usize,
    min_sim: f32,
    save_nbrs: Option<i64>,
}

#[pyfunction]
pub fn compute_similarities<'py>(
    ui_ratings: PyArrowType<ArrayData>,
    iu_ratings: PyArrowType<ArrayData>,
    shape: (usize, usize),
    min_sim: f32,
    save_nbrs: Option<i64>,
) -> PyResult<AccelTask> {
    let (nu, ni) = shape;

    // extract the data
    debug!("preparing {}x{} training", nu, ni);
    debug!(
        "resolving user-item matrix (type: {:#?})",
        ui_ratings.0.data_type()
    );
    let ui_mat = CSRMatrix::from_arrow(make_array(ui_ratings.0))?;
    debug!("resolving item-user matrix");
    let iu_mat = CSRMatrix::from_arrow(make_array(iu_ratings.0))?;
    assert_eq!(ui_mat.len(), nu);
    assert_eq!(ui_mat.n_cols, ni);
    assert_eq!(iu_mat.len(), ni);
    assert_eq!(iu_mat.n_cols, nu);

    Ok(AccelTask::wrap(ItemSimTask {
        ui_ratings: ui_mat,
        iu_ratings: iu_mat,
        n_items: ni,
        min_sim,
        save_nbrs,
    }))
}

impl AccelTaskImpl for ItemSimTask {
    fn invoke<'py>(&self, py: Python<'py>, task: &AccelTask) -> PyResult<Bound<'py, PyAny>> {
        // let's compute!
        debug!("computing similarity rows");
        let collector = ArrowCSRConsumer::new(self.n_items);

        let iter = (0..self.n_items).into_par_iter();
        let adapter = CancelAdapter::new(iter);
        task.set_cancel(IterCancel::from_adapter(&adapter));

        let chunks = py.detach(move || {
            adapter
                .map(|row| {
                    sim_row(
                        row,
                        &self.ui_ratings,
                        &self.iu_ratings,
                        self.min_sim,
                        self.save_nbrs,
                    )
                })
                .drive_unindexed(collector)
        });

        let chunks: Vec<PyArrowType<ArrayData>> =
            chunks.iter().map(|a| a.into_data().into()).collect();
        chunks.into_bound_py_any(py)
    }
}

fn sim_row(
    row: usize,
    ui_mat: &CSRMatrix,
    iu_mat: &CSRMatrix,
    min_sim: f32,
    save_nbrs: Option<i64>,
) -> Vec<(i32, f32)> {
    let (r_start, r_end) = iu_mat.extent(row);

    // accumulate count and inner products
    let mut counts = vec![0; ui_mat.n_cols];
    let mut dots = vec![0.0f32; ui_mat.n_cols];

    // track output slots in use
    let mut used = Vec::new();

    // loop over the users
    for i in r_start..r_end {
        let u = iu_mat.col_inds.value(i as usize);
        let r = iu_mat.values.value(i as usize);

        let (u_start, u_end) = ui_mat.extent(u as usize);
        // loop over the users' items
        for j in u_start..u_end {
            let j = j as usize;
            let other = ui_mat.col_inds.value(j) as usize;
            if other == row {
                continue;
            }
            let orate = ui_mat.values.value(j);
            if counts[other] == 0 {
                used.push(other);
            }
            counts[other] += 1;
            dots[other] += r * orate;
        }
    }

    // finish up and return!
    let mut sims: Vec<_> = used
        .into_iter()
        .filter(|i| dots[*i] >= min_sim)
        .map(|i| (i as i32, dots[i]))
        .collect();

    // truncate if needed
    if let Some(limit) = save_nbrs {
        if limit > 0 {
            // sort by value number
            sims.sort_by_key(|(_i, s)| Reverse(NotNan::new(*s).unwrap()));
            sims.truncate(limit as usize);
            sims.shrink_to_fit();
        }
    }
    // sort by column number
    sims.sort_by_key(|(i, _s)| *i);

    sims
}
