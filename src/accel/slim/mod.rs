// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse Linear Methods for recommendation.

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use log::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{
    parallel::maybe_fuse,
    progress::ProgressHandle,
    sparse::{ArrowCSRConsumer, CSRStructure, CSR},
};

const EPSILON: f64 = 1.0e-12;
const OPT_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct SLIMOptions {
    l1_reg: f64,
    l2_reg: f64,
    max_iters: u32,
}

struct SLIMWorkspace<'a> {
    options: SLIMOptions,
    ui_matrix: &'a CSRStructure<i64>,
    iu_matrix: &'a CSRStructure<i64>,
    n_users: usize,
    n_items: usize,
}

/// Register the lenskit._accel.slim module
pub fn register_slim(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let slim = PyModule::new(parent.py(), "slim")?;
    parent.add_submodule(&slim)?;
    slim.add_function(wrap_pyfunction!(train_slim, &slim)?)?;

    Ok(())
}

/// Learn SLIM regression weights.
///
/// This returns the **transpose** of the weight matrix, for convenient
/// implementation.
#[pyfunction]
fn train_slim<'py>(
    py: Python<'py>,
    ui_matrix: PyArrowType<ArrayData>,
    iu_matrix: PyArrowType<ArrayData>,
    l1_reg: f64,
    l2_reg: f64,
    max_iters: u32,
    progress: Bound<'py, PyAny>,
) -> PyResult<Vec<PyArrowType<ArrayData>>> {
    let ui_matrix = make_array(ui_matrix.0);
    let ui_matrix = CSRStructure::<i64>::from_arrow(ui_matrix)?;
    let iu_matrix = make_array(iu_matrix.0);
    let iu_matrix = CSRStructure::<i64>::from_arrow(iu_matrix)?;

    if ui_matrix.n_rows != iu_matrix.n_cols {
        return Err(PyValueError::new_err("user count mismatch"));
    }
    if ui_matrix.n_cols != iu_matrix.n_rows {
        return Err(PyValueError::new_err("item count mismatch"));
    }
    if ui_matrix.nnz() != iu_matrix.nnz() {
        return Err(PyValueError::new_err("rating count mismatch"));
    }

    let progress = ProgressHandle::from_input(progress);

    let options = SLIMOptions {
        l1_reg,
        l2_reg,
        max_iters,
    };

    debug!("computing similarity rows");
    let collector = ArrowCSRConsumer::new(ui_matrix.n_cols);

    let range = 0..ui_matrix.n_cols;
    let chunks = progress.process_iter(py, range.into_par_iter(), move |iter| {
        let chunks = maybe_fuse(iter)
            .map_init(
                || SLIMWorkspace::create(&ui_matrix, &iu_matrix, &options),
                SLIMWorkspace::compute_column,
            )
            .drive_unindexed(collector);
        Ok(chunks)
    })?;
    let result = py.detach(move || chunks.into_iter().map(|a| a.into_data().into()).collect());

    Ok(result)
}

impl<'a> SLIMWorkspace<'a> {
    fn create(
        ui_matrix: &'a CSRStructure<i64>,
        iu_matrix: &'a CSRStructure<i64>,
        options: &SLIMOptions,
    ) -> Self {
        let n_items = ui_matrix.n_cols;
        let n_users = ui_matrix.n_rows;
        SLIMWorkspace {
            options: *options,
            ui_matrix,
            iu_matrix,
            n_users,
            n_items,
        }
    }

    /// Train a single column of the SLIM weight matrix.
    ///
    /// This code was written from the papers, referencing Karypis's LIBSLIM for
    /// ideas on implementation details.  The relevant LIBSLIM source code
    /// is at https://github.com/KarypisLab/SLIM/tree/master/src/libslim.
    ///
    /// Our implementation differs by working directly with residuals, instead
    /// of working with estimates (yhat).
    ///
    /// **NOTE:** the Regularization Paths paper assumes standardized predictors,
    /// but the Karypis SLIM implementation does *not* center the vectors as near
    /// as I can tell.  We follow the implementation's direction here (which is
    /// also simpler).
    fn compute_column(&mut self, item: usize) -> Vec<(i32, f32)> {
        // get the active users for this item — indices where target vector is 1
        let i_users = self.iu_matrix.row_cols(item);

        // initialize our vectors
        let mut weights = vec![0.0; self.n_items];
        let mut resids = vec![0.0; self.n_users];

        // since our weights are initialized to zero, residuals are -1 for every
        // user who rated. we will aslo pre-compute lists of active items —
        // items that are never co-rated with the target item will have zero
        // weight.
        let mut act_mask = vec![false; self.n_items];
        let mut active = Vec::with_capacity(self.n_items / 4);

        // resid: rᵤᵢ - ∑ rᵤⱼwᵢⱼ, but all wᵢⱼ are initially 0
        for u in i_users {
            let u = *u as usize;
            resids[u] = 1.0;
            for j in self.ui_matrix.row_cols(u) {
                let j = *j as usize;
                if !act_mask[j] {
                    active.push(j);
                    act_mask[j] = true;
                }
            }
        }

        // iteratively apply coordinate descent until we converge
        let mut n_iters = 1;
        while n_iters <= self.options.max_iters {
            let max_upd = self.cd_round(item, &mut weights, &mut resids, &active);
            trace!("column {} iter {}: max update {}", item, n_iters, max_upd);

            if max_upd <= OPT_TOLERANCE {
                break;
            }
            n_iters += 1;
        }

        // sparsify weights for final result
        let mut res: Vec<_> = weights
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v >= EPSILON {
                    Some((i as i32, v as f32))
                } else {
                    None
                }
            })
            .collect();
        debug!(
            "column {}  trained in {} iters with {} nonzero entries",
            item,
            n_iters,
            res.len()
        );
        res.shrink_to_fit();

        // and we're done!
        res
    }

    /// Do one round of coordinate descent, returning the maximum coordinate change.
    fn cd_round(
        &mut self,
        item: usize,
        weights: &mut [f64],
        resids: &mut [f64],
        active: &[usize],
    ) -> f64 {
        let mut dmax = 0.0;
        for j in active {
            if *j != item {
                let j_users = self.iu_matrix.row_cols(*j);
                if !j_users.is_empty() {
                    let di = self.cd_single(*j, j_users, weights, resids);
                    if di > dmax {
                        dmax = di;
                    }
                }
            }
        }
        dmax
    }

    /// Do one parameter update of coordinate descent, returning the (absolute)
    /// coordinate change.
    fn cd_single(
        &mut self,
        j: usize,
        nz_rows: &[i32],
        weights: &mut [f64],
        resids: &mut [f64],
    ) -> f64 {
        let cur_w = weights[j];
        // Reg Paths divides by N to get a mean, but Karypis does *not*.

        // step 1: sum the residuals *without* each entry's contribution
        let mut upd = 0.0;
        for u in nz_rows {
            let u = *u as usize;
            upd += resids[u] + cur_w;
        }

        // step 2: update with soft-thresholded weight update
        let new = self.soft_thresh(upd, nz_rows.len() as f64);
        let diff = new - cur_w;
        weights[j] = new;

        // step 3: update residuals
        for u in nz_rows {
            let u = *u as usize;
            resids[u] -= diff;
        }

        diff.abs()
    }

    fn soft_thresh(&self, val: f64, sqnorm: f64) -> f64 {
        if val >= self.options.l1_reg {
            let num = val - self.options.l1_reg;
            let den = sqnorm + self.options.l2_reg;
            num / den
        } else {
            0.0
        }
    }
}
