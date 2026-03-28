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
use ndarray::Array1;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{
    parallel::maybe_fuse,
    progress::ProgressHandle,
    sparse::{ArrowCSRConsumer, CSRStructure, CSR},
};

const EPSILON: f64 = 1.0e-12;
// default value from Karypis code
const OPT_TOLERANCE: f64 = 1e-7;

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
    let chunks = progress.process_iter(py, range.into_par_iter(), |iter| {
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
    fn compute_column(&mut self, item: usize) -> Vec<(i32, f32)> {
        // get the active users for this item — indices where target vector is 1
        let i_users = self.iu_matrix.row_cols(item);

        // initialize our vectors
        let mut weights = Array1::zeros(self.n_items);
        let mut resids = Array1::zeros(self.n_users);

        // since our weights are initialized to zero, residuals are -1 for every user who rated
        // resid: r̂ᵤᵢ - ∑ rᵤⱼwᵢⱼ, but all r̂ᵤᵢ and wᵢⱼ are initially 0
        for i in i_users {
            resids[*i as usize] = -1.0;
        }

        // iteratively apply coordinate descent until we converge
        for iter in 0..self.options.max_iters {
            let max_upd = self.cd_round(&i_users, &mut weights, &mut resids);

            if max_upd <= OPT_TOLERANCE {
                debug!("finished column {} after {} iters", item, iter + 1);
                break;
            }
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
        res.shrink_to_fit();

        // and we're done!
        res
    }

    /// Do one round of coordinate descent, returning the maximum coordinate change.
    fn cd_round(
        &mut self,
        nz_rows: &[i32],
        weights: &mut Array1<f64>,
        resids: &mut Array1<f64>,
    ) -> f64 {
        let mut dmax = 0.0;
        for j in 0..self.n_items {
            let di = self.cd_single(j, nz_rows, weights, resids);
            if di > dmax {
                dmax = di;
            }
        }
        dmax
    }

    /// Do one parameter update of coordinate descent, returning the (absolute) coordinate change.
    fn cd_single(
        &mut self,
        j: usize,
        nz_rows: &[i32],
        weights: &mut Array1<f64>,
        resids: &mut Array1<f64>,
    ) -> f64 {
        let cur_w = weights[j];
        // step 1: *remove* this entry's contribution from residuals
        for u in nz_rows {
            let u = *u as usize;
            // since set ratings are 1, update is simple
            resids[u] += cur_w;
        }

        // step 2: compute update weight value from the users
        let upd: f64 = nz_rows.iter().map(|u| resids[*u as usize]).sum();
        let upd = upd / nz_rows.len() as f64;

        // step 3: update with soft-thresholded weight update
        let new = self.soft_thresh(upd, nz_rows.len() as f64);
        let diff = new - cur_w;
        weights[j] = new;

        // step 4: put *new* weight contribution back into residual
        for u in nz_rows {
            let u = *u as usize;
            resids[u] -= new;
        }

        diff.abs()
    }

    fn soft_thresh(&self, val: f64, norm: f64) -> f64 {
        if val >= self.options.l1_reg {
            let num = val - self.options.l1_reg;
            let den = norm + self.options.l2_reg;
            num / den
        } else {
            0.0
        }
    }
}
