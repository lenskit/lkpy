// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse Linear Methods for recommendation.
use std::time::Instant;

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{
    progress::ProgressHandle,
    sparse::{ArrowCSRConsumer, CSRStructure, CSR},
};

type FP = f32;
const EPSILON: FP = 1.0e-12;
const OPT_TOLERANCE: FP = 1e-3;

#[derive(Debug, Clone, Copy)]
struct SLIMOptions {
    l1_reg: FP,
    l2_reg: FP,
    max_iters: u32,
    max_nbrs: Option<usize>,
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
    l1_reg: FP,
    l2_reg: FP,
    max_iters: u32,
    max_nbrs: Option<usize>,
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
        max_nbrs,
    };

    debug!("computing similarity rows");
    let collector = ArrowCSRConsumer::with_progress(ui_matrix.n_cols, &progress);

    let range = 0..ui_matrix.n_cols;
    let result = py.allow_threads(|| {
        let chunks = range
            .into_par_iter()
            .map_init(
                || SLIMWorkspace::create(&ui_matrix, &iu_matrix, &options),
                SLIMWorkspace::compute_column,
            )
            .drive(collector);
        chunks.into_iter().map(|a| a.into_data().into()).collect()
    });

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
    #[inline(never)] // a lot of work is in here, let's make profiling easeir
    fn compute_column(&mut self, item: usize) -> Vec<(i32, FP)> {
        let start = Instant::now();
        // get the active users for this item — indices where target vector is 1
        let i_users = self.iu_matrix.row_cols(item);

        // initialize our vectors
        let mut weights = vec![0.0; self.n_items];
        let mut resids = vec![0.0; self.n_users];

        let active = self.prep_resid_and_active(item, &i_users, &mut resids);

        // iteratively apply coordinate descent until we converge
        let n_iters = self.run_cd(item, &mut weights, &mut resids, &active);

        // sparsify weights for final result
        let mut res: Vec<_> = weights
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v >= EPSILON {
                    Some((i as i32, v as FP))
                } else {
                    None
                }
            })
            .collect();
        let ms = Instant::now().duration_since(start).as_secs_f64() * 1000.0;
        debug!(
            "column {} trained with {} neighbors in {} iters ({:.1}ms) with {} nonzero entries",
            item,
            active.len(),
            n_iters,
            ms,
            res.len()
        );
        res.shrink_to_fit();

        // and we're done!
        res
    }

    /// Initialize residuals and compute the list of active items (items whose
    /// coefficient may be nonzero for this target item).
    ///
    /// Active items are determined by traversing the user-item graph, because
    /// an item that is never co-rated with the target item has no basis for a
    /// nonzero coefficient.  If fsSLIM is enabled, residuals are further
    /// limited to the top *k* most-similar items (by cosine similarity).
    #[inline(never)]
    fn prep_resid_and_active(&self, item: usize, i_users: &[i32], resids: &mut [FP]) -> Vec<usize> {
        let mut path_counts = vec![0; self.n_items];
        let mut active = Vec::with_capacity(self.n_items / 4);

        // Residuals are defined by rᵤᵢ - ∑ rᵤⱼwᵢⱼ, but all wᵢⱼ are initially 0,
        // so the residual is just 1 where the user rated the target item and 0
        // elsewhere.
        for u in i_users {
            let u = *u as usize;
            resids[u] = 1.0;
            for j in self.ui_matrix.row_cols(u) {
                let j = *j as usize;
                if j != item {
                    if path_counts[j] == 0 {
                        active.push(j);
                    }
                    path_counts[j] += 1;
                }
            }
        }

        if let Some(k) = self.options.max_nbrs {
            if k < active.len() {
                debug!("limiting column {} to {} active neighbors", item, k);
                // co-rating count is the numerator of cosine, so we just need
                // the denominators to sort the items & pick the top K.
                let i_norm = (i_users.len() as f64).sqrt();
                active.sort_by_key(|j| {
                    let j_norm = (self.iu_matrix.row_nnz(*j) as f64).sqrt();
                    NotNan::new(-(path_counts[*j] as f64) / (i_norm * j_norm))
                        .expect("NaN similarity")
                });
                active.resize(k, 0);
            }
        }

        active
    }

    #[inline(never)]
    fn run_cd(
        &mut self,
        item: usize,
        weights: &mut [FP],
        resids: &mut [FP],
        active: &[usize],
    ) -> u32 {
        let mut n_iters = self.options.max_iters;
        for it in 0..n_iters {
            let max_upd = self.cd_round(weights, resids, active);
            trace!("column {} iter {}: max update {}", item, it + 1, max_upd);

            if max_upd <= OPT_TOLERANCE {
                n_iters = it + 1;
                break;
            }
        }

        n_iters
    }

    /// Do one round of coordinate descent, returning the maximum coordinate change.
    fn cd_round(&mut self, weights: &mut [FP], resids: &mut [FP], active: &[usize]) -> FP {
        let mut dmax = 0.0;
        for j in active {
            let j_users = self.iu_matrix.row_cols(*j);
            let di = self.cd_single(*j, j_users, weights, resids);
            if di > dmax {
                dmax = di;
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
        weights: &mut [FP],
        resids: &mut [FP],
    ) -> FP {
        let cur_w = weights[j];
        // Reg Paths divides by N to get a mean, but Karypis does *not*.

        // step 1: sum the residuals *without* each entry's contribution
        let mut upd = 0.0;
        for u in nz_rows {
            let u = *u as usize;
            upd += resids[u] + cur_w;
        }

        // step 2: update with soft-thresholded weight update
        let new = self.soft_thresh(upd, nz_rows.len() as FP);
        let diff = new - cur_w;
        weights[j] = new;

        // step 3: update residuals
        for u in nz_rows {
            let u = *u as usize;
            resids[u] -= diff;
        }

        diff.abs()
    }

    fn soft_thresh(&self, val: FP, sqnorm: FP) -> FP {
        if val >= self.options.l1_reg {
            let num = val - self.options.l1_reg;
            let den = sqnorm + self.options.l2_reg;
            num / den
        } else {
            0.0
        }
    }
}
