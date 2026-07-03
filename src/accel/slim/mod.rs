// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse Linear Methods for recommendation.
use std::time::Instant;

use arrow::{
    array::{Array, ArrayData, make_array},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::{IntoPyObjectExt, exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use rayon_cancel::CancelAdapter;

use crate::{
    parallel::maybe_fuse,
    sparse::{ArrowCSRConsumer, CSR, CSRStructure},
    tasks::{AccelTask, AccelTaskImpl, IterCancel},
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

struct SLIMTask {
    options: SLIMOptions,
    ui_matrix: CSRStructure<i64>,
    iu_matrix: CSRStructure<i64>,
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
    ui_matrix: PyArrowType<ArrayData>,
    iu_matrix: PyArrowType<ArrayData>,
    l1_reg: FP,
    l2_reg: FP,
    max_iters: u32,
    max_nbrs: Option<usize>,
) -> PyResult<AccelTask> {
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

    let options = SLIMOptions {
        l1_reg,
        l2_reg,
        max_iters,
        max_nbrs,
    };

    Ok(AccelTask::wrap(SLIMTask {
        options,
        ui_matrix,
        iu_matrix,
    }))
}

impl AccelTaskImpl for SLIMTask {
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        task: &crate::tasks::AccelTask,
    ) -> PyResult<Bound<'py, PyAny>> {
        debug!("computing similarity rows");
        let n_items = self.ui_matrix.n_cols;
        let collector = ArrowCSRConsumer::new(n_items);

        let range = 0..n_items;
        let adapter = CancelAdapter::new(range.into_par_iter());
        task.set_cancel(IterCancel::from_adapter(&adapter));

        let chunks = py.detach(move || {
            let chunks = maybe_fuse(adapter)
                .map(|i| self.compute_column(i))
                .drive_unindexed(collector);
            chunks
        });
        let result: Vec<_> = py.detach(move || {
            chunks
                .into_iter()
                .map(|a| PyArrowType::from(a.into_data()))
                .collect()
        });
        result.into_bound_py_any(py)
    }
}

impl SLIMTask {
    fn n_items(&self) -> usize {
        self.ui_matrix.n_cols
    }

    fn n_users(&self) -> usize {
        self.ui_matrix.n_rows
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
    fn compute_column(&self, item: usize) -> Vec<(i32, FP)> {
        let start = Instant::now();
        // get the active users for this item — indices where target vector is 1
        let i_users = self.iu_matrix.row_cols(item);

        // initialize our vectors
        let mut weights = vec![0.0; self.n_items()];
        let mut resids = vec![0.0; self.n_users()];

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
        let mut path_counts = vec![0; self.n_items()];
        let mut active = Vec::with_capacity(self.n_items() / 4);

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
    fn run_cd(&self, item: usize, weights: &mut [FP], resids: &mut [FP], active: &[usize]) -> u32 {
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
    fn cd_round(&self, weights: &mut [FP], resids: &mut [FP], active: &[usize]) -> FP {
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
    fn cd_single(&self, j: usize, nz_rows: &[i32], weights: &mut [FP], resids: &mut [FP]) -> FP {
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
