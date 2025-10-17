// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Sparse Linear Methods for recommendation.

use log::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use ndarray::Array1;

use crate::{
    parallel::maybe_fuse,
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
    weights: Array1<f64>,
    estimates: Array1<f64>,
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

    let progress = if progress.is_none() {
        None
    } else {
        Some(progress.unbind())
    };
    let options = SLIMOptions {
        l1_reg,
        l2_reg,
        max_iters,
    };

    debug!("computing similarity rows");
    let collector = if let Some(pb) = progress {
        ArrowCSRConsumer::with_progress(ui_matrix.n_cols, pb)
    } else {
        ArrowCSRConsumer::new(ui_matrix.n_cols)
    };

    let result = py.allow_threads(move || {
        let range = 0..ui_matrix.n_cols;
        let chunks = maybe_fuse(range.into_par_iter())
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
            weights: Array1::zeros(n_items),
            estimates: Array1::zeros(n_users),
        }
    }

    /// Train a single column of the SLIM weight matrix.
    ///
    /// This code was written from the papers, referencing Karypis's LIBSLIM for
    /// ideas on implementation details.  The relevant LIBSLIM source code
    /// is at https://github.com/KarypisLab/SLIM/tree/master/src/libslim.
    fn compute_column(&mut self, item: usize) -> Vec<(i32, f32)> {
        // get the active users for this item
        let i_users = self.iu_matrix.row_cols(item);
        // since it's all 1s, the length of active entries is the squared norm
        let sq_cnorm = i_users.len() as f64;

        for iter in 0..self.options.max_iters {
            let mut sqdelta = 0.0;
            // coordinate descent - loop over items, learn that row in the weight vector
            for i in 0..self.n_items {
                let old_w = self.weights[i];
                // subtract this item's contribution to the estimate
                if old_w > 0.0 {
                    for c in i_users {
                        self.estimates[*c as usize] -= old_w
                    }
                }

                // compute the update value - sum errors where user is active (so rating is 1)
                let mut update = 0.0;
                for u in i_users {
                    let u = *u as usize;
                    update += 1.0 - self.estimates[u];
                }
                // convert to mean
                update /= self.n_users as f64;
                // soft-threshold and adjust
                let new = if update >= self.options.l1_reg {
                    let num = update - self.options.l1_reg;
                    num / (sq_cnorm - self.options.l2_reg)
                } else {
                    0.0
                };
                let delta = new - old_w;
                sqdelta += delta * delta;
                self.weights[i] = new;
                if new > 0.0 {
                    for c in i_users {
                        self.estimates[*c as usize] += new
                    }
                }
            }
            if sqdelta <= OPT_TOLERANCE {
                debug!("finished column {} after {} iters", item, iter + 1);
                break;
            }
        }

        // sparsify weights for final result
        let res: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if *v >= EPSILON {
                    Some((i as i32, *v as f32))
                } else {
                    None
                }
            })
            .collect();
        // reset workspace
        for u in i_users {
            self.estimates[*u as usize] = 0.0;
        }
        self.weights.fill(0.0);

        // and we're done!
        res
    }
}
