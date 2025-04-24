use std::cmp::Reverse;

use arrow::{
    array::{make_array, Array, ArrayData},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sparse::{ArrowCSRConsumer, CSRMatrix};

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    ui_ratings: PyArrowType<ArrayData>,
    iu_ratings: PyArrowType<ArrayData>,
    shape: (usize, usize),
    min_sim: f32,
    save_nbrs: Option<i64>,
    progress: Bound<'py, PyAny>,
) -> PyResult<Vec<PyArrowType<ArrayData>>> {
    let (nu, ni) = shape;
    let progress = if progress.is_none() {
        None
    } else {
        Some(progress.unbind())
    };

    py.allow_threads(|| {
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

        // let's compute!
        let range = 0..ni;
        debug!("computing similarity rows");
        let collector = if let Some(pb) = progress {
            ArrowCSRConsumer::with_progress(ni, pb)
        } else {
            ArrowCSRConsumer::new(ni)
        };
        let chunks = range
            .into_par_iter()
            .map(|row| sim_row(row, &ui_mat, &iu_mat, min_sim, save_nbrs))
            .drive(collector);

        Ok(chunks.iter().map(|a| a.into_data().into()).collect())
    })
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
