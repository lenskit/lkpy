// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::prelude::*;

use arrow::{
    array::{make_array, Array, ArrayData, Float32Array, Int32Array},
    pyarrow::PyArrowType,
};

use crate::{
    arrow::checked_array_ref,
    sparse::{CSRMatrix, CSRStructure, CSR},
};

use super::accum::{collect_items_averaged, collect_items_summed, ScoreAccumulator};

#[pyfunction]
pub fn user_score_items_explicit<'py>(
    tgt_items: PyArrowType<ArrayData>,
    nbr_rows: PyArrowType<ArrayData>,
    nbr_sims: PyArrowType<ArrayData>,
    ratings: PyArrowType<ArrayData>,
    max_nbrs: usize,
    min_nbrs: usize,
) -> PyResult<PyArrowType<ArrayData>> {
    let ratings = make_array(ratings.0);
    let rmat = CSRMatrix::<i32>::from_arrow(ratings)?;

    let tgt_items_ref = make_array(tgt_items.0);
    let tgt_is: &Int32Array = checked_array_ref("target item", "int32", &tgt_items_ref)?;

    let nbr_rows_ref = make_array(nbr_rows.0);
    let nbr_is: &Int32Array = checked_array_ref("neighbor index", "int32", &nbr_rows_ref)?;
    let nbr_sims_ref = make_array(nbr_sims.0);
    let nbr_sims: &Float32Array = checked_array_ref("neighbor sims", "float32", &nbr_sims_ref)?;

    let mut heaps = ScoreAccumulator::new_array(rmat.n_cols, tgt_is);
    let iter = nbr_is.iter().zip(nbr_sims.iter()).flat_map(|ns| match ns {
        (Some(n), Some(s)) => Some((n, s)),
        _ => None,
    });

    for (nbr, sim) in iter {
        let (sp, ep) = rmat.extent(nbr as usize);
        for i in sp..ep {
            let i = i as usize;
            let item = rmat.col_inds.value(i);
            let rating = rmat.values.value(i);
            heaps[item as usize].add_value(max_nbrs, sim, rating)?;
        }
    }

    let out = collect_items_averaged(&heaps, tgt_is, min_nbrs);

    Ok(out.into_data().into())
}

#[pyfunction]
pub fn user_score_items_implicit<'py>(
    tgt_items: PyArrowType<ArrayData>,
    nbr_rows: PyArrowType<ArrayData>,
    nbr_sims: PyArrowType<ArrayData>,
    ratings: PyArrowType<ArrayData>,
    max_nbrs: usize,
    min_nbrs: usize,
) -> PyResult<PyArrowType<ArrayData>> {
    let ratings = make_array(ratings.0);
    let rmat = CSRStructure::<i32>::from_arrow(ratings)?;

    let tgt_items_ref = make_array(tgt_items.0);
    let tgt_is: &Int32Array = checked_array_ref("target item", "int32", &tgt_items_ref)?;

    let nbr_rows_ref = make_array(nbr_rows.0);
    let nbr_is: &Int32Array = checked_array_ref("neighbor index", "int32", &nbr_rows_ref)?;
    let nbr_sims_ref = make_array(nbr_sims.0);
    let nbr_sims: &Float32Array = checked_array_ref("neighbor sims", "float32", &nbr_sims_ref)?;

    let mut heaps = ScoreAccumulator::new_array(rmat.n_cols, tgt_is);
    let iter = nbr_is.iter().zip(nbr_sims.iter()).flat_map(|ns| match ns {
        (Some(n), Some(s)) => Some((n, s)),
        _ => None,
    });

    for (nbr, sim) in iter {
        let (sp, ep) = rmat.extent(nbr as usize);
        for i in sp..ep {
            let i = i as usize;
            let item = rmat.col_inds.value(i);
            heaps[item as usize].add_weight(max_nbrs, sim)?;
        }
    }

    let out = collect_items_summed(&heaps, tgt_is, min_nbrs);

    Ok(out.into_data().into())
}
