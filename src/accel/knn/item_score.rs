use arrow::{
    array::{make_array, Array, ArrayData, Float32Array, Float32Builder, Int32Array},
    pyarrow::PyArrowType,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{sparse::CSRMatrix, types::checked_array_convert};

use super::accum::ScoreAccumulator;

/// Explicit-feedback scoring function.
#[pyfunction]
pub fn score_explicit<'py>(
    py: Python<'py>,
    sims: PyArrowType<ArrayData>,
    ref_items: PyArrowType<ArrayData>,
    ref_rates: PyArrowType<ArrayData>,
    tgt_items: PyArrowType<ArrayData>,
    max_nbrs: usize,
    min_nbrs: usize,
) -> PyResult<PyArrowType<ArrayData>> {
    py.allow_threads(|| {
        let sims = sim_matrix(sims.0)?;
        let ref_items = make_array(ref_items.0);
        let ref_rates = make_array(ref_rates.0);
        let tgt_items = make_array(tgt_items.0);

        let ref_is: &Int32Array = checked_array_convert("reference item", "Int32", &ref_items)?;
        let ref_vs: &Float32Array =
            checked_array_convert("reference ratings", "Float32", &ref_rates)?;
        let tgt_is: &Int32Array = checked_array_convert("target item", "Int32", &tgt_items)?;

        let mut heaps = ScoreAccumulator::new_array(sims.n_cols, tgt_is);

        // we loop reference items, looking for targets.
        // in the common (slow) top-N case, reference items are shorter than targets.
        for (ri, rv) in ref_is.iter().zip(ref_vs.iter()) {
            if let Some(ri) = ri {
                let rv = rv.ok_or_else(|| PyValueError::new_err("reference rating is null"))?;

                let (sp, ep) = sims.extent(ri as usize);
                for i in sp..ep {
                    let i = i as usize;
                    let ti = sims.col_inds.value(i);
                    let sim = sims.values.value(i);

                    // get the heap, initializing if needed.
                    let acc = &mut heaps[ti as usize];
                    acc.add_value(max_nbrs, sim, rv)?;
                }
            }
        }

        let mut out = Float32Builder::with_capacity(tgt_items.len());
        for ti in tgt_is {
            if let Some(ti) = ti {
                let acc = &heaps[ti as usize];
                if acc.len() >= min_nbrs {
                    let score = acc.weighted_sum() / acc.total_weight();
                    out.append_value(score);
                } else {
                    out.append_null();
                }
            } else {
                out.append_null();
            }
        }
        let out = out.finish();
        assert_eq!(out.len(), tgt_is.len());

        Ok(out.into_data().into())
    })
}

/// Implicit-feedback scoring function.
#[pyfunction]
pub fn score_implicit<'py>(
    py: Python<'py>,
    sims: PyArrowType<ArrayData>,
    ref_items: PyArrowType<ArrayData>,
    tgt_items: PyArrowType<ArrayData>,
    max_nbrs: usize,
    min_nbrs: usize,
) -> PyResult<PyArrowType<ArrayData>> {
    py.allow_threads(|| {
        let sims = sim_matrix(sims.0)?;
        let ref_items = make_array(ref_items.0);
        let tgt_items = make_array(tgt_items.0);

        let ref_is: &Int32Array = checked_array_convert("reference item", "Int32", &ref_items)?;
        let tgt_is: &Int32Array = checked_array_convert("target item", "Int32", &tgt_items)?;

        let mut heaps = ScoreAccumulator::new_array(sims.n_cols, tgt_is);

        // we loop reference items, looking for targets.
        // in the common (slow) top-N case, reference items are shorter than targets.
        for ref_item in ref_is {
            if let Some(ri) = ref_item {
                let (sp, ep) = sims.extent(ri as usize);
                for i in sp..ep {
                    let i = i as usize;
                    let ti = sims.col_inds.value(i);
                    let sim = sims.values.value(i);

                    // get the heap, initializing if needed.
                    let acc = &mut heaps[ti as usize];
                    acc.add_weight(max_nbrs, sim)?;
                }
            }
        }

        let mut out = Float32Builder::with_capacity(tgt_items.len());
        for ti in tgt_is {
            if let Some(ti) = ti {
                let acc = &heaps[ti as usize];
                if acc.len() >= min_nbrs {
                    let score = acc.total_weight();
                    out.append_value(score);
                } else {
                    out.append_null();
                }
            } else {
                out.append_null();
            }
        }
        let out = out.finish();
        assert_eq!(out.len(), tgt_is.len());

        Ok(out.into_data().into())
    })
}

fn sim_matrix(sims: ArrayData) -> PyResult<CSRMatrix<i64>> {
    let array = make_array(sims);
    let csr = CSRMatrix::from_arrow(array)?;
    assert_eq!(csr.n_rows, csr.n_cols);
    Ok(csr)
}
