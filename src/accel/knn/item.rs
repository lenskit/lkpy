use std::sync::Arc;
use std::{cmp::Reverse, collections::BinaryHeap};

use arrow::{
    array::{
        make_array, Array, ArrayData, Float32Array, Float32Builder, Int32Array, Int32Builder,
        LargeListArray, StructArray,
    },
    buffer::OffsetBuffer,
    datatypes::{DataType, Field, Fields},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{sparse::CSRMatrix, types::checked_array_convert};

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    ui_ratings: PyArrowType<ArrayData>,
    iu_ratings: PyArrowType<ArrayData>,
    shape: (usize, usize),
    min_sim: f32,
    save_nbrs: Option<i64>,
) -> PyResult<Vec<PyArrowType<ArrayData>>> {
    let (nu, ni) = shape;

    py.allow_threads(|| {
        // extract the data
        debug!("preparing {}x{} matrix", nu, ni);
        let ui_mat = CSRMatrix::from_arrow(make_array(ui_ratings.0), nu, ni)?;
        let iu_mat = CSRMatrix::from_arrow(make_array(iu_ratings.0), ni, nu)?;
        assert_eq!(ui_mat.array.len(), nu);
        assert_eq!(iu_mat.array.len(), ni);

        // let's compute!
        let range = 0..ni;
        debug!("computing similarity rows");
        let chunks = range
            .into_par_iter()
            .map(|row| sim_row(row, &ui_mat, &iu_mat, min_sim, save_nbrs))
            .collect_vec_list();
        // count the similarities
        let n_sim = chunks
            .iter()
            .flat_map(|v| v.iter().map(|v2| v2.len()))
            .sum::<usize>();
        debug!(
            "computed {} similarities in {} matrix chunks",
            n_sim,
            chunks.len()
        );

        // now we will post-process the chunks into Arrow arrays, one per chunk.
        // on the Python side, we will combine the chunks.
        let mut rv_chunks = Vec::new();
        for chunk in chunks {
            let lengths: Vec<usize> = chunk.iter().map(Vec::len).collect();
            let n_entries = lengths.iter().sum();
            let mut col_bld = Int32Builder::with_capacity(n_entries);
            let mut val_bld = Float32Builder::with_capacity(n_entries);

            for row in chunk.into_iter() {
                for (i, s) in row {
                    col_bld.append_value(i);
                    val_bld.append_value(s);
                }
            }

            let struct_fields = Fields::from(vec![
                Field::new("index", DataType::Int32, false),
                Field::new("value", DataType::Float32, false),
            ]);
            let list_field = Field::new("rows", DataType::Struct(struct_fields.clone()), false);
            let sa = StructArray::new(
                struct_fields,
                vec![Arc::new(col_bld.finish()), Arc::new(val_bld.finish())],
                None,
            );
            let list = LargeListArray::new(
                Arc::new(list_field),
                OffsetBuffer::from_lengths(lengths),
                Arc::new(sa),
                None,
            );
            rv_chunks.push(list.into_data().into());
        }

        Ok(rv_chunks)
    })
}

/// Entries in the accumulator heaps.
#[derive(Debug, Default, Clone)]
struct AccEntry<T> {
    weight: NotNan<f32>,
    data: T,
}

impl AccEntry<()> {
    fn weight_only(weight: f32) -> PyResult<AccEntry<()>> {
        Self::new(weight, ())
    }
}

impl<T> AccEntry<T> {
    fn new(weight: f32, payload: T) -> PyResult<AccEntry<T>> {
        Ok(AccEntry {
            weight: NotNan::new(weight)
                .map_err(|_e| PyValueError::new_err("similarity is null"))?,
            data: payload,
        })
    }

    fn get_weight(&self) -> f32 {
        self.weight.into_inner()
    }
}

impl<T> PartialEq for AccEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<T> Eq for AccEntry<T> {}

impl<T> PartialOrd for AccEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // reverse the ordering to make a min-heap
        other.weight.partial_cmp(&self.weight)
    }
}

impl<T> Ord for AccEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reverse the ordering to make a min-heap
        other.weight.cmp(&self.weight)
    }
}

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

        let mut heaps: Vec<Option<BinaryHeap<AccEntry<f32>>>> = vec![None; sims.n_cols];

        // we loop reference items, looking for targets.
        // in the common (slow) top-N case, reference items are shorter than targets.
        for (ri, rv) in ref_is.iter().zip(ref_vs.iter()) {
            let ri = ri.ok_or_else(|| PyValueError::new_err("reference item is null"))?;
            let rv = rv.ok_or_else(|| PyValueError::new_err("reference rating is null"))?;

            let (sp, ep) = sims.extent(ri as usize);
            for i in sp..ep {
                let i = i as usize;
                let ti = sims.col_inds.value(i);
                let sim = sims.values.value(i);

                // get the heap, initializing if needed.
                let heap = &mut heaps[ti as usize];
                if heap.is_none() {
                    *heap = Some(BinaryHeap::with_capacity(max_nbrs as usize + 1));
                }
                // add the item to the heap.
                let heap = heap.as_mut().unwrap();
                heap.push(AccEntry::new(sim, rv)?);
                if heap.len() > max_nbrs {
                    heap.pop();
                }
            }
        }

        let mut out = Float32Builder::with_capacity(tgt_items.len());
        for ti in tgt_is {
            let ti = ti.ok_or_else(|| PyValueError::new_err("target item is null"))? as usize;
            let heap = heaps[ti].take().filter(|h| h.len() >= min_nbrs);
            if let Some(heap) = heap {
                let mut sum = 0.0;
                let mut weight = 0.0;
                for a in heap {
                    sum += a.weight * a.data;
                    weight += a.weight.into_inner();
                }
                let score: f32 = sum / weight;
                out.append_value(score);
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

        let mut heaps: Vec<Option<BinaryHeap<AccEntry<()>>>> = vec![None; sims.n_cols];

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
                    let heap = &mut heaps[ti as usize];
                    if heap.is_none() {
                        *heap = Some(BinaryHeap::with_capacity(max_nbrs as usize + 1));
                    }
                    // add the item to the heap.
                    let heap = heap.as_mut().unwrap();
                    heap.push(AccEntry::weight_only(sim)?);
                    if heap.len() > max_nbrs {
                        heap.pop();
                    }
                }
            }
        }

        let mut out = Float32Builder::with_capacity(tgt_items.len());
        for ti in tgt_is {
            if let Some(ti) = ti {
                let heap = heaps[ti as usize].take().filter(|h| h.len() >= min_nbrs);
                if let Some(heap) = heap {
                    let score: f32 = heap.iter().map(AccEntry::get_weight).sum();
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
    let size = array.len();
    CSRMatrix::from_arrow(array, size, size)
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

    // sort by descending score
    sims.sort_by_key(|(_i, s)| Reverse(NotNan::new(*s).unwrap()));
    // truncate if needed
    if let Some(limit) = save_nbrs {
        if limit > 0 {
            sims.truncate(limit as usize);
            sims.shrink_to_fit();
        }
    }
    sims
}
