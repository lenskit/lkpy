use std::collections::BinaryHeap;

use arrow::{
    array::{make_array, Array, ArrayData, Float32Array, Float32Builder, Int32Array},
    pyarrow::PyArrowType,
};
use ordered_float::NotNan;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{sparse::CSRMatrix, types::checked_array_convert};

/// Accumulate scores.
#[derive(Clone)]
enum ScoreAccumulator<T> {
    Empty,
    Partial(Vec<AccEntry<T>>),
    Full(BinaryHeap<AccEntry<T>>),
}

impl<T> Default for ScoreAccumulator<T> {
    fn default() -> Self {
        ScoreAccumulator::Empty
    }
}

impl ScoreAccumulator<()> {
    fn add_weight(&mut self, limit: usize, weight: f32) -> PyResult<()> {
        self.add_value(limit, weight, ())
    }
}

impl<T> ScoreAccumulator<T> {
    fn len(&self) -> usize {
        match self {
            ScoreAccumulator::Empty => 0,
            ScoreAccumulator::Partial(v) => v.len(),
            ScoreAccumulator::Full(h) => h.len(),
        }
    }

    #[inline(never)]
    fn heap_mut(&mut self) -> &mut BinaryHeap<AccEntry<T>> {
        match self {
            ScoreAccumulator::Full(h) => h,
            ScoreAccumulator::Empty => {
                let heap = BinaryHeap::new();
                *self = ScoreAccumulator::Full(heap);
                self.heap_mut()
            }
            ScoreAccumulator::Partial(vec) => {
                let mut heap = BinaryHeap::with_capacity(vec.len() + 1);
                while let Some(v) = vec.pop() {
                    heap.push(v);
                }
                *self = ScoreAccumulator::Full(heap);
                self.heap_mut()
            }
        }
    }

    #[inline(never)]
    fn vector_mut(&mut self, limit: usize) -> Option<&mut Vec<AccEntry<T>>> {
        match self {
            ScoreAccumulator::Empty => {
                // make a vector!
                let vec = Vec::with_capacity(limit);
                *self = ScoreAccumulator::Partial(vec);
                self.vector_mut(limit)
            }
            ScoreAccumulator::Partial(vec) if vec.len() < limit => Some(vec),
            _ => None,
        }
    }

    fn add_value(&mut self, limit: usize, weight: f32, value: T) -> PyResult<()> {
        let entry = AccEntry::new(weight, value)?;
        if let Some(vec) = self.vector_mut(limit) {
            vec.push(entry);
        } else {
            let heap = self.heap_mut();
            heap.push(entry);
            while heap.len() > limit {
                heap.pop();
            }
        }

        Ok(())
    }

    fn total_weight(&self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Full(heap) => heap.iter().map(AccEntry::get_weight).sum(),
            Self::Partial(vec) => vec.iter().map(AccEntry::get_weight).sum(),
        }
    }
}

impl ScoreAccumulator<f32> {
    fn weighted_sum(&self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Full(heap) => heap.iter().map(|a| a.weight * a.data).sum(),
            Self::Partial(vec) => vec.iter().map(|a| a.weight * a.data).sum(),
        }
    }
}

/// Entries in the accumulator heaps.
#[derive(Debug, Default, Clone, Copy)]
struct AccEntry<T> {
    weight: NotNan<f32>,
    data: T,
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

        let mut heaps: Vec<ScoreAccumulator<f32>> = vec![ScoreAccumulator::default(); sims.n_cols];

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

        let mut heaps: Vec<ScoreAccumulator<()>> = vec![ScoreAccumulator::default(); sims.n_cols];

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
    let size = array.len();
    CSRMatrix::from_arrow(array, size, size)
}
