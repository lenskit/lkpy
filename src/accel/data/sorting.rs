// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cmp::{Ordering, Reverse};

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrowPrimitiveType, Int32Array, PrimitiveArray, RecordBatch,
    },
    pyarrow::PyArrowType,
};
#[cfg(test)]
use ntest::*;
use ordered_float::{FloatCore, NotNan};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::slice::ParallelSliceMut;

use crate::match_array_type;
use crate::{arrow::checked_array, ok_or_pyerr};

const PAR_SORT_THRESHOLD: usize = 10_000;

/// Check if two columns of a table are properly-sorted COO.
#[pyfunction]
pub(super) fn is_sorted_coo<'py>(
    data: Vec<PyArrowType<RecordBatch>>,
    c1: &'py str,
    c2: &'py str,
) -> PyResult<bool> {
    let mut last = None;
    for PyArrowType(batch) in data {
        let col1 = ok_or_pyerr!(
            batch.column_by_name(c1),
            PyValueError,
            "unknown column: {}",
            c1
        )?;
        let col2 = ok_or_pyerr!(
            batch.column_by_name(c2),
            PyValueError,
            "unknown column: {}",
            c2
        )?;

        let col1: Int32Array = checked_array(c1, col1)?;
        let col2: Int32Array = checked_array(c2, col2)?;

        for i in 0..col1.len() {
            let v1 = col1.value(i);
            let v2 = col2.value(i);
            let k = (v1, v2);
            if let Some(lk) = last {
                if k <= lk {
                    // found a key out-of-order, we're done
                    return Ok(false);
                }
            }
            last = Some(k);
        }
    }

    // got this far, we're sorted
    Ok(true)
}

#[pyfunction]
pub(crate) fn argsort_descending<'py>(
    py: Python<'py>,
    scores: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let scores = make_array(scores.0);
    let array = py.detach(|| {
        let indices = match_array_type!(scores, {
            floating(arr) => argsort_float(arr),
            integer(arr) => argsort_int(arr),
        })?;

        PyResult::Ok(Int32Array::from(indices))
    })?;
    Ok(array.into_data().into())
}

fn argsort_float<T: ArrowPrimitiveType>(scores: &PrimitiveArray<T>) -> Vec<i32>
where
    T::Native: FloatCore,
{
    let sbuf = scores.values();

    let mut indices = Vec::with_capacity(scores.len());
    for (i, v) in scores.iter().enumerate() {
        if let Some(v) = v {
            if !v.is_nan() {
                indices.push(i as i32);
            }
        }
    }

    if scores.len() >= PAR_SORT_THRESHOLD {
        indices.par_sort_unstable_by_key(|i| Reverse(NotNan::new(sbuf[*i as usize]).unwrap()));
    } else {
        indices.sort_unstable_by_key(|i| Reverse(NotNan::new(sbuf[*i as usize]).unwrap()));
    }

    indices
}

fn argsort_int<T: ArrowPrimitiveType>(scores: &PrimitiveArray<T>) -> Vec<i32>
where
    T::Native: Ord,
{
    let sbuf = scores.values();

    let mut indices = Vec::with_capacity(scores.len());
    for (i, v) in scores.iter().enumerate() {
        if let Some(_v) = v {
            indices.push(i as i32);
        }
    }

    if scores.len() >= PAR_SORT_THRESHOLD {
        indices.par_sort_unstable_by_key(|i| Reverse(sbuf[*i as usize]));
    } else {
        indices.sort_unstable_by_key(|i| Reverse(sbuf[*i as usize]));
    }

    indices
}

#[pyfunction]
pub(crate) fn argtopn<'py>(
    py: Python<'py>,
    scores: PyArrowType<ArrayData>,
    n: usize,
) -> PyResult<PyArrowType<ArrayData>> {
    if n <= 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }

    let scores = make_array(scores.0);
    let array = py.detach(|| {
        let indices = match_array_type!(scores, {
            floating(arr) => argtopn_impl(arr, n),
            integer(arr) => argtopn_impl(arr, n),
        })?;

        PyResult::Ok(Int32Array::from(indices))
    })?;
    Ok(array.into_data().into())
}

fn argtopn_impl<T: ArrowPrimitiveType>(scores: &PrimitiveArray<T>, n: usize) -> Vec<i32>
where
    T::Native: PartialOrd,
{
    let sbuf = scores.values();

    let mut heap = IndirectHeap::create(n, sbuf);
    for i in 0..scores.len() {
        heap.insert(i as i32);
    }

    heap.topn_vec()
}

struct IndirectHeap<'v, V: PartialOrd> {
    size: usize,
    keys: Vec<i32>,
    values: &'v [V],
}

impl<'v, V: PartialOrd + Copy> IndirectHeap<'v, V> {
    fn create(size: usize, values: &'v [V]) -> Self {
        IndirectHeap {
            size,
            keys: Vec::with_capacity(size + 1),
            values,
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn _value_at(&self, idx: usize) -> V {
        self.values[self.keys[idx] as usize]
    }

    fn insert(&mut self, key: i32) {
        let n = self.keys.len();
        if n < self.size {
            // heap has space, add
            self.keys.push(key);
            self.upheap(n);
        } else {
            // heap is full, new value belongs — replace + adjust
            let kv = self.values[key as usize];
            match kv.partial_cmp(&self._value_at(0)) {
                Some(Ordering::Greater) => {
                    self.keys[0] = key;
                    self.downheap(0, self.size);
                }
                _ => (),
            }
        }
    }

    fn topn_vec(mut self) -> Vec<i32> {
        let mut n = self.keys.len();
        while n > 0 {
            n -= 1;
            self.keys.swap(0, n);
            self.downheap(0, n);
        }
        self.keys
    }

    fn downheap(&mut self, pos: usize, lim: usize) {
        let mut min = pos;
        let mut mv = self._value_at(min);
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;

        if left < lim {
            let lv = self._value_at(left);
            match lv.partial_cmp(&mv) {
                Some(Ordering::Less) => {
                    min = left;
                    mv = lv;
                }
                _ => (),
            }
        }
        if right < lim {
            let rv = self._value_at(right);
            match rv.partial_cmp(&mv) {
                Some(Ordering::Less) => {
                    min = right;
                }
                _ => (),
            }
        }

        if min != pos {
            self.keys.swap(pos, min);
            self.downheap(min, lim);
        }
    }

    fn upheap(&mut self, pos: usize) {
        if pos > 0 {
            let parent = (pos - 1) / 2;
            let pv = self._value_at(parent);
            let mv = self._value_at(pos);
            match pv.partial_cmp(&mv) {
                Some(Ordering::Greater) => {
                    self.keys.swap(pos, parent);
                    self.upheap(parent);
                }
                _ => (),
            }
        }
    }
}

#[test]
fn test_heap_empty() {
    let scores = [10];
    let heap = IndirectHeap::create(5, &scores);
    assert_eq!(heap.len(), 0);
    assert_eq!(heap.topn_vec().len(), 0);
}

#[test]
fn test_heap_one() {
    let scores = [10];
    let mut heap = IndirectHeap::create(5, &scores);
    heap.insert(0);
    assert_eq!(heap.len(), 1);
    let vals = heap.topn_vec();
    assert_eq!(vals.len(), 1);
    assert_eq!(&vals, &[0]);
}

#[test]
#[timeout(100)]
fn test_heap_two() {
    let scores = [10, 20];
    let mut heap = IndirectHeap::create(5, &scores);
    heap.insert(0);
    heap.insert(1);
    assert_eq!(heap.len(), 2);
    let vals = heap.topn_vec();
    assert_eq!(vals.len(), 2);
    assert_eq!(&vals, &[1, 0]);
}

#[test]
#[timeout(100)]
fn test_heap_two_alt() {
    let scores = [10, 20];
    let mut heap = IndirectHeap::create(5, &scores);
    heap.insert(1);
    heap.insert(0);
    assert_eq!(heap.len(), 2);
    let vals = heap.topn_vec();
    assert_eq!(vals.len(), 2);
    assert_eq!(&vals, &[1, 0]);
}

#[test]
#[timeout(100)]
fn test_heap_sort() {
    let scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut heap = IndirectHeap::create(5, &scores);
    for i in 0..scores.len() {
        heap.insert(i as i32);
    }
    assert_eq!(heap.len(), 5);
    let vals = heap.topn_vec();
    assert_eq!(vals.len(), 5);
    assert_eq!(&vals, &[9, 8, 7, 6, 5]);
}
