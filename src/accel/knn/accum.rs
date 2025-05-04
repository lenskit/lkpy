// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accumulator for scores in k-NN.
use std::collections::BinaryHeap;

use arrow::array::{Float32Array, Float32Builder, Int32Array};
use ordered_float::NotNan;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Accumulate scores.
#[derive(Clone)]
pub(super) enum ScoreAccumulator<T> {
    Disabled,
    Empty,
    Partial(Vec<AccEntry<T>>),
    Full(BinaryHeap<AccEntry<T>>),
}

impl<T> Default for ScoreAccumulator<T> {
    fn default() -> Self {
        Self::Disabled
    }
}

impl ScoreAccumulator<()> {
    pub fn add_weight(&mut self, limit: usize, weight: f32) -> PyResult<()> {
        self.add_value(limit, weight, ())
    }
}

impl<T: Clone> ScoreAccumulator<T> {
    pub fn new_array(n: usize, active: &Int32Array) -> Vec<ScoreAccumulator<T>> {
        // create accumulators for all items, and enable the targets
        let mut heaps: Vec<ScoreAccumulator<T>> = vec![ScoreAccumulator::disabled(); n];
        for i in active.iter() {
            if let Some(i) = i {
                heaps[i as usize].enable()
            }
        }
        heaps
    }
}

impl<T> ScoreAccumulator<T> {
    /// Create a disabled score accumulator.
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// Enable a score accumulator.
    pub fn enable(&mut self) {
        match self {
            Self::Disabled => *self = Self::Empty,
            _ => (),
        }
    }

    pub fn enabled(&self) -> bool {
        match self {
            ScoreAccumulator::Disabled => false,
            _ => true,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ScoreAccumulator::Empty | ScoreAccumulator::Disabled => 0,
            ScoreAccumulator::Partial(v) => v.len(),
            ScoreAccumulator::Full(h) => h.len(),
        }
    }

    fn heap_mut(&mut self) -> &mut BinaryHeap<AccEntry<T>> {
        match self {
            ScoreAccumulator::Disabled => {
                panic!("mutable heaps not available on disabled accumulators")
            }
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

    pub fn add_value(&mut self, limit: usize, weight: f32, value: T) -> PyResult<()> {
        if self.enabled() {
            let entry = AccEntry::new(weight, value)?;
            if let Some(vec) = self.vector_mut(limit) {
                vec.push(entry);
            } else {
                let heap = self.heap_mut();
                if entry.weight > heap.peek().unwrap().weight {
                    heap.push(entry);
                    while heap.len() > limit {
                        heap.pop();
                    }
                }
            }
        }

        Ok(())
    }

    pub fn total_weight(&self) -> f32 {
        match self {
            Self::Empty | Self::Disabled => 0.0,
            Self::Full(heap) => heap.iter().map(AccEntry::get_weight).sum(),
            Self::Partial(vec) => vec.iter().map(AccEntry::get_weight).sum(),
        }
    }
}

impl ScoreAccumulator<f32> {
    pub fn weighted_sum(&self) -> f32 {
        match self {
            Self::Disabled | Self::Empty => 0.0,
            Self::Full(heap) => heap.iter().map(|a| a.weight * a.data).sum(),
            Self::Partial(vec) => vec.iter().map(|a| a.weight * a.data).sum(),
        }
    }
}

/// Entries in the accumulator heaps.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct AccEntry<T> {
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

pub(super) fn collect_items_averaged(
    heaps: &[ScoreAccumulator<f32>],
    tgt_is: &Int32Array,
    min_nbrs: usize,
) -> Float32Array {
    let mut out = Float32Builder::with_capacity(tgt_is.len());
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
    out.finish()
}

pub(super) fn collect_items_summed(
    heaps: &[ScoreAccumulator<()>],
    tgt_is: &Int32Array,
    min_nbrs: usize,
) -> Float32Array {
    let mut out = Float32Builder::with_capacity(tgt_is.len());
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
    out.finish()
}
