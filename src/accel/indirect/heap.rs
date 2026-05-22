// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;

#[cfg(test)]
use ntest::*;

/// Indirect min-heap for sorting and limiting.
pub struct IndirectMinHeap<K: Copy, V: PartialOrd + Copy, F: Fn(K) -> V> {
    size: usize,
    keys: Vec<K>,
    lookup: F,
}

impl<'f, K: Copy, V: PartialOrd + Copy, F: Fn(K) -> V> IndirectMinHeap<K, V, F> {
    /// Create a new min heap with the specified size and value lookup function.
    pub fn create(size: usize, lookup: F) -> Self {
        IndirectMinHeap {
            size,
            keys: Vec::with_capacity(size + 1),
            lookup,
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    fn value_for_position(&self, idx: usize) -> V {
        (self.lookup)(self.keys[idx])
    }

    /// Insert a new entry into the min-heap.
    pub fn insert(&mut self, key: K) {
        let n = self.keys.len();
        if n < self.size {
            // heap has space, add
            self.keys.push(key);
            self.upheap(n);
        } else {
            // heap is full, new value belongs — replace + adjust
            let kv = (self.lookup)(key);
            match kv.partial_cmp(&self.value_for_position(0)) {
                Some(Ordering::Greater) => {
                    self.keys[0] = key;
                    self.downheap(0, self.size);
                }
                _ => (),
            }
        }
    }

    /// Convert this heap to a sorted (descending) vector of keys.
    pub fn topn_vec(mut self) -> Vec<K> {
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
        let mut mv = self.value_for_position(min);
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;

        if left < lim {
            let lv = self.value_for_position(left);
            match lv.partial_cmp(&mv) {
                Some(Ordering::Less) => {
                    min = left;
                    mv = lv;
                }
                _ => (),
            }
        }
        if right < lim {
            let rv = self.value_for_position(right);
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
            let pv = self.value_for_position(parent);
            let mv = self.value_for_position(pos);
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
    let heap = IndirectMinHeap::create(5, |i: &usize| &scores[*i]);
    assert_eq!(heap.len(), 0);
    assert_eq!(heap.topn_vec().len(), 0);
}

#[test]
fn test_heap_one() {
    let scores = [10];
    let mut heap = IndirectMinHeap::create(5, |i: usize| scores[i]);
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
    let mut heap = IndirectMinHeap::create(5, |i: usize| scores[i]);
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
    let mut heap = IndirectMinHeap::create(5, |i: usize| scores[i]);
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
    let mut heap = IndirectMinHeap::create(5, |i: usize| scores[i]);
    for i in 0..scores.len() {
        heap.insert(i);
    }
    assert_eq!(heap.len(), 5);
    let vals = heap.topn_vec();
    assert_eq!(vals.len(), 5);
    assert_eq!(&vals, &[9, 8, 7, 6, 5]);
}
