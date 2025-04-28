// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Row-column sets for quick masking.

use std::collections::HashSet;

use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use pyo3::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashSet};

use crate::sparse::{CSRStructure, CSR};

enum RCSEntry {
    Empty,
    Single(i32),
    Double(i32, i32),
    Many(FxHashSet<i32>),
}

impl Default for RCSEntry {
    fn default() -> Self {
        Self::Empty
    }
}

impl RCSEntry {
    fn insert(&mut self, val: i32, expected: usize) {
        match self {
            Self::Empty => *self = Self::Single(val),
            Self::Single(v1) => {
                *self = Self::Double(*v1, val);
            }
            Self::Double(v1, v2) => {
                let mut set = HashSet::with_capacity_and_hasher(expected, FxBuildHasher);
                set.insert(*v1);
                set.insert(*v2);
                set.insert(val);
                *self = Self::Many(set);
            }
            Self::Many(set) => {
                set.insert(val);
            }
        }
    }

    fn contains(&self, val: i32) -> bool {
        match self {
            Self::Empty => false,
            Self::Single(v) => *v == val,
            Self::Double(v1, v2) => *v1 == val || *v2 == val,
            Self::Many(s) => s.contains(&val),
        }
    }
}

#[pyclass]
pub struct RowColumnSet {
    sets: Vec<RCSEntry>,
}

impl RowColumnSet {
    pub(crate) fn contains_pair(&self, row: i32, col: i32) -> bool {
        self.sets[row as usize].contains(col)
    }
}

#[pymethods]
impl RowColumnSet {
    #[new]
    fn new(matrix: PyArrowType<ArrayData>) -> PyResult<Self> {
        let matrix = make_array(matrix.0);
        let matrix: CSRStructure<i32> = CSRStructure::from_arrow(matrix)?;

        let mut sets = Vec::with_capacity(matrix.len());

        for r in 0..matrix.len() {
            let (sp, ep) = matrix.extent(r);
            let n = (ep - sp) as usize;
            let mut set = RCSEntry::default();
            for ci in sp..ep {
                set.insert(matrix.col_inds.value(ci as usize), n);
            }
            sets.push(set)
        }

        Ok(RowColumnSet { sets })
    }

    #[pyo3(name = "contains_pair")]
    pub(crate) fn contains_pair_py(&self, row: i32, col: i32) -> bool {
        self.sets[row as usize].contains(col)
    }
}
